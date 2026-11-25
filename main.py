from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class ChatProvider(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> str: ...


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        app_name: str = "Telegram AI Bot",
        site_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self._api_key = api_key
        self._model = model
        self._app_name = app_name
        self._site_url = site_url
        self._endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self._timeout = timeout

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "X-Title": self._app_name,
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"AI provider returned HTTP {exc.code}: {self._error_detail(body)}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Could not reach AI provider: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderError("AI provider returned invalid JSON") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("AI provider response has no message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("AI provider returned an empty response")
        return content.strip()

    @staticmethod
    def _error_detail(body: str) -> str:
        try:
            data = json.loads(body)
            detail = data.get("error", {}).get("message")
            if detail:
                return str(detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        return body[:500] or "No error details provided"


@dataclass
class AIReply:
    text: str
    sources: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DealIdea:
    what: str
    deadline: str | None


class AICore:
    _RULES = (
        "Ты — ИИ-менеджер рабочих чатов строительной компании.\n"
        "История чата дана строками вида: [#<id> | <дата время> | <автор>] <текст>.\n"
        "Жёсткие правила:\n"
        "1. Чётко разделяй ситуации: разные объекты, клиенты, задачи и сделки — это разные ситуации. "
        "Никогда не смешивай договорённости и вопросы из разных ситуаций в один пункт.\n"
        "2. Каждый факт подтверждай ссылкой на исходное сообщение в формате [#<id>]. "
        "Используй только id, которые реально есть в истории. Ничего не выдумывай.\n"
        "3. Если информации не хватает — прямо скажи об этом, не додумывай.\n"
        "4. Отвечай по-русски, простым текстом, без Markdown-разметки.\n"
        '5. Верни строго JSON вида {"answer": "<текст>", "sources": [<id>, ...]}. '
        "Переносы строк внутри answer экранируй как \\n. Никакого текста вне JSON."
    )

    _SUMMARY_TASK = (
        "Сделай структурированную сводку по истории чата.\n"
        "Разбей её на отдельные ситуации (объект / клиент / задача). Для каждой ситуации укажи:\n"
        "- название ситуации;\n"
        "- договорённости: кто, о чём, к какому сроку [#id];\n"
        "- открытые вопросы и споры [#id];\n"
        "- что требует решения руководителя.\n"
        "В sources перечисли все id, на которые ты сослался."
    )

    _ASK_TASK = (
        "Ответь на вопрос пользователя строго по истории чата.\n"
        "Определи, к какой ситуации относится вопрос, и отвечай только по ней. "
        "Если вопрос затрагивает несколько ситуаций — ответь по каждой отдельно, не смешивая их. "
        "Не подмешивай договорённости из других ситуаций.\n"
        "Если в истории нет ответа — так и скажи.\n"
        "Вопрос: {question}"
    )

    _DEAL_TASK = (
        "Ты — детектор договорённостей в рабочем чате строительной компании.\n"
        "Тебе дают одно сообщение и его автора. Определи, содержит ли оно конкретное "
        "обещание или договорённость о работе: что сделать, поставить, завершить, передать.\n"
        "НЕ считаются договорённостью: вопросы, жалобы, приветствия, обсуждения без "
        "обязательства, команды бота.\n"
        'Верни строго JSON: {"is_deal": true или false, '
        '"what": "краткая суть обязательства до 120 символов", '
        '"deadline": "срок словами из сообщения или null"}.\n'
        "Никакого текста вне JSON."
    )

    def __init__(self, provider: ChatProvider) -> None:
        self._provider = provider

    def summarize(self, transcript: str) -> AIReply:
        return self._run(self._SUMMARY_TASK, transcript)

    def answer(self, transcript: str, question: str) -> AIReply:
        return self._run(self._ASK_TASK.format(question=question), transcript)

    def detect_deal(self, author: str, text: str) -> DealIdea | None:
        messages = [
            ChatMessage("system", self._DEAL_TASK),
            ChatMessage("user", f"Автор: {author}\nСообщение: {text}"),
        ]
        data = self._json(self._provider.complete(messages))
        if not isinstance(data, dict) or not data.get("is_deal"):
            return None
        what = data.get("what")
        if not isinstance(what, str) or not what.strip():
            return None
        deadline = data.get("deadline")
        if (
            not isinstance(deadline, str)
            or not deadline.strip()
            or deadline.strip().lower() in ("null", "none", "-")
        ):
            deadline = None
        return DealIdea(
            what=what.strip()[:160],
            deadline=deadline.strip() if deadline else None,
        )

    def _run(self, task: str, transcript: str) -> AIReply:
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("Transcript cannot be empty")
        messages = [
            ChatMessage("system", self._RULES + "\n\n" + task),
            ChatMessage("user", transcript),
        ]
        return self._parse(self._provider.complete(messages))

    @staticmethod
    def _clean(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text

    @staticmethod
    def _json(raw: str) -> Any:
        text = AICore._clean(raw)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse(raw: str) -> AIReply:
        data = AICore._json(raw)
        sources: list[int] = []
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            answer = data["answer"].strip()
            raw_sources = data.get("sources")
            if isinstance(raw_sources, list):
                for s in raw_sources:
                    if isinstance(s, int):
                        sources.append(s)
                    elif isinstance(s, str) and s.isdigit():
                        sources.append(int(s))
        else:
            answer = AICore._clean(raw)  # модель вернула не JSON — берём текст как есть
        for match in re.findall(r"\[#(\d+)\]", answer):
            mid = int(match)
            if mid not in sources:
                sources.append(mid)
        return AIReply(text=answer, sources=sources)