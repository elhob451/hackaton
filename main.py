from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


class ChatProvider(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> str:
        ...


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
                {"role": message.role, "content": message.content}
                for message in messages
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


class AICore:
    def __init__(
        self,
        provider: ChatProvider,
        system_prompt: str = "You are a chat assistant, you get chat history from corp chat and then you summarize it",
        history_size: int = 20,
    ) -> None:
        if history_size < 2:
            raise ValueError("history_size must be at least 2")
        self._provider = provider
        self._system_prompt = system_prompt
        self._history_size = history_size
        self._histories: dict[str, list[ChatMessage]] = {}

    def respond(self, conversation_id: str, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Message text cannot be empty")

        history = self._histories.get(conversation_id, [])
        user_message = ChatMessage(role="user", content=text)
        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            *history,
            user_message,
        ]
        reply = self._provider.complete(messages)
        assistant_message = ChatMessage(role="assistant", content=reply)
        self._histories[conversation_id] = [
            *history,
            user_message,
            assistant_message,
        ][-self._history_size :]
        return reply

    def reset(self, conversation_id: str) -> None:
        self._histories.pop(conversation_id, None)
