from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from main import AICore, OpenRouterProvider, ProviderError

logger = logging.getLogger(__name__)

# Добирать ли в память сообщения, пришедшие пока бот был выключен
# (они всё равно только после добавления бота в чат). Команды из простоя игнорируются всегда.
REMEMBER_BACKLOG_MESSAGES = True
MAX_TRANSCRIPT_CHARS = 60000
# Какие команды бот стирает из группы (нужны права админа на удаление сообщений)
DELETE_COMMANDS_IN_GROUP = {"/summary", "/ask", "/deals", "/confirm"}

# Дешёвый пре-фильтр перед вызовом ИИ-радара: сообщение похоже на обещание/срок
DEAL_HINTS = re.compile(
    r"обеща|сдела|сдадим|сдаю|успеем|успею|будет готов|готово будет|договор|завтра|"
    r"до (пн|вт|ср|чт|пт|сб|вс|понед|втор|сред|четв|пятн|субб|воскр|завтра|конца|недели|месяца|\d)|"
    r"к (завтра|пятниц|сред|конц|\d)",
    re.IGNORECASE,
)

COMMANDS_HELP = (
    "Команды:\n"
    "/summary — сводка по чату в личные сообщения\n"
    "/summary_md — сводка сюда, в чат\n"
    "/ask <вопрос> — ответ по истории в личные сообщения\n"
    "/ask_md <вопрос> — ответ сюда, в чат\n"
    "/deals — реестр договорённостей в личные сообщения\n"
    "/deals_md — реестр сюда, в чат\n"
    "/confirm <номер> — подтвердить договорённость\n"
    "Договорённости я ловлю в чате сам и фиксирую со ссылкой на сообщение."
)


def load_env_file() -> None:
    env_path = Path(__file__).resolve().with_name(".env")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid .env entry on line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


class TelegramError(RuntimeError):
    pass


class TelegramBot:
    def __init__(
        self,
        token: str,
        ai_core: AICore,
        poll_timeout: int = 30,
        log_limit: int = 5000,
    ) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._ai_core = ai_core
        self._poll_timeout = poll_timeout
        self._log_limit = log_limit
        self._state_path = Path(__file__).resolve().with_name("bot_state.json")
        self._chats: dict[int, dict[str, Any]] = {}
        self._user_last_chat: dict[int, int] = {}
        self._bot_id = 0
        self._bot_username = ""
        self._offset: int | None = None
        self._load_state()

    # ---------- состояние ----------

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path.read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        for chat_id, chat in (data.get("chats") or {}).items():
            self._chats[int(chat_id)] = chat
        for user_id, chat_id in (data.get("user_last_chat") or {}).items():
            self._user_last_chat[int(user_id)] = int(chat_id)

    def _save_state(self) -> None:
        data = {"chats": self._chats, "user_last_chat": self._user_last_chat}
        try:
            self._state_path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        except OSError as exc:
            logger.error("Could not save state: %s", exc)

    # ---------- главный цикл ----------

    def run(self) -> None:
        while True:
            try:
                me = self._api_request("getMe", {})
                break
            except TelegramError as exc:
                logger.error("getMe failed: %s — retry in 5s (проверь VPN/интернет)", exc)
                time.sleep(5)
        self._bot_id = me["id"]
        self._bot_username = me.get("username", "")
        logger.info("Bot @%s started", self._bot_username)
        self._drain_backlog()
        logger.info("Listening for new messages")
        while True:
            payload: dict[str, Any] = {
                "timeout": self._poll_timeout,
                "allowed_updates": ["message", "my_chat_member"],
            }
            if self._offset is not None:
                payload["offset"] = self._offset
            try:
                updates = self._api_request(
                    "getUpdates", payload, timeout=self._poll_timeout + 10
                )
            except TelegramError as exc:
                logger.error("Telegram polling failed: %s", exc)
                time.sleep(5)
                continue
            for update in updates:
                self._offset = update["update_id"] + 1
                self.handle_update(update, stale=False)

    def _drain_backlog(self) -> None:
        """Слить апдейты, накопившиеся пока бот был выключен.
        Команды из них игнорируются, сообщения — только в память."""
        while True:
            payload: dict[str, Any] = {
                "timeout": 0,
                "allowed_updates": ["message", "my_chat_member"],
            }
            if self._offset is not None:
                payload["offset"] = self._offset
            try:
                updates = self._api_request("getUpdates", payload, timeout=15)
            except TelegramError as exc:
                logger.error("Backlog drain failed: %s", exc)
                return
            if not updates:
                return
            for update in updates:
                self._offset = update["update_id"] + 1
                self.handle_update(update, stale=True)
            logger.info("Processed %d stale updates (commands skipped)", len(updates))

    # ---------- разбор апдейтов ----------

    def handle_update(self, update: dict[str, Any], stale: bool = False) -> None:
        member = update.get("my_chat_member")
        if isinstance(member, dict):
            self._handle_membership(member, stale)
            return
        message = update.get("message")
        if isinstance(message, dict):
            self._handle_message(message, stale)

    def _handle_membership(self, member: dict[str, Any], stale: bool) -> None:
        new = member.get("new_chat_member") or {}
        user = new.get("user") or {}
        if user.get("id") != self._bot_id:
            return
        chat = member.get("chat") or {}
        status = new.get("status")
        if status in ("member", "administrator"):
            self._on_joined(chat, member.get("date") or int(time.time()), stale)
        elif status in ("left", "kicked"):
            self._on_left(chat.get("id"))

    def _on_joined(self, chat: dict[str, Any], date: int, stale: bool) -> None:
        chat_id = chat.get("id")
        if chat_id is None:
            return
        existing = self._chats.get(chat_id)
        if existing is not None and existing.get("joined_at") == date:
            return  # дубль события (message + my_chat_member)
        # память чата начинается строго с момента добавления
        self._chats[chat_id] = {
            "id": chat_id,
            "title": chat.get("title") or "",
            "username": chat.get("username"),
            "type": chat.get("type"),
            "joined_at": date,
            "messages": [],
            "deals": [],
            "deal_seq": 0,
        }
        self._save_state()
        logger.info("Joined chat %s, memory starts now", chat_id)
        if not stale:
            self._try_send(
                chat_id,
                "Я подключился и запоминаю сообщения с этого момента.\n\n" + COMMANDS_HELP,
            )

    def _on_left(self, chat_id: int | None) -> None:
        if chat_id is not None and chat_id in self._chats:
            del self._chats[chat_id]
            self._save_state()
            logger.info("Left chat %s, memory cleared", chat_id)

    def _handle_message(self, message: dict[str, Any], stale: bool) -> None:
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return
        chat_type = chat.get("type")
        sender = message.get("from") or {}
        user_id = sender.get("id")

        for member in message.get("new_chat_members") or []:
            if member.get("id") == self._bot_id:
                self._on_joined(chat, message.get("date") or int(time.time()), stale)
        left = message.get("left_chat_member") or {}
        if left.get("id") == self._bot_id:
            self._on_left(chat_id)
            return

        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        text = text.strip()

        if chat_type == "private":
            if text.startswith("/"):
                if stale:
                    logger.info("Ignored stale private command")
                    return
                self._safe_command(chat, sender, text)
            return

        if chat_type not in ("group", "supergroup"):
            return

        state = self._chats.get(chat_id)
        if state is None:
            # бот был в чате ещё до первого запуска этого кода:
            # считаем моментом добавления первое увиденное сообщение
            state = {
                "id": chat_id,
                "title": chat.get("title") or "",
                "username": chat.get("username"),
                "type": chat_type,
                "joined_at": message.get("date") or int(time.time()),
                "messages": [],
                "deals": [],
                "deal_seq": 0,
            }
            self._chats[chat_id] = state
            self._save_state()
        if chat.get("title"):
            state["title"] = chat["title"]
        if chat.get("username"):
            state["username"] = chat["username"]

        date = message.get("date") or 0
        joined_at = state.get("joined_at")
        if joined_at is not None and date < joined_at:
            return  # сообщение раньше добавления бота — не запоминаем

        if user_id is not None and self._user_last_chat.get(user_id) != chat_id:
            self._user_last_chat[user_id] = chat_id
            self._save_state()

        if text.startswith("/"):
            if stale:
                logger.info("Ignored stale command in chat %s", chat_id)
                return
            base = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
            if base in DELETE_COMMANDS_IN_GROUP:
                self._delete_message(chat_id, message.get("message_id"))
            self._safe_command(chat, sender, text)
            return

        if stale and not REMEMBER_BACKLOG_MESSAGES:
            return
        self._record_message(chat_id, message)
        if not stale:
            self._radar(chat_id, message)

    def _record_message(self, chat_id: int, message: dict[str, Any]) -> None:
        sender = message.get("from") or {}
        entry = {
            "message_id": message.get("message_id"),
            "date": message.get("date"),
            "from": self._sender_name(sender),
            "text": message.get("text", ""),
        }
        log = self._chats[chat_id]["messages"]
        log.append(entry)
        if len(log) > self._log_limit:
            del log[: len(log) - self._log_limit]
        self._save_state()

    @staticmethod
    def _sender_name(sender: dict[str, Any]) -> str:
        return (
            sender.get("username")
            or " ".join(
                p for p in (sender.get("first_name"), sender.get("last_name")) if p
            )
            or str(sender.get("id", "unknown"))
        )

    # ---------- радар договорённостей ----------

    def _radar(self, chat_id: int, message: dict[str, Any]) -> None:
        """Пассивно ловит обещания в чате. Никогда не роняет основной поток."""
        try:
            text = (message.get("text") or "").strip()
            sender = message.get("from") or {}
            if not text or sender.get("is_bot") or not DEAL_HINTS.search(text):
                return
            state = self._chats.get(chat_id)
            if state is None:
                return
            author = self._sender_name(sender)
            idea = self._ai_core.detect_deal(author, text)
            if idea is None:
                return
            deal = {
                "n": state.get("deal_seq", 0) + 1,
                "author": author,
                "author_id": sender.get("id"),
                "what": idea.what,
                "deadline": idea.deadline,
                "source_id": message.get("message_id"),
                "status": "pending",
                "confirmed_by": None,
                "card_id": None,
            }
            state["deal_seq"] = deal["n"]
            state.setdefault("deals", []).append(deal)
            self._save_state()
            # карточка остаётся в чате...
            result = self._try_send_html(chat_id, self._deal_card(deal, state, html=True))
            if isinstance(result, dict):
                deal["card_id"] = result.get("message_id")
                self._save_state()
            # ...и дублируется в лс автору обещания
            if deal["author_id"] is not None:
                self._try_send(deal["author_id"], self._deal_card(deal, state, html=False))
            logger.info("Deal #%s registered in chat %s", deal["n"], chat_id)
        except ProviderError as exc:
            logger.warning("Deal radar AI failed: %s", exc)
        except Exception:
            logger.exception("Deal radar unexpected error")

    def _deal_card(self, deal: dict[str, Any], state: dict[str, Any], html: bool) -> str:
        esc = self._escape if html else (lambda s: str(s))
        link = (
            self._message_link(state, deal["source_id"])
            if deal.get("source_id") is not None
            else None
        )
        if html and link:
            source = f'<a href="{link}">исходное сообщение</a>'
        else:
            source = link or "ссылка недоступна"
        if deal["status"] == "confirmed":
            head = f"✅ Договорённость №{deal['n']} — подтверждена"
            extra = f"Подтвердил: {esc(deal['confirmed_by'])}"
        else:
            head = f"🤝 Договорённость №{deal['n']} — ожидает подтверждения"
            extra = f"Подтвердить: /confirm {deal['n']}"
        return "\n".join(
            [
                head,
                f"Кто: {esc(deal['author'])}",
                f"Что: {esc(deal['what'])}",
                f"Срок: {esc(deal['deadline'] or 'не указан')}",
                extra,
                f"Источник: {source}",
            ]
        )

    def _deals_report(self, state: dict[str, Any], html: bool) -> str:
        deals = state.get("deals") or []
        if not deals:
            return "Пока не зафиксировал ни одной договорённости в этом чате."
        esc = self._escape if html else (lambda s: str(s))
        lines = [f"📋 Договорённости чата «{esc(state.get('title') or state['id'])}»:"]
        for d in deals:
            status = "✅ подтверждена" if d["status"] == "confirmed" else "⏳ ожидает"
            link = (
                self._message_link(state, d["source_id"])
                if d.get("source_id") is not None
                else None
            )
            src = f'<a href="{link}">источник</a>' if (html and link) else (link or "")
            lines.append(
                f"№{d['n']} [{status}] {esc(d['author'])} — {esc(d['what'])}; "
                f"срок: {esc(d['deadline'] or 'не указан')}. {src}"
            )
        lines.append("Подтвердить: /confirm <номер>")
        return "\n".join(lines)

    def _confirm(
        self,
        state: dict[str, Any],
        target_id: int,
        sender: dict[str, Any],
        argument: str,
        in_private: bool,
        source_chat_id: int,
    ) -> None:
        user_id = sender.get("id")
        # ошибки и фидбек — в лс (или в личку, если команда оттуда), чтобы не сорить в группе
        feedback_id = source_chat_id if in_private else (user_id if user_id is not None else target_id)
        if not argument.isdigit():
            self._try_send(feedback_id, "Использование: /confirm <номер договорённости>")
            return
        n = int(argument)
        deal = next((d for d in state.get("deals") or [] if d["n"] == n), None)
        if deal is None:
            self._try_send(feedback_id, f"Договорённость №{n} не найдена. Список: /deals")
            return
        if deal["status"] == "confirmed":
            self._try_send(
                feedback_id,
                f"Договорённость №{n} уже подтверждена ({deal['confirmed_by']}).",
            )
            return
        deal["status"] = "confirmed"
        deal["confirmed_by"] = self._sender_name(sender)
        self._save_state()

        # обновлённая карточка остаётся в чате (редактируем старую на месте)
        card_html = self._deal_card(deal, state, html=True)
        edited = False
        if deal.get("card_id"):
            try:
                self._api_request(
                    "editMessageText",
                    {
                        "chat_id": target_id,
                        "message_id": deal["card_id"],
                        "text": card_html,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                edited = True
            except TelegramError as exc:
                logger.warning("Could not edit deal card: %s", exc)
        if not edited:
            self._try_send_html(target_id, card_html)

        # дубль в лс автору и подтвердившему
        for rid in {deal.get("author_id"), user_id}:
            if rid is not None:
                self._try_send(rid, self._deal_card(deal, state, html=False))

    # ---------- команды ----------

    def _safe_command(self, chat: dict[str, Any], sender: dict[str, Any], text: str) -> None:
        chat_id = chat["id"]
        try:
            self._command(chat, sender, text)
        except ProviderError as exc:
            logger.error("AI request failed: %s", exc)
            self._try_send(chat_id, "ИИ-сервис сейчас недоступен, попробуй позже.")
        except TelegramError as exc:
            logger.error("Telegram request failed: %s", exc)
        except Exception:
            logger.exception("Unexpected error while processing a command")
            self._try_send(chat_id, "Что-то пошло не так при обработке команды.")

    def _command(self, chat: dict[str, Any], sender: dict[str, Any], text: str) -> None:
        parts = text.split(maxsplit=1)
        command = parts[0].split("@", 1)[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        user_id = sender.get("id")
        source_chat_id = chat["id"]
        in_private = chat.get("type") == "private"

        if command == "/start":
            if in_private:
                self._send_text(
                    source_chat_id,
                    "Привет! Теперь я могу писать тебе в личку.\n\n" + COMMANDS_HELP,
                )
            else:
                self._send_text(source_chat_id, COMMANDS_HELP)
            return
        if command == "/help":
            self._send_text(source_chat_id, COMMANDS_HELP)
            return
        if command not in (
            "/summary", "/summary_md", "/ask", "/ask_md",
            "/deals", "/deals_md", "/confirm",
        ):
            return

        # историю какого чата анализируем
        if in_private:
            target_id = self._user_last_chat.get(user_id)
            if target_id is None or target_id not in self._chats:
                self._send_text(
                    source_chat_id,
                    "Я пока не знаю твой рабочий чат. Напиши команду в группе, где я состою.",
                )
                return
        else:
            target_id = source_chat_id
        state = self._chats[target_id]
        to_group = command.endswith("_md")

        if command == "/confirm":
            self._confirm(state, target_id, sender, argument, in_private, source_chat_id)
            return

        if command in ("/deals", "/deals_md"):
            dest_id = target_id if to_group else (user_id if user_id is not None else target_id)
            try:
                self._send_html(dest_id, self._deals_report(state, html=True))
            except TelegramError as exc:
                if not to_group and self._is_dm_blocked(exc):
                    self._send_text(
                        source_chat_id,
                        f"Не могу написать тебе в личку: открой @{self._bot_username}, "
                        "нажми «Запустить» (Start) и повтори команду.",
                    )
                    return
                raise
            if in_private and dest_id != source_chat_id:
                self._try_send(source_chat_id, "📬 Ответ отправил в рабочий чат.")
            return

        dest_id = target_id if to_group else (user_id if user_id is not None else target_id)

        if command in ("/ask", "/ask_md") and not argument:
            self._send_text(source_chat_id, "Использование: /ask <вопрос> или /ask_md <вопрос>")
            return
        if not state["messages"]:
            self._send_text(source_chat_id, "Пока нет сообщений с момента моего добавления в чат.")
            return

        transcript = self._transcript(state)
        try:
            self._api_request("sendChatAction", {"chat_id": dest_id, "action": "typing"})
        except TelegramError:
            pass

        if command in ("/summary", "/summary_md"):
            reply = self._ai_core.summarize(transcript)
        else:
            reply = self._ai_core.answer(transcript, argument)

        answer_text, sources_html = self._render(reply, state)
        header = ""
        if in_private or dest_id != target_id:
            header = f"Чат: {state.get('title') or target_id}\n\n"
        try:
            self._send_text(dest_id, header + answer_text)
            if sources_html:
                self._send_html(dest_id, sources_html)
        except TelegramError as exc:
            if not to_group and self._is_dm_blocked(exc):
                self._send_text(
                    source_chat_id,
                    f"Не могу написать тебе в личку: открой @{self._bot_username}, "
                    "нажми «Запустить» (Start) и повтори команду.",
                )
                return
            raise
        if in_private and dest_id != source_chat_id:
            self._try_send(source_chat_id, "📬 Ответ отправил в рабочий чат.")

    @staticmethod
    def _is_dm_blocked(exc: TelegramError) -> bool:
        s = str(exc).lower()
        return any(w in s for w in ("403", "initiate", "blocked", "deactivated"))

    # ---------- ответы со ссылками ----------

    def _transcript(self, state: dict[str, Any]) -> str:
        lines = []
        for e in state["messages"]:
            if not e.get("text"):
                continue
            lines.append(
                f"[#{e['message_id']} | {self._fmt_time(e.get('date'))} | {e['from']}] {e['text']}"
            )
        transcript = "\n".join(lines)
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[-MAX_TRANSCRIPT_CHARS:]
            transcript = transcript[transcript.find("\n") + 1 :]
        return transcript

    def _render(self, reply: Any, state: dict[str, Any]) -> tuple[str, str]:
        known = {
            e["message_id"]: e
            for e in state["messages"]
            if e.get("message_id") is not None
        }
        ids: list[int] = []
        for mid in reply.sources:
            if mid in known and mid not in ids:
                ids.append(mid)
        index = {mid: i + 1 for i, mid in enumerate(ids)}

        def repl(match: re.Match) -> str:
            n = index.get(int(match.group(1)))
            return f"[{n}]" if n else ""

        text = re.sub(r"\[#(\d+)\]", repl, reply.text).strip()
        text = re.sub(r"[ \t]{2,}", " ", text)
        if not text:
            text = "ИИ вернул пустой ответ, попробуй ещё раз."

        lines = []
        for mid in ids:
            e = known[mid]
            who = self._escape(e["from"])
            when = self._fmt_time(e.get("date"))
            link = self._message_link(state, mid)
            if link:
                label = f'{index[mid]}. <a href="{link}">{who}, {when}</a>'
            else:
                label = f"{index[mid]}. {who}, {when} (ссылка недоступна в этом типе чата)"
            quote = self._escape((e.get("text") or "")[:90])
            lines.append(f"{label}: «{quote}»")
        html = "<b>Источники:</b>\n" + "\n".join(lines) if lines else ""
        return text, html

    @staticmethod
    def _message_link(state: dict[str, Any], message_id: int) -> str | None:
        username = state.get("username")
        if username:
            return f"https://t.me/{username}/{message_id}"
        chat_id = str(state.get("id", ""))
        if chat_id.startswith("-100"):
            return f"https://t.me/c/{chat_id[4:]}/{message_id}"
        return None

    @staticmethod
    def _fmt_time(ts: Any) -> str:
        if not ts:
            return "??"
        return datetime.fromtimestamp(ts).strftime("%d.%m %H:%M")

    @staticmethod
    def _escape(text: Any) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # ---------- отправка и удаление ----------

    def _delete_message(self, chat_id: int, message_id: int | None) -> bool:
        if message_id is None:
            return False
        try:
            self._api_request(
                "deleteMessage",
                {"chat_id": chat_id, "message_id": message_id},
            )
            return True
        except TelegramError as exc:
            logger.warning("Could not delete command message: %s", exc)
            return False

    def _try_send(self, chat_id: int | str, text: str) -> None:
        try:
            self._send_text(chat_id, text)
        except TelegramError as exc:
            logger.error("Could not send message: %s", exc)

    def _try_send_html(self, chat_id: int | str, html: str) -> Any:
        try:
            return self._send_html(chat_id, html)
        except TelegramError as exc:
            logger.error("Could not send HTML message: %s", exc)
            return None

    def _send_text(self, chat_id: int | str, text: str) -> None:
        for start in range(0, len(text), 4096):
            self._api_request(
                "sendMessage",
                {"chat_id": chat_id, "text": text[start : start + 4096]},
            )

    def _send_html(self, chat_id: int | str, html: str) -> Any:
        result = None
        chunk = ""
        for line in html.split("\n"):
            if len(chunk) + len(line) + 1 > 3500:
                result = self._api_request(
                    "sendMessage",
                    {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML",
                     "disable_web_page_preview": True},
                )
                chunk = ""
            chunk += line + "\n"
        if chunk.strip():
            result = self._api_request(
                "sendMessage",
                {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML",
                 "disable_web_page_preview": True},
            )
        return result

    def _api_request(self, method: str, payload: dict[str, Any], timeout: float = 15) -> Any:
        request = Request(
            f"{self._base_url}/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise TelegramError(f"HTTP {exc.code}: {body[:500]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TelegramError(f"Connection failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TelegramError("Telegram returned invalid JSON") from exc
        if not isinstance(data, dict) or not data.get("ok"):
            description = data.get("description", "Unknown Telegram API error")
            raise TelegramError(description)
        return data.get("result")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env_file()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    missing = [
        name
        for name, value in {
            "TELEGRAM_BOT_TOKEN": token,
            "OPENROUTER_API_KEY": openrouter_api_key,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} in the .env file before starting the bot.")
    provider = OpenRouterProvider(
        api_key=openrouter_api_key,
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        app_name=os.getenv("OPENROUTER_APP_NAME", "Telegram AI Bot"),
        site_url=os.getenv("OPENROUTER_SITE_URL") or None,
    )
    TelegramBot(token, AICore(provider)).run()


if __name__ == "__main__":
    main()