from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from main import AICore, OpenRouterProvider, ProviderError


logger = logging.getLogger(__name__)


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
        self._message_log: dict[int, list[dict[str, Any]]] = {}
        self.dumped_messages: dict[int, list[dict[str, Any]]] = {}

    def run(self) -> None:
        offset: int | None = None
        logger.info("Telegram bot started")

        while True:
            payload: dict[str, Any] = {
                "timeout": self._poll_timeout,
                "allowed_updates": ["message"],
            }
            if offset is not None:
                payload["offset"] = offset

            try:
                updates = self._api_request(
                    "getUpdates",
                    payload,
                    timeout=self._poll_timeout + 10,
                )
            except TelegramError as exc:
                logger.error("Telegram polling failed: %s", exc)
                time.sleep(5)
                continue

            for update in updates:
                offset = update["update_id"] + 1
                self.handle_update(update)

    def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return

        text = message.get("text")
        chat = message.get("chat")
        if not isinstance(text, str) or not text.strip() or not isinstance(chat, dict):
            return

        chat_id = chat.get("id")
        if chat_id is None:
            return

        if chat.get("type") in ("group", "supergroup"):
            self._record_message(chat_id, message)

        try:
            self._handle_text(chat_id, text.strip())
        except ProviderError as exc:
            logger.error("AI request failed: %s", exc)
            self._send_text(
                chat_id,
                "The AI service is currently unavailable. Please try again later.",
            )
        except TelegramError as exc:
            logger.error("Telegram request failed: %s", exc)
        except Exception:
            logger.exception("Unexpected error while processing an update")
            try:
                self._send_text(
                    chat_id,
                    "Something went wrong while processing your message.",
                )
            except TelegramError as exc:
                logger.error("Could not send error message: %s", exc)

    def _handle_text(self, chat_id: int | str, text: str) -> None:
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        conversation_id = str(chat_id)

        if command == "/start":
            self._send_text(
                chat_id,
                "Hello! Send me a message and I will forward it to the AI core.",
            )
            return
        if command == "/help":
            self._send_text(
                chat_id,
                "Send any text to chat with the AI. Use /reset to clear this chat's history. "
                "In groups, all messages are collected automatically; use /dump to save them to memory.",
            )
            return
        if command == "/reset":
            self._ai_core.reset(conversation_id)
            self._send_text(chat_id, "Conversation history cleared.")
            return
        if command == "/dump":
            messages = list(self._message_log.get(chat_id, []))
            self.dumped_messages[chat_id] = messages
            self._send_text(
                chat_id,
                f"Saved {len(messages)} messages from this chat to memory.",
            )
            return

        self._api_request(
            "sendChatAction",
            {"chat_id": chat_id, "action": "typing"},
        )
        reply = self._ai_core.respond(conversation_id, text)
        self._send_text(chat_id, reply)

    def _record_message(self, chat_id: int, message: dict[str, Any]) -> None:
        sender = message.get("from") or {}
        name = (
            sender.get("username")
            or " ".join(
                part
                for part in (sender.get("first_name"), sender.get("last_name"))
                if part
            )
            or str(sender.get("id", "unknown"))
        )
        entry = {
            "message_id": message.get("message_id"),
            "date": message.get("date"),
            "from": name,
            "text": message.get("text", ""),
        }
        log = self._message_log.setdefault(chat_id, [])
        log.append(entry)
        if len(log) > self._log_limit:
            del log[: len(log) - self._log_limit]

    def _send_text(self, chat_id: int | str, text: str) -> None:
        for start in range(0, len(text), 4096):
            self._api_request(
                "sendMessage",
                {"chat_id": chat_id, "text": text[start : start + 4096]},
            )

    def _api_request(
        self,
        method: str,
        payload: dict[str, Any],
        timeout: float = 15,
    ) -> Any:
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
    ai_core = AICore(provider)
    TelegramBot(token, ai_core).run()


if __name__ == "__main__":
    main()
