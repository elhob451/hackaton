from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from telethon.errors import RPCError

from main import AICore, OpenRouterProvider, ProviderError
from user_bot import UserBotError, UserBotManager


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
    _PRIVATE_ONLY_COMMANDS = {
        "/login",
        "/code",
        "/password",
        "/logout",
        "/folders",
        "/folder",
        "/limit",
        "/status",
    }
    _SENSITIVE_COMMANDS = {"/login", "/code", "/password"}

    def __init__(
        self,
        token: str,
        ai_core: AICore,
        user_bot_manager: UserBotManager,
        poll_timeout: int = 30,
        log_limit: int = 5000,
    ) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._ai_core = ai_core
        self._manager = user_bot_manager
        self._poll_timeout = poll_timeout
        self._log_limit = log_limit
        self._message_log: dict[int, list[dict[str, Any]]] = {}
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._bot_username: str | None = None

    async def run(self) -> None:
        offset: int | None = None
        try:
            me = await self._api_request_async("getMe", {})
            if isinstance(me, dict) and me.get("username"):
                self._bot_username = str(me["username"])
        except TelegramError as exc:
            logger.warning("Could not fetch bot username: %s", exc)

        logger.info("Telegram bot started (username=%s)", self._bot_username or "unknown")

        try:
            while True:
                payload: dict[str, Any] = {
                    "timeout": self._poll_timeout,
                    "allowed_updates": ["message"],
                }
                if offset is not None:
                    payload["offset"] = offset

                try:
                    updates = await self._api_request_async(
                        "getUpdates",
                        payload,
                        timeout=self._poll_timeout + 10,
                    )
                except TelegramError as exc:
                    logger.error("Telegram polling failed: %s", exc)
                    await asyncio.sleep(5)
                    continue

                for update in updates:
                    offset = update["update_id"] + 1
                    await self.handle_update(update)
        finally:
            for task in list(self._active_tasks):
                task.cancel()
            if self._active_tasks:
                await asyncio.gather(*self._active_tasks, return_exceptions=True)

    async def handle_update(self, update: dict[str, Any]) -> None:
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

        chat_type = str(chat.get("type", ""))
        message_id = message.get("message_id")
        sender = message.get("from") or {}
        sender_id = sender.get("id")

        self._record_message(chat_id, message)

        try:
            await self._handle_text(chat_id, chat_type, message_id, sender_id, text.strip())
        except (UserBotError, ValueError) as exc:
            await self._send_text_async(chat_id, f"Ошибка: {exc}")
        except ProviderError as exc:
            logger.error("AI request failed: %s", exc)
            await self._send_text_async(
                chat_id,
                "Сервис ИИ временно недоступен. Попробуйте позже.",
            )
        except RPCError as exc:
            logger.error("Telethon RPC error: %s", exc)
            await self._send_text_async(chat_id, f"Ошибка Telegram API: {exc}")
        except TelegramError as exc:
            logger.error("Telegram request failed: %s", exc)
        except Exception:
            logger.exception("Unexpected error while processing an update")
            try:
                await self._send_text_async(
                    chat_id,
                    "Произошла непредвиденная ошибка при обработке сообщения.",
                )
            except TelegramError as exc:
                logger.error("Could not send error message: %s", exc)

    async def _handle_text(
        self,
        chat_id: int | str,
        chat_type: str,
        message_id: int | None,
        sender_id: int | None,
        text: str,
    ) -> None:
        parts = text.split(maxsplit=1)
        command = parts[0].split("@", 1)[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command in {"/start", "/help"}:
            await self._send_help(chat_id, chat_type)
            return

        if command == "/summary":
            await self._handle_chat_summary(chat_id)
            return

        if command not in self._PRIVATE_ONLY_COMMANDS:
            return

        if chat_type != "private":
            if command in self._SENSITIVE_COMMANDS and message_id is not None:
                try:
                    await self._api_request_async(
                        "deleteMessage",
                        {"chat_id": chat_id, "message_id": message_id},
                    )
                except TelegramError:
                    pass
            bot_ref = f"@{self._bot_username}" if self._bot_username else "боту"
            await self._send_text_async(
                chat_id,
                f"Для безопасности команда {command} работает только в личных сообщениях — напишите {bot_ref}.",
            )
            return

        if not isinstance(sender_id, int):
            raise UserBotError("Не удалось определить ваш Telegram ID.")

        if command == "/login":
            reply = await self._manager.start_login(sender_id, args)
            await self._send_text_async(chat_id, reply)
        elif command == "/code":
            reply = await self._manager.submit_code(sender_id, args)
            await self._send_text_async(chat_id, reply)
        elif command == "/password":
            reply = await self._manager.submit_password(sender_id, args)
            await self._send_text_async(chat_id, reply)
        elif command == "/logout":
            await self._manager.logout(sender_id)
            await self._send_text_async(
                chat_id,
                "Вы вышли из аккаунта. Сессия завершена и удалена с сервера.",
            )
        elif command == "/folders":
            await self._handle_folders(chat_id, sender_id)
        elif command == "/folder":
            await self._handle_folder_summary(chat_id, sender_id, args)
        elif command == "/limit":
            await self._handle_limit(chat_id, sender_id, args)
        elif command == "/status":
            await self._handle_status(chat_id, sender_id)

    async def _send_help(self, chat_id: int | str, chat_type: str) -> None:
        bot_ref = f"@{self._bot_username}" if self._bot_username else "боту в ЛС"
        if chat_type != "private":
            text = (
                "Команды в группе:\n"
                "/summary — сделать саммари последних сообщений этого чата\n\n"
                f"Чтобы подключить свои папки Telegram и делать по ним саммари, откройте личный чат с {bot_ref} и нажмите /start."
            )
            await self._send_text_async(chat_id, text)
            return

        text = (
            "Как пользоваться ботом:\n\n"
            "1. Подключите свой аккаунт Telegram:\n"
            "/login +79991234567 — отправить код входа\n"
            "/code 12-34-5 — подтвердить код (с пробелами или дефисами)\n"
            "/password <пароль> — если включена двухэтапная аутентификация\n\n"
            "2. Работа с папками:\n"
            "/folders — список ваших папок Telegram\n"
            "/folder <ID или название> [--limit N] — саммари чатов из папки\n"
            "/limit [число] — лимит сообщений на чат (по умолчанию 100)\n\n"
            "3. Управление сессией:\n"
            "/status — статус подключения\n"
            "/logout — выйти и полностью удалить вашу сессию с сервера\n\n"
            "В общих группах также работает /summary — саммари сообщений группы."
        )
        await self._send_text_async(chat_id, text)

    async def _handle_chat_summary(self, chat_id: int | str) -> None:
        messages = self._message_log.get(chat_id, [])
        if not messages:
            await self._send_text_async(
                chat_id,
                "Пока нет сохраненных сообщений в этом чате для саммари.",
            )
            return

        transcript = "\n".join(
            f"{entry['from']}: {entry['text']}" for entry in messages if entry["text"]
        )
        if not transcript.strip():
            await self._send_text_async(
                chat_id,
                "Пока нет сохраненных сообщений в этом чате для саммари.",
            )
            return

        await self._api_request_async(
            "sendChatAction",
            {"chat_id": chat_id, "action": "typing"},
        )
        summary = await asyncio.to_thread(self._ai_core.summarize, transcript)
        await self._send_text_async(chat_id, summary)

    async def _handle_folders(self, chat_id: int | str, user_id: int) -> None:
        await self._api_request_async(
            "sendChatAction",
            {"chat_id": chat_id, "action": "typing"},
        )
        folders = await self._manager.list_folders(user_id)
        if not folders:
            await self._send_text_async(
                chat_id,
                "У вас не найдено пользовательских папок Telegram.",
            )
            return

        lines = ["Ваши папки Telegram:"]
        lines.extend(f"• {folder_id} — {title}" for folder_id, title in folders)
        lines.append("\nДля саммари отправьте: /folder <ID или название>")
        await self._send_text_async(chat_id, "\n".join(lines))

    async def _handle_folder_summary(
        self,
        chat_id: int | str,
        user_id: int,
        args: str,
    ) -> None:
        folder_ref, limit = self._parse_folder_args(args)
        if not self._manager.is_connected(user_id):
            raise UserBotError(
                "Сначала подключите аккаунт: /login +79991234567"
            )
        if self._manager.is_busy(user_id):
            raise UserBotError(
                "У вас уже запущено саммари другой папки. Дождитесь завершения."
            )

        effective_limit = limit or self._manager.get_limit(user_id)
        await self._send_text_async(
            chat_id,
            f"Начинаю саммари папки «{folder_ref}» (лимит: {effective_limit} сообщений на чат)...",
        )

        task = asyncio.create_task(
            self._run_folder_summary(chat_id, user_id, folder_ref, effective_limit)
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _run_folder_summary(
        self,
        chat_id: int | str,
        user_id: int,
        folder_ref: str,
        limit: int,
    ) -> None:
        try:
            await self._api_request_async(
                "sendChatAction",
                {"chat_id": chat_id, "action": "typing"},
            )
            title, summaries = await self._manager.summarize_folder(
                user_id,
                folder_ref,
                limit=limit,
            )
            if not summaries:
                await self._send_text_async(
                    chat_id,
                    f"В папке «{title}» не нашлось текстовых сообщений для саммари.",
                )
                return

            await self._send_text_async(
                chat_id,
                f"Готово! Папка «{title}», обработано чатов: {len(summaries)}.",
            )
            for chat_name, summary in summaries:
                await self._send_text_async(
                    chat_id,
                    f"=== {chat_name} ===\n{summary}",
                )
        except asyncio.CancelledError:
            raise
        except (UserBotError, ValueError) as exc:
            await self._send_text_async(chat_id, f"Ошибка: {exc}")
        except RPCError as exc:
            logger.error("Telethon RPC error while summarizing folder: %s", exc)
            await self._send_text_async(chat_id, f"Ошибка Telegram API: {exc}")
        except Exception:
            logger.exception("Unexpected error during folder summarization")
            try:
                await self._send_text_async(
                    chat_id,
                    "Произошла непредвиденная ошибка при саммари папки.",
                )
            except TelegramError as exc:
                logger.error("Could not send error message: %s", exc)

    async def _handle_limit(
        self,
        chat_id: int | str,
        user_id: int,
        args: str,
    ) -> None:
        if not args:
            await self._send_text_async(
                chat_id,
                f"Ваш текущий лимит: {self._manager.get_limit(user_id)} сообщений на чат.\n"
                "Изменить: /limit 50",
            )
            return

        try:
            new_limit = int(args)
        except ValueError as exc:
            raise ValueError("Лимит должен быть положительным целым числом. Пример: /limit 50") from exc

        self._manager.set_limit(user_id, new_limit)
        await self._send_text_async(
            chat_id,
            f"Лимит сообщений на чат обновлен: {new_limit}.",
        )

    async def _handle_status(self, chat_id: int | str, user_id: int) -> None:
        status = self._manager.get_status(user_id)
        state = "выполняет саммари" if status["busy"] else "ожидает команд"
        text = (
            f"Статус подключения: {status['auth_state']}\n"
            f"Аккаунт: {status['account']}\n"
            f"Состояние: {state}\n"
            f"Лимит сообщений на чат: {status['limit']}"
        )
        await self._send_text_async(chat_id, text)

    @staticmethod
    def _parse_folder_args(raw_args: str) -> tuple[str, int | None]:
        raw = raw_args.strip()
        if not raw:
            raise ValueError(
                "Укажите ID или название папки. Пример: /folder Работа или /folder 2 --limit 50"
            )

        limit: int | None = None
        match = re.search(r"(?:^|\s)--limit(?:=|\s+)(\S+)\s*$", raw)
        if match:
            try:
                limit = int(match.group(1))
            except ValueError as exc:
                raise ValueError("Параметр --limit должен быть целым числом.") from exc
            if limit <= 0:
                raise ValueError("Параметр --limit должен быть больше нуля.")
            raw = raw[: match.start()].strip()

        if not raw:
            raise ValueError("Укажите ID или название папки перед --limit.")
        return raw, limit

    def _record_message(self, chat_id: int, message: dict[str, Any]) -> None:
        text = message.get("text", "")
        if text.startswith("/"):
            return
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
            "text": text,
        }
        log = self._message_log.setdefault(chat_id, [])
        log.append(entry)
        if len(log) > self._log_limit:
            del log[: len(log) - self._log_limit]

    async def _send_text_async(self, chat_id: int | str, text: str) -> None:
        for start in range(0, len(text), 4096):
            await self._api_request_async(
                "sendMessage",
                {"chat_id": chat_id, "text": text[start : start + 4096]},
            )

    async def _api_request_async(
        self,
        method: str,
        payload: dict[str, Any],
        timeout: float = 15,
    ) -> Any:
        return await asyncio.to_thread(self._api_request, method, payload, timeout)

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


async def run_bot() -> None:
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
        raise UserBotError(
            f"Укажите {', '.join(missing)} в файле .env перед запуском бота."
        )

    provider = OpenRouterProvider(
        api_key=openrouter_api_key,
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        app_name=os.getenv("OPENROUTER_APP_NAME", "Telegram AI Bot"),
        site_url=os.getenv("OPENROUTER_SITE_URL") or None,
    )
    ai_core = AICore(provider)
    manager = UserBotManager.from_env(ai_core)

    try:
        await manager.load_existing_sessions()
        await TelegramBot(token, ai_core, manager).run()
    finally:
        await manager.shutdown()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        load_env_file()
        asyncio.run(run_bot())
    except (UserBotError, RPCError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("Interrupted.") from None


if __name__ == "__main__":
    main()