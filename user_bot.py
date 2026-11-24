from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telethon import TelegramClient, functions, types, utils
from telethon.errors import (
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    PhoneNumberInvalidError,
    RPCError,
    SessionPasswordNeededError,
)

from main import AICore, OpenRouterProvider, ProviderError


logger = logging.getLogger(__name__)


class UserBotError(RuntimeError):
    pass


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


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def folder_title(folder: Any) -> str:
    title = folder.title
    return str(getattr(title, "text", title))


def format_account_label(me: Any) -> str:
    name = " ".join(part for part in (me.first_name, me.last_name) if part)
    username = f"@{me.username}" if me.username else ""
    label = " ".join(part for part in (name, username) if part)
    return f"{label} (ID: {me.id})".strip()


class TelegramUserBot:
    def __init__(
        self,
        client: TelegramClient,
        ai_core: AICore,
        message_limit: int = 100,
    ) -> None:
        if message_limit <= 0:
            raise ValueError("Message limit must be positive")
        self._client = client
        self._ai_core = ai_core
        self._message_limit = message_limit

    async def get_folders(self) -> list[Any]:
        result = await self._client(functions.messages.GetDialogFiltersRequest())
        return [
            folder
            for folder in result.filters
            if isinstance(folder, (types.DialogFilter, types.DialogFilterChatlist))
        ]

    async def list_folders(self) -> None:
        folders = await self.get_folders()
        if not folders:
            print("No custom Telegram folders found.")
            return
        for folder in folders:
            print(f"{folder.id}: {folder_title(folder)}")

    async def select_folder(self, reference: str) -> Any:
        folders = await self.get_folders()
        if reference.isdecimal():
            matches = [folder for folder in folders if folder.id == int(reference)]
        else:
            normalized = reference.casefold()
            matches = [
                folder for folder in folders if folder_title(folder).casefold() == normalized
            ]

        if len(matches) == 1:
            return matches[0]

        available = ", ".join(
            f"{folder.id}: {folder_title(folder)}" for folder in folders
        ) or "none"
        if not matches:
            raise UserBotError(
                f"Папка {reference!r} не найдена. Доступные папки: {available}"
            )
        raise UserBotError(
            f"Найдено несколько папок с названием {reference!r}; укажите её ID. "
            f"Доступные папки: {available}"
        )

    async def get_folder_dialogs(self, folder: Any) -> list[Any]:
        dialogs = await self._client.get_dialogs(folder=0)
        dialogs.extend(await self._client.get_dialogs(folder=1))

        pinned_ids = self._peer_ids(folder.pinned_peers)
        included_ids = self._peer_ids(folder.include_peers)
        excluded_ids = self._peer_ids(getattr(folder, "exclude_peers", []))

        pinned: dict[int, Any] = {}
        regular: list[Any] = []
        for dialog in dialogs:
            dialog_id = dialog.id
            if dialog_id in excluded_ids or not self._is_in_folder(
                dialog, folder, pinned_ids, included_ids
            ):
                continue
            if self._is_excluded_by_flags(dialog, folder):
                continue
            if dialog_id in pinned_ids:
                pinned[dialog_id] = dialog
            else:
                regular.append(dialog)

        ordered_pinned = [
            pinned[dialog_id]
            for dialog_id in self._peer_ids(folder.pinned_peers)
            if dialog_id in pinned
        ]
        return ordered_pinned + regular

    async def build_transcript(
        self,
        dialog: Any,
        limit: int | None = None,
    ) -> str | None:
        effective_limit = self._message_limit if limit is None else limit
        if effective_limit <= 0:
            raise ValueError("Message limit must be positive")

        messages = [
            message
            async for message in self._client.iter_messages(
                dialog.entity, limit=effective_limit
            )
        ]
        messages.reverse()

        lines = [f"Chat: {dialog.name}"]
        sender_names: dict[int, str] = {}
        for message in messages:
            text = message.message
            if not isinstance(text, str) or not text.strip():
                continue

            sender_id = message.sender_id
            if sender_id not in sender_names:
                sender = await message.get_sender()
                sender_names[sender_id] = self._sender_name(sender, sender_id)
            date = message.date.isoformat(timespec="seconds")
            lines.append(f"[{date}] {sender_names[sender_id]}: {text.strip()}")

        if len(lines) == 1:
            return None
        return "\n".join(lines)

    async def summarize_selected_folder(
        self,
        folder: Any,
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        dialogs = await self.get_folder_dialogs(folder)
        logger.info(
            "Found %d chats in Telegram folder %s", len(dialogs), folder_title(folder)
        )

        summaries: list[tuple[str, str]] = []
        for dialog in dialogs:
            try:
                transcript = await self.build_transcript(dialog, limit=limit)
                if transcript is None:
                    logger.info("Skipping %s: no text messages", dialog.name)
                    continue
                summary = await asyncio.to_thread(self._ai_core.summarize, transcript)
                summaries.append((dialog.name, summary))
            except ProviderError as exc:
                logger.error("AI request failed for %s: %s", dialog.name, exc)
            except RPCError as exc:
                logger.error("Telegram request failed for %s: %s", dialog.name, exc)
            except Exception:
                logger.exception("Unexpected error while processing %s", dialog.name)
        return summaries

    async def summarize_folder(
        self,
        reference: str,
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        folder = await self.select_folder(reference)
        return await self.summarize_selected_folder(folder, limit=limit)

    @staticmethod
    def _peer_ids(peers: list[Any]) -> list[int]:
        return list(dict.fromkeys(utils.get_peer_id(peer) for peer in peers))

    @staticmethod
    def _is_in_folder(
        dialog: Any,
        folder: Any,
        pinned_ids: list[int],
        included_ids: list[int],
    ) -> bool:
        if dialog.id in pinned_ids or dialog.id in included_ids:
            return True

        entity = dialog.entity
        if isinstance(entity, types.User):
            if entity.bot:
                return bool(getattr(folder, "bots", False))
            if entity.contact:
                return bool(getattr(folder, "contacts", False))
            return bool(getattr(folder, "non_contacts", False))
        if isinstance(entity, types.Chat):
            return bool(getattr(folder, "groups", False))
        if isinstance(entity, types.Channel):
            if entity.broadcast:
                return bool(getattr(folder, "broadcasts", False))
            return bool(getattr(folder, "groups", False))
        return False

    @staticmethod
    def _is_excluded_by_flags(dialog: Any, folder: Any) -> bool:
        if getattr(folder, "exclude_archived", False) and dialog.dialog.folder_id == 1:
            return True
        if getattr(folder, "exclude_muted", False):
            mute_until = dialog.dialog.notify_settings.mute_until
            if isinstance(mute_until, datetime):
                if mute_until.tzinfo is None:
                    mute_until = mute_until.replace(tzinfo=timezone.utc)
                if mute_until > datetime.now(timezone.utc):
                    return True
            elif mute_until:
                return True
        if getattr(folder, "exclude_read", False):
            if not dialog.unread_count and not dialog.dialog.unread_mark:
                return True
        return False

    @staticmethod
    def _sender_name(sender: Any, sender_id: int) -> str:
        if sender is None:
            return str(sender_id)

        if isinstance(sender, types.User):
            name = " ".join(
                part for part in (sender.first_name, sender.last_name) if part
            )
            username = f"@{sender.username}" if sender.username else ""
            return " ".join(part for part in (name, username) if part) or str(sender_id)

        return getattr(sender, "title", None) or str(sender_id)


@dataclass
class PendingAuth:
    client: TelegramClient
    phone: str
    phone_code_hash: str
    awaiting_password: bool = False


class UserBotManager:
    """Управляет изолированными сессиями Telethon для каждого пользователя бота."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        ai_core: AICore,
        sessions_dir: Path | None = None,
        default_limit: int = 100,
    ) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._ai_core = ai_core
        self._sessions_dir = sessions_dir or (
            Path(__file__).resolve().parent / "sessions"
        )
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._default_limit = default_limit

        self._clients: dict[int, TelegramClient] = {}
        self._account_labels: dict[int, str] = {}
        self._pending: dict[int, PendingAuth] = {}
        self._limits: dict[int, int] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    @classmethod
    def from_env(
        cls,
        ai_core: AICore,
        default_limit: int = 100,
    ) -> UserBotManager:
        missing = [
            name
            for name in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH")
            if not os.getenv(name)
        ]
        if missing:
            raise UserBotError(
                f"Укажите {', '.join(missing)} в файле .env на сервере (один раз для всего приложения)."
            )

        try:
            api_id = int(os.environ["TELEGRAM_API_ID"])
        except ValueError as exc:
            raise UserBotError("TELEGRAM_API_ID должен быть целым числом.") from exc

        return cls(
            api_id=api_id,
            api_hash=os.environ["TELEGRAM_API_HASH"],
            ai_core=ai_core,
            default_limit=default_limit,
        )

    def _session_base(self, user_id: int) -> Path:
        return self._sessions_dir / str(user_id)

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock

    def is_connected(self, user_id: int) -> bool:
        client = self._clients.get(user_id)
        return client is not None and client.is_connected()

    def is_busy(self, user_id: int) -> bool:
        lock = self._locks.get(user_id)
        return lock is not None and lock.locked()

    def get_limit(self, user_id: int) -> int:
        return self._limits.get(user_id, self._default_limit)

    def set_limit(self, user_id: int, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Лимит должен быть положительным целым числом.")
        self._limits[user_id] = limit

    def get_status(self, user_id: int) -> dict[str, Any]:
        pending = self._pending.get(user_id)
        if self.is_connected(user_id):
            auth_state = "подключен"
        elif pending is not None:
            auth_state = (
                "ожидает облачный пароль (/password)"
                if pending.awaiting_password
                else "ожидает код подтверждения (/code)"
            )
        else:
            auth_state = "не подключен"

        return {
            "connected": self.is_connected(user_id),
            "auth_state": auth_state,
            "busy": self.is_busy(user_id),
            "account": self._account_labels.get(user_id, "—"),
            "limit": self.get_limit(user_id),
        }

    async def load_existing_sessions(self) -> None:
        for session_file in self._sessions_dir.glob("*.session"):
            stem = session_file.stem
            if not stem.isdigit():
                continue
            user_id = int(stem)
            client = TelegramClient(
                str(self._session_base(user_id)),
                self._api_id,
                self._api_hash,
            )
            try:
                await client.connect()
                if await client.is_user_authorized():
                    me = await client.get_me()
                    if me is not None:
                        self._clients[user_id] = client
                        self._account_labels[user_id] = format_account_label(me)
                        logger.info(
                            "Restored userbot session for bot_user=%s (%s)",
                            user_id,
                            self._account_labels[user_id],
                        )
                        continue
                await client.disconnect()
            except Exception:
                logger.exception("Failed to restore session %s", session_file.name)
                try:
                    await client.disconnect()
                except Exception:
                    pass

    async def start_login(self, user_id: int, raw_phone: str) -> str:
        phone = re.sub(r"[\s\-()]", "", raw_phone.strip())
        if not phone:
            raise ValueError("Укажите номер телефона. Пример: /login +79991234567")
        if not phone.startswith("+"):
            phone = f"+{phone}"

        if self.is_connected(user_id):
            label = self._account_labels.get(user_id, "аккаунт")
            return (
                f"Вы уже подключены как {label}.\n"
                "Чтобы сменить аккаунт, сначала выполните /logout."
            )

        await self._cleanup_pending(user_id)

        client = TelegramClient(
            str(self._session_base(user_id)),
            self._api_id,
            self._api_hash,
        )
        await client.connect()

        if await client.is_user_authorized():
            me = await client.get_me()
            self._clients[user_id] = client
            self._account_labels[user_id] = format_account_label(me)
            return f"Сессия восстановлена: {self._account_labels[user_id]}"

        try:
            sent = await client.send_code_request(phone)
        except PhoneNumberInvalidError as exc:
            await client.disconnect()
            raise UserBotError("Некорректный номер телефона. Проверьте формат (+79991234567).") from exc
        except Exception:
            await client.disconnect()
            raise

        self._pending[user_id] = PendingAuth(
            client=client,
            phone=phone,
            phone_code_hash=sent.phone_code_hash,
            awaiting_password=False,
        )
        return (
            f"Код подтверждения отправлен в Telegram на номер {phone}.\n\n"
            "Важно: отправьте код с пробелами или дефисами (например: /code 12-34-5), "
            "иначе сам Telegram автоматически аннулирует код при отправке в чат."
        )

    async def submit_code(self, user_id: int, raw_code: str) -> str:
        pending = self._pending.get(user_id)
        if pending is None:
            raise UserBotError("Сначала начните вход командой: /login +79991234567")

        code = re.sub(r"\D", "", raw_code)
        if not code:
            raise ValueError("Укажите цифры кода. Пример: /code 12-34-5")

        try:
            await pending.client.sign_in(
                phone=pending.phone,
                code=code,
                phone_code_hash=pending.phone_code_hash,
            )
        except SessionPasswordNeededError:
            pending.awaiting_password = True
            return (
                "На аккаунте включена двухэтапная аутентификация (облачный пароль).\n"
                "Отправьте: /password ваш_облачный_пароль"
            )
        except PhoneCodeInvalidError as exc:
            raise UserBotError("Неверный код. Проверьте цифры и попробуйте ещё раз.") from exc
        except PhoneCodeExpiredError as exc:
            await self._cleanup_pending(user_id)
            raise UserBotError(
                "Код истек или был аннулирован Telegram. Запросите новый через /login <номер>."
            ) from exc

        return await self._finalize_login(user_id, pending.client)

    async def submit_password(self, user_id: int, password: str) -> str:
        pending = self._pending.get(user_id)
        if pending is None or not pending.awaiting_password:
            raise UserBotError("Сейчас ввод облачного пароля не требуется.")

        if not password:
            raise ValueError("Укажите облачный пароль: /password ваш_пароль")

        try:
            await pending.client.sign_in(password=password)
        except PasswordHashInvalidError as exc:
            raise UserBotError("Неверный облачный пароль. Попробуйте ещё раз.") from exc

        return await self._finalize_login(user_id, pending.client)

    async def _finalize_login(self, user_id: int, client: TelegramClient) -> str:
        self._pending.pop(user_id, None)
        me = await client.get_me()
        if me is None:
            await client.disconnect()
            raise UserBotError("Не удалось получить профиль аккаунта после входа.")

        self._clients[user_id] = client
        self._account_labels[user_id] = format_account_label(me)
        logger.info(
            "User %s logged into userbot as %s",
            user_id,
            self._account_labels[user_id],
        )
        return (
            f"Вход выполнен! Подключен аккаунт: {self._account_labels[user_id]}\n"
            "Теперь доступны команды /folders и /folder <ID или название>."
        )

    async def logout(self, user_id: int) -> None:
        if self.is_busy(user_id):
            raise UserBotError("Дождитесь завершения текущего саммари перед выходом.")

        await self._cleanup_pending(user_id)
        client = self._clients.pop(user_id, None)
        self._account_labels.pop(user_id, None)

        if client is not None:
            try:
                if client.is_connected():
                    await client.log_out()
            except Exception:
                logger.warning("log_out() failed for user %s, disconnecting locally", user_id)
            finally:
                if client.is_connected():
                    await client.disconnect()

        for suffix in (".session", ".session-journal"):
            path = Path(f"{self._session_base(user_id)}{suffix}")
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete %s", path)

    async def list_folders(self, user_id: int) -> list[tuple[int, str]]:
        client = self._require_client(user_id)
        user_bot = TelegramUserBot(
            client,
            self._ai_core,
            message_limit=self.get_limit(user_id),
        )
        folders = await user_bot.get_folders()
        return [(folder.id, folder_title(folder)) for folder in folders]

    async def summarize_folder(
        self,
        user_id: int,
        reference: str,
        limit: int | None = None,
    ) -> tuple[str, list[tuple[str, str]]]:
        lock = self._get_lock(user_id)
        if lock.locked():
            raise UserBotError("У вас уже запущено саммари другой папки. Дождитесь завершения.")

        async with lock:
            client = self._require_client(user_id)
            effective_limit = limit or self.get_limit(user_id)
            user_bot = TelegramUserBot(
                client,
                self._ai_core,
                message_limit=effective_limit,
            )
            folder = await user_bot.select_folder(reference)
            title = folder_title(folder)
            summaries = await user_bot.summarize_selected_folder(
                folder,
                limit=effective_limit,
            )
            return title, summaries

    def _require_client(self, user_id: int) -> TelegramClient:
        client = self._clients.get(user_id)
        if client is None or not client.is_connected():
            raise UserBotError(
                "Ваш аккаунт не подключен. Войдите в личке с ботом: /login +79991234567"
            )
        return client

    async def _cleanup_pending(self, user_id: int) -> None:
        pending = self._pending.pop(user_id, None)
        if pending is not None and pending.client.is_connected():
            await pending.client.disconnect()

    async def shutdown(self) -> None:
        for user_id in list(self._pending):
            await self._cleanup_pending(user_id)
        for client in list(self._clients.values()):
            if client.is_connected():
                await client.disconnect()
        self._clients.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize recent messages from a custom Telegram chat folder."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--folder", help="Exact folder title (case-insensitive) or folder ID")
    selection.add_argument(
        "--list-folders", action="store_true", help="List custom Telegram folders and exit"
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=100,
        help="Maximum recent messages to read per chat (default: 100)",
    )
    return parser


def build_ai_core() -> AICore:
    provider = OpenRouterProvider(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        app_name=os.getenv("OPENROUTER_APP_NAME", "Telegram AI Userbot"),
        site_url=os.getenv("OPENROUTER_SITE_URL") or None,
    )
    return AICore(provider)


async def run_cli(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "OPENROUTER_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise UserBotError(f"Set {', '.join(missing)} in the .env file before starting.")

    client = TelegramClient(
        os.getenv("TELEGRAM_SESSION", "user_bot"),
        int(os.environ["TELEGRAM_API_ID"]),
        os.environ["TELEGRAM_API_HASH"],
    )
    user_bot = TelegramUserBot(client, build_ai_core(), message_limit=args.limit)
    try:
        phone = os.getenv("TELEGRAM_PHONE")
        if phone:
            await client.start(phone=phone)
        else:
            await client.start()

        if args.list_folders:
            await user_bot.list_folders()
            return

        summaries = await user_bot.summarize_folder(args.folder)
        if not summaries:
            print("No chat summaries were produced.")
            return
        for chat_name, summary in summaries:
            print(f"\n=== {chat_name} ===\n{summary}")
    finally:
        await client.disconnect()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    try:
        load_env_file()
        asyncio.run(run_cli(args))
    except (UserBotError, RPCError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("Interrupted.") from None


if __name__ == "__main__":
    main()