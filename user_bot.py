from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telethon import TelegramClient, functions, types, utils
from telethon.errors import RPCError

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
                f"Telegram folder {reference!r} was not found. Available folders: {available}"
            )
        raise UserBotError(
            f"Telegram folder name {reference!r} is ambiguous; use its ID. "
            f"Available folders: {available}"
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

    async def build_transcript(self, dialog: Any) -> str | None:
        messages = [
            message
            async for message in self._client.iter_messages(
                dialog.entity, limit=self._message_limit
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

    async def summarize_folder(self, reference: str) -> list[tuple[str, str]]:
        folder = await self.select_folder(reference)
        dialogs = await self.get_folder_dialogs(folder)
        logger.info(
            "Found %d chats in Telegram folder %s", len(dialogs), folder_title(folder)
        )

        summaries: list[tuple[str, str]] = []
        for dialog in dialogs:
            try:
                transcript = await self.build_transcript(dialog)
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


async def run(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "OPENROUTER_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise UserBotError(f"Set {', '.join(missing)} in the .env file before starting.")

    try:
        api_id = int(os.environ["TELEGRAM_API_ID"])
    except ValueError as exc:
        raise UserBotError("TELEGRAM_API_ID must be an integer.") from exc

    client = TelegramClient(
        os.getenv("TELEGRAM_SESSION", "user_bot"),
        api_id,
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
        asyncio.run(run(args))
    except (UserBotError, RPCError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("Interrupted.") from None


if __name__ == "__main__":
    main()
