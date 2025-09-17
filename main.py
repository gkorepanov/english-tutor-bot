from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import redis.asyncio as redis
from litellm import completion
from pydantic import BaseModel, ConfigDict, ValidationError
from telegram import Chat, Update, constants
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

LOGGER = logging.getLogger(__name__)
BOT_DISPLAY_NAME = "English Tutor"
MAX_HISTORY_MESSAGES = 20
MAX_TEXT_LENGTH = 2000


class TutorReply(BaseModel):
    need_to_comment: bool
    comment: str | None = None

    model_config = ConfigDict(extra="ignore")


class ChatHistoryStorage:
    def __init__(self, redis_client: redis.Redis, max_messages: int = MAX_HISTORY_MESSAGES) -> None:
        self._redis = redis_client
        self._max_messages = max_messages

    def _key(self, chat_id: int) -> str:
        return f"chat_history:{chat_id}"

    async def add_message(
        self,
        chat_id: int,
        *,
        author: str,
        timestamp: datetime,
        text: str,
        is_bot: bool,
    ) -> None:
        payload = {
            "author": author,
            "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
            "text": text,
            "is_bot": is_bot,
        }
        key = self._key(chat_id)
        await self._redis.lpush(key, json.dumps(payload, ensure_ascii=False))
        await self._redis.ltrim(key, 0, self._max_messages - 1)

    async def get_history(self, chat_id: int) -> List[Dict[str, Any]]:
        key = self._key(chat_id)
        raw_items = await self._redis.lrange(key, 0, self._max_messages - 1)
        history = [json.loads(item) for item in reversed(raw_items)]
        return history


def get_env(name: str, *, required: bool = True) -> str:
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or ""


def build_conversation(history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in history:
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
        except (KeyError, ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)
        author = entry.get("author", "Unknown")
        if entry.get("is_bot"):
            author = f"{author} (you)"
        text = entry.get("text", "").replace("\n", " ").strip()
        display_time = timestamp.astimezone().strftime("%H:%M")
        lines.append(f"[{display_time}] {author}: {text}")
    return "\n".join(lines)


def sanitize_text(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH] + "…"
    return text


def parse_tutor_reply(raw: str) -> TutorReply:
    cleaned = raw.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        without_ticks = cleaned.strip("`")
        if "\n" in without_ticks:
            _, cleaned = without_ticks.split("\n", 1)
        else:
            cleaned = without_ticks
        cleaned = cleaned.strip()
    try:
        return TutorReply.model_validate_json(cleaned)
    except ValidationError as exc:
        LOGGER.warning("Validation error from LLM response: %s", exc)
        raise


SYSTEM_MESSAGE = (
    "You are an expert English language assistant designed to help users improve real-world communication in English, particularly U.S. English. "
    "Messages labeled 'English Tutor (you)' were written by you earlier. Be concise, supportive, and actionable. "
    "You may include short Russian explanations when they significantly clarify complex nuances. Use Telegram HTML formatting such as <b>bold</b> and emoji when it helps. "
    "Respond only with JSON matching the provided schema."
)


def build_user_prompt(history: List[Dict[str, Any]]) -> str:
    conversation = build_conversation(history)
    last_entry = history[-1]
    last_author = last_entry.get("author", "Unknown")
    last_text = last_entry.get("text", "")
    prompt = f"""Review the following chat conversation.\n<conversation>\n{conversation}\n</conversation>\n\n"""
    prompt += (
        "Focus exclusively on the most recent message from the user. "
        f"The last message is from {last_author}: \"{last_text}\".\n"
        "Assess if the user would benefit from feedback related to grammar, word choice, tone, or clarity. "
        "If everything is natural for contemporary U.S. English and no feedback is required, choose not to comment.\n\n"
        "Return a JSON object that adheres to this schema:\n"
        "{\n  \"need_to_comment\": boolean,\n  \"comment\": string | null\n}\n\n"
        "Set \"need_to_comment\" to true only when you will provide guidance. When commenting, write a short, encouraging note in English, optionally adding a brief Russian aside for complex explanations. "
        "Use Telegram HTML formatting (e.g., <b>bold</b>) and emoji only when they genuinely add value."
    )
    return prompt


async def request_tutor_reply(
    history: List[Dict[str, Any]],
    *,
    model: str,
    temperature: float,
) -> TutorReply | None:
    if not history:
        return None

    try:
        response = await asyncio.to_thread(
            completion,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": build_user_prompt(history)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    except Exception:
        LOGGER.exception("Failed to call LLM provider via LiteLLM")
        return None

    try:
        raw_content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        LOGGER.error("Unexpected response structure from LLM: %s", response)
        return None

    try:
        return parse_tutor_reply(raw_content)
    except ValidationError:
        return None


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if not message or not chat or not user:
        return

    if user.is_bot:
        return

    admin_user_id = context.application.bot_data["admin_user_id"]

    if chat.type == Chat.PRIVATE and user.id != admin_user_id:
        await message.reply_text("This bot is private. Access is restricted.")
        return

    storage: ChatHistoryStorage = context.application.bot_data["history_storage"]

    text = sanitize_text(message.text or "")
    timestamp = message.date or datetime.now(timezone.utc)
    author_name = user.full_name or user.username or f"User {user.id}"

    await storage.add_message(
        chat.id,
        author=author_name,
        timestamp=timestamp,
        text=text,
        is_bot=False,
    )

    history = await storage.get_history(chat.id)

    tutor_reply = await request_tutor_reply(
        history,
        model=context.application.bot_data["llm_model"],
        temperature=context.application.bot_data["llm_temperature"],
    )

    if not tutor_reply or not tutor_reply.need_to_comment:
        return

    comment = (tutor_reply.comment or "").strip()
    if not comment:
        return

    await message.reply_text(comment, parse_mode=constants.ParseMode.HTML)

    await storage.add_message(
        chat.id,
        author=BOT_DISPLAY_NAME,
        timestamp=datetime.now(timezone.utc),
        text=comment,
        is_bot=True,
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if not message or not user:
        return

    admin_user_id = context.application.bot_data["admin_user_id"]
    if user.id != admin_user_id:
        await message.reply_text("This bot is private. Access is restricted.")
        return

    await message.reply_text(
        "Hi! Add me to a chat and I'll keep an eye on the conversation to help with English when it's helpful."
    )


async def enforce_admin_only(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_member_update = update.my_chat_member
    if not chat_member_update:
        return

    new_status = chat_member_update.new_chat_member.status
    if new_status not in {"member", "administrator", "creator"}:
        return

    admin_user_id = context.application.bot_data["admin_user_id"]
    actor = update.effective_user

    if not actor or actor.id == admin_user_id:
        return

    chat = update.effective_chat
    if not chat:
        return

    LOGGER.info("Leaving chat %s because unauthorized user tried to add the bot", chat.id)
    try:
        await context.bot.send_message(
            chat.id,
            "Only the admin user can add this bot to chats. Leaving now.",
        )
    except Exception:
        LOGGER.debug("Failed to send unauthorized notice before leaving chat", exc_info=True)

    await context.bot.leave_chat(chat.id)


async def close_redis(application: Application) -> None:
    redis_client: redis.Redis = application.bot_data.get("redis_client")
    if redis_client:
        await redis_client.close()


async def on_startup(application: Application) -> None:
    redis_client: redis.Redis = application.bot_data.get("redis_client")
    if redis_client:
        try:
            await redis_client.ping()
        except Exception:
            LOGGER.exception("Failed to connect to Redis")
            raise


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    bot_token = get_env("BOT_TOKEN")
    admin_user_raw = get_env("ADMIN_USER")
    try:
        admin_user_id = int(admin_user_raw)
    except ValueError as exc:
        raise RuntimeError("ADMIN_USER must be an integer Telegram user ID") from exc

    redis_url = get_env("REDIS_URL")
    llm_api_key = get_env("LITELLM_API_KEY")
    os.environ.setdefault("LITELLM_API_KEY", llm_api_key)
    llm_model = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
    llm_temperature = float(os.getenv("LITELLM_TEMPERATURE", "0.2"))

    redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    history_storage = ChatHistoryStorage(redis_client)

    application = ApplicationBuilder().token(bot_token).build()

    application.bot_data["admin_user_id"] = admin_user_id
    application.bot_data["history_storage"] = history_storage
    application.bot_data["llm_model"] = llm_model
    application.bot_data["llm_temperature"] = llm_temperature
    application.bot_data["redis_client"] = redis_client

    application.add_handler(ChatMemberHandler(enforce_admin_only, ChatMemberHandler.MY_CHAT_MEMBER))
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.post_init.append(on_startup)
    application.post_shutdown.append(close_redis)

    application.run_polling(allowed_updates=["message", "my_chat_member"], drop_pending_updates=True)


if __name__ == "__main__":
    main()
