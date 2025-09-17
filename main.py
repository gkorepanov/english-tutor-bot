from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import redis.asyncio as redis
from litellm import acompletion
from pydantic import BaseModel, ConfigDict, ValidationError
from telegram import Update, constants
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


class ChatSettingsStorage:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def _key(self, chat_id: int) -> str:
        return f"chat_settings:{chat_id}"

    async def set_enabled(self, chat_id: int, enabled: bool) -> None:
        key = self._key(chat_id)
        await self._redis.hset(key, mapping={"enabled": "1" if enabled else "0"})

    async def is_enabled(self, chat_id: int) -> bool:
        key = self._key(chat_id)
        val = await self._redis.hget(key, "enabled")
        if val is None:
            return True
        return val == "1"


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


def shorten(text: str, limit: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


SYSTEM_MESSAGE = (
    "You are an expert English language tutor who has been added to a chat where users communicate naturally. "
    "You can see the context of recent messages and your role is to help users learn more native-like, real-world U.S. English "
    "without being intrusive or annoying. Messages labeled 'English Tutor (you)' were written by you earlier. "
    "Be concise, supportive, and actionable in your feedback."
)


def build_user_prompt(history: List[Dict[str, Any]]) -> str:
    conversation = build_conversation(history)
    prompt = f"""Review the following chat conversation.\n<conversation>\n{conversation}\n</conversation>\n\n"""
    prompt += (
        "Focus exclusively on the most recent message from the user. Other messages are given only for context. "
        "Assess if the user would benefit from feedback related to grammar, word choice, tone, or clarity. "
        "If everything is natural for contemporary U.S. English and no feedback is required, choose not to comment.\n\n"
        "When providing feedback, suggest how to say the same thing in both informal spoken English and written contexts when appropriate. "
        "Show how a native speaker would naturally express the same idea in conversation or in a text message - "
        "this might include shorter forms, contractions, phrasal verbs, or common abbreviations that natives use. "
        "Provide practical examples that demonstrate natural, native-like usage.\n\n"
        "Set need_to_comment to true only when you will provide guidance. When commenting, write a short, encouraging note in English, "
        "optionally adding a brief Russian aside for complex words user might not know (e.g. explain \"brevity\", \"abundance\"). "
        "Use Telegram HTML formatting (e.g., <b>bold</b>) and emoji only when they genuinely add value."
    )
    return prompt


async def request_tutor_reply(
    history: List[Dict[str, Any]],
    *,
    model: str,
) -> TutorReply | None:
    if not history:
        return None

    try:
        prompt_text = build_user_prompt(history)
        last_entry = history[-1]
        last_author = last_entry.get("author", "Unknown")
        last_text = last_entry.get("text", "")
        LOGGER.info(
            "LLM request: model=%s conv_size=%s last_author=%s last_text='%s' prompt_chars=%s",
            model,
            len(history),
            last_author,
            shorten(last_text, 200),
            len(prompt_text),
        )
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt_text},
            ],
            response_format=TutorReply,
            api_key=get_env("LITELLM_API_KEY"),
        )
    except Exception:
        LOGGER.exception("Failed to call LLM provider via LiteLLM")
        return None

    try:
        message = response["choices"][0]["message"].content
        LOGGER.info("LLM response: %s", message)
        reply = TutorReply.model_validate_json(message)
        LOGGER.info(
            "LLM response: need_to_comment=%s comment='%s'",
            reply.need_to_comment,
            shorten(reply.comment or "", 200),
        )
        return reply
    except (KeyError, IndexError, TypeError, ValidationError):
        LOGGER.error("Unexpected structured response from LLM: %s", response)
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

    if chat.type == constants.ChatType.PRIVATE and user.id != admin_user_id:
        LOGGER.info(
            "Rejecting private message from non-admin: chat_id=%s user_id=%s",
            chat.id,
            user.id,
        )
        await message.reply_text("This bot is private. Access is restricted.")
        return

    storage: ChatHistoryStorage = context.application.bot_data["history_storage"]
    settings: ChatSettingsStorage = context.application.bot_data["settings_storage"]

    LOGGER.info(
        "Incoming message: chat_id=%s user_id=%s msg_id=%s text='%s'",
        chat.id,
        user.id,
        getattr(message, "message_id", None),
        shorten(message.text or "", 200),
    )

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

    if not await settings.is_enabled(chat.id):
        LOGGER.info("Chat %s is disabled; skipping.", chat.id)
        return

    tutor_reply = await request_tutor_reply(
        history,
        model=context.application.bot_data["llm_model"],
    )

    if not tutor_reply or not tutor_reply.need_to_comment:
        LOGGER.info(
            "No tutor comment: chat_id=%s msg_id=%s",
            chat.id,
            getattr(message, "message_id", None),
        )
        return

    comment = (tutor_reply.comment or "").strip()
    if not comment:
        return

    LOGGER.info(
        "Sending tutor comment: chat_id=%s reply_len=%s excerpt='%s'",
        chat.id,
        len(comment),
        shorten(comment, 200),
    )
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
        LOGGER.info("Rejecting /start from non-admin user_id=%s", user.id)
        await message.reply_text("This bot is private. Access is restricted.")
        return

    LOGGER.info("Handling /start for admin user_id=%s", user.id)
    await message.reply_text(
        "Hi! Add me to a chat and I'll keep an eye on the conversation to help with English when it's helpful."
    )


async def enable_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if not message or not chat:
        return
    settings: ChatSettingsStorage = context.application.bot_data["settings_storage"]
    await settings.set_enabled(chat.id, True)
    LOGGER.info("Enabled tutor in chat %s", chat.id)
    await message.reply_text("English Tutor is now enabled in this chat.")


async def disable_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if not message or not chat:
        return
    settings: ChatSettingsStorage = context.application.bot_data["settings_storage"]
    await settings.set_enabled(chat.id, False)
    LOGGER.info("Disabled tutor in chat %s", chat.id)
    await message.reply_text("English Tutor is now disabled in this chat.")


async def enforce_admin_only(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_member_update = update.my_chat_member
    if not chat_member_update:
        return

    new_status = chat_member_update.new_chat_member.status
    old_status = chat_member_update.old_chat_member.status
    if new_status not in {"member", "administrator", "creator"}:
        return

    admin_user_id = context.application.bot_data["admin_user_id"]
    actor = update.effective_user

    chat = update.effective_chat
    if not chat or not actor:
        return

    if actor.id != admin_user_id:
        LOGGER.info("Leaving chat %s because unauthorized user tried to add the bot", chat.id)
        try:
            await context.bot.send_message(
                chat.id,
                "Only the admin user can add this bot to chats. Leaving now.",
            )
        except Exception:
            LOGGER.debug("Failed to send unauthorized notice before leaving chat", exc_info=True)
        await context.bot.leave_chat(chat.id)
        return

    if old_status in {"left", "kicked"} and new_status in {"member", "administrator", "creator"}:
        try:
            LOGGER.info("Added to chat %s by admin; greeting sent", chat.id)
            await context.bot.send_message(
                chat.id,
                (
                    "Hello! I'm your <b>English Tutor</b>.\n\n"
                    "I'll monitor messages and provide concise feedback on grammar, word choice, tone, or clarity when helpful.\n\n"
                    "Use /on to enable and /off to disable my feedback in this chat."
                ),
                parse_mode=constants.ParseMode.HTML,
            )
        except Exception:
            LOGGER.debug("Failed to send greeting message after being added", exc_info=True)


async def close_redis(application: Application) -> None:
    redis_client: redis.Redis = application.bot_data.get("redis_client")
    if redis_client:
        LOGGER.info("Closing Redis client")
        await redis_client.close()


async def on_startup(application: Application) -> None:
    redis_client: redis.Redis = application.bot_data.get("redis_client")
    if redis_client:
        try:
            await redis_client.ping()
            LOGGER.info("Connected to Redis successfully")
        except Exception:
            LOGGER.exception("Failed to connect to Redis")
            raise


def main() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bot_token = get_env("BOT_TOKEN")
    admin_user_raw = get_env("ADMIN_USER")
    try:
        admin_user_id = int(admin_user_raw)
    except ValueError as exc:
        raise RuntimeError("ADMIN_USER must be an integer Telegram user ID") from exc

    redis_url = get_env("REDIS_URL")
    llm_model = os.getenv("LITELLM_MODEL", "openai/gpt-5")
    LOGGER.info("Starting English Tutor bot with model=%s", llm_model)

    redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    history_storage = ChatHistoryStorage(redis_client)
    settings_storage = ChatSettingsStorage(redis_client)

    application = ApplicationBuilder().token(bot_token).post_init(on_startup).post_shutdown(close_redis).build()

    application.bot_data["admin_user_id"] = admin_user_id
    application.bot_data["history_storage"] = history_storage
    application.bot_data["settings_storage"] = settings_storage
    application.bot_data["llm_model"] = llm_model
    application.bot_data["redis_client"] = redis_client

    application.add_handler(ChatMemberHandler(enforce_admin_only, ChatMemberHandler.MY_CHAT_MEMBER))
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("on", enable_command))
    application.add_handler(CommandHandler("off", disable_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))


    application.run_polling(allowed_updates=["message", "my_chat_member"], drop_pending_updates=True)


if __name__ == "__main__":
    main()
