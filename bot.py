"""Логика Telegram-бота: приём сообщений, статусы инструментов, ответы пользователя."""

from __future__ import annotations

import logging
from typing import Any

import telebot
from telebot import TeleBot

from agent import AgentResult, reset_chat, run_agent
from config import get_settings
from pinecone_manager import get_pinecone_manager
from service_messages import (
    HELP_MESSAGE,
    SERVICE_ECHO_REPLY,
    TOOL_STATUS,
    WELCOME_MESSAGE,
    is_service_text,
)

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096

# Текст, когда у пользователя вообще нет ни имени, ни юзернейма.
FALLBACK_USER_NAME = "друг"


def _edit_safely(bot: TeleBot, chat_id: int, message_id: int, text: str) -> bool:
    """Отредактировать статусное сообщение. Вернуть True при успехе."""
    try:
        bot.edit_message_text(text, chat_id, message_id)
        return True
    except Exception:  # noqa: BLE001 — например, «message is not modified»
        logger.debug("Не удалось отредактировать сообщение %d", message_id, exc_info=True)
        return False


def _send_long_message(bot: TeleBot, chat_id: int, text: str) -> None:
    """Отправить ответ, разрезав его по лимиту Telegram (4096 символов)."""
    for start in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        bot.send_message(chat_id, text[start : start + TELEGRAM_MESSAGE_LIMIT])


def _user_display_name(message: Any) -> str:
    """Определить имя собеседника: имя+фамилия → @username → id → fallback.

    Обрабатывает случай, когда у пользователя нет ни имени/фамилии,
    ни юзернейма: тогда возвращается нейтральное обращение (не падаем).
    """
    user = getattr(message, "from_user", None)
    if user is None:
        return FALLBACK_USER_NAME

    first = str(getattr(user, "first_name", "") or "").strip()
    last = str(getattr(user, "last_name", "") or "").strip()
    full_name = " ".join(part for part in (first, last) if part)
    if full_name:
        return full_name

    username = str(getattr(user, "username", "") or "").strip()
    if username:
        return f"@{username}"

    user_id = getattr(user, "id", None)
    if user_id is not None:
        return f"пользователь {user_id}"
    return FALLBACK_USER_NAME


def _welcome_message(name: str) -> str:
    """Собрать приветствие с обращением по имени (или без него)."""
    if not name or name == FALLBACK_USER_NAME:
        return WELCOME_MESSAGE
    return WELCOME_MESSAGE.replace("Привет!", f"Привет, {name}!", 1)


def _save_user_message_to_memory(chat_id: int, user_text: str) -> None:
    """Сохранить ТЕКСТ сообщения пользователя в Pinecone (контекст диалога).

    Служебные тексты бота сюда не доходят (фильтруются в handle_text);
    сам менеджер дополнительно пропускает служебный текст и дубликаты
    (косинусное сходство выше порога → update/skip).
    """
    try:
        result = get_pinecone_manager().save_user_message(
            user_text, user_id=str(chat_id)
        )
        logger.info(
            "Контекст диалога: action=%s, similarity=%.4f, text=%r",
            result.action,
            result.similarity,
            user_text,
        )
    except Exception:  # noqa: BLE001 — сбой памяти не должен ломать ответ
        logger.exception("Не удалось сохранить сообщение в Pinecone")


def create_bot() -> TeleBot:
    """Создать TeleBot с настроенной обработкой сообщений."""
    token = get_settings().require_telegram_token()
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def handle_start(message: Any) -> None:
        # Имя может отсутствовать (нет имени/фамилии/username) — есть fallback.
        name = _user_display_name(message)
        bot.reply_to(message, _welcome_message(name))

    @bot.message_handler(commands=["help"])
    def handle_help(message: Any) -> None:
        bot.reply_to(message, HELP_MESSAGE)

    @bot.message_handler(commands=["reset"])
    def handle_reset(message: Any) -> None:
        chat_id = message.chat.id
        reset_chat(chat_id)
        bot.reply_to(message, "✅ История диалога очищена.")

    @bot.message_handler(func=lambda m: True, content_types=["text"])
    def handle_text(message: Any) -> None:
        chat_id = message.chat.id
        user_text = (message.text or "").strip()
        if not user_text:
            return

        # Служебные ответы бота не обрабатываются и не сохраняются в память.
        if is_service_text(user_text):
            logger.info("Служебное сообщение из чата %s пропущено: %r", chat_id, user_text)
            bot.reply_to(message, SERVICE_ECHO_REPLY)
            return

        # Тексты сообщений пользователя сохраняются в Pinecone (контекст
        # диалога с косинусным дедупом); ответы бота — только в историю чата.
        _save_user_message_to_memory(chat_id, user_text)

        user_name = _user_display_name(message)
        status_message = bot.send_message(chat_id, "🤔 Думаю…")

        def on_tool_called(tool_name: str) -> None:
            _edit_safely(
                bot,
                chat_id,
                status_message.message_id,
                TOOL_STATUS.get(tool_name, "⚙️ Работаю…"),
            )

        try:
            result: AgentResult = run_agent(
                chat_id,
                user_text,
                user_name=user_name,
                on_tool_called=on_tool_called,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при обработке сообщения из чата %s", chat_id)
            _edit_safely(
                bot,
                chat_id,
                status_message.message_id,
                "⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.",
            )
            return

        answer = result.text

        # Заменяем статусное сообщение финальным ответом (разрезая по лимиту
        # Telegram); если редактирование не удалось — отправляем отдельно.
        if len(answer) > TELEGRAM_MESSAGE_LIMIT:
            _edit_safely(bot, chat_id, status_message.message_id, "✅ Готово, отправляю ответ…")
            _send_long_message(bot, chat_id, answer)
        elif not _edit_safely(bot, chat_id, status_message.message_id, answer):
            _send_long_message(bot, chat_id, answer)

        # Если агент достал фото собаки — отправляем его отдельным сообщением.
        if result.photo_url:
            try:
                bot.send_photo(chat_id, result.photo_url, caption="🐶 Фото собаки")
            except Exception:  # noqa: BLE001
                logger.exception("Не удалось отправить фото собаки")

    return bot
