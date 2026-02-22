import os
import logging
from aiohttp import web
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, Update
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from openai import AsyncOpenAI
import aiosqlite
import re

# ================= КОНФИГУРАЦИЯ =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BOT_USERNAME = os.getenv("BOT_USERNAME", "my_bot")

DB_NAME = "/tmp/group_history.db"
MAX_CONTEXT_MESSAGES = 15
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"

BASE_SYSTEM_PROMPT = """
Ты полезный ассистент в чате. Отвечай кратко и по-русски.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ================= БАЗА ДАННЫХ =================
async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                user_name TEXT,
                text TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_prompts (
                chat_id INTEGER PRIMARY KEY,
                custom_prompt TEXT
            )
        """)
        await db.commit()

async def save_message(chat_id: int, user_name: str, text: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("INSERT INTO messages (chat_id, user_name, text) VALUES (?, ?, ?)",
                        (chat_id, user_name, text))
        await db.execute("""
            DELETE FROM messages WHERE id NOT IN (
                SELECT id FROM messages ORDER BY id DESC LIMIT ?
            )
        """, (MAX_CONTEXT_MESSAGES * 10,))
        await db.commit()

async def get_history(chat_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT user_name, text FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, MAX_CONTEXT_MESSAGES)
        ) as cursor:
            rows = await cursor.fetchall()
            return list(reversed(rows))

async def save_custom_prompt(chat_id: int, prompt: str):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("""
            INSERT INTO chat_prompts (chat_id, custom_prompt) 
            VALUES (?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET custom_prompt = ?
        """, (chat_id, prompt, prompt))
        await db.commit()

async def get_custom_prompt(chat_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute(
            "SELECT custom_prompt FROM chat_prompts WHERE chat_id = ?", 
            (chat_id,)
        ) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else None

async def delete_custom_prompt(chat_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("DELETE FROM chat_prompts WHERE chat_id = ?", (chat_id,))
        await db.commit()

# ================= КОМАНДЫ =================
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я на Vercel.\n\n"
        "📌 Команды:\n"
        "• `запомни: <инструкция>` — задать правило\n"
        "• `/prompt` — показать инструкцию\n"
        "• `/forget` — сбросить\n"
        "• `/ask <вопрос>` — вопрос\n"
        "• Упомяни @ или реплай — отвечу"
    )

@dp.message(Command("prompt"))
async def cmd_show_prompt(message: Message):
    prompt = await get_custom_prompt(message.chat.id)
    if prompt:
        await message.answer(f"📋 Инструкция:\n```\n{prompt}\n```", parse_mode="Markdown")
    else:
        await message.answer("❌ Нет инструкций.")

@dp.message(Command("forget"))
async def cmd_forget_prompt(message: Message):
    await delete_custom_prompt(message.chat.id)
    await message.answer("🗑️ Сброшено.")

@dp.message(Command("ask"))
async def cmd_ask(message: Message):
    if len(message.text.split()) < 2:
        await message.answer("Используй: `/ask <вопрос>`")
        return
    await process_ai_response(message, message.text.split(maxsplit=1)[1])

@dp.message()
async def handle_message(message: Message):
    if message.from_user and message.from_user.id == bot.id:
        return

    text = message.text or message.caption or ""
    if not text:
        return

    user_name = message.from_user.username or message.from_user.first_name or "User"
    
    remember_match = re.match(r'^запомни:\s*(.+)$', text.strip(), re.IGNORECASE)
    if remember_match:
        instruction = remember_match.group(1).strip()
        await save_custom_prompt(message.chat.id, instruction)
        await message.answer(f"✅ Запомнил: `{instruction}`", parse_mode="Markdown")
        await save_message(message.chat.id, user_name, text)
        return

    await save_message(message.chat.id, user_name, text)

    is_mentioned = False
    if message.entities:
        for entity in message.entities:
            if entity.type == "mention" and entity.user and entity.user.id == bot.id:
                is_mentioned = True
    
    if f"@{BOT_USERNAME}" in text:
        is_mentioned = True

    is_reply = (message.reply_to_message and 
                message.reply_to_message.from_user and 
                message.reply_to_message.from_user.id == bot.id)

    if not (is_mentioned or is_reply):
        return

    await process_ai_response(message, text)

async def process_ai_response(message: Message, user_text: str):
    try:
        custom_prompt = await get_custom_prompt(message.chat.id)
        system_prompt = BASE_SYSTEM_PROMPT
        if custom_prompt:
            system_prompt += f"\n\n❗ ПРАВИЛО: {custom_prompt}"

        history = await get_history(message.chat.id)
        context = [{"role": "user", "content": f"{u}: {t}"} for u, t in history]
        current_user = message.from_user.username or message.from_user.first_name or "User"
        context.append({"role": "user", "content": f"{current_user}: {user_text}"})

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, *context],
            temperature=0.7,
            max_tokens=400
        )
        await message.answer(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"AI Error: {e}")
        await message.answer("⚠️ Ошибка. Попробуй позже.")

# ================= ВЕБХУК =================
async def webhook_handler(request):
    update = await request.json()
    await dp.feed_update(bot, Update(**update))
    return web.Response()

async def on_startup(app):
    await init_db()
    
    # Получаем URL от Vercel
    vercel_url = os.getenv("VERCEL_URL", "")
    if vercel_url:
        webhook_url = f"https://{vercel_url}{WEBHOOK_PATH}"
        await bot.set_webhook(webhook_url, drop_pending_updates=True)
        logger.info(f"✅ Webhook установлен: {webhook_url}")
    else:
        logger.info("⚠️ VERCEL_URL не найден (локальный запуск)")

async def on_shutdown(app):
    await bot.session.close()

def create_app():
    app = web.Application()
    app.add_routes([web.post(WEBHOOK_PATH, webhook_handler)])
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app

app = create_app()
