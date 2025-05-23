import asyncio
import logging
import os
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.types import BotCommand
from aiogram.client.default import DefaultBotProperties

from app.bot.handlers import register_handlers
from app.bot.keyboards import set_default_commands
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Bot token from BotFather (from environment variable)
API_TOKEN = os.getenv("TG_BOT_TOKEN")

async def main():
    # Initialize Bot instance with the default parse mode
    bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    
    # Create Dispatcher
    dp = Dispatcher()
    
    # Register all handlers
    register_handlers(dp)
    
    # Set default commands
    await set_default_commands(bot)
    
    # Start polling
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main()) 