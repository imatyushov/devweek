from aiogram.types import (
    ReplyKeyboardMarkup, 
    KeyboardButton,
    BotCommand
)

def create_start_keyboard() -> ReplyKeyboardMarkup:
    """
    Creates the main keyboard with the "New Issue" button
    
    Returns:
        ReplyKeyboardMarkup: The keyboard markup
    """
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Новое обращение")]],
        resize_keyboard=True
    )
    return keyboard

def create_issue_keyboard() -> ReplyKeyboardMarkup:
    """
    Creates the keyboard for when user is submitting an issue
    
    Returns:
        ReplyKeyboardMarkup: The keyboard markup
    """
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Завершить обращение")]],
        resize_keyboard=True
    )
    return keyboard

async def set_default_commands(bot):
    """
    Sets the default commands for the bot in the menu button
    
    Args:
        bot: Bot instance
    """
    commands = [
        BotCommand(command="/start", description="Начать работу с ботом"),
        BotCommand(command="/help", description="Получить справку")
    ]
    await bot.set_my_commands(commands) 