from aiogram import Router, F
from aiogram.types import Message, InputMediaPhoto
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
import os
import uuid
import io
import asyncio

from states import SupportState
from keyboards import create_start_keyboard, create_issue_keyboard
from utils import handle_issue

# Create router
router = Router()

@router.message(Command("start", "help"))
async def cmd_start(message: Message):
    """
    Handler for /start and /help commands
    """
    keyboard = create_start_keyboard()
    await message.answer(
        "Привет! Это бот поддержки Rutube. Я помогу вам с любыми вопросами или проблемами.\n\n"
        "Чтобы начать новое обращение, нажмите кнопку ниже.", 
        reply_markup=keyboard
    )

@router.message(F.text == "Новое обращение")
async def new_issue(message: Message, state: FSMContext):
    """
    Handler for "New Issue" button
    """
    await state.set_state(SupportState.issue)
    keyboard = create_issue_keyboard()
    await message.answer(
        "Пожалуйста, опишите вашу проблему более подробно. Вы также можете отправить фото или документы.", 
        reply_markup=keyboard
    )

@router.message(SupportState.issue, F.text == "Завершить обращение")
async def finish_issue(message: Message, state: FSMContext):
    """
    Handler for "Finish Issue" button when in issue state
    """
    keyboard = create_start_keyboard()
    await message.answer(
        "Обращение завершено. Если у вас есть еще вопросы, нажмите кнопку 'Новое обращение'.",
        reply_markup=keyboard
    )
    await state.clear()

@router.message(SupportState.issue)
async def process_issue(message: Message, state: FSMContext):
    """
    Handler for processing messages while in the issue state
    """
    try:
        # Получаем бота из контекста сообщения
        bot = message.bot
        
        # Генерируем уникальный ID для обращения
        issue_id = str(uuid.uuid4())
        
        # Get text from either the message text or caption (for media messages)
        text = message.text or message.caption or ""
        
        # Скачиваем фотографии в память, если они есть
        photos = []
        if message.photo:
            # Берем только последнюю (самую качественную) фотографию
            photo = message.photo[-1]
            file_id = photo.file_id
            
            try:
                file_info = await bot.get_file(file_id)
                
                # Скачиваем в BytesIO вместо файла
                photo_bytes = io.BytesIO()
                await bot.download_file(file_info.file_path, destination=photo_bytes)
                photo_bytes.seek(0)  # Возвращаем указатель в начало
                
                # Получаем расширение файла из пути
                _, ext = os.path.splitext(file_info.file_path)
                if not ext:
                    ext = '.jpg'  # Используем .jpg если расширение не определено
                
                # Сохраняем информацию о фото
                photos.append({
                    'file_bytes': photo_bytes,
                    'file_name': f'photo_{len(photos) + 1}{ext}',
                    'mime_type': 'image/jpeg' if ext.lower() in ['.jpg', '.jpeg'] else 'image/png'
                })
                print(f"Скачано фото размером {photo_bytes.getbuffer().nbytes} байт")
            except Exception as e:
                print(f"Ошибка при скачивании фото: {e}")
        
        # Скачиваем документы в память, если они есть
        documents = []
        if message.document:
            file_id = message.document.file_id
            file_name = message.document.file_name
            mime_type = message.document.mime_type
            
            try:
                file_info = await bot.get_file(file_id)
                
                # Скачиваем в BytesIO вместо файла
                doc_bytes = io.BytesIO()
                await bot.download_file(file_info.file_path, destination=doc_bytes)
                doc_bytes.seek(0)  # Возвращаем указатель в начало
                
                # Сохраняем информацию о документе
                documents.append({
                    'file_bytes': doc_bytes,
                    'file_name': file_name,
                    'mime_type': mime_type
                })
                print(f"Скачан документ {file_name} размером {doc_bytes.getbuffer().nbytes} байт")
            except Exception as e:
                print(f"Ошибка при скачивании документа: {e}")
        
        # Вызываем функцию обработки с передачей объектов файлов
        # и получаем текст ответа
        response_text, issue_info = await handle_issue(
            issue_id=issue_id,
            text=text,
            photos=photos, 
            documents=documents
        )
        
        # Отправляем ответ пользователю с текстом из handle_issue
        await message.answer(response_text)
    except Exception as e:
        # В случае ошибки логируем и сообщаем пользователю
        print(f"Ошибка при обработке обращения: {e}")
        await message.answer(
            "Произошла ошибка при обработке вашего обращения. Пожалуйста, попробуйте еще раз позже."
        )
    
    # Clear state and return to start
    await state.clear()
    keyboard = create_start_keyboard()
    await message.answer(
        "Если у вас есть еще вопросы, нажмите кнопку 'Новое обращение'.",
        reply_markup=keyboard
    )

def register_handlers(dp):
    """
    Registers all handlers to the dispatcher
    
    Args:
        dp: The dispatcher instance
    """
    dp.include_router(router) 