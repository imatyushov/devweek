import asyncio
from typing import List, Optional, Dict, Any
import io
from app.ml.rag.issue_processor import process_issue

async def handle_issue(*, issue_id: str, text: str = "", 
                       photos: List[Dict[str, Any]] = None, 
                       documents: List[Dict[str, Any]] = None):
    """
    Функция для обработки обращений пользователей с файлами в памяти
    
    Args:
        issue_id: Уникальный идентификатор обращения
        text: Текст обращения
        photos: Список словарей с данными о фотографиях
            [{'file_bytes': BytesIO, 'file_name': str, 'mime_type': str}, ...]
        documents: Список словарей с данными о документах
            [{'file_bytes': BytesIO, 'file_name': str, 'mime_type': str}, ...]
    
    Returns:
        Tuple[str, Dict]: Кортеж из (текст ответа пользователю, информация об обращении)
    """
    photos = photos or []
    documents = documents or []
    
    print(f"Обработка обращения ID: {issue_id}")
    
    # Формируем текст с информацией о полученном обращении
    debug_info = []
    
    # Выводим информацию о тексте обращения
    if text:
        debug_info.append(f"Получен текст обращения: {text}")
        print(f"Текст обращения: {text}")
    else:
        debug_info.append("Текст обращения отсутствует")
        print("Текст обращения отсутствует")
    
    # Выводим информацию о фотографиях
    if photos:
        photo_info = f"Получено фотографий: {len(photos)}"
        debug_info.append(photo_info)
        print(f"\nФотографии ({len(photos)}):")
        for i, photo_data in enumerate(photos):
            photo_bytes = photo_data['file_bytes']
            photo_size = photo_bytes.getbuffer().nbytes
            photo_detail = f"  {i+1}. {photo_data['file_name']} - {photo_size} байт, тип: {photo_data['mime_type']}"
            print(photo_detail)
            
            # Здесь можно добавить логику анализа изображений
            # Например: print(f"     - Анализ изображения: обнаружены объекты X, Y, Z")
    else:
        debug_info.append("Фотографий нет")
        print("\nФотографий нет")
    
    # Выводим информацию о документах
    if documents:
        doc_info = f"Получено документов: {len(documents)}"
        debug_info.append(doc_info)
        print(f"\nДокументы ({len(documents)}):")
        for i, doc_data in enumerate(documents):
            doc_bytes = doc_data['file_bytes']
            doc_size = doc_bytes.getbuffer().nbytes
            doc_detail = f"  {i+1}. {doc_data['file_name']} - {doc_size} байт, тип: {doc_data['mime_type']}"
            print(doc_detail)
            
            # Выполняем специальную обработку для разных типов документов
            if doc_data['mime_type'] == 'application/pdf':
                debug_info.append(f"Получен PDF документ: {doc_data['file_name']}")
                print(f"     - PDF документ: можно извлечь текст или проанализировать содержимое")
    else:
        debug_info.append("Документов нет")
        print("\nДокументов нет")
    
    print(f"\nОбращение {issue_id} успешно обработано")
    
    # Формируем информацию об обращении
    issue_info = {
        "issue_id": issue_id,
        "text": text,
        "photo_count": len(photos),
        "document_count": len(documents),
        "processed": True
    }
    
    result = await asyncio.to_thread(process_issue, issue_info) # since the LLMs calls are sync for now we should move this out of the event loop
    return result, issue_info
    
    # Формируем ответ пользователю
    if text or photos or documents:
        response_text = (
            f"Спасибо за ваше обращение! Оно зарегистрировано под номером: {issue_id[:8]}.\n\n"
            f"Мы получили следующую информацию:\n"
        )
        
        if text:
            response_text += f"• Описание проблемы\n"
        
        if photos:
            response_text += f"• Фотографии: {len(photos)} шт.\n"
            
        if documents:
            response_text += f"• Документы: {len(documents)} шт.\n"
            
        response_text += "\nВаше обращение будет рассмотрено в ближайшее время."
    else:
        response_text = "Извините, но в вашем обращении не было никакой информации. Пожалуйста, опишите вашу проблему или приложите соответствующие фото/документы."
    
    return response_text, issue_info 