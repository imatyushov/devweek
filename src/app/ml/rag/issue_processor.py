
from app.ml.ocr.photos_get_text import get_answer_from_image
#from app.ml.rag.rag_search import rag_service
from app.ml.rag.rag import get_answer
import tempfile
from .chat_history import storage

def process_issue(issue: dict) -> str:
    text = issue.get("text")
    photo = issue.get("photo")
    user_id = issue.get("user_id")
    if user_id is None:
        return "Не удалось определить идентификатор пользователя. Обратитесь к разработчикам приложения"
    if photo is None:
        storage.add_message(user_id, text)
        return get_answer(storage.get_user_history(user_id))
    else:
        photo_ext = "." + photo['file_name'].split(".")[1]
        photo_bytesIO = photo['file_bytes']
        temp_file = tempfile.NamedTemporaryFile(delete_on_close=True, suffix=photo_ext)
        temp_file.write(photo_bytesIO.read())
        temp_file.flush()
        result = get_answer_from_image(temp_file.name)
        return result