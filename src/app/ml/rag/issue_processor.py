import io
from app.ml.ocr.photos_get_text import get_answer_from_image
#from app.ml.rag.rag_search import rag_service
#from app.ml.rag.rag import get_answer
from .rag_final import LlamaAssistant
import tempfile
from .chat_history import storage
from PIL import Image

llama_assistant = LlamaAssistant()

def clear_user_history(user_id: int):
    storage.clear_user_history(user_id)
    

def process_issue(issue: dict) -> str:
    text = issue.get("text")
    photo = issue.get("photo")
    user_id = issue.get("user_id")
    if user_id is None:
        return "Не удалось определить идентификатор пользователя. Обратитесь к разработчикам приложения"
    if photo is None:
        storage.add_message(user_id, text)
        llama_assistant.answer_question(storage.get_user_history(user_id))
        return llama_assistant.answer_question(storage.get_user_history(user_id))
    else:
        photo_ext = "." + photo['file_name'].split(".")[1]
        photo_bytesIO = photo['file_bytes']
        temp_file = tempfile.NamedTemporaryFile(delete_on_close=False, suffix=photo_ext)
        temp_file.write(photo_bytesIO.read())
        temp_file.flush()
        result = get_answer_from_image(temp_file.name)
        return result
if __name__=='__mane__':
    photo_path = photo_path
    img = Image.open(photo_path)
    img_bytes=io.BytesIO()
    img.save(img_bytes)
    process_issue({'text':'','photo':img_bytes, 'user_id':123})
