
from app.ml.ocr.photos_get_text import get_answer_from_image
from app.ml.rag.rag_search import rag_service
import tempfile

def process_issue(issue: dict) -> str:
    text = issue.get("text")
    photo = issue.get("photo")
    if photo is None:
        return rag_service.get_answer(text)
    else:
        photo_ext = photo['file_name'].split(".")[1]
        photo_bytesIO = photo['file_bytes']
        temp_file = tempfile.NamedTemporaryFile(delete_on_close=True, suffix=photo_ext)
        temp_file.write(photo_bytesIO.read())
        temp_file.flush()
        result = get_answer_from_image(temp_file.name)
        return result