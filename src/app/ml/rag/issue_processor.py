
from app.ml.ocr.ocr import process_photo
from app.ml.rag.rag_faq import inference as faq_inference

def process_issue(issue: dict) -> str:
    text = issue.get("text")
    photos = issue.get("photos")
    if photos is None:
        return faq_inference(text)
    else:
        # search in Stubs
        return None