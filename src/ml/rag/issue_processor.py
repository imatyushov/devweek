
from src.ml.ocr import process_photo

def process_issue(issue: dict) -> str:
    text = issue["text"]
    photos = issue["photos"]
    if photos in None:
        # search in FAQ
        pass
    else:
        # search in Stubs
        pass