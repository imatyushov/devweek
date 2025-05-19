import os
import re
from pathlib import Path
from PIL import Image, ImageOps
import pyocr
import pyocr.builders
from fuzzywuzzy import fuzz
import multiprocessing
import threading

tools = pyocr.get_available_tools()
if not tools:
    raise RuntimeError("OCR инструмент не найден.")
tool = tools[0]

TIMEOUT = 2.0
ERRORS = [
    r"упс, неправильная ссылка.*рекомендациях",
    r"автор удалил это видео",
    r"видео на модерации",
    r"видео ещё не опубликовано.*откроется позже",
    r"автор скрыл это видео",
    r"данное видео временно недоступно.*",
    r"оформить за \d+ ?₽",
    r"оформить подписку",
    r"видео 18\+?.*несовершеннолетним пользователям",
    r"видео недоступно по решению правообладателя или у вас включен vpn",
    r"закончились права на показ этого видео или у вас включен vpn",
    r"трансляция ещё не началась",
    r"до начала трансляции осталось",
    r"трансляция закончилась.*запись",
    r"просмотреть это видео только на",
    r"такого видео нет",
    r"мне уже есть.*18.*лет",
]
compiled_patterns = [re.compile(p, re.IGNORECASE) for p in ERRORS]
FUZZY_THRESHOLD = 70
TIMEOUT_SENTINEL = object()

FOLDER = r"Data/03_dialog_files"
SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}


def preprocess_image(path):
    img = Image.open(path)
    gray = ImageOps.grayscale(img)
    bw = gray.point(lambda x: 0 if x < 128 else 255, mode='1')
    return bw.convert('L')


def ocr_worker(path, return_dict):
    try:
        img = preprocess_image(path)
        text = "" if img is None else tool.image_to_string(
            img, lang='rus', builder=pyocr.builders.TextBuilder()
        )
        return_dict['text'] = text
    except Exception:
        return_dict['text'] = ""


def run_with_timeout(path):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    proc = multiprocessing.Process(target=ocr_worker, args=(path, return_dict))
    proc.start()

    def watcher():
        proc.join(TIMEOUT)
        if proc.is_alive():
            proc.terminate()
            return_dict['timeout'] = True

    watch_thread = threading.Thread(target=watcher)
    watch_thread.start()

    proc.join()
    watch_thread.join()

    if return_dict.get('timeout'):
        return TIMEOUT_SENTINEL
    return return_dict.get('text', "")


def find_matches(text: str):
    if not text:
        return []
    norm = ' '.join(text.lower().split())
    results = []
    for pat in compiled_patterns:
        if pat.search(norm):
            results.append((pat.pattern, 100))
    for pat in ERRORS:
        raw = re.sub(r"[^\w\s]", "", pat.lower())
        score = fuzz.partial_ratio(raw, norm)
        if score >= FUZZY_THRESHOLD:
            results.append((pat, score))
    return results


def process_single(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    text = run_with_timeout(path)
    if text is TIMEOUT_SENTINEL:
        print(f"[TIMEOUT] {path}")
        return None

    matches = find_matches(text)
    return text, matches


def main():
    for fname in sorted(os.listdir(FOLDER)):
        ext = Path(fname).suffix.lower()
        if ext not in SUPPORTED_EXT:
            continue
        path = os.path.join(FOLDER, fname)
        if not os.path.isfile(path):
            continue

        print(f"\n==== {fname} ====")
        result = process_single(path)
        if result is None:
            continue
        text, matches = result

        print(text.strip() or "[нет текста]")
        if matches:
            print('+' * 30)
            print("Найдены совпадения:")
            for pat, score in sorted(matches, key=lambda x: -x[1]):
                print(f"«{pat}» ({score})")
            print('+' * 30)
        else:
            print("Совпадений не найдено.")


if __name__ == '__main__':
    main()
    # a=process_single(r'D:\itmo\Data\05_Errors\All_images\1.png')
    # print(a[1][0][0])