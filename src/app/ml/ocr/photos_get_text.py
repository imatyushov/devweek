import os
import re
from pathlib import Path
from PIL import Image, ImageOps
import easyocr
from fuzzywuzzy import fuzz
import multiprocessing
import threading
import pandas as pd


TIMEOUT = 20.0
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
    r"видео недоступно"
]
compiled_patterns = [re.compile(p, re.IGNORECASE) for p in ERRORS]
FUZZY_THRESHOLD = 70
TIMEOUT_SENTINEL = object()

FOLDER = r"app/ml/images"
SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

reader = easyocr.Reader(['ru'])

def preprocess_image(path):
    ### change path to bin photo
    img = Image.open(path) # del row

    gray = ImageOps.grayscale(img)
    bw = gray.point(lambda x: 0 if x < 128 else 255, mode='1')
    return bw.convert('L')

def ocr_worker(path, return_dict):
    try:
        img = preprocess_image(path)
        text = ""
        if img is not None:
            text = " ".join([line[1] for line in reader.readtext(img)])
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
    ###
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

def is_similar_question(input_text, questions, threshold=85):
    for question in questions:
        similarity = fuzz.token_set_ratio(input_text.lower(), question.lower())
        if similarity >= threshold:
            return True, question, similarity
    return False, None, 0

def extract_quoted_text(text):
    match = re.search(r'[\"«](.*?)[\"»]', text)
    return match.group(1) if match else text

# if __name__ == '__main__':
#     # main()
#     a = process_single(r'D:\itmo\Data\05_Errors\All_images\3.png')
#     text, matches = a
#     print(text)
#     print(matches)

#     df = pd.read_csv('database/02_Stubs.csv', sep='|')
#     df['question_clean'] = df['question'].apply(extract_quoted_text)

#     threshold = 70

#     match_found, matched_question, similarity = is_similar_question(text, df['question_clean'], threshold)

#     if match_found:
#         print(f"Найдено совпадение: '{matched_question}' (схожесть: {similarity}%)")
#     else:
#         print("Совпадений не найдено")

def get_answer_from_image(image_path, df_path='app/ml/db/csv/02_Stubs.csv', threshold=70):
    # Обработка изображения и извлечение текста
    result = process_single(image_path)
    print(result)
    if result is None:
        return None
    text, matches = result
    input_text = extract_quoted_text(text)

    # Загрузка и подготовка датафрейма
    df = pd.read_csv(df_path, sep='|')
    df['question_clean'] = df['question'].apply(extract_quoted_text)

    # Поиск максимальной схожести
    max_similarity = -1
    best_answer = None
    for index, row in df.iterrows():
        question = row['question_clean']
        similarity = fuzz.token_set_ratio(input_text.lower(), question.lower())
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = row['answer']

    return best_answer if max_similarity >= 30 else 'На удлось разобрать текст на фотографии, прикрипите изображение лучшего качества'

if __name__ == '__main__':
    answer = get_answer_from_image(photo_path)
    print(answer)
