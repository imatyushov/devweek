import torch
import pickle
import numpy as np
import json
from transformers import pipeline, AutoTokenizer, AutoModel
from natasha import Doc, NewsEmbedding, NewsNERTagger, NewsMorphTagger, Segmenter
import stanza
from collections import defaultdict
import pandas as pd
from langchain_ollama import ChatOllama


class TextVectorizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализация компонентов Natasha
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)
        self.morph_tagger = NewsMorphTagger(self.emb)
        
        # Инициализация NER пайплайна
        self.ner_pipeline = pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            tokenizer="Gherman/bert-base-NER-Russian",
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="simple"
        )
        
        # Инициализация Stanza
        self.nlp = stanza.Pipeline(
            lang='ru', 
            processors='tokenize,lemma,ner', 
            use_gpu=True,
            tokenize_pretokenized=True
        )
        
        # Загрузка модели SBERT
        model_name = 'ai-forever/sbert_large_mt_nlu_ru'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def _neural_normalize(self, text):
        """Лемматизация текста с помощью Stanza"""
        doc = self.nlp(text)
        return " ".join([word.lemma for sent in doc.sentences for word in sent.words])
    
    def _replace_entities(self, text):
        """Замена именованных сущностей"""
        try:
            text = str(text).strip()
            if not text:
                return text
            
            entities = []
            
            # Извлечение сущностей из обеих моделей
            hf_entities = self.ner_pipeline(text)
            for ent in hf_entities:
                if isinstance(ent, dict):
                    entities.append((ent['start'], ent['end'], ent['entity_group']))
            
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.ner_tagger)
            for span in doc.spans:
                entities.append((span.start, span.stop, span.type))
            
            # Сортировка и замена
            entities = sorted(entities, key=lambda x: x[1]-x[0], reverse=True)
            counters = defaultdict(int)
            replacements = []
            
            for start, end, ent_type in entities:
                counters[ent_type] += 1
                replacement = f"{ent_type}-{chr(64 + counters[ent_type])}"
                replacements.append((start, end, replacement))
            
            text_list = list(text)
            for start, end, replacement in sorted(replacements, reverse=True, key=lambda x: x[0]):
                text_list[start:end] = list(replacement)
            
            return ''.join(text_list)
        
        except Exception as e:
            print(f"Ошибка замены сущностей: {str(e)}")
            return text
    
    def _get_embeddings(self, texts, batch_size=8):
        """Генерация эмбеддингов для списка текстов"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
            
            del inputs, outputs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return np.array(embeddings)
    
    def process(self, text):
        """
        Основной метод обработки текста
        Возвращает усредненный вектор-эмбеддинг
        """
        # Шаг 1: Нормализация текста
        normalized = self._neural_normalize(text)
        
        # Шаг 2: Замена сущностей
        ner_replaced = self._replace_entities(text)
        
        # Шаг 3: Генерация эмбеддингов
        embeddings = self._get_embeddings([text, normalized, ner_replaced], batch_size=3)
        
        # Шаг 4: Усреднение эмбеддингов
        return np.mean(embeddings, axis=0)
    
vectorizer = TextVectorizer()
with open('ml/df_embed.pkl', 'rb') as f:
    df=pickle.load(f)
    
def process_sample(df_database: pd.DataFrame, text:str) -> pd.DataFrame:
   
    vector = vectorizer.process(text)
    print(vector.shape)

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    df_new=df_database.copy()
    df_new['similarity'] = df_new['average_embedding'].apply(lambda e: cosine_similarity(e, vector))
    top5 = df_new.nlargest(5, 'similarity').copy()
    top5['distance'] = 1 - top5['similarity']
    top5 = top5.drop(columns=['similarity'])

    return top5

from dotenv import load_dotenv

load_dotenv()

import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import os
import asyncio
API_TOKEN = os.getenv("TG_BOT_TOKEN")
logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Вас приветствует чат-бот поддержки Rutube. Буду рад ответить на любые вопросы!")
    

@dp.message()
async def help_user(message: types.Message):
    try:
        top5 = process_sample(df, message.text)
        await message.answer(top5.iloc[0]["answer"])
    except:
        await message.answer("При работе сервиса возникла ошибка. Мы уже сообщили в техподдержку")    
async def main():
    await dp.start_polling(bot)       
        
if __name__ == "__main__":
    asyncio.run(main())
    
    
    
#def main():
#    while True:
#        print(">", end="")
#        query = input()
#        top5 = process_sample(df, query)
#        scripts = []
#        for id, row in top5.iterrows():
#            scripts.append(json.dumps({"id": id, " question" : row["question"]}))
#        llm = ChatOllama(
#            model="llama3.1",
#            temperature=0
#        )
#        scripts = "\n".join(scripts)
#        prompt = f'''Найди id вопроса, который похож на представленный. {query}
#        Вопросы: {scripts}'''
#        question = llm.invoke(prompt).text()
#        print(f"LLM id: {question}")
#        print("Chat: " + top5.iloc[0]["answer"])