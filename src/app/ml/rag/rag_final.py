from langchain_community.chat_models import ChatOllama
import numpy as np
import faiss
import pandas as pd
import pickle
import re
from sentence_transformers import SentenceTransformer
import app.ml.llm.llm_factory as llm_factory

class LlamaAssistant:
    def __init__(self,
                 embedding_model_name='ai-forever/FRIDA',
                 db_path='app/ml/db/embeddings/frida.pkl',
                 llm_model="llama3:8b"):
        # Инициализация языковой модели
        self.llm = llm_factory.get_configuration_llm()

        # Загрузка эмбеддинговой модели
        self.embedding_model = SentenceTransformer(embedding_model_name).to('cpu')

        # Загрузка базы данных с эмбеддингами
        with open(db_path, 'rb') as f:
            self.df = pickle.load(f)

        # Создаем faiss индексы
        self.question_index = self.create_faiss_index(self.df["question_embedding"])
        self.content_index = self.create_faiss_index(self.df["content_embedding"])

    @staticmethod
    def create_faiss_index(embeddings_series):
        embeddings = np.stack(embeddings_series.values).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def rag_search(self, query: str, search_type: str = 'question', top_k: int = 6) -> pd.DataFrame:
        prefix = 'search_question:' if search_type == 'question' else 'search_content:'
        emb = self.embedding_model.encode(f"{prefix} {query}").astype('float32')
        idx = self.question_index if search_type == 'question' else self.content_index
        dists, idxs = idx.search(np.array([emb]), top_k)
        res = self.df.iloc[idxs[0]].copy()
        res['similarity_score'] = 1 - dists[0] / 2
        return res.sort_values('similarity_score', ascending=False).reset_index(drop=True)

    def combined_search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        q_res = self.rag_search(query, 'question', top_k)
        c_res = self.rag_search(query, 'content', top_k)
        q_res['source'] = 'question'
        c_res['source'] = 'context'

        combined = pd.concat([q_res, c_res], ignore_index=True)
        combined = combined.sort_values('similarity_score', ascending=False)
        combined = combined.drop_duplicates('question').reset_index(drop=True)

        return combined.head(top_k)

    def get_most_relevant_answer(self, df: pd.DataFrame, user_question: str) -> str:
        contexts = [
            f"Контекст #{idx+1}:\nВопрос: {row['question']}\nОтвет: {row['answer']}\n"
            for idx, row in df.iterrows()
        ]
        contexts_block = "\n".join(contexts)

        system_prompt = (
            "Ты — AI-ассистент, отвечающий на вопросы пользователей о платформе Rutube. "
            "Твои ответы должны основываться исключительно на предоставленном контексте. "
            "Не делай предположений, если информации нет. "
            "Отвечай строго на русском языке, профессионально и информативно."
        )

        user_prompt = f"""
    Контекст:
    {contexts_block}

    Вопрос пользователя:
    \"{user_question}\"

    Пожалуйста, ответь максимально конкретно на основе только предоставленного контекста.  
    Если ответ по запросу отсутствует — задай один уточняющий вопрос, не добавляя объяснений и рассуждений.  
    Отвечай только по существу, без излишних деталей.
    """
        print(system_prompt)
        print(user_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.invoke(messages)
        return response.content.strip()

    def answer_question(self, question: str, top_k=3) -> str:
        results = self.combined_search(question, top_k)
        raw_answer = self.get_most_relevant_answer(results, question)
        
        # Пост-обработка ответа
        if "Ответ:" in raw_answer:
            return raw_answer.split("Ответ:")[-1].strip()
        return raw_answer.strip()


# Пример использования:
# llama_assistant = LlamaAssistant()
# answer = assistant.answer_question("Как создать плейлист?")
# print(answer)