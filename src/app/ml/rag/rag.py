# rag_module.py
import numpy as np
import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from langchain_deepseek import ChatDeepSeek
import app.ml.llm.llm_factory as llm_factory

class RAGSystem:
    def __init__(self, model_name='ai-forever/FRIDA', db_path='app/ml/db/embeddings/frida.pkl'):
        self.embedding_model = SentenceTransformer(model_name)
        with open(db_path, 'rb') as f:
            self.df = pickle.load(f)
        self.question_index = self._create_faiss_index(self.df["question_embedding"])
        self.content_index = self._create_faiss_index(self.df["content_embedding"])

    def _create_faiss_index(self, embeddings_series):
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

    def combined_search(self, query: str, top_k: int = 6) -> pd.DataFrame:
        q_res = self.rag_search(query, 'question', top_k)
        c_res = self.rag_search(query, 'content', top_k)
        q_res['source'] = 'question'
        c_res['source'] = 'context'
        combined = pd.concat([q_res, c_res], ignore_index=True)
        combined = combined.sort_values('similarity_score', ascending=False).drop_duplicates('question').reset_index(drop=True)
        return combined


def get_most_relevant_answer(df: pd.DataFrame, user_question: str) -> str:
    contexts = [
        f"Контекст #{idx+1}:\nВопрос: {row['question']}\nОтвет: {row['answer']}\n"
        for idx, row in df.iterrows()
    ]
    contexts_block = "\n".join(contexts)

    system_prompt = (
        "Ты — AI-ассистент, отвечающий на вопросы пользователей о платформе Rutube."
        "Твои ответы должны основываться исключительно на предоставленном контексте."
        "Строго соблюдай следующие правила:"
        "Не делай предположений о наличии функционала на платформе, если это не указано явно в контексте."
        "Не предполагай взаимозаменяемость или аналогичность функционала между различными платформами (например, Android и iOS), если это не подтверждено в контексте."
        "Если в контексте отсутствует информация о запрашиваемом функционале или платформе, сообщи пользователю об отсутствии данных или задай уточняющий вопрос."
        "Не используй внешние знания или общие предположения."
        "Отвечай на русском языке, соблюдая профессиональный и информативный тон."
    )

    user_prompt = f"""
    Контекст:
    {contexts_block}

    Вопрос пользователя:
    "{user_question}"

Инструкции:
    Анализируй контекст:
        Используй только предоставленные пары вопрос-ответ.
        Не делай предположений о наличии или отсутствии функционала на платформе, если это не указано явно в контексте.
        Не предполагай взаимозаменяемость или аналогичность функционала между различными платформами, если это не подтверждено в контексте.

    Формат ответа:
        Если есть подходящий ответ — предоставь его.
        Если функция недоступна для указанной платформы или типа пользователя — укажи это.
        Если контекст не покрывает запрос или не хватает данных — задай один уточняющий вопрос.
        Избегай предоставления дополнительной информации, не относящейся напрямую к вопросу пользователя.
"""

    client = llm_factory.get_configuration_llm()
    response = client.invoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    return response.content

rag = RAGSystem()

def get_answer(question: str) -> str:
    results = rag.combined_search(question, 5)
    answer = get_most_relevant_answer(results, question)
    return answer
# Пример использования:
if __name__ == '__main__':
    question = "Как загрузить видео на Rutube?"
    get_answer(question)

    