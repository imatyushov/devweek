import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import numpy as np
import faiss
import pandas as pd
import pickle
import re
from sentence_transformers import SentenceTransformer


class DeepSeekAssistant:
    def __init__(self,
                 model_name="deepseek-ai/deepseek-llm-7b-chat",
                 embedding_model_name='ai-forever/FRIDA',
                 db_path="app/ml/db/embeddings/frida.pkl"):
        # Загрузка модели и токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

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

    def combined_search(self, query: str, top_k: int = 6) -> pd.DataFrame:
        q_res = self.rag_search(query, 'question', top_k)
        c_res = self.rag_search(query, 'content', top_k)
        q_res['source'] = 'question'
        c_res['source'] = 'context'
        combined = pd.concat([q_res, c_res], ignore_index=True)
        combined = combined.sort_values('similarity_score', ascending=False).drop_duplicates('question').reset_index(drop=True)
        return q_res

    def get_most_relevant_answer(self, df: pd.DataFrame, user_question: str) -> str:
        # Формируем контексты
        contexts = []
        for idx, row in df.iterrows():
            contexts.append(
                f"Контекст #{idx+1}:\nВопрос: {row['question']}\nОтвет: {row['answer']}\n"
            )
        contexts_block = "\n".join(contexts)

        # Системный промт
        system_prompt = (
            "Ты — AI-ассистент, отвечающий на вопросы пользователей о платформе Rutube. "
            "Твои ответы должны основываться исключительно на предоставленном контексте. "
            "Строго соблюдай следующие правила: "
            "Не делай предположений о наличии функционала на платформе, если это не указано явно в контексте. "
            "Не предполагай взаимозаменяемость или аналогичность функционала между различными платформами (например, Android и iOS), если это не подтверждено в контексте. "
            "Если в контексте отсутствует информация о запрашиваемом функционале или платформе, сообщи пользователю об отсутствии данных или задай уточняющий вопрос. "
            "Не используй внешние знания или общие предположения. "
            "Отвечай на русском языке, соблюдая профессиональный и информативный тон."
        )

        # Пользовательский промт
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Токенизация с шаблоном чата
        input_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Генерация
        outputs = self.model.generate(input_tensor, max_new_tokens=512)

        # Декодируем только сгенерированную часть
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result

    def answer_question(self, question: str, top_k=4) -> str:
        results = self.combined_search(question, top_k)
        raw_answer = self.get_most_relevant_answer(results, question)
        match = re.search(r'Ответ:\s*(.*)', raw_answer, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # Если "Ответ:" нет, возвращаем весь ответ
            return raw_answer.strip()


deepseek_assistant = DeepSeekAssistant()
# Пример использования:
# assistant = DeepSeekAssistant()
# answer = assistant.answer_question("доставка воды")
# print(answer)
