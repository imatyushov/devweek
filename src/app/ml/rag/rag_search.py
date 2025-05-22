import numpy as np
import faiss
import pandas as pd
import pickle
import re
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import app.ml.llm.llm_factory as llm_factory


PATH = 'app/ml/db/embeddings/frida_new.pkl'

class RAGService:
    def __init__(self, text_db_path: str):
        self.df = self._load_data(text_db_path)
        self.embedding_model = SentenceTransformer('ai-forever/FRIDA').to('cpu')
        self.question_index = self._create_faiss_index(self.df['question_embedding'])
        self.content_index = self._create_faiss_index(self.df['content_embedding'])
        self.llm = llm_factory.get_configuration_llm()

    def _load_data(self, path: str) -> pd.DataFrame:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _create_faiss_index(self, embeddings_series: pd.Series) -> faiss.IndexFlatL2:
        embeddings = np.stack(embeddings_series.values).astype('float32')
        dim = embeddings.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(embeddings)
        return idx

    def rag_search(self, query: str, search_type: str = 'question', top_k: int = 6) -> pd.DataFrame:
        prefix = 'search_question:' if search_type == 'question' else 'search_content:'
        emb = self.embedding_model.encode(f"{prefix} {query}").astype('float32')
        idx = self.question_index if search_type == 'question' else self.content_index
        dists, idxs = idx.search(np.array([emb]), top_k)
        res = self.df.iloc[idxs[0]].copy()
        res['similarity_score'] = 1 - dists[0] / 2
        # sort by similarity and reset index
        res = res.sort_values('similarity_score', ascending=False).reset_index(drop=True)
        return res

    def generate_prompt(seld, user_question: str, top5: pd.DataFrame) -> str:

        # Формируем пронумерованный список кандидатов
        candidates = top5['question'].tolist()
        formatted_list = []
        for idx, q in enumerate(candidates, start=1):
            formatted_list.append(f"{idx}. {q}")
        candidates_text = "\n".join(formatted_list)

        # Обновлённый prompt
        prompt = (
            f"You are a semantic matcher.\n"
            f"User's question: \"{user_question}\"\n\n"
            f"From the following list of candidate questions, select the one that is most semantically similar to the user's question.\n"
            f"If none of the candidates are relevant or similar enough, respond with 'number 0'.\n"
            f"Respond with only the number of the chosen question after the word 'number' (e.g., 'number 1', 'number 2', etc.) without any additional text.\n\n"
            f"{candidates_text}"
        )
        print(prompt)
        return prompt

    def extract_reversed_result_number(self, text: str) -> int:
        rev = text[::-1]
        m = re.search(r"rebmun", rev)
        if not m:
            return 0
        num_rev = rev[:m.start()].strip()
        num_str = num_rev[::-1]
        m2 = re.search(r"\d+", num_str)
        return int(m2.group()) if m2 else 0

    def find_match_index(self, question: str, candidates: pd.DataFrame) -> int:
        prompt = self.generate_prompt(question, candidates)
        msg = HumanMessage(content=prompt)
        try:
            result = self.llm.generate([[msg]])
            resp = result.generations[0][0].text.strip()
            return self.extract_reversed_result_number(resp)
        except Exception as e:
            # Log or handle LLM errors; default to no match
            print(f"LLM error: {e}")
            return 0

    def get_answer(self, user_question: str, search_type: str = 'question', top_k: int = 6) -> str:
        results = self.rag_search(user_question, search_type, top_k)
        choice = self.find_match_index(user_question, results)
        print(choice)
        if choice < 1 or choice > len(results):
            return "К сожалению, я не смог найти ответ на ваш запрос. Перефразируйте его или обратитесь к оператору."
        return results.loc[choice - 1, 'answer']

rag_service = RAGService(PATH)