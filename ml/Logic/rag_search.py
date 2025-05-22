import numpy as np
import faiss
import pandas as pd
import pickle
import re
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class RAGService:
    def __init__(self, text_db_path: str):
        self.df = self._load_data(text_db_path)
        self.embedding_model = SentenceTransformer('ai-forever/FRIDA').to('cpu')
        self.question_index = self._create_faiss_index(self.df['question_embedding'])
        self.content_index = self._create_faiss_index(self.df['content_embedding'])
        self.llm = ChatOllama(
            model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K",
            baseUrl="http://localhost:11434",
            memory=None,
        )

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
        return res.sort_values('similarity_score', ascending=False).reset_index(drop=True)

    def combined_search(self, query: str, top_k: int = 6) -> pd.DataFrame:
        q_res = self.rag_search(query, 'question', top_k)
        c_res = self.rag_search(query, 'content', top_k)
        q_res['source'] = 'question'
        c_res['source'] = 'context'
        combined = pd.concat([q_res, c_res], ignore_index=True)
        combined = combined.sort_values('similarity_score', ascending=False).drop_duplicates('question').reset_index(drop=True)
        return combined

    def generate_prompt(self, user_question: str, candidates: pd.DataFrame) -> str:
        candidates_list = candidates['question'].tolist()[:6]
        numbered = [f"{i+1}. {q}" for i, q in enumerate(candidates_list)]
        text_list = "\n".join(numbered)
        return (
            f"You are a semantic matcher.\n"
            f"User's question: \"{user_question}\"\n\n"
            f"From the following list of candidate questions, select the one most semantically similar to the user's question."
            f" If none match well enough, answer 'number 0'.\n\n"
            f"{text_list}" 
        )

    def extract_reversed_result_number(self, text: str) -> int:
        rev = text[::-1]
        m = re.search(r"rebmun", rev)
        if not m:
            return 0
        num_rev = rev[:m.start()].strip()[::-1]
        m2 = re.search(r"\d+", num_rev)
        return int(m2.group()) if m2 else 0

    def find_match_index(self, question: str, candidates: pd.DataFrame) -> int:
        prompt_text = self.generate_prompt(question, candidates)
        system_msg = SystemMessage(content="Select the number of the most semantically similar question from the list.")
        human_msg = HumanMessage(content=prompt_text)
        try:
            result = self.llm.generate([[system_msg, human_msg]])
            resp = result.generations[0][0].text.strip()
            return self.extract_reversed_result_number(resp)
        except Exception as e:
            print(f"LLM error: {e}")
            return 0

    def get_answer(self, user_question: str, search_type: str = 'content', top_k: int = 6) -> str:
        if search_type in ('question', 'content'):
            results = self.rag_search(user_question, search_type, top_k)
        else:
            results = self.combined_search(user_question, top_k)
        choice = self.find_match_index(user_question, results)
        if choice < 1 or choice > len(results):
            return "К сожалению, я не смог найти ответ. Перефразируйте запрос или обратитесь к оператору."
        return results.loc[choice-1, 'answer']

if __name__ == '__main__':
    svc = RAGService('database/frida.pkl')
    print(svc.get_answer('хочу новый аккаунт'))