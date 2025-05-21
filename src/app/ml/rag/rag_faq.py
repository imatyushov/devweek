from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import re
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
from natasha import Doc, NewsEmbedding, NewsNERTagger, NewsMorphTagger, Segmenter
import stanza
from collections import defaultdict
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import app.ml.llm.llm_factory as llm_factory

import os
DATABASE_PATH='app/ml/db/df_embed_frida.pkl'

#connect local llm
llm = llm_factory.get_configuration_llm()

#promt for preprocessing
# def build_prompt(user_query: str) -> str:
#     return (
#         "You are a user-query classifier. Follow these rules:\n"
#         "1. If the query contains profanity or non‑constructive negative criticism, return exactly this text: “'Script response to profanity and non‑constructive negative criticism'”.\n"
#         "2. Otherwise, return the original query unchanged.\n"
#         "3. Under no circumstances answer the user’s question itself.\n\n"
#         f"User query:\n{user_query}\n\n"
#         "Result: only one of the two options—either the original query or the specified template text. No additional text. Write the answer after the word result:\n"
#         "result:"
#     )

def build_prompt(user_query: str) -> str:
    return (
        "You are a user‑query classifier. Follow these instructions **literally**, without any deviation:\n"
        "\n"
        "1. If the user query contains any profanity or non‑constructive negative criticism, you **must** return exactly **one** of the following strings (no quotes, no extra whitespace, no punctuation):\n"
        "   Script response to profanity and non‑constructive negative criticism\n"
        "\n"
        "2. Otherwise, you **must** return exactly the original user query **character for character**, preserving casing, punctuation, spaces and line breaks.\n"
        "\n"
        "3. You **must not** answer or modify the content of the query itself.\n"
        "\n"
        "**Important constraints**:\n"
        "- Do **not** paraphrase, synonymize, translate or otherwise alter any wording.\n"
        "- Do **not** add or remove any characters (except when replacing the entire query with the exact fixed script above).\n"
        "- Output **only** a single line beginning with `result:` followed immediately (no space) by the required text.\n"
        "- No additional lines, no explanations, no quotes around the result.\n"
        "\n"
        "User query:\n"
        "----\n"
        f"{user_query}\n"
        "----\n"
        "\n"
        "Now produce exactly one line in this format:\n"
        "result:<your output as specified above>"
    )


def preprocess_question(text:str):
    msg = HumanMessage(content=build_prompt(text))
    chat_result = llm.generate([[msg]])
    response_text = chat_result.generations[0][0].text.strip()
    return response_text   

#extract after preprocessing
def extract_reversed_result(text: str) -> str:

    reversed_text = text[::-1]

    match = re.search(r":tluser\S*", reversed_text)
    
    if not match:
        return ""
    
    before_marker = reversed_text[:match.start()]
    return before_marker[::-1].strip()

#on cpu to let deepseek work
class TextVectorizer:
    def __init__(self):
        # Принудительно используем CPU для NLP-пайплайнов
        self.device = torch.device("cpu")

        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)
        self.morph_tagger = NewsMorphTagger(self.emb)

        self.ner_pipeline = pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            tokenizer="Gherman/bert-base-NER-Russian",
            device=-1,
            aggregation_strategy="simple"
        )

        self.nlp = stanza.Pipeline(
            lang='ru', 
            processors='tokenize,lemma,ner',
            use_gpu=False,
            tokenize_pretokenized=True
        )

        # SentenceTransformer для FRIDA
        self.embedder = SentenceTransformer('ai-forever/FRIDA', device='cpu')

    def _neural_normalize(self, text):
        doc = self.nlp(text)
        return " ".join([word.lemma for sent in doc.sentences for word in sent.words])

    def _replace_entities(self, text):
        try:
            text = str(text).strip()
            if not text:
                return text
            entities = []

            hf_entities = self.ner_pipeline(text)
            for ent in hf_entities:
                if isinstance(ent, dict):
                    entities.append((ent['start'], ent['end'], ent['entity_group']))

            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.ner_tagger)
            for span in doc.spans:
                entities.append((span.start, span.stop, span.type))

            entities = sorted(entities, key=lambda x: x[1] - x[0], reverse=True)
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
        # Используем SentenceTransformer внутри класса
        return self.embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True)

    def process(self, text):
        normalized = self._neural_normalize(text)
        ner_replaced = self._replace_entities(text)
        embeddings = self._get_embeddings([text, normalized, ner_replaced], batch_size=3)
        return np.mean(embeddings, axis=0)



#init models
vectorizer = TextVectorizer()    
#opening database
with open(DATABASE_PATH, 'rb') as f:
    df=pickle.load(f)



def process_sample(df_database: pd.DataFrame, text:str) -> pd.DataFrame:
   
    vector = vectorizer.process(text)
    print(vector.shape)

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    df_new=df_database.copy()
    df_new['similarity'] = df_new['average_embedding'].apply(lambda e: cosine_similarity(e, vector))
    top10 = df_new.nlargest(10, 'similarity').copy()
    top10['distance'] = 1 - top10['similarity']
    top10 = top10.drop(columns=['similarity'])

    return top10

#for final match
def generate_prompt(user_question: str, top5: pd.DataFrame) -> str:

    candidates = top5['question'].tolist()
    formatted_list = []
    for idx, q in enumerate(candidates, start=1):
        formatted_list.append(f"{idx}. {q}")
    candidates_text = "\n".join(formatted_list)

    prompt = (
        f"You are a semantic matcher.\n"
        f"User's question: \"{user_question}\"\n\n"
        f"From the following list of candidate questions, select the one that is most semantically similar to the user's question.\n"
        f"Respond with only the number of the chosen question after the word 'number' (e.g., 'number 1', 'number 2', etc.) without any additional text.\n\n"
        f"{candidates_text}"
    )

    return prompt

def find_match(USER_QUESTION, top10):
    msg = HumanMessage(content=generate_prompt(USER_QUESTION, top10))
    chat_result = llm.generate([[msg]])
    response_text = chat_result.generations[0][0].text.strip()
    return response_text   

def extract_reversed_result_number(text: str) -> str:

    reversed_text = text[::-1]
    print(reversed_text)
    # ищем первый фрагмент начинающийся с :tluser
    match = re.search(r"rebmun\S*", reversed_text)
    
    if not match:
        return ""
    
    before_marker = reversed_text[:match.start()]
    return before_marker[::-1].strip()

def inference(USER_QUESTION:str) -> str:

    """
    Main method for rag
    """
    clean_question=preprocess_question(USER_QUESTION)
    text_for_rag=extract_reversed_result(clean_question)

    print(text_for_rag)

    top10=process_sample(df, text=text_for_rag)
    print(top10)
    final_answer=find_match(USER_QUESTION, top10)
    print(final_answer)
    number=extract_reversed_result_number(final_answer)
    print(number)
    return top10.iloc[int(number)-1].answer