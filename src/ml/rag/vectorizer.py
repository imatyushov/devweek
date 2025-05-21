from transformers import pipeline
from natasha import Doc, NewsEmbedding, NewsNERTagger, NewsMorphTagger, Segmenter
import torch
import stanza
import numpy as np

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