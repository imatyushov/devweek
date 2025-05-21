from .local_deepseek import LocalDeepseek
from .cloud_deekpseek import CloudDeepseek
from dotenv import load_dotenv
import os
load_dotenv()

CLOUD_REASONER = "faq-rag-cloud-deepseek-reasoner"
LOCAL_QWEN = "faq-rag-local-deepseek-qwen-14B"

llms = {
        CLOUD_REASONER: CloudDeepseek(API_KEY=os.getenv("DEEPSEEK_API_KEY")),
        LOCAL_QWEN: LocalDeepseek()
}

def get_llm(name):
    return llms.get(name)

def get_configuration_llm():
    if os.getenv("APP_ENV") == "prod" or os.getenv("DEEPSEEK_API_KEY") is None:
        print("LLMFactory: using local LLM")
        return llms.get(LOCAL_QWEN)
    else:
        print("LLMFactory: using cloud LLM")
        return llms.get(CLOUD_REASONER)