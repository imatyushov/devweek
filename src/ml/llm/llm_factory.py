from local_deepseek import LocalDeepseek
from cloud_deekpseek import CloudDeepseek
from dotenv import load_dotenv
import os

class LLMFactory:
    def __init__(self):
        load_dotenv()
        self.llms = {
            "cloud-deepseek-reasoner": CloudDeepseek(API_KEY=os.getenv("DEEPSEEK_API_KEY")),
            "local-deepseek-qwen-14B": LocalDeepseek()
        }
    def get_llm(self, name):
        return self.llms[name] if name in self.llms.keys else None