from langchain_ollama import ChatOllama

def LocalDeepseek():
    return ChatOllama(
        model="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K", #need 16gm
        temperature=1,
        baseUrl="http://localhost:11434",
        memory=None,
    )