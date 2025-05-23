from langchain_community.chat_models import ChatOllama

def LocalDeepseek():
    return ChatOllama(
            model="llama3:8b",
            save_history=False,
            max_tokens=512,
            system="You are a helpful AI assistant for Rutube platform."
        )