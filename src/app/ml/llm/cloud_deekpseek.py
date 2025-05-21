from langchain_deepseek import ChatDeepSeek

def CloudDeepseek(API_KEY):
    return ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=0.5,
        top_p=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=API_KEY
    )
