from rag_search import RAGService
USER_QUESTION = 'шла саша по шоссе и сосала сушку'
PATH = 'database/frida.pkl'


service=RAGService(PATH)


while True:
    user_question = input()
    print(service.get_answer(user_question))
