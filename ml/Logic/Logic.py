from rag_search import RAGService

# Logic_with_clear_history.py
# Инициализация сервиса и истории
service = RAGService('database/frida.pkl')
chat_histories = {'default_user': []}  # хранит список предыдущих запросов пользователя
MAX_HISTORY = 3  # максимальное число записей в истории

# Функции для работы с историей

def add_to_history(user_id, text):
    """Добавляет запрос пользователя в историю и ограничивает её длину"""
    chat_histories.setdefault(user_id, []).append(text)
    if len(chat_histories[user_id]) > MAX_HISTORY:
        chat_histories[user_id].pop(0)


def build_context(user_id):
    """Возвращает строку из предыдущих запросов, разделённых переводом строки"""
    return "\n".join(chat_histories.get(user_id, []))

if __name__ == '__main__':
    user_id = 'default_user'
    print("Введите вопрос (или '-' для очистки истории):")

    while True:
        user_input = input().strip()

        # Очистка истории
        if user_input == '-':
            chat_histories[user_id] = []
            print("История запросов очищена.")
            print("Введите следующий вопрос (или '-' для очистки истории):")
            continue

        # Добавляем текущий запрос в историю
        add_to_history(user_id, user_input)

        # Формируем только историю запросов без префиксов
        history_ctx = build_context(user_id)
        full_query = f"{history_ctx}\n{user_input}" if history_ctx else user_input

        # Передаём в RAG только историю запросов + текущий запрос
        answer = service.get_answer(full_query)

        # Выводим ответ
        print("Bot:", answer)
        print("---")
        print("Введите следующий вопрос (или '-' для очистки истории):")
