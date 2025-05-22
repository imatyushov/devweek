import threading

class UserChatHistory:
    def __init__(self, max_messages=3):
        self.max_messages = max_messages
        self.__messages = []
        self.__lock = threading.Lock()

    def add_message(self, message: str):
        with self.__lock:
            if len(self.__messages) == self.max_messages:
                self.__messages.pop(0)
            self.__messages.append(message)

    def get_history(self) -> str:
        with self.__lock:
            return "\n".join(self.__messages)


class UserChatHistoryStorage:
    def __init__(self):
        self.__user_histories = {}
        self.__lock = threading.Lock()

    def add_message(self, user_id, message: str):
        with self.__lock:
            history = self.__user_histories.get(user_id)
            if history is None:
                history = UserChatHistory()
                self.__user_histories[user_id] = history
        # Do message addition outside the global lock to avoid lock contention
        history.add_message(message)

    def get_user_history(self, user_id):
        with self.__lock:
            history = self.__user_histories.get(user_id)
        return None if history is None else history.get_history()
    
    def clear_user_history(self, user_id):
        with self.__lock:
            self.__user_histories[user_id] = UserChatHistory()
    
storage = UserChatHistoryStorage()
