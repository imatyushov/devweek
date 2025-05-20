from aiogram.fsm.state import StatesGroup, State

class SupportState(StatesGroup):
    issue = State()  # State for processing the user's issue 