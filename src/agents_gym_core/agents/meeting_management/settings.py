from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # GetOutlookEvents Document
    GET_OUTLOOK_EVENTS_TOOL_DOCUMENT: str = "handcrafted_outlook_events.json"


meeting_management_settings = Settings()
