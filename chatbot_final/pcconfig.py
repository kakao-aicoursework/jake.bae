import pynecone as pc

class ChatbotFinalConfig(pc.Config):
    pass

config = ChatbotFinalConfig(
    app_name="chatbot_final",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)