import pynecone as pc

class ChatbotLangChainConfig(pc.Config):
    pass

config = ChatbotLangChainConfig(
    app_name="chatbot_langchain",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)