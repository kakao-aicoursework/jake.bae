"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.

import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain.prompts.chat import ChatPromptTemplate
from pprint import pprint

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import numpy as np

# openai.api_key = "<YOUR_OPENAI_API_KEY
key = open('../api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key

PROMPT_DIR = os.path.abspath("prompt_template")
INTENT_PROMPT_TEMPLATE = os.path.join(PROMPT_DIR, "parse_intent.txt")
INTENT_LIST_TXT = os.path.join(PROMPT_DIR, "intent_list.txt")
KAKAO_SINK_PROMPT = os.path.join(PROMPT_DIR, "kakao_sink_prompt.txt")
KAKAO_SOCIAL_PROMPT = os.path.join(PROMPT_DIR, "kakao_social_prompt.txt")
KAKAO_CHANNEL_PROMPT = os.path.join(PROMPT_DIR, "kakao_channel_prompt.txt")

CHROMA_COLLECTION_NAME = "kakao-bot"
CHROMA_PERSIST_DIR = os.path.abspath("chroma-persist")
_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()

llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

default_chain = ConversationChain(llm=llm, output_key="output")

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)

kakao_sink_chain = create_chain(
    llm=llm,
    template_path=KAKAO_SINK_PROMPT,
    output_key="output",
)

kakao_social_chain = create_chain(
    llm=llm,
    template_path=KAKAO_SOCIAL_PROMPT,
    output_key="output",
)

kakao_channel_chain = create_chain(
    llm=llm,
    template_path=KAKAO_CHANNEL_PROMPT,
    output_key="output",
)


def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs


def kakao_chatbot_answer(user_message: str) -> str:
    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    intent = parse_intent_chain.run(context)

    if intent == "sink":
        context["related_documents"] = query_db(context["user_message"])
        answer = kakao_sink_chain.run(context)
    elif intent == "social":
        context["related_documents"] = query_db(context["user_message"])
        answer = kakao_social_chain.run(context)
    elif intent == "channel":
        context["related_documents"] = query_db(context["user_message"])
        answer = kakao_channel_chain.run(context)
    else:
        answer = default_chain.run(context["user_message"])

    return answer


def chatbot_answer_using_chatgpt(question: str) -> str:
    response = kakao_chatbot_answer(question)
    return response


class Message(Base):
    question: str
    answer: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    # @pc.var
    # def output(self) -> str:
    #     if not self.text.strip():
    #         return "Chatbot Answer will appear here."
    #     answer = ""
    #     return answer

    def post(self):
        if not self.text.strip():
            return

        self.messages = [
            Message(
                question=self.text,
                answer= chatbot_answer_using_chatgpt(self.text),
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
            )
        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Chatbot 🗺", font_size="2rem"),
        pc.text(
            "Send messages to this chatbot",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.question),
            down_arrow(),
            text_box(message.answer),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="send a message",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Chatbot")
app.compile()
