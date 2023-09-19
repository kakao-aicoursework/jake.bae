"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.

import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from langchain.prompts.chat import ChatPromptTemplate
from pprint import pprint

import numpy as np

# openai.api_key = "<YOUR_OPENAI_API_KEY
# >"
key = open('../api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key

PROMPT_TEMPLATE = os.path.join("project_data_카카오싱크.txt")


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def kakao_sink_answer(question: str) -> str:
    system = "assistant는 카카오싱크 api 사용법을 설명 해주는 고객 지원 챗봇으로 동작한다. 고객의카카오싱크 api 사용법에 관한 질문에 대해 가장 적절하고 간결한 답변을 출력한다."

    chat_llm = ChatOpenAI(streaming=True, verbose=True, temperature=1, max_tokens=500, model='gpt-3.5-turbo')
    chat_llm([SystemMessage(content=system)])
    prompt_template = ChatPromptTemplate.from_template(
        template=read_prompt_template(PROMPT_TEMPLATE))
    chat_chain = LLMChain(llm=chat_llm, prompt=prompt_template, output_key = 'output' )
    result = chat_chain(dict(question = question))

    return result['output']

def chatbot_answer_using_chatgpt(question: str) -> str:
    response = kakao_sink_answer(question)
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
