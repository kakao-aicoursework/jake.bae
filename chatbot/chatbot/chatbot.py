"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

import numpy as np

# openai.api_key = "<YOUR_OPENAI_API_KEY>"
key = open('../api-key', 'r').readline()
openai.api_key = key




parallel_example = {
    "한국어": ["오늘 날씨 어때", "딥러닝 기반의 AI기술이 인기를끌고 있다."],
    "영어": ["How is the weather today", "Deep learning-based AI technology is gaining popularity."],
    "일본어": ["今日の天気はどうですか", "ディープラーニングベースのAIテクノロジーが人気を集めています。"]
}


def chatbot_answer_using_chatgpt(text, messages) -> str:
    # fewshot 예제를 만들고
    fewshot_messages = []

    def build_fewshot():

        for message in messages:
            fewshot_messages.append({"role": "user", "content": message.question})
            fewshot_messages.append({"role": "assistant", "content": message.answer})

        return fewshot_messages

    # system instruction 만들고
    system_instruction = f"assistant는 챗봇으로 동작한다. 이전에 나누었던 대화를 바탕으로 질문에 대한 가장 적절한 답변을 출력한다."

    # messages를만들고
    fewshot_messages = build_fewshot()

    messages = [{"role": "system", "content": system_instruction},
                *fewshot_messages,
                {"role": "user", "content": text}]

    # API 호출
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
    answer = response['choices'][0]['message']['content']
    # Return
    return answer


class Message(Base):
    question: str
    answer: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Chatbot Answer will appear here."
        answer = chatbot_answer_using_chatgpt(self.text, self.messages)
        return answer

    def post(self):
        self.messages = [
            Message(
                question=self.text,
                answer=self.output,
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
