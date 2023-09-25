import os


# openai.api_key = "<YOUR_OPENAI_API_KEY
key = open('../api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key

CHROMA_COLLECTION_NAME = "kakao-bot"
CHROMA_PERSIST_DIR = os.path.abspath("chroma-persist")
DATA_DIR = os.path.abspath("datas")
from langchain.document_loaders import (
    NotebookLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

LOADER_DICT = {
    "txt": TextLoader
}


def upload_embedding_from_file(file_path):
    loader = LOADER_DICT.get(file_path.split(".")[-1])
    if loader is None:
        raise ValueError("Not supported file type")
    documents = loader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []
    print(dir_path)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                try:
                    upload_embedding_from_file(file_path)
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")
                    failed_upload_files.append(file_path)


upload_embeddings_from_dir(DATA_DIR)
