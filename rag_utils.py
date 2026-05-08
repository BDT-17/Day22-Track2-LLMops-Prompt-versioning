import hashlib
import math
import re
import json
from pathlib import Path
from collections import Counter
from typing import Callable, List

from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import PrivateAttr

from config import AppConfig, DATA_DIR


KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 3


class SimpleHashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        counts = Counter(tokens)
        for token, count in counts.items():
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[index] += float(count)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


def create_llm(config: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.openai_model,
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        temperature=0,
    )


class RemoteChatClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def invoke(self, payload):
        messages = payload if isinstance(payload, list) else payload.to_messages()
        result = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        parts = []
        for chunk in result:
            if isinstance(chunk, str):
                if not chunk.startswith("data: "):
                    continue
                body = chunk[6:].strip()
                if body == "[DONE]":
                    continue
                try:
                    chunk = json.loads(body)
                except Exception:
                    continue
            choices = getattr(chunk, "choices", None)
            if choices is None and isinstance(chunk, dict):
                choices = chunk.get("choices", [])
            if choices is None:
                continue
            for choice in choices:
                delta = choice.get("delta", {}) if isinstance(choice, dict) else getattr(choice, "delta", {})
                content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                if content:
                    parts.append(content)
        if parts:
            return "".join(parts)
        choices = getattr(result, "choices", [])
        if choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is not None:
                return getattr(message, "content", "")
        return str(result)

    def __call__(self, payload):
        return self.invoke(payload)

    def answer(self, question: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Answer using only the provided context. "
                    "If the answer is not in the context, say you do not have enough information.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ]
        return self.invoke(messages)


class StreamChatModel(BaseChatModel):
    _client: OpenAI = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__()
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def _llm_type(self) -> str:
        return "stream-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
        parts = []
        for chunk in result:
            if isinstance(chunk, str):
                if not chunk.startswith("data: "):
                    continue
                body = chunk[6:].strip()
                if body == "[DONE]":
                    continue
                try:
                    chunk = json.loads(body)
                except Exception:
                    continue
            choices = getattr(chunk, "choices", None)
            if choices is None and isinstance(chunk, dict):
                choices = chunk.get("choices", [])
            if choices is None:
                continue
            for choice in choices:
                delta = choice.get("delta", {}) if isinstance(choice, dict) else getattr(choice, "delta", {})
                content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                if content:
                    parts.append(content)
        text = "".join(parts).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def create_embeddings(config: AppConfig) -> OpenAIEmbeddings:
    if config.embedding_mode == "local":
        return SimpleHashEmbeddings()
    return OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
    )


def create_eval_llm(config: AppConfig) -> ChatOpenAI:
    return StreamChatModel(
        api_key=config.ragas_llm_api_key,
        base_url=config.ragas_base_url,
        model=config.ragas_llm_model,
    )


def create_eval_embeddings(config: AppConfig) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.ragas_embeddings_model,
        api_key=config.ragas_llm_api_key,
        base_url=config.ragas_base_url,
    )


def load_knowledge_base(path: Path = KNOWLEDGE_BASE_PATH) -> str:
    return path.read_text(encoding="utf-8")


def build_vectorstore(config: AppConfig) -> FAISS:
    text = load_knowledge_base()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    embeddings = create_embeddings(config)
    return FAISS.from_texts(chunks, embeddings)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_prompt_templates() -> dict[str, ChatPromptTemplate]:
    system_v1 = (
        "You are a helpful AI assistant. Answer the user using only the provided context. "
        "Keep the response concise, factual, and within 2 to 4 sentences. "
        "If the answer is not supported by the context, say you do not have enough information.\n\n"
        "Context:\n{context}"
    )
    system_v2 = (
        "You are an expert AI tutor. Use only the provided context.\n"
        "Write a structured answer with a short lead sentence followed by 2 concise supporting sentences. "
        "Be explicit when the context does not contain enough information.\n\n"
        "Context:\n{context}"
    )

    return {
        "v1": ChatPromptTemplate.from_messages(
            [("system", system_v1), ("human", "{question}")]
        ),
        "v2": ChatPromptTemplate.from_messages(
            [("system", system_v2), ("human", "{question}")]
        ),
    }


def build_rag_chain(vectorstore: FAISS, llm: ChatOpenAI, prompt: ChatPromptTemplate):
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def create_remote_chat_client(config: AppConfig) -> RemoteChatClient:
    return RemoteChatClient(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.openai_model,
    )


def run_prompt_with_retriever(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_text = format_docs(docs)
    if hasattr(llm, "invoke") and not isinstance(llm, ChatOpenAI):
        prompt_value = prompt.invoke({"context": context_text, "question": question})
        answer = llm.invoke(prompt_value)
    else:
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": context_text, "question": question}
        )
    if hasattr(answer, "content"):
        answer = answer.content
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }


def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return "v1" if hash_int % 2 == 0 else "v2"


def build_prompt_hub_names(project_name: str) -> dict[str, str]:
    slug = project_name.strip().lower().replace(" ", "-")
    return {
        "v1": f"{slug}-rag-prompt-v1",
        "v2": f"{slug}-rag-prompt-v2",
    }
