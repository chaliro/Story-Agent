import json
import os

import dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings




dotenv.load_dotenv()

def get_llm_user(model: str, temperature: float, type: str):
    dotenv.load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    if type == "ollama":
        return ChatOllama(model=model, temperature=temperature)
    elif type == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        raise TypeError(f"不支持的 LLM 类型: '{type}'。请选择 'ollama' 或 'openai'。")

def get_llm_temperature(model: str, temperature: float, type: str):
    dotenv.load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    if type == "ollama":
        return ChatOllama(model=model, temperature=temperature)
    elif type == "openai":
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        raise TypeError(f"不支持的 LLM 类型: '{type}'。请选择 'ollama' 或 'openai'。")

def get_llm():
    dotenv.load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    type = os.getenv("LLM_TYPE")
    if type == "ollama":
        return ChatOllama(model=os.getenv("LLM_MODEL"), temperature=os.getenv("LLM_TEMPERATURE"))
    elif type == "openai":
        return ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=os.getenv("LLM_TEMPERATURE"))
    else:
        raise TypeError(f"不支持的 LLM 类型: '{type}'。请选择 'ollama' 或 'openai'。")

def get_evaluation_llm():
    dotenv.load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    type = os.getenv("LLM_TYPE")
    if type == "ollama":
        return ChatOllama(model=os.getenv("LLM_MODEL"), temperature=os.getenv("LLM_EVALUATION_TEMPERATURE"))
    elif type == "openai":
        return ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=os.getenv("LLM_EVALUATION_TEMPERATURE"))
    else:
        raise TypeError(f"不支持的 LLM 类型: '{type}'。请选择 'ollama' 或 'openai'。")
def get_embedding_llm():
    dotenv.load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    type = os.getenv("LLM_EMBEDDING_TYPE")
    if type == "ollama":
        return OllamaEmbeddings(model=os.getenv("LLM_EMBEDDING_MODEL"))
    elif type == "openai":
        return OpenAIEmbeddings(model=os.getenv("LLM_EMBEDDING_MODEL"))
    else:
        raise TypeError(f"不支持的 LLM 类型: '{type}'。请选择 'ollama' 或 'openai'。")




