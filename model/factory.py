from abc import ABC, abstractmethod
from typing import Optional
import os
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config_handler import rag_conf

# 自动将 MOONSHOT_API_KEY 映射为 OPENAI_API_KEY（ChatOpenAI 读取的变量名）
_moonshot_key = os.environ.get("MOONSHOT_API_KEY", "")
if _moonshot_key and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = _moonshot_key


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["chat_base_url"],
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return HuggingFaceEmbeddings(
            model_name=rag_conf["embedding_model_path"],
        )


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()
