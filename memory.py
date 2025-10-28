import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 环境设置 (请确保您的.env文件中有OPENAI_API_KEY) ---
from utils import get_embedding_llm

class MemorySystem:
    """
    一个管理小说创作记忆的系统，实现了技术报告中的“摘要层”和“分块层”。
    - 摘要层 (Summary Layer): 存储每个章节的摘要，用于快速检索高级情节。
    - 分块层 (Chunk Layer): 存储每个章节的详细文本块，用于检索具体描写和对话。
    """

    def __init__(self, embedding_model=None):
        """
        初始化记忆系统。
        """
        print("Initializing MemorySystem...")
        self.embeddings = embedding_model if embedding_model else get_embedding_llm()

        # --- 【核心修改 2】增加 chunk_size 来优化记忆的完整性 ---
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # 将每个文本块的大小从300增加到800
            chunk_overlap=150,     # 相应地增加重叠部分
            length_function=len,
            is_separator_regex=False,
        )

        # 初始化两个空的向量数据库
        embedding_dimension = len(self.embeddings.embed_query("test"))
        self.summary_store = FAISS.from_texts([" "], self.embeddings, metadatas=[{}])
        self.summary_store.delete(self.summary_store.index_to_docstore_id.values())
        self.full_text_store = FAISS.from_texts([" "], self.embeddings, metadatas=[{}])
        self.full_text_store.delete(self.full_text_store.index_to_docstore_id.values())

        print("MemorySystem initialized successfully.")

    def add_chapter(self, chapter_index: int, chapter_title: str, full_text: str, summary: str):
        """
        将一个新完成的章节添加到记忆系统中。
        这包括将其摘要存入摘要库，并将全文分块后存入分块库。

        Args:
            chapter_index (int): 章节编号。
            chapter_title (str): 章节标题。
            full_text (str): 章节的完整原文。
            summary (str): 章节的摘要。
        """
        print(f"--- Adding Chapter {chapter_index}: '{chapter_title}' to Memory ---")

        # [cite_start]1. 添加到摘要层 (Summary Layer) [cite: 123]
        summary_doc = Document(
            page_content=summary,
            metadata={
                "chapter_index": chapter_index,
                "chapter_title": chapter_title,
                "type": "chapter_summary"
            }
        )
        self.summary_store.add_documents([summary_doc])
        print(f"Added summary for Chapter {chapter_index} to the summary store.")

        # [cite_start]2. 添加到分块层 (Chunk Layer) [cite: 137]
        # 对全文进行分块
        chunks = self.text_splitter.split_text(full_text)

        # [cite_start]为每个分块创建 LangChain Document 对象，并附带元数据 [cite: 145]
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chapter_index": chapter_index,
                    "chapter_title": chapter_title,
                    "chunk_index": i,
                    # chunk_type, emotional_tone 等元数据可通过LLM分析添加，此处简化
                    "chunk_type": "叙述"
                }
            )
            chunk_docs.append(doc)

        self.full_text_store.add_documents(chunk_docs)
        print(f"Split full text of Chapter {chapter_index} into {len(chunk_docs)} chunks and added to the full-text store.")
        print("-" * 20)

    def retrieve_context_for_writer(
        self,
        query: str,
        top_k_summaries: int = 10,
        top_k_chunks: int = 10
    ) -> str:
        """
        为作家Agent检索相关的历史上下文。
        它会并行地从摘要层和分块层检索信息，然后整合成一份结构化的上下文。
        [cite_start]这模拟了“融合检索Agent”的简化工作流程 [cite: 156, 162]。

        Args:
            query (str): 当前的创作目标或章节大纲，用于检索。
            top_k_summaries (int): 要检索的相关章节摘要数量。
            top_k_chunks (int): 要检索的相关文本分块数量。

        Returns:
            str: 格式化后的历史上下文，可直接注入到作家的Prompt中。
        """
        print(f"--- Retrieving Context for Query: '{query[:50]}...' ---")

        # 并行检索
        # [cite_start]从摘要层检索，获取宏观情节线索 [cite: 172]
        summary_results = self.summary_store.similarity_search_with_score(query, k=top_k_summaries)

        # [cite_start]从分块层检索，获取具体的细节、对话和描写 [cite: 173]
        chunk_results = self.full_text_store.similarity_search_with_score(query, k=top_k_chunks)

        # [cite_start]结果融合与格式化，模拟“上下文压缩与注入” [cite: 164, 181]
        context_parts = []

        if summary_results:
            context_parts.append("## 相关历史章节摘要回顾:")
            for doc, score in summary_results:
                metadata = doc.metadata
                context_parts.append(
                    f"- **第{metadata['chapter_index']}章: {metadata['chapter_title']}** (相关度: {score:.2f})\n"
                    f"  摘要: {doc.page_content}"
                )

        if chunk_results:
            context_parts.append("\n## 过去章节中的相关细节片段:")
            for doc, score in chunk_results:
                metadata = doc.metadata
                context_parts.append(
                    f"- **出自第{metadata['chapter_index']}章: {metadata['chapter_title']}** (相关度: {score:.2f})\n"
                    f"  片段: \"...{doc.page_content}...\""
                )

        if not context_parts:
            return "记忆库中暂无相关历史信息。"

        final_context = "\n".join(context_parts)
        print("Successfully retrieved and formatted context.")
        print("-" * 20)
        return final_context
    # +++ 新增方法 +++
    def save(self, directory: str):
        """将FAISS向量数据库保存到本地目录。"""
        print(f"--- 💾 Saving MemorySystem to {directory} ---")
        os.makedirs(directory, exist_ok=True)
        # 使用FAISS的专用方法保存索引
        self.summary_store.save_local(os.path.join(directory, "summary_store"))
        self.full_text_store.save_local(os.path.join(directory, "full_text_store"))
        print("--- ✅ MemorySystem vector stores saved successfully. ---")

    # +++ 新增类方法 +++
    @classmethod
    def load(cls, directory: str, embedding_model) -> "MemorySystem":
        """
        从本地目录加载或创建一个新的MemorySystem实例。
        """
        summary_path = os.path.join(directory, "summary_store")
        full_text_path = os.path.join(directory, "full_text_store")

        # 检查持久化文件是否存在
        if os.path.exists(summary_path) and os.path.exists(full_text_path):
            print(f"--- 📂 Loading MemorySystem from {directory} ---")
            # 创建一个新的实例，但不调用 __init__
            memory_system = super(MemorySystem, cls).__new__(cls)
            memory_system.embeddings = embedding_model
            memory_system.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

            # 从本地文件加载FAISS索引
            memory_system.summary_store = FAISS.load_local(
                summary_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
            memory_system.full_text_store = FAISS.load_local(
                full_text_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
            print("--- ✅ MemorySystem loaded successfully from disk. ---")
            return memory_system
        else:
            print(f"--- ⚠️ Memory store not found at '{directory}'. Creating a new one. ---")
            return cls(embedding_model=embedding_model)

if __name__ == "__main__":
    memory_system = MemorySystem()
    memory_system.save("./vector_memory/8")


