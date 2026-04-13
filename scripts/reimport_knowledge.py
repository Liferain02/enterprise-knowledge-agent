"""
重新导入文档到知识库，并自动添加 confidentiality 字段

问题：所有文档的 confidentiality 字段未设置，导致 ACL 过滤器过滤掉所有文档
解决：重新导入文档并添加默认 confidentiality = "internal"
"""
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'

from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

from src.models.embeddings import get_embeddings
from src.rag.processing.document_loader import get_document_loader_manager


def reimport_knowledge_base():
    """重新导入知识库文档"""
    print("=" * 50)
    print("开始重新导入知识库文档")
    print("=" * 50)

    # 获取 embeddings
    print("\n步骤 1: 初始化 embeddings...")
    embeddings = get_embeddings()
    print("  embeddings 初始化完成")

    # 初始化向量存储
    print("\n步骤 2: 初始化向量存储...")
    vectorstore = Chroma(
        collection_name='enterprise_knowledge',
        embedding_function=embeddings,
        persist_directory='./chroma_db',
        client_settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    print("  向量存储初始化完成")

    # 重置集合
    print("\n步骤 3: 重置集合...")
    try:
        vectorstore.delete_collection()
        print("  集合已重置")
    except Exception as e:
        print(f"  重置集合: {e}")

    # 加载文档
    print("\n步骤 4: 加载文档...")
    loader = get_document_loader_manager()

    knowledge_dir = './data/knowledge'
    supported_extensions = ['.md', '.txt', '.pdf']

    all_docs = []
    file_count = 0

    for filename in os.listdir(knowledge_dir):
        filepath = os.path.join(knowledge_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue

        try:
            docs = loader.load_file(filepath)
            for doc in docs:
                # 添加 confidentiality 字段
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['confidentiality'] = 'internal'
                doc.metadata['source'] = filepath
            all_docs.extend(docs)
            file_count += 1
            print(f"  已加载: {filename} ({len(docs)} chunks)")
        except Exception as e:
            print(f"  加载失败: {filename} - {e}")

    print(f"\n总计: {file_count} 个文件, {len(all_docs)} 个文档块")

    if not all_docs:
        print("\n没有找到可导入的文档")
        return

    # 导入文档
    print("\n步骤 5: 导入文档到向量库...")
    try:
        vectorstore.add_documents(documents=all_docs)
        print(f"  成功导入 {len(all_docs)} 个文档")
    except Exception as e:
        print(f"  导入失败: {e}")
        return

    # 验证
    print("\n验证导入结果...")
    import chromadb
    client = chromadb.PersistentClient(
        path='./chroma_db',
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_collection('enterprise_knowledge')
    count = collection.count()
    print(f"  集合文档数: {count}")

    # 验证 confidentiality
    results = collection.get(limit=min(count, 10), include=['metadatas'])
    conf_count = sum(1 for m in results['metadatas']
                    if m and m.get('confidentiality') == 'internal')
    print(f"  有 confidentiality=internal 的文档: {conf_count}/{len(results['metadatas'])}")

    print("\n" + "=" * 50)
    print("导入完成！")
    print("=" * 50)


if __name__ == '__main__':
    reimport_knowledge_base()
