"""
修复脚本：为现有文档添加默认 confidentiality 字段

问题：所有文档的 confidentiality 字段未设置，导致 ACL 过滤器过滤掉所有文档
解决：为文档添加默认 confidentiality = "internal"
"""
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'


def fix_confidentiality_field():
    """修复文档的 confidentiality 字段"""
    print("=" * 50)
    print("开始修复 confidentiality 字段")
    print("=" * 50)

    # 连接 ChromaDB
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path='./chroma_db',
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    collection = client.get_collection('enterprise_knowledge')
    print(f"\n集合名称: {collection.name}")
    print(f"文档数量: {collection.count()}")

    if collection.count() == 0:
        print("\n集合为空，需要重新入库文档。请运行知识库导入命令。")
        return

    # 步骤 1: 获取所有文档 id
    print("\n步骤 1: 获取文档列表...")
    peek_results = collection.peek(limit=1000)
    all_ids = peek_results.get('ids', [])

    if not all_ids:
        print("没有找到文档 ID")
        return

    print(f"  找到 {len(all_ids)} 个文档")

    # 步骤 2: 获取所有文档的 metadata
    print("\n步骤 2: 获取文档 metadata...")
    get_results = collection.get(limit=1000, include=['metadatas'])

    # 步骤 3: 构建更新数据
    print("\n步骤 3: 构建更新数据...")
    missing_count = 0
    update_ids = []
    update_metadatas = []

    for doc_id, meta in zip(all_ids, get_results['metadatas']):
        if meta is None:
            meta = {}

        if 'confidentiality' not in meta:
            missing_count += 1
            new_meta = dict(meta)
            new_meta['confidentiality'] = 'internal'
            update_ids.append(doc_id)
            update_metadatas.append(new_meta)

    print(f"  需要修复的文档: {missing_count}")
    print(f"  无需修复的文档: {len(all_ids) - missing_count}")

    if missing_count == 0:
        print("\n所有文档已有 confidentiality 字段，无需修复")
        return

    # 步骤 4: 删除并重新添加（ChromaDB 1.5+ 不支持直接更新 metadata）
    print("\n步骤 4: 删除并重新添加文档...")

    # 获取完整文档数据
    full_results = collection.get(limit=1000, include=['documents', 'metadatas'])
    docs_to_readd = []

    for doc_id in update_ids:
        # 找到对应的文档
        idx = all_ids.index(doc_id)
        docs_to_readd.append({
            'id': doc_id,
            'content': full_results['documents'][idx],
            'old_meta': full_results['metadatas'][idx],
            'new_meta': {'confidentiality': 'internal'}
        })

    # 删除旧文档
    collection.delete(ids=all_ids)
    print(f"  已删除 {len(all_ids)} 个文档")

    # 步骤 5: 重新添加文档
    print("\n步骤 5: 重新添加文档...")

    from src.models.embeddings import get_embeddings
    embeddings = get_embeddings()

    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    vectorstore = Chroma(
        collection_name='enterprise_knowledge',
        embedding_function=embeddings,
        persist_directory='./chroma_db',
        client_settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
    )

    # 构建 Document 对象
    docs = []
    valid_ids = []

    for item in docs_to_readd:
        new_meta = dict(item['old_meta']) if item['old_meta'] else {}
        new_meta['confidentiality'] = 'internal'

        doc = Document(
            page_content=item['content'],
            metadata=new_meta
        )
        docs.append(doc)
        valid_ids.append(item['id'])

    # 添加修复后的文档
    vectorstore.add_documents(documents=docs, ids=valid_ids)
    print(f"  已添加 {len(docs)} 个文档")

    # 验证
    print("\n验证修复结果...")
    verify_results = collection.get(limit=100, include=['metadatas'])
    fixed_count = sum(1 for m in verify_results['metadatas']
                     if m and m.get('confidentiality') == 'internal')
    total = len([m for m in verify_results['metadatas'] if m])
    print(f"  修复后有 confidentiality=internal 的文档: {fixed_count}/{total}")

    print("\n" + "=" * 50)
    print("修复完成！")
    print("=" * 50)


if __name__ == '__main__':
    fix_confidentiality_field()
