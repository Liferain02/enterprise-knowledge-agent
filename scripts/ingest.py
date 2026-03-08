"""
文档嵌入脚本
将知识库文档向量化并存储到向量数据库
"""
import sys
import asyncio
from pathlib import Path

from src.rag.processing.document_loader import get_document_loader_manager
from src.rag.storage.vectorstore import get_vectorstore_manager
from config.settings import get_settings


def ingest_knowledge_base(
    knowledge_dir: str = None,
    collection_name: str = "enterprise_knowledge",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunking_strategy: str = "recursive",
    reset: bool = False
):
    """
    执行知识库文档嵌入

    Args:
        knowledge_dir: 知识库目录路径
        collection_name: Chroma 集合名称
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小
        chunking_strategy: 分块策略: recursive / markdown / semantic / hybrid
        reset: 是否重置向量数据库
    """
    settings = get_settings()

    # 确定知识库目录
    if knowledge_dir is None:
        knowledge_dir = str(settings.knowledge_base_dir)

    print(f"=" * 50)
    print(f"开始文档嵌入...")
    print(f"知识库目录: {knowledge_dir}")
    print(f"集合名称: {collection_name}")
    print(f"分块大小: {chunk_size}")
    print(f"分块策略: {chunking_strategy}")
    print(f"=" * 50)
    
    # 检查目录是否存在
    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists():
        print(f"错误: 知识库目录不存在: {knowledge_dir}")
        return
    
    # 获取文档加载器
    loader_manager = get_document_loader_manager()
    
    # 加载目录下所有文档
    print(f"\n[1/4] 加载文档...")
    try:
        documents = loader_manager.load_directory(knowledge_dir)
        print(f"      成功加载 {len(documents)} 个文档")
    except Exception as e:
        print(f"      加载文档失败: {e}")
        return
    
    if not documents:
        print(f"      知识库目录为空，没有文档需要嵌入")
        return
    
    # 打印加载的文档列表
    unique_files = set()
    for doc in documents:
        source = doc.metadata.get("source_file", "未知")
        unique_files.add(source)
    
    print(f"      文档文件:")
    for f in unique_files:
        print(f"        - {Path(f).name}")
    
    # 分割文档
    print(f"\n[2/4] 分割文档...")
    try:
        split_docs = loader_manager.split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=chunking_strategy
        )
        print(f"      分割完成: {len(split_docs)} 个文本块")
    except Exception as e:
        print(f"      分割文档失败: {e}")
        return
    
    # 获取向量存储管理器
    print(f"\n[3/4] 向量化文档...")
    vectorstore_manager = get_vectorstore_manager(collection_name)
    
    # 如果需要重置，先删除集合
    if reset:
        print(f"      重置向量数据库...")
        try:
            vectorstore_manager.reset()
            print(f"      向量数据库已重置")
        except Exception as e:
            print(f"      重置失败: {e}")
    
    # 添加文档到向量存储
    try:
        # 提取文本和元数据
        texts = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]
        
        # 添加到向量数据库
        vectorstore_manager.add_texts(texts, metadatas)
        print(f"      向量化完成: {len(texts)} 个文本块已添加到向量数据库")
    except Exception as e:
        print(f"      向量化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取集合信息
    print(f"\n[4/4] 验证结果...")
    try:
        info = vectorstore_manager.get_collection_info()
        print(f"      集合名称: {info['name']}")
        print(f"      文档数量: {info['count']}")
        print(f"      存储路径: {info['persist_directory']}")
    except Exception as e:
        print(f"      获取集合信息失败: {e}")
    
    print(f"\n{'=' * 50}")
    print(f"文档嵌入完成!")
    print(f"{'=' * 50}")


def main():
    """主函数"""
    import argparse

    # 初始化 settings 以获取默认配置
    settings = get_settings()

    parser = argparse.ArgumentParser(description="知识库文档嵌入工具")
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default=None,
        help="知识库目录路径 (默认: data/knowledge)"
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default="enterprise_knowledge",
        help="Chroma 集合名称 (默认: enterprise_knowledge)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="分块大小 (默认: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="分块重叠大小 (默认: 200)"
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="",
        help=f"分块策略 (默认: {settings.chunking_strategy}): recursive(固定长度) / markdown(标题) / semantic(语义) / hybrid(混合)"
    )
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="是否重置向量数据库"
    )

    args = parser.parse_args()

    # 如果未指定策略，使用配置文件中的默认值
    chunking_strategy = args.strategy if args.strategy.strip() else settings.chunking_strategy

    ingest_knowledge_base(
        knowledge_dir=args.dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=chunking_strategy,
        reset=args.reset
    )


if __name__ == "__main__":
    main()

