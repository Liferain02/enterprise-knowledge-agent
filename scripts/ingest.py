"""
文档嵌入脚本
将知识库文档向量化并存储到向量数据库
支持 Vision LLM 图片理解（文档中的图片将被自动理解并入库）
"""
import sys
import asyncio
from pathlib import Path

from src.rag.processing.document_loader import get_document_loader_manager
from src.rag.processing.multimodal import get_multimodal_processor
from src.rag.storage.vectorstore import get_vectorstore_manager
from config.settings import get_settings


def ingest_knowledge_base(
    knowledge_dir: str = None,
    collection_name: str = "enterprise_knowledge",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunking_strategy: str = "recursive",
    reset: bool = False,
    enable_vision: bool = None,
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
        enable_vision: 是否启用 Vision LLM 图片理解（默认使用配置）
    """
    settings = get_settings()
    enable_vision = enable_vision if enable_vision is not None else settings.vision_ingestion_enabled

    # 确定知识库目录
    if knowledge_dir is None:
        knowledge_dir = str(settings.knowledge_base_dir)

    print(f"=" * 60)
    print(f"开始文档嵌入...")
    print(f"知识库目录: {knowledge_dir}")
    print(f"集合名称: {collection_name}")
    print(f"分块大小: {chunk_size}")
    print(f"分块策略: {chunking_strategy}")
    print(f"Vision LLM 图片理解: {'启用' if enable_vision else '禁用'}")
    print(f"=" * 60)

    # 检查目录是否存在
    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists():
        print(f"错误: 知识库目录不存在: {knowledge_dir}")
        return

    # 获取文档加载器
    loader_manager = get_document_loader_manager()

    # 加载目录下所有文档
    print(f"\n[1/6] 加载文档...")
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

    # ============================================================
    # [2/6] Vision LLM 图片理解
    # ============================================================
    # 注意：enhance_documents 会按 file_path 去重，同一文件的多个页面
    # 只在第一个页面追加图片描述；其他页面仅继承元数据标记
    image_descriptions = {}  # {file_path: description_text}
    if enable_vision:
        print(f"\n[2/6] Vision LLM 理解文档图片...")
        try:
            multimodal_processor = get_multimodal_processor()
            documents = multimodal_processor.enhance_documents(documents)

            # 提取每个文档的图片描述（用于广播到其他分块）
            for doc in documents:
                fp = doc.metadata.get("file_path") or doc.metadata.get("source_file")
                if doc.metadata.get("has_images") and fp and fp not in image_descriptions:
                    # 找到图片描述内容（在 page_content 末尾）
                    marker = "## 文档图片内容"
                    idx = doc.page_content.rfind(marker)
                    if idx >= 0:
                        image_descriptions[fp] = doc.page_content[idx:]

            print(f"      已提取 {len(image_descriptions)} 个文档的图片描述")
            print(f"      Vision LLM 图片理解完成")
        except Exception as e:
            print(f"      Vision LLM 处理失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"      继续处理（跳过图片理解）...")
    else:
        print(f"\n[2/6] Vision LLM 图片理解（已跳过）")

    # 分割文档
    print(f"\n[3/6] 分割文档...")
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

    # ============================================================
    # [4/6] 将图片描述广播到同一文档的所有分块
    # ============================================================
    if image_descriptions:
        print(f"\n[4/6] 广播图片描述到所有分块...")
        chunks_with_description = 0
        for doc in split_docs:
            fp = doc.metadata.get("file_path") or doc.metadata.get("source_file")
            if fp in image_descriptions:
                doc.page_content += "\n\n" + image_descriptions[fp]
                chunks_with_description += 1
        print(f"      已将图片描述追加到 {chunks_with_description} 个分块")

    # 获取向量存储管理器
    print(f"\n[5/6] 向量化文档...")
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
    print(f"\n[6/6] 验证结果...")
    try:
        info = vectorstore_manager.get_collection_info()
        print(f"      集合名称: {info['name']}")
        print(f"      文档数量: {info['count']}")
        print(f"      存储路径: {info['persist_directory']}")
    except Exception as e:
        print(f"      获取集合信息失败: {e}")

    print(f"\n{'=' * 60}")
    print(f"文档嵌入完成!")
    print(f"{'=' * 60}")


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
        default=2000,
        help="分块大小（字符数，仅用于 markdown 策略，默认: 2000）"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=300,
        help="分块重叠大小（字符数，默认: 300）"
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
    vision_group = parser.add_mutually_exclusive_group()
    vision_group.add_argument(
        "--vision",
        action="store_true",
        default=None,
        help="启用 Vision LLM 图片理解 (默认: 使用配置)"
    )
    vision_group.add_argument(
        "--no-vision",
        action="store_true",
        help="禁用 Vision LLM 图片理解"
    )

    args = parser.parse_args()

    # 如果指定了 --no-vision，强制禁用
    if getattr(args, "no_vision", False):
        enable_vision = False
    elif getattr(args, "vision", None) is not None:
        enable_vision = args.vision
    else:
        enable_vision = settings.vision_ingestion_enabled

    # 如果未指定策略，使用配置文件中的默认值
    chunking_strategy = args.strategy if args.strategy.strip() else settings.chunking_strategy

    ingest_knowledge_base(
        knowledge_dir=args.dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=chunking_strategy,
        reset=args.reset,
        enable_vision=enable_vision,
    )


if __name__ == "__main__":
    main()

