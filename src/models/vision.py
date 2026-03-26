"""
Vision LLM 模块 - 图片理解
使用 qwen-vl-plus / qwen-vl-max / gpt-4o 等多模态模型理解图片
"""
import base64
import io
from typing import List, Optional, Union, Any
from functools import lru_cache
from PIL import Image
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_vision_llm(
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Any:
    """
    获取 Vision LLM 实例

    Args:
        model: 模型名称，默认使用配置中的 vision_model
        temperature: 温度参数

    Returns:
        ChatOpenAI 实例（支持 vision）
    """
    settings = get_settings()
    vision_model = model or settings.vision_model

    # 使用通用的 ChatOpenAI（qwen-vl 系列和 gpt-4o 都兼容）
    return _create_vision_llm(
        model=vision_model,
        api_key=settings.dashscope_api_key if settings.llm_provider == "qwen" else settings.openai_api_key,
        base_url=settings.dashscope_base_url if settings.llm_provider == "qwen" else settings.openai_base_url,
        temperature=temperature,
    )


@lru_cache(maxsize=4)
def _create_vision_llm(
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
) -> Any:
    """创建 Vision LLM 实例（带缓存）"""
    import os as _os

    # 设置代理
    http_proxy = _os.environ.get("http_proxy") or _os.environ.get("HTTP_PROXY")
    https_proxy = _os.environ.get("https_proxy") or _os.environ.get("HTTPS_PROXY")
    if http_proxy and not _os.environ.get("HTTP_PROXY"):
        _os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy and not _os.environ.get("HTTPS_PROXY"):
        _os.environ["HTTPS_PROXY"] = https_proxy

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        api_key=api_key,
        base_url=base_url,
    )


# ============================================================
# 图片处理工具
# ============================================================

def compress_image_if_needed(
    image_data: Union[bytes, str],
    max_size_mb: int = 5,
    max_dimension: int = 2048,
) -> bytes:
    """
    如果图片过大，压缩到指定大小

    Args:
        image_data: 图片数据（bytes 或 base64 str）
        max_size_mb: 最大大小（MB）
        max_dimension: 最大边长（像素）

    Returns:
        压缩后的图片 bytes
    """
    # 解析图片数据
    if isinstance(image_data, str):
        # 可能是 base64 或 URL
        if image_data.startswith("data:image"):
            # data:image/png;base64,xxxx
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data

    # 检查大小
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb <= max_size_mb:
        return image_bytes

    # 压缩图片
    img = Image.open(io.BytesIO(image_bytes))

    # 缩放
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # 转为 bytes
    output = io.BytesIO()
    img_format = img.format or "JPEG"
    img.save(output, format=img_format, quality=85)
    return output.getvalue()


def encode_image_to_base64(image_bytes: bytes) -> str:
    """将图片 bytes 转为 base64 字符串（带 MIME 前缀）"""
    # 自动检测格式
    img = Image.open(io.BytesIO(image_bytes))
    img_format = img.format or "jpeg"
    mime_type = f"image/{img_format.lower()}"
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64_str}"


def parse_image_content(
    image_data: Union[bytes, str],
    filename: Optional[str] = None,
) -> str:
    """
    将图片数据转换为 Vision LLM 所需的格式

    Args:
        image_data: 图片数据（bytes / base64 str / data URI）
        filename: 文件名

    Returns:
        Vision LLM 格式的图片数据（data:image/xxx;base64,xxx）
    """
    # 如果是字符串
    if isinstance(image_data, str):
        if image_data.startswith("data:image"):
            return image_data  # 已经是目标格式
        if image_data.startswith("http"):
            # URL 格式暂不支持，直接返回
            return image_data
        # base64 字符串
        try:
            img_bytes = base64.b64decode(image_data)
            return encode_image_to_base64(img_bytes)
        except Exception:
            return f"data:image/jpeg;base64,{image_data}"

    # bytes
    img_bytes = compress_image_if_needed(image_data)
    return encode_image_to_base64(img_bytes)


# ============================================================
# 图片理解 API
# ============================================================

async def understand_images(
    images: List[Any],
    prompt: str = "请详细描述这张图片的内容，包括文字、图表、布局等所有细节。",
    model: Optional[str] = None,
) -> str:
    """
    使用 Vision LLM 理解一张或多张图片

    Args:
        images: 图片数据列表（支持 bytes / base64 str / ImageContent）
        prompt: 提问提示词
        model: Vision 模型名称

    Returns:
        Vision LLM 对图片的理解结果
    """
    if not images:
        return ""

    settings = get_settings()
    if not getattr(settings, "vision_enabled", True):
        return "[图片理解功能已禁用]"

    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.messages import AIMessage

        # 解析所有图片为 Vision LLM 格式
        parsed_images = []
        for img in images:
            if hasattr(img, "data"):
                # ImageContent 对象
                parsed = parse_image_content(img.data, img.filename)
            else:
                parsed = parse_image_content(img)
            parsed_images.append(parsed)

        # 构建多模态消息
        if len(parsed_images) == 1:
            content = [
                {"type": "image_url", "image_url": {"url": parsed_images[0]}},
                {"type": "text", "text": prompt},
            ]
        else:
            # 多张图片
            content = []
            for img_url in parsed_images:
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            content.append({"type": "text", "text": prompt})

        llm = get_vision_llm(model=model, temperature=0.1)
        message = HumanMessage(content=content)

        response = await llm.ainvoke([message])

        logger.info(f"[Vision] 理解 {len(images)} 张图片，成功")
        return response.content

    except Exception as e:
        logger.warning(f"[Vision] 图片理解失败: {e}")
        import traceback
        traceback.print_exc()
        return f"[图片理解失败: {str(e)}]"


def understand_images_sync(
    images: List[Any],
    prompt: str = "请详细描述这张图片的内容，包括文字、图表、布局等所有细节。",
    model: Optional[str] = None,
) -> str:
    """
    同步版本（使用 asyncio.run 在新循环中执行）
    用于在同步上下文中调用 Vision LLM
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(understand_images(images, prompt, model))

    # 已有循环 → 用 ThreadPoolExecutor
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(
            asyncio.run,
            understand_images(images, prompt, model)
        )
        return future.result()
