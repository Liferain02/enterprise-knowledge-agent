"""
Vision 路由 - 图片理解专用接口
"""
import logging
from fastapi import APIRouter, HTTPException, Depends

from src.api.schemas import ImageContent
from src.models.vision import get_vision_llm
from src.api.security import get_current_user
from config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])


@router.get("/health")
async def vision_health():
    """Vision 服务健康检查"""
    settings = get_settings()
    vision_enabled = getattr(settings, "vision_enabled", True)
    vision_model = getattr(settings, "vision_model", "unknown")

    return {
        "status": "ok" if vision_enabled else "disabled",
        "vision_enabled": vision_enabled,
        "vision_model": vision_model,
        "service": "vision-multimodal",
        "supported_formats": ["jpeg", "png", "gif", "webp", "bmp"],
        "max_size_mb": getattr(settings, "vision_max_image_size", 5),
    }


@router.post("/understand")
async def understand_image(
    image: ImageContent,
    question: str = "请详细描述这张图片的内容",
    current_user: dict = Depends(get_current_user),
):
    """
    直接图片理解接口（无需对话）

    Args:
        image: 单张图片（base64 编码）
        question: 关于图片的问题

    Returns:
        图片理解结果
    """
    try:
        from src.models.vision import understand_images

        result = await understand_images(
            images=[image],
            prompt=question,
        )

        return {
            "success": True,
            "question": question,
            "description": result,
        }

    except Exception as e:
        logger.exception(f"图片理解失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图片理解出错: {str(e)}")


@router.post("/describe")
async def describe_image(
    image: ImageContent,
    current_user: dict = Depends(get_current_user),
):
    """
    图片描述接口

    自动对图片进行详细描述，适合用于图片索引和理解。
    """
    try:
        from src.models.vision import understand_images

        prompt = (
            "请对这张图片进行详细描述，包括：\n"
            "1. 图片主体内容\n"
            "2. 包含的文字内容（请完整提取）\n"
            "3. 图表、表格内容（如果有）\n"
            "4. 布局和结构\n"
            "5. 图片类型（截图、照片、文档等）\n"
            "请用中文详细描述。"
        )

        result = await understand_images(
            images=[image],
            prompt=prompt,
        )

        return {
            "success": True,
            "description": result,
        }

    except Exception as e:
        logger.exception(f"图片描述失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图片描述出错: {str(e)}")
