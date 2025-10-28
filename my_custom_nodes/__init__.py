# 导入具体的节点类
from .nodes.gemini_image_analysis import GeminiImageAnalysis
from .nodes.gemini_image_generator import GeminiImageGenerator
from .nodes.placeholder_image_generator import PlaceholderImageGenerator

# 节点类映射 - 这是ComfyUI识别节点的关键
NODE_CLASS_MAPPINGS = {
    "GeminiImageAnalysis": GeminiImageAnalysis,
    "GeminiImageGenerator": GeminiImageGenerator,
    "PlaceholderImageGenerator": PlaceholderImageGenerator,
}

# 节点显示名称映射 - 在ComfyUI界面中显示的中文名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageAnalysis": "Gemini 图像分析",
    "GeminiImageGenerator": "Gemini 图像生成器", 
    "PlaceholderImageGenerator": "占位图片生成器",
}

# 导出所有必要的变量
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
