import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

class PlaceholderImageGenerator:
    """
    ComfyUI节点：生成指定尺寸的蓝色背景占位图片
    图片上有白色文字"REPLACE THIS IMAGE ENTIRELY WITH THE NEW ONE"
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1920,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 1080,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "preset_size": (["1280x720", "1920x1080", "1024x1024", "720x1280", "1080x1920", "custom"], {
                    "default": "1920x1080"
                }),
                "text": ("STRING", {
                    "default": "REPLACE THIS IMAGE ENTIRELY WITH THE NEW ONE",
                    "multiline": True,
                    "placeholder": "输入要显示的文本"
                }),
                "background_color": (["blue", "red", "green", "black", "white"], {
                    "default": "blue"
                }),
                "text_color": (["white", "black", "red", "green", "blue"], {
                    "default": "white"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("placeholder_image",)
    FUNCTION = "generate_placeholder"
    CATEGORY = "Image/Generation"
    
    def _get_color_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """将颜色名称转换为RGB元组"""
        color_map = {
            "blue": (0, 0, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
        }
        return color_map.get(color_name, (0, 0, 255))  # 默认蓝色
    
    def _get_preset_dimensions(self, preset: str) -> Tuple[int, int]:
        """根据预设获取尺寸"""
        preset_map = {
            "1280x720": (1280, 720),
            "1920x1080": (1920, 1080),
            "1024x1024": (1024, 1024),
            "720x1280": (720, 1280),
            "1080x1920": (1080, 1920),
        }
        return preset_map.get(preset, (1920, 1080))
    
    def _pil_to_tensor_channel_last(self, pil_img):
        """
        Convert PIL.Image -> torch.Tensor (1, H, W, 3), float32 in [0,1].
        """
        arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0  # H,W,3
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1,H,W,3
        return tensor
    
    def generate_placeholder(self, width: int, height: int, preset_size: str = "1920x1080",
                           text: str = "REPLACE THIS IMAGE ENTIRELY WITH THE NEW ONE",
                           background_color: str = "blue", text_color: str = "white") -> Tuple[torch.Tensor]:
        """
        生成占位图片
        """
        # 如果选择了预设尺寸，使用预设尺寸
        if preset_size != "custom":
            width, height = self._get_preset_dimensions(preset_size)
        
        # 获取颜色
        bg_color = self._get_color_rgb(background_color)
        txt_color = self._get_color_rgb(text_color)
        
        # 创建蓝色背景图片
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # 尝试使用系统字体，如果失败则使用默认字体
        try:
            # 根据图片尺寸计算字体大小
            font_size = min(width, height) // 20
            font_size = max(font_size, 12)  # 最小字体大小
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("Arial.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 计算文本位置（居中）
        # 将长文本分行
        words = text.split()
        lines = []
        current_line = ""
        
        # 简单的文本换行逻辑
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= width * 0.9:  # 留10%边距
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(current_line)
        
        # 计算总文本高度
        line_height = font_size + 5
        total_text_height = len(lines) * line_height
        
        # 计算起始Y位置（垂直居中）
        start_y = (height - total_text_height) // 2
        
        # 绘制每一行文本
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算X位置（水平居中）
            x = (width - text_width) // 2
            y = start_y + i * line_height
            
            # 绘制文本
            draw.text((x, y), line, fill=txt_color, font=font)
        
        # 转换为torch张量
        placeholder_image = self._pil_to_tensor_channel_last(image)
        
        return (placeholder_image,)
