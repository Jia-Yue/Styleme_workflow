import torch
import numpy as np
from PIL import Image
import base64
import io
import requests
import json
from typing import Dict, Any, List, Tuple, Optional
import os
import traceback

# 尝试导入google.genai库
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    try:
        import genai
        GENAI_AVAILABLE = True
    except Exception:
        genai = None
        GENAI_AVAILABLE = False

class GeminiImageAnalysis:
    """
    ComfyUI节点：使用Google Gemini 2.5 Flash Image API进行图像分析，只输出文字描述
    """
    
    def __init__(self):
        self.api_key = None
        self.model_name = "gemini-2.5-flash-image-preview"
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入您的Google AI API密钥"
                }),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], {
                    "default": "gemini-2.5-flash-image-preview"
                }),
                "input_image": ("IMAGE",),
                "analysis_prompt": ("STRING", {
                    "default": "请详细描述这张图片的内容，包括主要对象、场景、颜色、构图等元素",
                    "multiline": True,
                    "placeholder": "输入图像分析提示词"
                }),
            },
            "optional": {
                "input_image_2": ("IMAGE",),
                "analysis_type": (["detailed_description", "object_detection", "scene_analysis", "color_analysis", "style_analysis", "custom"], {
                    "default": "detailed_description"
                }),
                "language": (["中文", "English", "auto"], {
                    "default": "中文"
                }),
                "detail_level": (["brief", "moderate", "detailed", "comprehensive"], {
                    "default": "moderate"
                }),
                "safety_settings": (["block_few", "block_some", "block_most", "block_none"], {
                    "default": "block_some"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "max_output_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "seed": ("STRING", {
                    "default": "0",
                    "placeholder": "随机种子 (0=随机)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("analysis_text", "analysis_success")
    FUNCTION = "analyze_image"
    CATEGORY = "AI/Image Analysis"
    
    def _to_pil(self, image):
        """Convert ComfyUI IMAGE (tensor / numpy / PIL) -> PIL.Image (RGB)."""
        # Torch tensor
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            # if batch dim present (1, C, H, W) or (1, H, W, C)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            # channel-first (C,H,W) -> (H,W,C)
            if (
                img.ndim == 3
                and img.shape[0] in (1, 3, 4)
                and img.shape[0] < img.shape[1]
            ):
                img = img.permute(1, 2, 0)
            arr = img.numpy()
            # floats in [0..1] -> scale to 0..255
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            # grayscale -> 3ch
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            return Image.fromarray(arr).convert("RGB")

        # Numpy array
        if isinstance(image, np.ndarray):
            arr = image
            # channel-first heuristic
            if (
                arr.ndim == 3
                and arr.shape[0] in (1, 3, 4)
                and arr.shape[0] < arr.shape[1]
            ):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            return Image.fromarray(arr).convert("RGB")

        # PIL
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def enhance_analysis_prompt(self, prompt: str, analysis_type: str, detail_level: str, language: str) -> str:
        """
        根据分析类型、详细程度和语言增强提示词
        """
        # 分析类型模板
        analysis_templates = {
            "detailed_description": "请详细描述这张图片的内容，包括主要对象、场景、颜色、构图等元素",
            "object_detection": "请识别并列出图片中的所有主要对象和物品",
            "scene_analysis": "请分析这张图片的场景设置、环境背景和整体氛围",
            "color_analysis": "请分析这张图片的主要颜色、色调和色彩搭配",
            "style_analysis": "请分析这张图片的艺术风格、构图技巧和视觉特点",
            "custom": prompt  # 使用用户自定义提示词
        }
        
        # 详细程度修饰词
        detail_modifiers = {
            "brief": "简洁地",
            "moderate": "详细地",
            "detailed": "非常详细地",
            "comprehensive": "全面深入地"
        }
        
        # 语言指示
        language_instructions = {
            "中文": "请用中文回答",
            "English": "Please answer in English",
            "auto": ""
        }
        
        # 构建增强提示词
        base_prompt = analysis_templates.get(analysis_type, prompt)
        detail_modifier = detail_modifiers.get(detail_level, "详细地")
        language_instruction = language_instructions.get(language, "")
        
        enhanced_prompt = f"{detail_modifier}{base_prompt}"
        
        if language_instruction:
            enhanced_prompt = f"{enhanced_prompt}。{language_instruction}"
        
        return enhanced_prompt

    def analyze_image(self, api_key: str, model: str, input_image: torch.Tensor, analysis_prompt: str,
                     input_image_2: torch.Tensor = None, analysis_type: str = "detailed_description",
                     language: str = "中文", detail_level: str = "moderate", 
                     safety_settings: str = "block_some", temperature: float = 0.3,
                     top_k: int = 40, top_p: float = 0.95, max_output_tokens: int = 2048,
                     seed: str = "0") -> Tuple[str, bool]:
        """
        主要的图像分析函数 - 使用google.genai库进行图像分析
        """
        if not GENAI_AVAILABLE:
            raise RuntimeError("google.genai库未找到，请安装: pip install google-genai")
        
        if not api_key.strip():
            raise ValueError("请提供有效的Google AI API密钥")
        
        # 处理可能的None值 - ComfyUI可选参数可能传递None
        print(f"原始参数 - model: {model}, seed: {seed} (type: {type(seed)})")
        
        # 设置模型名称
        self.model_name = model or "gemini-2.5-flash-image-preview"
        
        analysis_type = analysis_type or "detailed_description"
        language = language or "中文"
        detail_level = detail_level or "moderate"
        safety_settings = safety_settings or "block_some"
        temperature = temperature if temperature is not None else 0.3
        top_k = top_k if top_k is not None else 40
        top_p = top_p if top_p is not None else 0.95
        max_output_tokens = max_output_tokens if max_output_tokens is not None else 2048
        seed_str = seed or "0"
        
        # 转换seed字符串为整数
        try:
            seed_int = int(seed_str) if seed_str.strip() else 0
        except ValueError:
            print(f"警告: 无效的seed值 '{seed_str}'，使用默认值0")
            seed_int = 0
        
        print(f"处理后参数 - seed: {seed_int} (type: {type(seed_int)})")
        
        try:
            # 增强分析提示词
            enhanced_prompt = self.enhance_analysis_prompt(analysis_prompt, analysis_type, detail_level, language)
            
            print(f"使用模型: {self.model_name}")
            print(f"分析类型: {analysis_type}")
            print(f"详细程度: {detail_level}")
            print(f"语言: {language}")
            print(f"提示词: {enhanced_prompt[:100]}...")
            
            # 转换输入图像为PIL格式
            pil_image = self._to_pil(input_image)
            print(f"输入图像1尺寸: {pil_image.size}")
            
            # 处理第二张图片（如果提供）
            contents = [enhanced_prompt, pil_image]
            if input_image_2 is not None:
                pil_image_2 = self._to_pil(input_image_2)
                print(f"输入图像2尺寸: {pil_image_2.size}")
                contents.append(pil_image_2)
                print("使用双图分析模式")
            else:
                print("使用单图分析模式")
            
            # 初始化客户端
            client = genai.Client(api_key=api_key.strip())
            
            # 调用模型进行图像分析
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            
            # 提取响应数据
            analysis_text = ""
            analysis_success = False
            
            if getattr(response, "candidates", None):
                for cand in response.candidates:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    
                    # 提取文本响应
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            analysis_text += part.text
                            analysis_success = True
                            break
                    if analysis_success:
                        break
            
            if not analysis_success or not analysis_text.strip():
                raise RuntimeError("图像分析失败：API响应中没有文本数据")
            
            print(f"分析成功，文本长度: {len(analysis_text)} 字符")
            print(f"分析结果预览: {analysis_text[:200]}...")
            
            return (analysis_text, analysis_success)
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"图像分析失败: {str(e)}\n{tb}"
            print(error_msg)
            return (error_msg, False)
