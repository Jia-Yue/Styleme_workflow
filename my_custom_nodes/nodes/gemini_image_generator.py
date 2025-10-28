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

class GeminiImageGenerator:
    """
    ComfyUI节点：使用Google Gemini 2.5 Flash Image API进行图+文字生成图片
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
                "prompt": ("STRING", {
                    "default": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
                    "multiline": True,
                    "placeholder": "输入图片生成提示词"
                }),
                "input_image": ("IMAGE",),
            },
            "optional": {
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "style": (["realistic", "artistic", "cartoon", "anime", "photographic"], {
                    "default": "realistic"
                }),
                "quality": (["standard", "high"], {
                    "default": "high"
                }),
                "safety_settings": (["block_few", "block_some", "block_most", "block_none"], {
                    "default": "block_some"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
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
                    "default": 1024,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "seed": ("STRING", {
                    "default": "0",
                    "placeholder": "随机种子 (0=随机)"
                }),
                "output_mode": (["auto", "image_generation", "image_analysis"], {
                    "default": "auto"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("generated_image", "response_text", "has_image")
    FUNCTION = "generate_image"
    CATEGORY = "AI/Image Generation"
    
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

    def _pil_to_tensor_channel_last(self, pil_img):
        """
        Convert PIL.Image -> torch.Tensor (1, H, W, 3), float32 in [0,1].
        """
        arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0  # H,W,3
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1,H,W,3
        return tensor

    def enhance_prompt(self, prompt: str, style: str) -> str:
        """
        根据选择的风格增强提示词
        """
        style_enhancements = {
            "realistic": "photorealistic, high detail, professional photography",
            "artistic": "artistic style, creative composition, painterly",
            "cartoon": "cartoon style, colorful, animated",
            "anime": "anime style, manga art, Japanese animation",
            "photographic": "professional photography, high resolution, sharp focus"
        }
        
        enhancement = style_enhancements.get(style, "")
        enhanced_prompt = prompt
        
        if enhancement:
            enhanced_prompt = f"{prompt}, {enhancement}"
        
        return enhanced_prompt


    def generate_image(self, api_key: str, model: str, prompt: str, input_image: torch.Tensor,
                      input_image_2: torch.Tensor = None, input_image_3: torch.Tensor = None, 
                      style: str = "realistic", quality: str = "high", 
                      safety_settings: str = "block_some", temperature: float = 0.7,
                      top_k: int = 40, top_p: float = 0.95, max_output_tokens: int = 1024,
                      seed: str = "0", output_mode: str = "auto") -> Tuple[torch.Tensor, str, bool]:
        """
        主要的图像生成函数 - 使用google.genai库
        """
        if not GENAI_AVAILABLE:
            raise RuntimeError("google.genai库未找到，请安装: pip install google-genai")
        
        if not api_key.strip():
            raise ValueError("请提供有效的Google AI API密钥")
        
        # 处理可能的None值 - ComfyUI可选参数可能传递None
        print(f"原始参数 - model: {model}, seed: {seed} (type: {type(seed)}), output_mode: {output_mode}")
        
        # 设置模型名称
        self.model_name = model or "gemini-2.5-flash-image-preview"
        
        style = style or "realistic"
        quality = quality or "high"
        safety_settings = safety_settings or "block_some"
        temperature = temperature if temperature is not None else 0.7
        top_k = top_k if top_k is not None else 40
        top_p = top_p if top_p is not None else 0.95
        max_output_tokens = max_output_tokens if max_output_tokens is not None else 1024
        seed_str = seed or "0"
        output_mode = output_mode or "auto"
        
        # 转换seed字符串为整数
        try:
            seed_int = int(seed_str) if seed_str.strip() else 0
        except ValueError:
            print(f"警告: 无效的seed值 '{seed_str}'，使用默认值0")
            seed_int = 0
        
        print(f"处理后参数 - seed: {seed_int} (type: {type(seed_int)})")
        
        try:
            # 根据输出模式调整提示词
            if output_mode == "image_analysis":
                # 图像分析模式 - 不增强提示词，让模型描述图像
                enhanced_prompt = prompt
                print(f"使用图像分析模式")
            else:
                # 图像生成模式 - 使用风格增强
                enhanced_prompt = self.enhance_prompt(prompt, style)
                print(f"使用图像生成模式")
            
            print(f"使用模型: {self.model_name}")
            print(f"提示词: {enhanced_prompt[:100]}...")
            
            # 转换输入图像为PIL格式
            pil_image = self._to_pil(input_image)
            print(f"输入图像1尺寸: {pil_image.size}")
            
            # 处理第二张和第三张图片（如果提供）
            contents = [enhanced_prompt, pil_image]
            image_count = 1
            
            if input_image_2 is not None:
                pil_image_2 = self._to_pil(input_image_2)
                print(f"输入图像2尺寸: {pil_image_2.size}")
                contents.append(pil_image_2)
                image_count += 1
            
            if input_image_3 is not None:
                pil_image_3 = self._to_pil(input_image_3)
                print(f"输入图像3尺寸: {pil_image_3.size}")
                contents.append(pil_image_3)
                image_count += 1
            
            print(f"使用{image_count}图输入模式")
            
            # 初始化客户端
            client = genai.Client(api_key=api_key.strip())
            
            # 调用模型 - 支持1-3张图片输入
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            
            # 提取响应数据
            img_bytes = None
            response_text = ""
            has_image = False
            
            if getattr(response, "candidates", None):
                for cand in response.candidates:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    
                    # 提取文本响应
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                        
                        # 提取图像数据
                        inline = getattr(part, "inline_data", None)
                        if inline is not None and getattr(inline, "data", None):
                            img_bytes = bytes(inline.data)
                            has_image = True
                            break
                    if img_bytes:
                        break
            
            # 根据输出模式处理响应
            if output_mode == "image_generation" and not has_image:
                raise RuntimeError("图像生成模式但API响应中没有图像数据")
            elif output_mode == "image_analysis" and not response_text:
                raise RuntimeError("图像分析模式但API响应中没有文本数据")
            
            # 处理图像输出
            if has_image and img_bytes:
                # 打开PIL图像
                out_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                print(f"生成图像尺寸: {out_pil.size}")
                
                # 转换为torch张量
                generated_image = self._pil_to_tensor_channel_last(out_pil)
            else:
                # 如果没有图像，返回一个占位符图像
                print("没有图像输出，返回占位符图像")
                placeholder_img = Image.new('RGB', (512, 512), (128, 128, 128))
                generated_image = self._pil_to_tensor_channel_last(placeholder_img)
            
            return (generated_image, response_text, has_image)
            
        except Exception as e:
            tb = traceback.format_exc()
            raise Exception(f"图像生成失败: {str(e)}\n{tb}")
