# 我的自定义节点

这个目录包含三个自定义ComfyUI节点，采用标准的ComfyUI节点结构：

```
my_custom_nodes/
├── __init__.py                    # 节点注册文件
├── nodes/                         # 节点实现目录
│   ├── __init__.py               # 包标识文件
│   ├── gemini_image_analysis.py  # Gemini图像分析节点
│   ├── gemini_image_generator.py # Gemini图像生成节点
│   └── placeholder_image_generator.py # 占位图片生成节点
└── README.md                     # 使用说明
```

## 节点列表

### 1. Gemini 图像分析 (GeminiImageAnalysis)
- **功能**: 使用Google Gemini 2.5 Flash Image API进行图像分析
- **输入**: API密钥、模型选择、输入图像、分析提示词
- **输出**: 分析文本、分析成功状态
- **分类**: AI/Image Analysis

### 2. Gemini 图像生成器 (GeminiImageGenerator)  
- **功能**: 使用Google Gemini 2.5 Flash Image API进行图像生成
- **输入**: API密钥、模型选择、提示词、输入图像
- **输出**: 生成图像、响应文本、是否有图像
- **分类**: AI/Image Generation

### 3. 占位图片生成器 (PlaceholderImageGenerator)
- **功能**: 生成指定尺寸的占位图片
- **输入**: 宽度、高度、预设尺寸、文本、背景颜色、文字颜色
- **输出**: 占位图片
- **分类**: Image/Generation

## 安装要求

确保已安装以下依赖：
```bash
pip install google-genai
```

## 使用方法

1. 重启ComfyUI
2. 在节点菜单中找到相应的节点
3. 配置API密钥（对于Gemini节点）
4. 连接节点并运行工作流

## 注意事项

- Gemini节点需要有效的Google AI API密钥
- 确保网络连接正常以访问Google API
- 图像分析节点支持单图或双图输入
- 图像生成节点支持多种风格和质量设置
