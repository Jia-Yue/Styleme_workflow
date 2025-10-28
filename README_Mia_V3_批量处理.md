# Mia_V3 批量处理脚本使用说明

## 功能特点

- 支持使用 `Mia_V3_api.json` 工作流进行批量图片处理
- 自动读取 `image_prompts.txt` 配置文件
- 动态替换工作流中的 API key
- 智能处理图片输入：空值自动跳过
- 支持多批次处理

## 文件说明

- `batch_process_mia_v3.py` - 主处理脚本
- `run_mia_v3_batch.bat` - Windows批处理启动脚本
- `Mia_V3_api.json` - ComfyUI工作流模板
- `image_prompts.txt` - 配置文件
- `README_Mia_V3_批量处理.md` - 本说明文档

## 配置文件格式

`image_prompts.txt` 文件格式如下：

```
api_key=你的API密钥

Batch1
image1=test.png
image2=test2.jpg
image3=test3.png
prompt=把图2模特的衣服穿到图3的人身上，注意把背景变成图1海边的背景

Batch2
image1=test.png
image2=
image3=
prompt=change cloth to red
```

### 配置说明

- `api_key` - Gemini API密钥
- `Batch1`, `Batch2` - 批次标识
- `image1`, `image2`, `image3` - 每个批次的三个图片
- `prompt` - 每个批次的提示词

**注意**：如果某个图片字段为空（如 `image2=`），脚本会自动跳过该图片输入。

## 使用方法

### 方法1：使用批处理脚本（推荐）

1. 确保 ComfyUI 正在运行
2. 双击运行 `run_mia_v3_batch.bat`
3. 等待处理完成

### 方法2：直接运行Python脚本

1. 确保 ComfyUI 正在运行
2. 打开命令行，进入脚本目录
3. 运行：`python batch_process_mia_v3.py`

## 工作流程

1. **读取配置**：从 `image_prompts.txt` 读取API密钥和批次信息
2. **加载工作流**：加载 `Mia_V3_api.json` 工作流模板
3. **检查服务器**：验证 ComfyUI 服务器是否运行
4. **处理批次**：
   - 替换工作流中的 API key
   - 设置对应批次的提示词
   - 智能处理图片输入（空值跳过）
   - 提交工作流到 ComfyUI
   - 等待处理完成
5. **输出结果**：生成的文件会保存到 ComfyUI 的 output 目录

## 处理示例

### 批次1处理
- 输入：`image1=test.png`, `image2=test2.jpg`, `image3=test3.png`
- 工作流会使用所有三个图片，Gemini会看到三张不同的图片

### 批次2处理
- 输入：`image1=test.png`, `image2=`, `image3=`
- 工作流只会使用 `image1`，跳过空的 `image2` 和 `image3`，Gemini只会看到一张图片

## 输出文件

处理完成后，生成的文件会保存在 ComfyUI 的 output 目录中，文件名格式为：
- `MiaV3_Batch1_xxxxx.png` - 批次1的输出
- `MiaV3_Batch2_xxxxx.png` - 批次2的输出

## 注意事项

1. 确保 ComfyUI 正在运行在 `http://127.0.0.1:8000`
2. 图片文件需要放在 ComfyUI 的 input 目录中
3. 确保 API 密钥有效且有足够的配额
4. 处理时间取决于图片大小和复杂度

## 故障排除

### 常见问题

1. **无法连接到 ComfyUI 服务器**
   - 检查 ComfyUI 是否正在运行
   - 确认端口 8000 未被占用

2. **图片文件未找到**
   - 确保图片文件在 ComfyUI 的 input 目录中
   - 检查文件名是否正确

3. **API 错误**
   - 检查 API 密钥是否正确
   - 确认 API 配额是否充足

4. **工作流处理失败**
   - 检查 ComfyUI 日志
   - 确认工作流模板是否正确

## 技术细节

- 使用 Python requests 库与 ComfyUI API 通信
- 支持深拷贝工作流模板避免数据污染
- 智能处理空值图片输入
- 动态调整节点连接：根据实际图片数量自动连接或删除输入
- 支持超时和错误处理
- 每个批次使用独立的提示词，通过 `Batch` 标识区分


