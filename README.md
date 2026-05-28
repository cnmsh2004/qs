# ComfyUI-Qs-LLM/README.md

# ComfyUI-Qs-LLM

Qs-LLM 是一个 ComfyUI 自定义节点，用于多图视觉分析与中文提示词反推，后端调用本地 Ollama。

## 功能

- 多图输入：`image1..image8`
- 批量图输入：`image_batch`
- 自动读取本地 Ollama 模型列表并作为 `model` 下拉选项
- 默认优先展示视觉相关模型
- 固定中文输出约束，适合反推提示词

## 环境要求

- Ollama 已运行（默认地址 `http://127.0.0.1:11434`）
- 已安装视觉模型，例如 `qwen2.5vl:7b`
- Python 依赖：
  - requests
  - Pillow
  - numpy

## 安装

1. 将本目录放到 `ComfyUI/custom_nodes/ComfyUI-Qs-LLM`
2. 安装依赖：
   - `pip install -r requirements.txt`
3. 重启 ComfyUI

## 使用建议

- 初次测试先接 1-2 张图
- `max_tokens` 建议 256-512
- 如果遇到 500，先降低图像分辨率或减少图片数量

## 连通性检查

- `ollama list`
- `curl.exe http://127.0.0.1:11434/api/tags`