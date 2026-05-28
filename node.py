import base64
import io
import json
import os
import re
from typing import List

import numpy as np
import requests
from PIL import Image


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _normalize_one(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image ndim after split: {arr.ndim}, shape={arr.shape}")

    if arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported channel shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return arr


def _split_images(x) -> List[np.ndarray]:
    arr = _to_numpy(x)
    if arr.ndim == 3:
        return [_normalize_one(arr)]
    if arr.ndim == 4:
        return [_normalize_one(arr[i]) for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported image tensor shape: {arr.shape}")


def _to_jpeg_b64(arr: np.ndarray, quality: int = 90) -> str:
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gather_images(image_slots, image_batch):
    images = []
    for item in image_slots:
        if item is not None:
            images.extend(_split_images(item))
    if image_batch is not None:
        images.extend(_split_images(image_batch))
    return images


def _is_vl_model(name: str) -> bool:
    n = name.lower()
    keys = ["vl", "vision", "llava", "minicpm-v", "qwen2.5vl", "qwen2vl", "qwen3-vl"]
    return any(k in n for k in keys)


def _fetch_ollama_models(base_url: str) -> List[str]:
    base = (base_url or "http://127.0.0.1:11434").rstrip("/")
    url = f"{base}/api/tags"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name")
            if isinstance(name, str) and name.strip():
                models.append(name.strip())

        seen = set()
        uniq = []
        for m in models:
            if m not in seen:
                uniq.append(m)
                seen.add(m)

        vl = [m for m in uniq if _is_vl_model(m)]
        others = [m for m in uniq if m not in vl]
        ordered = vl + others
        return ordered if ordered else ["qwen2.5vl:7b"]
    except Exception:
        return ["qwen2.5vl:7b"]


def _remove_think_blocks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```think[\s\S]*?```", "", text, flags=re.IGNORECASE)
    return text.strip()


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text.strip()


def _looks_like_json(text: str) -> bool:
    text = (text or "").strip()
    return text.startswith("{") and text.endswith("}")


def _extract_content_by_regex(text: str) -> str:
    text = text or ""
    match = re.search(r'"content"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.S)
    if not match:
        return ""

    value = match.group(1)
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\").strip()


def _extract_message_text(msg: dict) -> str:
    if not isinstance(msg, dict):
        return ""

    content = msg.get("content", "")
    if isinstance(content, str) and content.strip():
        return content

    thinking = msg.get("thinking", "")
    if isinstance(thinking, str) and thinking.strip():
        return thinking

    reasoning = msg.get("reasoning_content", "")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    return ""


def _try_extract_nested_json_content(text: str) -> str:
    text = _strip_code_fence(text)
    if not _looks_like_json(text):
        return ""

    try:
        obj = json.loads(text)
    except Exception:
        return ""

    if not isinstance(obj, dict):
        return ""

    msg = obj.get("message")
    if isinstance(msg, dict):
        text = _extract_message_text(msg)
        if text:
            return text

    response = obj.get("response")
    if isinstance(response, str) and response.strip():
        return response

    return ""


def _clean_output_text(raw: str) -> str:
    if raw is None:
        return ""

    text = str(raw).strip()
    text = _strip_code_fence(text)
    text = _remove_think_blocks(text)

    regex_content = _extract_content_by_regex(text)
    if regex_content:
        text = regex_content.strip()
        text = _strip_code_fence(text)
        text = _remove_think_blocks(text)

    nested = _try_extract_nested_json_content(text)
    if nested:
        text = nested.strip()
        text = _strip_code_fence(text)
        text = _remove_think_blocks(text)

    text = re.sub(r'"thinking"\s*:\s*"((?:\\.|[^"\\])*)"\s*,?', "", text, flags=re.S)
    text = re.sub(r'"model"\s*:\s*"[^"]*"\s*,?', "", text)
    text = re.sub(r'"created_at"\s*:\s*"[^"]*"\s*,?', "", text)
    text = re.sub(r'"done"\s*:\s*(true|false)\s*,?', "", text)
    text = re.sub(r'"done_reason"\s*:\s*"[^"]*"\s*,?', "", text)
    text = re.sub(r'"total_duration"\s*:\s*\d+\s*,?', "", text)
    text = re.sub(r'"load_duration"\s*:\s*\d+\s*,?', "", text)
    text = re.sub(r'"prompt_eval_count"\s*:\s*\d+\s*,?', "", text)
    text = re.sub(r'"prompt_eval_duration"\s*:\s*\d+\s*,?', "", text)
    text = re.sub(r'"eval_count"\s*:\s*\d+\s*,?', "", text)
    text = re.sub(r'"eval_duration"\s*:\s*\d+\s*,?', "", text)

    text = text.replace("\\n", "\n").replace('\\"', '"')
    text = text.strip()

    if _looks_like_json(text):
        extracted = _extract_content_by_regex(text)
        if extracted:
            text = extracted.strip()

    return text.strip()


def _extract_final_content(data) -> str:
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict):
            text = _extract_message_text(msg)
            if text:
                cleaned = _clean_output_text(text)
                if cleaned:
                    return cleaned

        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            m = choices[0].get("message", {})
            if isinstance(m, dict):
                text = _extract_message_text(m)
                if text:
                    cleaned = _clean_output_text(text)
                    if cleaned:
                        return cleaned

        response = data.get("response")
        if isinstance(response, str):
            cleaned = _clean_output_text(response)
            if cleaned:
                return cleaned

    raw = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
    regex_content = _extract_content_by_regex(raw)
    if regex_content:
        return _clean_output_text(regex_content)

    return ""


class QsLlmNode:
    @classmethod
    def INPUT_TYPES(cls):
        default_base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_list = _fetch_ollama_models(default_base)

        return {
            "required": {
                "model": (model_list, {"default": model_list[0]}),
                "role": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 1325706302, "min": -1, "max": 2147483647, "step": 1}),
                "control_after_generate": (["randomize", "fixed", "increment", "decrement"], {"default": "randomize"}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "reasoning_effort": (["none", "low", "medium", "high"], {"default": "none"}),
                "skip_error": ("BOOLEAN", {"default": False}),
                "require_image": ("BOOLEAN", {"default": False}),
                "ollama_base_url": ("STRING", {"default": default_base}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image_batch": ("IMAGE",),
                "video": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run"
    CATEGORY = "LLM"

    def run(
        self,
        model,
        role,
        prompt,
        temperature,
        seed,
        control_after_generate,
        max_tokens,
        top_p,
        presence_penalty,
        frequency_penalty,
        reasoning_effort,
        skip_error,
        require_image,
        ollama_base_url,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image_batch=None,
        video="",
    ):
        try:
            images = _gather_images(
                [image1, image2, image3, image4, image5, image6, image7, image8],
                image_batch,
            )
            has_images = len(images) > 0

            if require_image and not has_images:
                msg = "[ERROR] No image input. Connect image1..image8 or image_batch."
                if skip_error:
                    return (msg,)
                raise ValueError(msg)

            real_seed = int(seed)
            if control_after_generate == "randomize":
                real_seed = -1
            elif control_after_generate == "increment":
                real_seed = int(seed) + 1
            elif control_after_generate == "decrement":
                real_seed = int(seed) - 1

            repeat_penalty = 1.0 + max(0.0, float(frequency_penalty)) * 0.2

            effort_hint = ""
            if reasoning_effort != "none":
                effort_hint = f"\n推理强度：{reasoning_effort}。"

            output_rule = (
                "\n只输出最终答案，不要输出JSON，不要输出字段名。"
                "\n不要解释你为何这样做。"
            )

            user_prompt = (prompt or "").strip()
            if not user_prompt:
                if has_images:
                    user_prompt = "请分析这些图片并输出可复用的中文提示词描述。"
                else:
                    user_prompt = "请根据输入文本输出可复用的中文提示词描述。"

            full_prompt = (user_prompt + effort_hint + output_rule).strip()

            base = (ollama_base_url or "").strip().rstrip("/")
            if not base:
                base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

            system_role = (role or "").strip()
            if system_role:
                system_role += (
                    "\n只输出最终答案。"
                    "\n禁止输出JSON、字段名。"
                )
            else:
                system_role = "只输出最终答案。禁止输出JSON、字段名。"

            user_message = {"role": "user", "content": full_prompt}
            if has_images:
                b64_images = [_to_jpeg_b64(x, quality=90) for x in images]
                user_message["images"] = b64_images

            payload = {
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_role},
                    user_message,
                ],
                "options": {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "num_predict": int(max_tokens),
                    "seed": int(real_seed),
                    "repeat_penalty": float(repeat_penalty),
                },
            }

            if float(presence_penalty) != 0.0:
                payload["messages"][0]["content"] += f"\n新颖度偏好：{presence_penalty:.2f}。"

            url = f"{base}/api/chat"
            resp = requests.post(url, json=payload, timeout=300)

            if resp.status_code >= 400:
                try:
                    err_json = resp.json()
                    err_text = json.dumps(err_json, ensure_ascii=False)
                except Exception:
                    err_text = resp.text
                raise RuntimeError(f"Ollama {resp.status_code} error: {err_text}")

            data = resp.json()
            content = _extract_final_content(data)

            if not str(content).strip():
                content = "未提取到最终正文。请降低max_tokens、简化prompt，或改用qwen2.5vl模型。"

            return (str(content),)

        except Exception as e:
            msg = f"[ERROR] {e}"
            if skip_error:
                return (msg,)
            raise RuntimeError(msg)