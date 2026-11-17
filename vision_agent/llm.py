"""大模型客户端。

Agent 在 S1（调研）、S5（训练脚本/S8 报告撰写）等环节需要大模型做「理解 + 归纳 + 生成」。

支持三种协议：
    anthropic        x-api-key 头 + {base}/messages        （MiniMax Code 默认配置即此协议）
    chat-completions Authorization: Bearer + {base}/chat/completions （OpenAI 兼容）
    none             不调用大模型，全部走确定性模板（离线可用）

凭据解析顺序：
    1. task.yaml 里的 llm.base_url / llm.api_key
    2. 环境变量 llm.api_key_env（默认 VISION_AGENT_LLM_API_KEY）
    3. ~/.minimax/config.yaml 里的 provider.minimax（baseURL + apiKey + defaultModel）
    4. 都找不到 -> 降级为 template 模式，流程仍然跑完，只是文案更朴素

模板降级不是「假装调用了模型」：模板模式会明确标注 `source=template`，
产物里也会写明「本节由确定性模板生成，未经过大模型润色」。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import LLMConfig
from .httpclient import get_json

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


@dataclass
class LLMResult:
    ok: bool
    text: str
    source: str  # "llm" | "template" | "error"
    error: str = ""
    model: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _default_minimax_config_path() -> Path:
    return Path.home() / ".minimax" / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


class LLMClient:
    """统一的聊天调用入口。"""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._resolved: Optional[Tuple[str, str, str, str]] = None  # protocol, base, key, model
        self.last_error = ""

    # ------------------------------------------------------------------
    # 凭据解析
    # ------------------------------------------------------------------
    def _resolve(self) -> Optional[Tuple[str, str, str, str]]:
        if self._resolved is not None:
            return self._resolved

        cfg = self.cfg
        provider = (cfg.provider or "auto").lower()

        # 1) 显式配置
        if cfg.base_url:
            key = cfg.api_key or os.environ.get(cfg.api_key_env, "")
            protocol = provider if provider in {"anthropic", "chat-completions"} else "chat-completions"
            model = cfg.model or "gpt-4o-mini"
            self._resolved = (protocol, cfg.base_url.rstrip("/"), key, model)
            return self._resolved

        # 2) 环境变量
        env_url = os.environ.get("VISION_AGENT_LLM_BASE_URL", "")
        env_key = os.environ.get(cfg.api_key_env, "") or os.environ.get("VISION_AGENT_LLM_API_KEY", "")
        if env_url:
            protocol = provider if provider in {"anthropic", "chat-completions"} else "chat-completions"
            self._resolved = (protocol, env_url.rstrip("/"), env_key, cfg.model or "gpt-4o-mini")
            return self._resolved

        # 3) ~/.minimax/config.yaml
        if provider in {"auto", "anthropic"}:
            cfg_path = Path(cfg.config_path).expanduser() if cfg.config_path else _default_minimax_config_path()
            if cfg_path.exists():
                data = _load_yaml(cfg_path)
                prov = (data.get("provider") or {}).get("minimax") or {}
                options = prov.get("options") or {}
                base = options.get("baseURL") or ""
                key = options.get("apiKey") or ""
                if base and key and key != "sk-xxx":
                    npm = prov.get("npm", "")
                    protocol = "anthropic" if "anthropic" in npm else "chat-completions"
                    model = cfg.model or (data.get("defaultModel") or "").split("/")[-1] or "MiniMax-M3.1-Flash"
                    self._resolved = (protocol, base.rstrip("/"), key, model)
                    return self._resolved

        if provider == "none":
            self._resolved = ("none", "", "", "")
            return self._resolved

        return None

    @property
    def available(self) -> bool:
        resolved = self._resolve()
        return bool(resolved and resolved[0] != "none" and resolved[1])

    @property
    def model_name(self) -> str:
        resolved = self._resolve()
        return resolved[3] if resolved else "(未配置)"

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        """发起一次对话。失败时按 fallback 策略返回模板/错误。"""
        resolved = self._resolve()
        if not resolved or resolved[0] == "none":
            self.last_error = "未配置可用的大模型"
            return LLMResult(False, "", "error", self.last_error, self.model_name)

        protocol, base, key, model = resolved
        payload_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature

        try:
            if protocol == "anthropic":
                url = f"{base}/messages"
                headers = {"Content-Type": "application/json", "x-api-key": key}
                body = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": user}],
                    "system": system,
                }
                if temperature is not None:
                    body["temperature"] = temperature
            else:
                url = f"{base}/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
                body = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": payload_messages,
                }
                if temperature is not None:
                    body["temperature"] = temperature

            data = self._post_json(url, headers, body)
            text = self._extract_text(data, protocol)
            if not text:
                raise RuntimeError(f"响应里没有文本内容: {str(data)[:300]}")
            return LLMResult(True, text, "llm", "", model)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            return LLMResult(False, "", "error", str(exc), model)

    def _post_json(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("需要 requests 库才能调用大模型: pip install requests")
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.cfg.timeout)
        except requests.exceptions.SSLError:
            # conda 环境的 CA 证书链常不完整，降级为不校验证书
            resp = requests.post(url, headers=headers, json=body, timeout=self.cfg.timeout, verify=False)
        if resp.status_code >= 400:
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    "大模型鉴权失败（401/403）。"
                    "config.yaml 里的 apiKey 可能是占位符（sk-xxx），"
                    "请在 task.yaml 的 llm.api_key 填写真实密钥，"
                    "或设置环境变量 VISION_AGENT_LLM_API_KEY。"
                    f" 服务端返回: {resp.text[:200]}"
                )
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    @staticmethod
    def _extract_text(data: Dict[str, Any], protocol: str) -> str:
        if protocol == "anthropic":
            blocks = data.get("content", []) or []
            parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            return "\n".join(p for p in parts if p).strip()
        choices = data.get("choices", []) or []
        if choices:
            msg = choices[0].get("message", {}) or {}
            return (msg.get("content") or "").strip()
        return ""

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------
    def complete_or_template(
        self,
        system: str,
        user: str,
        template: str,
        label: str = "",
    ) -> LLMResult:
        """调用大模型；失败时回落到 template 文本。"""
        result = self.chat(system, user)
        if result.ok:
            return result
        if self.cfg.fallback == "strict":
            return result
        return LLMResult(True, template, "template", result.error, self.model_name)


def extract_json_block(text: str) -> Optional[Any]:
    """从大模型输出里抠出 JSON（容忍 ```json 包裹和前后解释文字）。"""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 退而求其次：找第一对平衡的大括号
    start = cleaned.find("{")
    if start < 0:
        start = cleaned.find("[")
    if start < 0:
        return None
    opener = cleaned[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    for idx in range(start, len(cleaned)):
        if cleaned[idx] == opener:
            depth += 1
        elif cleaned[idx] == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def chat_sessions_available() -> List[str]:
    """列出本机可用的模型来源（用于 doctor 展示）。"""
    found = []
    cfg_path = _default_minimax_config_path()
    if cfg_path.exists():
        data = _load_yaml(cfg_path)
        if data.get("provider"):
            found.append(f"minimax config: {cfg_path}")
    for env in ("VISION_AGENT_LLM_API_KEY", "VISION_AGENT_LLM_BASE_URL"):
        if os.environ.get(env):
            found.append(f"env: {env}")
    return found
