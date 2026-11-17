"""轻量 HTTP 工具：检索、下载、连通性探测。

优先使用 requests（几乎所有 Python 环境都有），缺失时回退到 urllib，
保证 Agent 在「只有标准库」的裸环境里依然能跑完流程。
"""

from __future__ import annotations

import json
import shutil
import ssl
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 vision-agent/0.1"
)

try:  # pragma: no cover - 取决于运行环境
    import requests

    _HAS_REQUESTS = True
    _SSL_ERRORS: tuple = (requests.exceptions.SSLError,)
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
    _HAS_REQUESTS = False
    _SSL_ERRORS = ()

import urllib.error
import urllib.request


class HttpError(RuntimeError):
    """HTTP 请求失败。"""


def _ssl_ctx() -> ssl.SSLContext:
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    except Exception:  # pragma: no cover
        return ssl._create_unverified_context()  # type: ignore


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 1.5,
) -> Optional[Any]:
    """GET 并解析 JSON。失败返回 None（不抛异常，方便批量检索时忽略单点失败）。"""
    text = get_text(url, params=params, headers=headers, timeout=timeout, retries=retries, backoff=backoff)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def get_text(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 1.5,
    allow_insecure_fallback: bool = True,
) -> Optional[str]:
    """GET 文本内容。

    Windows + conda 环境下 requests 自带的 CA 证书链经常不全（或被代理 MITM），
    表现为 SSLCertVerificationError。这里做一次降级重试：关闭证书校验再试，
    只影响读取公开数据，不会用于传输凭据。
    """
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    if headers:
        hdrs.update(headers)

    last_err: Optional[Exception] = None
    insecure = False
    for attempt in range(retries + 1):
        try:
            if _HAS_REQUESTS:
                resp = requests.get(url, params=params, headers=hdrs, timeout=timeout, verify=not insecure)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.text
            full = url
            if params:
                from urllib.parse import urlencode

                full = f"{url}?{urlencode(params)}"
            req = urllib.request.Request(full, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as fh:
                return fh.read().decode("utf-8", errors="replace")
        except _SSL_ERRORS as exc:
            last_err = exc
            if allow_insecure_fallback and not insecure:
                insecure = True
                continue
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - 网络异常种类多，统一降级
            last_err = exc
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise HttpError(f"GET {url} 失败: {last_err}")


def download(
    url: str,
    dest: Path,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    overwrite: bool = False,
    chunk: int = 1 << 20,
) -> Path:
    """下载文件到 dest，支持断点续传。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)

    mode = "wb"
    offset = 0
    if tmp.exists() and tmp.stat().st_size > 0:
        mode = "ab"
        offset = tmp.stat().st_size
        hdrs["Range"] = f"bytes={offset}-"

    def _fetch(verify: bool):
        if _HAS_REQUESTS:
            with requests.get(url, headers=hdrs, timeout=timeout, stream=True, verify=verify) as resp:
                if resp.status_code == 416:  # 已经下完
                    return
                resp.raise_for_status()
                with open(tmp, mode) as fh:
                    for piece in resp.iter_content(chunk_size=chunk):
                        if piece:
                            fh.write(piece)
        else:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp, open(
                tmp, mode
            ) as fh:
                shutil.copyfileobj(resp, fh, chunk)

    try:
        try:
            _fetch(verify=True)
        except _SSL_ERRORS:
            _fetch(verify=False)
    except Exception:
        if tmp.exists() and tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
        raise

    tmp.replace(dest)
    return dest


def url_host(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return url


def is_offline() -> bool:
    """粗略判断当前是否离线（用于给出友好提示）。"""
    try:
        get_text("https://huggingface.co/api/models?limit=1", timeout=8, retries=0)
        return False
    except Exception:
        return True
