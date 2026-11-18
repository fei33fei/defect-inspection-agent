"""可视化检测系统（Flask）。

功能：
    /                 上传图片 -> 检测 -> 标注图 + 框信息 + 下载
    /api/predict      检测接口（JSON）
    /download/<file>  下载标注结果
    /camera           摄像头页面（可选摄像头序号）
    /video_feed       MJPEG 视频流（电脑内置摄像头 / 外接 USB 摄像头 / 视频文件）
    /api/cameras      枚举可用摄像头
    /health           健康检查

依赖是「懒加载」的：没有 torch/ultralytics/cv2 时不会在导入阶段崩溃，
而是返回一张说明页面，告诉用户该装什么。

环境变量：
    VA_MODEL   权重路径（best.pt）
    VA_CONF    置信度阈值
    VA_IOU     NMS IoU 阈值
    VA_CAMERA  默认摄像头序号
    VA_HOST / VA_PORT
"""

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODEL_PATH = os.environ.get("VA_MODEL", "")
CONF = float(os.environ.get("VA_CONF", "0.25"))
IOU = float(os.environ.get("VA_IOU", "0.45"))
CAMERA_INDEX = int(os.environ.get("VA_CAMERA", "0"))
TITLE = os.environ.get("VA_TITLE", "视觉检测系统")
UPLOAD_DIR = Path(os.environ.get("VA_UPLOAD_DIR", "app/uploads"))
OUTPUT_DIR = Path(os.environ.get("VA_OUTPUT_DIR", "app/outputs"))
MAX_CONTENT_LENGTH = int(os.environ.get("VA_MAX_UPLOAD_MB", "20")) * 1024 * 1024

_model_cache: Dict[str, Any] = {}


def _ensure_agent_on_path() -> None:
    """把 vision_agent 包所在目录加入 sys.path。

    本文件会被复制成独立脚本（工作区 scripts/app_xxx.py）运行，
    那时包本身不在 sys.path 上，import vision_agent.attention 会失败。
    这里向上回溯几级找到含 vision_agent/ 的目录并补进路径。

    另外两种情况也一并处理：
    - ``VA_AGENT_DIR`` 环境变量显式指定（生成脚本时会写进去）；
    - 桌面版运行时布局：``agent/`` 与 ``server/`` 平级，向上回溯找不到。
    """
    import sys

    try:
        import vision_agent  # noqa: F401

        return
    except ImportError:
        pass

    here = Path(__file__).resolve().parent

    def _add(candidate: Path) -> bool:
        if (candidate / "vision_agent" / "attention.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return True
        return False

    env_dir = os.environ.get("VA_AGENT_DIR", "").strip()
    if env_dir and _add(Path(env_dir)):
        return

    for candidate in (here.parent / "agent", here.parent):
        if _add(candidate):
            return

    for parent in here.parents:
        if _add(parent):
            return
        # 只回溯有限层数，避免在巨大的目录树里空转
        if len(str(parent)) < len(str(here)) - 200:
            return


# ---------------------------------------------------------------------------
# 注意力模块补丁
# ---------------------------------------------------------------------------
# 必须在加载 SE/EMA/CBAM 权重**之前**装好：ultralytics 反序列化 .pt 时
# 会按类名回查 ultralytics.nn.modules.conv，补丁晚一步就会报
# "Can't get attribute 'SEBlock'"。
_ensure_agent_on_path()
try:
    from vision_agent.attention import install as _install_attention

    _install_attention()
except Exception as _exc:  # noqa: BLE001 - 补丁失败不阻断启动，/health 会显示原因
    _ATTENTION_PATCH_ERROR = str(_exc)
else:
    _ATTENTION_PATCH_ERROR = ""


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


def load_model(model_path: str):
    """加载 YOLO 模型（带缓存）。失败返回 (None, 错误信息)。"""
    if not model_path:
        return None, "未指定权重路径（请设置环境变量 VA_MODEL）"
    if "model" in _model_cache:
        return _model_cache["model"], ""
    path = Path(model_path)
    if not path.exists():
        return None, f"权重文件不存在: {path}"
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return None, f"未安装 ultralytics: {exc}"
    try:
        model = YOLO(str(path))
    except Exception as exc:  # noqa: BLE001
        hint = f"（注意力补丁未生效: {_ATTENTION_PATCH_ERROR}）" if _ATTENTION_PATCH_ERROR else ""
        return None, f"模型加载失败: {exc}{hint}"
    _model_cache["model"] = model
    return model, ""


# ---------------------------------------------------------------------------
# 摄像头
# ---------------------------------------------------------------------------


def list_cameras(max_index: int = 5) -> List[Dict[str, Any]]:
    """尝试打开 0..max_index-1 号摄像头，返回可用列表。"""
    try:
        import cv2
    except ImportError:
        return []
    cams = []
    for idx in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cams.append({"index": idx, "available": True})
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()
    return cams


def video_stream(source: str, conf: float = CONF, iou: float = IOU, width: int = 960):
    """MJPEG 视频流生成器。source 可以是摄像头序号或视频文件路径。"""
    try:
        import cv2
    except ImportError as exc:
        yield _error_frame(f"缺少 opencv: {exc}")
        return

    model, err = load_model(MODEL_PATH)
    if model is None:
        yield _error_frame(err)
        return

    src = source.strip()
    is_file = not src.isdigit()
    cap = cv2.VideoCapture(int(src) if not is_file else src)
    if not cap.isOpened():
        yield _error_frame(f"无法打开视频源: {source}")
        return

    names = getattr(model, "names", {}) or {}

    def stream():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                results = model.predict(frame, conf=conf, iou=iou, verbose=False)
                frame = results[0].plot()
            except Exception as exc:  # noqa: BLE001
                cv2.putText(
                    frame, f"detect error: {exc}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
            buf = io.BytesIO()
            cv2.imencode(".jpg", frame)[1].tofile(buf)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
        cap.release()

    return stream()


def _error_frame(message: str) -> bytes:
    """把错误信息渲染成一张图片帧（视频流里直接能看到原因）。"""
    try:
        import cv2
        import numpy as np

        frame = np.zeros((240, 960, 3), dtype="uint8")
        cv2.putText(frame, "Vision Agent", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2)
        cv2.putText(frame, message[:110], (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    except Exception:
        pass
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#262b36; --fg:#e6e9ef; --muted:#8b94a7; --accent:#4c8dff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  header { padding: 18px 28px; border-bottom: 1px solid var(--line); display:flex; align-items:center; gap:16px; }
  header h1 { font-size: 18px; margin:0; font-weight:600; }
  header nav a { color:var(--muted); text-decoration:none; margin-right:14px; font-size:14px; }
  header nav a:hover { color:var(--accent); }
  .wrap { max-width: 1080px; margin: 28px auto; padding: 0 20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius: 12px; padding: 22px; margin-bottom: 20px; }
  h2 { font-size:15px; margin:0 0 14px; color:var(--muted); font-weight:600; letter-spacing:.04em; }
  input[type=file] { color:var(--muted); }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px; padding:9px 20px; cursor:pointer; font-size:14px; }
  button:hover { filter: brightness(1.1); }
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
  .preview img { max-width: 520px; border-radius:10px; border:1px solid var(--line); display:block; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; }
  .tag { display:inline-block; padding:2px 9px; border-radius:20px; background:#223; color:var(--accent); font-size:12px; margin:2px; }
  .muted { color:var(--muted); font-size:13px; }
  a.dl { color:var(--accent); text-decoration:none; font-size:14px; }
  video, img.stream { max-width:100%; border-radius:10px; border:1px solid var(--line); }
  select { background:#1e222b; color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:7px 10px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <nav>
    <a href="/">图片检测</a>
    <a href="/camera">摄像头实时检测</a>
    <a href="/health">状态</a>
  </nav>
</header>
<div class="wrap">
  <div class="card">
    <h2>上传图片检测</h2>
    <form id="f" enctype="multipart/form-data">
      <input type="file" name="image" id="file" multiple required>
      <button type="submit" style="margin-left:12px">开始检测</button>
      <button type="button" id="batchBtn" style="margin-left:10px;background:#3a4152">批量检测 + 对比图</button>
    </form>
    <p class="muted" id="msg"></p>
    <div class="row" id="result"></div>
    <div id="batch" style="margin-top:18px"></div>
  </div>
</div>
<script>
const form = document.getElementById('f');
const msg = document.getElementById('msg');
const result = document.getElementById('result');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  msg.textContent = '检测中…';
  result.innerHTML = '';
  const files = document.getElementById('file').files;
  if (files.length > 1) { msg.textContent = '检测到多张图片，请点「批量检测 + 对比图」'; return; }
  const fd = new FormData();
  fd.append('image', files[0]);
  try {
    const resp = await fetch('/api/predict', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok || data.error) { msg.textContent = '失败: ' + (data.error || resp.statusText); return; }
    msg.textContent = `完成，共 ${data.boxes.length} 个目标`;
    let tags = data.boxes.map(b => `<span class="tag">${b.class_name} ${(b.confidence*100).toFixed(1)}%</span>`).join('');
    result.innerHTML = `
      <div class="preview"><img src="${data.image_url}"><p style="margin-top:10px"><a class="dl" href="${data.download_url}" download>下载标注图</a></p></div>
      <div style="flex:1;min-width:260px"><h2>检测结果</h2>${tags || '<p class="muted">未检测到目标</p>'}
      <p class="muted" style="margin-top:14px">耗时 ${data.elapsed_ms} ms</p></div>`;
  } catch (err) { msg.textContent = '请求失败: ' + err; }
});

document.getElementById('batchBtn').addEventListener('click', async () => {
  const files = document.getElementById('file').files;
  if (!files.length) { msg.textContent = '请先选择图片'; return; }
  msg.textContent = `批量检测中（${files.length} 张）…`;
  document.getElementById('batch').innerHTML = '';
  const fd = new FormData();
  for (const f of files) fd.append('images', f);
  try {
    const resp = await fetch('/api/predict_batch', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok || data.error) { msg.textContent = '失败: ' + (data.error || resp.statusText); return; }
    msg.textContent = `完成：${data.count} 张图，共 ${data.total_boxes} 个目标`;
    const rows = data.results.map(r => `<tr><td>${r.name}</td><td>${r.boxes ? r.boxes.length : '-'}</td>
      <td>${r.summary || (r.error ? r.error : '-')}</td><td>${r.elapsed_ms ? r.elapsed_ms + ' ms' : '-'}</td>
      <td>${r.image_url ? `<a class="dl" href="${r.download_url}" download>下载</a>` : '-'}</td></tr>`).join('');
    document.getElementById('batch').innerHTML = `
      ${data.comparison_url ? `<h2>对比图</h2><img class="stream" src="${data.comparison_url}">
        <p style="margin:10px 0 16px"><a class="dl" href="${data.comparison_url}" download>下载对比图</a></p>` : ''}
      <table><thead><tr><th>图片</th><th>目标数</th><th>类别</th><th>耗时</th><th>结果</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  } catch (err) { msg.textContent = '请求失败: ' + err; }
});
</script>
</body></html>
"""

CAMERA_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · 摄像头</title>
<style>
  body { margin:0; background:#0f1115; color:#e6e9ef; font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
  header { padding:18px 28px; border-bottom:1px solid #262b36; display:flex; gap:16px; align-items:center; }
  header h1 { font-size:18px; margin:0; }
  header a { color:#8b94a7; text-decoration:none; font-size:14px; margin-right:12px; }
  .wrap { max-width:980px; margin:24px auto; padding:0 20px; }
  .bar { display:flex; gap:12px; align-items:center; margin-bottom:16px; }
  select, button { background:#1e222b; color:#e6e9ef; border:1px solid #262b36; border-radius:8px; padding:8px 12px; }
  button { background:#4c8dff; color:#fff; border:0; cursor:pointer; }
  .muted { color:#8b94a7; font-size:13px; }
</style>
</head>
<body>
<header><h1>__TITLE__</h1><a href="/">图片检测</a><a href="/camera">摄像头</a></header>
<div class="wrap">
  <div class="bar">
    <label class="muted" for="src">视频源</label>
    <select id="src">__CAMERA_OPTIONS__</select>
    <input id="file" type="file" accept="video/*" style="display:none">
    <button onclick="document.getElementById('file').click()">选择视频文件</button>
    <button onclick="start()">开始检测</button>
    <button onclick="stop()">停止</button>
  </div>
  <img class="stream" id="stream" alt="video stream">
  <p class="muted" id="tip">外接 USB 摄像头通常在 1 号及之后；也可直接选择视频文件做流式检测。</p>
</div>
<script>
function start() {
  const src = document.getElementById('src').value;
  document.getElementById('stream').src = '/video_feed?source=' + encodeURIComponent(src) + '&t=' + Date.now();
}
function stop() { document.getElementById('stream').src = ''; }
document.getElementById('file').addEventListener('change', function() {
  if (this.files.length) { document.getElementById('tip').textContent = '已选择: ' + this.files[0].name + '（复制到工作目录后以文件名方式访问）'; }
});
start();
</script>
</body></html>
"""


def _save_annotated(result, path: Path) -> Path:
    """保存标注图。

    兼容三条路径：
      1. ultralytics >= 8.1 的 Results.save()
      2. cv2.imwrite(result.plot())
      3. PIL（cv2 也没装时的最后退路）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.save(filename=str(path))
        return path
    except AttributeError:
        pass
    except Exception:  # noqa: BLE001 - save() 存在但失败时也走退路
        pass

    annotated = result.plot()
    try:
        import cv2

        cv2.imwrite(str(path), annotated)
        return path
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass

    try:
        from PIL import Image

        img = annotated[:, :, ::-1] if annotated.ndim == 3 else annotated  # BGR -> RGB
        Image.fromarray(img).save(str(path))
        return path
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"保存标注图失败（cv2 与 PIL 都不可用）: {exc}") from exc


def detect_one(model, src_path: Path, uid: str) -> Dict[str, Any]:
    """对单张图片做检测，返回标注图路径 + 框信息。"""
    t0 = time.time()
    result = model.predict(str(src_path), conf=CONF, iou=IOU, verbose=False)[0]
    annotated = OUTPUT_DIR / f"{uid}.jpg"
    _save_annotated(result, annotated)
    elapsed = int((time.time() - t0) * 1000)

    boxes: List[Dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes):
        for b in result.boxes:
            cls_id = int(b.cls[0])
            boxes.append(
                {
                    "class_id": cls_id,
                    "class_name": result.names.get(cls_id, str(cls_id)),
                    "confidence": round(float(b.conf[0]), 4),
                    "xyxy": [round(float(v), 2) for v in b.xyxy[0]],
                }
            )
    return {
        "boxes": boxes,
        "annotated": annotated,
        "elapsed_ms": elapsed,
        "source": str(src_path),
    }


def build_comparison_grid(
    items: List[Dict[str, Any]],
    out_path: Path,
    columns: int = 2,
    cell_height: int = 320,
) -> Optional[Path]:
    """把多张标注图拼成一张对比图（带标题条）。

    items: [{"image": Path, "label": str}, ...]
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    cells = []
    for item in items:
        path = Path(item.get("image", ""))
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = cell_height / max(1, h)
        img = cv2.resize(img, (max(1, int(w * scale)), cell_height))
        bar = np.zeros((28, img.shape[1], 3), dtype="uint8")
        label = str(item.get("label", ""))[:60]
        cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
        cells.append(np.vstack([bar, img]))
    if not cells:
        return None

    cell_w = max(c.shape[1] for c in cells)
    padded = []
    for c in cells:
        if c.shape[1] < cell_w:
            pad = np.zeros((c.shape[0], cell_w - c.shape[1], 3), dtype="uint8")
            c = np.hstack([c, pad])
        padded.append(c)

    rows = []
    for i in range(0, len(padded), max(1, columns)):
        chunk = padded[i : i + max(1, columns)]
        while len(chunk) < max(1, columns):
            chunk.append(np.zeros_like(chunk[0]))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path


def create_app():
    """创建 Flask 应用。Flask 缺失时返回 None。"""
    try:
        from flask import Flask, jsonify, render_template_string, request, send_from_directory
    except ImportError:
        return None

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML.replace("__TITLE__", TITLE))

    @app.get("/camera")
    def camera():
        options = []
        for cam in list_cameras():
            options.append(f'<option value="{cam["index"]}">摄像头 {cam["index"]}</option>')
        if not options:
            options.append(f'<option value="{CAMERA_INDEX}">摄像头 {CAMERA_INDEX}（未探测到设备）</option>')
        html = CAMERA_HTML.replace("__TITLE__", TITLE).replace("__CAMERA_OPTIONS__", "".join(options))
        return html

    @app.get("/video_feed")
    def video_feed():
        source = request.args.get("source", str(CAMERA_INDEX))
        stream = video_stream(source)
        return stream

    @app.post("/api/predict")
    def api_predict():
        from flask import jsonify

        model, err = load_model(MODEL_PATH)
        if model is None:
            return jsonify({"error": err}), 500
        if "image" not in request.files:
            return jsonify({"error": "没有收到图片文件 (字段名 image)"}), 400
        file = request.files["image"]
        if not file or not file.filename:
            return jsonify({"error": "空文件"}), 400

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename).suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            ext = ".jpg"
        uid = uuid.uuid4().hex[:12]
        src_path = UPLOAD_DIR / f"{uid}{ext}"
        file.save(src_path)

        try:
            info = detect_one(model, src_path, uid)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"推理失败: {exc}"}), 500

        annotated = info["annotated"]
        return jsonify(
            {
                "boxes": info["boxes"],
                "image_url": f"/download/{annotated.name}",
                "download_url": f"/download/{annotated.name}",
                "elapsed_ms": info["elapsed_ms"],
                "model": MODEL_PATH,
                "conf": CONF,
            }
        )

    @app.post("/api/predict_batch")
    def api_predict_batch():
        """批量检测：一次上传多张图，逐张检测并生成对比图。"""
        from flask import jsonify

        model, err = load_model(MODEL_PATH)
        if model is None:
            return jsonify({"error": err}), 500
        files = request.files.getlist("images") or request.files.getlist("image")
        files = [f for f in files if f and f.filename]
        if not files:
            return jsonify({"error": "没有收到图片文件 (字段名 images)"}), 400

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        results = []
        grid_items = []
        for file in files:
            ext = Path(file.filename).suffix.lower() or ".jpg"
            if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                ext = ".jpg"
            uid = uuid.uuid4().hex[:12]
            src_path = UPLOAD_DIR / f"{uid}{ext}"
            file.save(src_path)
            try:
                info = detect_one(model, src_path, uid)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {"name": file.filename, "error": str(exc), "boxes": [], "image_url": ""}
                )
                continue
            names = ",".join(
                sorted({b["class_name"] for b in info["boxes"]})[:4]
            ) or "无目标"
            results.append(
                {
                    "name": file.filename,
                    "boxes": info["boxes"],
                    "image_url": f"/download/{info['annotated'].name}",
                    "download_url": f"/download/{info['annotated'].name}",
                    "elapsed_ms": info["elapsed_ms"],
                    "summary": names,
                }
            )
            grid_items.append(
                {
                    "image": info["annotated"],
                    "label": f"{file.filename[:28]} | {names} | {info['elapsed_ms']}ms",
                }
            )

        grid_url = ""
        if len(grid_items) > 1:
            grid = build_comparison_grid(
                grid_items, OUTPUT_DIR / f"compare_{uuid.uuid4().hex[:8]}.jpg"
            )
            if grid is not None:
                grid_url = f"/download/{grid.name}"

        return jsonify(
            {
                "results": results,
                "comparison_url": grid_url,
                "count": len(results),
                "total_boxes": sum(len(r.get("boxes", [])) for r in results),
            }
        )

    @app.get("/download/<path:filename>")
    def download(filename: str):
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

    @app.get("/api/cameras")
    def api_cameras():
        return jsonify({"cameras": list_cameras()})

    @app.get("/health")
    def health():
        model, err = load_model(MODEL_PATH)
        return jsonify(
            {
                "status": "ok" if model is not None else "degraded",
                "model": MODEL_PATH,
                "model_loaded": model is not None,
                "error": err,
                "conf": CONF,
                "iou": IOU,
            }
        )

    return app


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    global MODEL_PATH, CONF, IOU, CAMERA_INDEX, TITLE

    parser = argparse.ArgumentParser(description="视觉检测可视化系统")
    parser.add_argument("--model", default=MODEL_PATH, help="权重路径")
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--host", default=os.environ.get("VA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VA_PORT", "8000")))
    parser.add_argument("--title", default=TITLE)
    args = parser.parse_args(argv)

    MODEL_PATH = args.model
    CONF = args.conf
    IOU = args.iou
    CAMERA_INDEX = args.camera
    TITLE = args.title
    _model_cache.clear()

    app = create_app()
    if app is None:
        print("未安装 flask，执行: pip install flask", flush=True)
        return 1
    if not MODEL_PATH:
        print("提示: 未指定 --model，页面仍可打开，但检测接口会返回错误。", flush=True)
    print(f"[app] 打开 http://{args.host}:{args.port}/  (Ctrl+C 退出)", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
