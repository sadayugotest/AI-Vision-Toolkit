# -*- coding: utf-8 -*-
"""
app_detection.py  –  Model Detection Web App
ทดสอบ model (YOLO detect/segment/classify หรือ Anomalib) กับรูปภาพที่อัปโหลด
รัน: uvicorn app_detection:app --host 0.0.0.0 --port 5631 --reload
"""
import os
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import io
import uuid
import base64
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Model Detection Web")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TEMP_DIR = Path(tempfile.gettempdir()) / "det_app"
TEMP_DIR.mkdir(exist_ok=True)

# ─── Helper: encode ndarray → base64 PNG ──────────────────────────────────────
def _img_to_b64(arr_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("imencode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

def _bgr_to_b64(arr_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr_bgr)
    if not ok:
        raise RuntimeError("imencode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

# ─── Inference functions ───────────────────────────────────────────────────────

def run_yolo(task: str, model_path: str, image_path: str) -> dict:
    """YOLO detect / segment / classify"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model(image_path)
    r = results[0]

    orig_bgr = cv2.imread(image_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    h, w = orig_bgr.shape[:2]

    if task == "classify":
        # ── Classification ──────────────────────────────────────────────────
        top1_idx  = int(r.probs.top1)
        top1_conf = float(r.probs.top1conf)
        top5_idx  = r.probs.top5
        top5_conf = r.probs.top5conf.tolist()
        names     = r.names

        label = names.get(top1_idx, str(top1_idx))
        top5  = [(names.get(i, str(i)), float(c)) for i, c in zip(top5_idx, top5_conf)]

        # Bar chart for top-5
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(orig_rgb); axes[0].axis('off')
        axes[0].set_title("Original Image", fontweight='bold')

        t5_names = [x[0] for x in top5]
        t5_vals  = [x[1] for x in top5]
        colors   = ['#22c55e' if i == 0 else '#3b82f6' for i in range(len(t5_names))]
        bars = axes[1].barh(t5_names[::-1], t5_vals[::-1], color=colors[::-1])
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Confidence")
        axes[1].set_title(f"Top-5 Predictions\nResult: {label} ({top1_conf:.1%})", fontweight='bold')
        for bar, val in zip(bars, t5_vals[::-1]):
            axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                         f"{val:.1%}", va='center', fontsize=10)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close('all')
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return {
            "task": "classify",
            "label": label,
            "confidence": round(top1_conf, 4),
            "top5": top5,
            "result_image": b64,
            "summary": f"Class: <b>{label}</b>  Confidence: <b>{top1_conf:.1%}</b>",
        }

    else:
        # ── Detect / Segment ────────────────────────────────────────────────
        plotted = r.plot()            # BGR with boxes/masks drawn by ultralytics
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        boxes  = r.boxes
        n_det  = len(boxes) if boxes is not None else 0
        names  = r.names
        labels = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                labels.append(f"{names.get(cls_id, cls_id)} {conf:.1%}")

        # Side-by-side: original | result
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(orig_rgb); axes[0].axis('off')
        axes[0].set_title("Original Image", fontweight='bold')
        axes[1].imshow(plotted_rgb); axes[1].axis('off')
        t = "Segmentation" if task == "segment" else "Detection"
        axes[1].set_title(f"{t} Result  ({n_det} object{'s' if n_det!=1 else ''})", fontweight='bold')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close('all')
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        summary_txt = f"Detected <b>{n_det}</b> object(s)"
        if labels:
            summary_txt += ": " + ", ".join(labels[:10])
            if len(labels) > 10:
                summary_txt += f" ... (+{len(labels)-10} more)"

        return {
            "task": task,
            "detections": n_det,
            "labels": labels,
            "result_image": b64,
            "summary": summary_txt,
        }


def run_anomalib(model_type: str, ckpt_path: str, image_path: str) -> dict:
    """Anomalib predict + Overlay visualization (3 panels)"""
    from anomalib.engine import Engine
    from anomalib.data import PredictDataset
    from torch.utils.data import DataLoader

    _mdl = model_type.lower()
    if _mdl == "patchcore":
        from anomalib.models import Patchcore as AnomalibModel
    elif _mdl == "stfpm":
        from anomalib.models import Stfpm as AnomalibModel
    elif _mdl == "fastflow":
        from anomalib.models import Fastflow as AnomalibModel
    else:
        from anomalib.models import Padim as AnomalibModel

    model = AnomalibModel(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=False,
    )
    engine = Engine(task="classification")

    dataset    = PredictDataset(path=image_path)
    dataloader = DataLoader(dataset, batch_size=1)

    predictions = engine.predict(
        model=model,
        ckpt_path=ckpt_path,
        dataloaders=dataloader,
    )

    res         = predictions[0]
    pred_score  = float(res["pred_scores"][0].item())
    pred_label  = bool(res["pred_labels"][0].item())
    anomaly_map = res["anomaly_maps"][0].cpu().numpy()

    while anomaly_map.ndim > 2:
        anomaly_map = anomaly_map.squeeze(0)

    orig_bgr = cv2.imread(image_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    hx, wx   = orig_bgr.shape[:2]
    amap     = cv2.resize(anomaly_map, (wx, hx))

    status       = "ABNORMAL" if pred_label else "NORMAL"
    status_color = "red"      if pred_label else "green"

    # ── 3-panel figure: Original | Heatmap | Overlay ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original Image", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(amap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f"Predicted Heat Map\nScore: {pred_score:.4f}",
                      fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(orig_rgb)
    heatmap_colored = plt.cm.jet(amap)[:, :, :3]
    axes[2].imshow(heatmap_colored, alpha=0.5)
    axes[2].set_title(f"Overlay\nStatus: {status}",
                      fontsize=13, fontweight='bold', color=status_color)
    axes[2].axis('off')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close('all')
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "task": "anomalib",
        "model_type": _mdl,
        "score": round(pred_score, 4),
        "label": pred_label,
        "status": status,
        "result_image": b64,
        "summary": (
            f"Anomaly Score: <b>{pred_score:.4f}</b> &nbsp;|&nbsp; "
            f"Status: <b style='color:{status_color}'>{status}</b>"
        ),
    }


# ─── API: /api/detect ──────────────────────────────────────────────────────────
@app.post("/api/detect")
async def detect(
    task:         str        = Form(...),           # detect | segment | classify | anomalib
    anomalib_model: str      = Form("padim"),        # padim | patchcore | stfpm | fastflow
    model_file:   UploadFile = File(...),
    image_file:   UploadFile = File(...),
):
    uid = uuid.uuid4().hex

    # บันทึก model file
    model_ext  = Path(model_file.filename or "model.bin").suffix or ".pt"
    model_path = TEMP_DIR / f"model_{uid}{model_ext}"
    model_path.write_bytes(await model_file.read())

    # บันทึก image file
    img_ext   = Path(image_file.filename or "img.jpg").suffix or ".jpg"
    img_path  = TEMP_DIR / f"img_{uid}{img_ext}"
    img_path.write_bytes(await image_file.read())

    try:
        if task == "anomalib":
            result = run_anomalib(anomalib_model, str(model_path), str(img_path))
        elif task in ("detect", "segment", "classify"):
            result = run_yolo(task, str(model_path), str(img_path))
        else:
            raise HTTPException(400, f"task ไม่รู้จัก: {task}")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(500, detail=f"{e}\n\n{tb}")
    finally:
        try: model_path.unlink()
        except: pass
        try: img_path.unlink()
        except: pass

    return JSONResponse(result)


# ─── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""
<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8"/>
<title>Model Detection</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
:root{
  --bg:#0b1220;--bg2:#0e1526;--card:#111827;--card2:#0f172a;
  --text:#e5e7eb;--muted:#94a3b8;
  --accent:#22c55e;--accent2:#3b82f6;--warn:#f59e0b;--danger:#f43f5e;
  --border:#1f2937;--gap:14px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;}
.topbar{background:var(--card2);border-bottom:1px solid var(--border);
  padding:14px 28px;display:flex;align-items:center;gap:14px;}
.topbar h1{font-size:1.25rem;font-weight:700;color:var(--accent);}
.topbar .sub{font-size:.82rem;color:var(--muted);}
.wrap{max-width:960px;margin:0 auto;padding:28px var(--gap);}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:24px;margin-bottom:var(--gap);}
.card h2{font-size:1rem;font-weight:600;margin-bottom:16px;color:var(--accent2);}
label{display:block;font-size:.83rem;color:var(--muted);margin-bottom:5px;margin-top:10px;}
label:first-child{margin-top:0}
input[type=file],select{
  width:100%;padding:9px 12px;
  background:#1e293b;border:1px solid var(--border);border-radius:8px;
  color:var(--text);font-size:.9rem;cursor:pointer;
}
input[type=file]::-webkit-file-upload-button{
  background:var(--accent2);border:none;color:#fff;
  padding:5px 14px;border-radius:6px;cursor:pointer;margin-right:10px;
}
select{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%2394a3b8'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;}

.tab-bar{display:flex;gap:6px;margin-bottom:20px;flex-wrap:wrap;}
.tab-btn{padding:8px 20px;border-radius:30px;border:1.5px solid var(--border);
  background:transparent;color:var(--muted);font-size:.88rem;cursor:pointer;transition:.2s;}
.tab-btn:hover{border-color:var(--accent2);color:var(--text);}
.tab-btn.active{background:var(--accent2);border-color:var(--accent2);color:#fff;font-weight:600;}

.btn{display:inline-block;padding:10px 28px;border-radius:8px;border:none;
  background:var(--accent);color:#000;font-size:.95rem;font-weight:700;cursor:pointer;
  transition:.2s;text-decoration:none;}
.btn:hover{filter:brightness(1.1);}
.btn:disabled{opacity:.4;cursor:not-allowed;}
.btn.secondary{background:var(--accent2);color:#fff;}

.preview-box{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;}
.preview-box img{max-height:180px;max-width:280px;border-radius:8px;
  border:1px solid var(--border);object-fit:contain;background:#0d1526;}

.result-card{display:none;background:var(--card2);border:1px solid var(--border);
  border-radius:14px;padding:24px;margin-top:var(--gap);}
.result-card.show{display:block;}
.result-img{width:100%;border-radius:10px;margin-top:14px;border:1px solid var(--border);}
.summary-box{margin-top:14px;padding:14px 18px;background:#0d1a2e;border-radius:10px;
  border-left:4px solid var(--accent2);font-size:1rem;line-height:1.7;}
.summary-box .score-row{display:flex;gap:24px;flex-wrap:wrap;margin-top:8px;}
.score-badge{padding:4px 14px;border-radius:20px;font-size:.85rem;font-weight:600;}
.normal{background:#052e16;color:#4ade80;}
.abnormal{background:#3b0a0a;color:#f87171;}
.tag{background:#172554;color:#93c5fd;padding:3px 10px;border-radius:12px;font-size:.8rem;}

.spinner{display:none;text-align:center;padding:32px;color:var(--muted);}
.spinner.show{display:block;}
.spinner-ring{width:44px;height:44px;border:4px solid #1e293b;border-top-color:var(--accent2);
  border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 12px;}
@keyframes spin{to{transform:rotate(360deg)}}

.err{background:#1c0a0a;border:1px solid #7f1d1d;border-radius:10px;
  padding:14px;color:#fca5a5;font-size:.85rem;white-space:pre-wrap;margin-top:12px;display:none;}
.err.show{display:block;}

.anom-field{display:none;}
.file-hint{font-size:.75rem;color:var(--muted);margin-top:3px;}
</style>
</head>
<body>

<div class="topbar">
  <span style="font-size:1.5rem">🔬</span>
  <div>
    <h1>Model Detection Web</h1>
    <div class="sub">ทดสอบโมเดล YOLO / Anomalib กับรูปภาพ</div>
  </div>
</div>

<div class="wrap">

  <!-- ── Task Selector ── -->
  <div class="card">
    <h2>1. เลือกประเภทโมเดล</h2>
    <div class="tab-bar">
      <button class="tab-btn active" onclick="setTask('detect')"   id="tb-detect">🔍 Detection</button>
      <button class="tab-btn"        onclick="setTask('segment')"  id="tb-segment">✏️ Segmentation</button>
      <button class="tab-btn"        onclick="setTask('classify')" id="tb-classify">✨ Classification</button>
      <button class="tab-btn"        onclick="setTask('anomalib')" id="tb-anomalib">📊 Anomalib</button>
    </div>

    <!-- Anomalib sub-model selector -->
    <div class="anom-field" id="anomModelField">
      <label>Anomalib Model Architecture</label>
      <select id="anomalib_model">
        <option value="padim">PaDiM</option>
        <option value="patchcore">PatchCore</option>
        <option value="stfpm">STFPM</option>
        <option value="fastflow">FastFlow</option>
      </select>
    </div>
  </div>

  <!-- ── Upload ── -->
  <div class="card">
    <h2>2. อัปโหลดไฟล์</h2>

    <label id="modelLabel">ไฟล์ Model (.pt)</label>
    <input type="file" id="modelFile" accept=".pt,.ckpt,.pth"/>
    <div class="file-hint" id="modelHint">YOLO: .pt &nbsp;|&nbsp; Anomalib: .ckpt</div>

    <label style="margin-top:16px">รูปภาพที่ต้องการทดสอบ (.jpg / .png)</label>
    <input type="file" id="imageFile" accept="image/*" onchange="previewImage(this)"/>

    <div class="preview-box" id="previewBox"></div>
  </div>

  <!-- ── Run ── -->
  <div class="card" style="text-align:center">
    <button class="btn" id="runBtn" onclick="runDetect()">▶ เริ่มตรวจสอบ</button>
  </div>

  <!-- ── Spinner ── -->
  <div class="spinner" id="spinner">
    <div class="spinner-ring"></div>
    กำลังประมวลผล...
  </div>

  <!-- ── Error ── -->
  <div class="err" id="errBox"></div>

  <!-- ── Result ── -->
  <div class="result-card" id="resultCard">
    <h2 id="resultTitle">ผลลัพธ์</h2>
    <div class="summary-box" id="summaryBox"></div>
    <img class="result-img" id="resultImg" src="" alt="result"/>
  </div>

</div><!-- /wrap -->

<script>
let currentTask = 'detect';

function setTask(t){
  currentTask = t;
  ['detect','segment','classify','anomalib'].forEach(x=>{
    document.getElementById('tb-'+x).classList.toggle('active', x===t);
  });
  const isAnom = t==='anomalib';
  document.getElementById('anomModelField').style.display = isAnom ? 'block' : 'none';
  document.getElementById('modelLabel').textContent =
    isAnom ? 'ไฟล์ Model (.ckpt)' : 'ไฟล์ Model (.pt)';
  document.getElementById('modelHint').textContent =
    isAnom ? 'Anomalib checkpoint: .ckpt' : 'YOLO weight: .pt';
  // reset result
  document.getElementById('resultCard').classList.remove('show');
  document.getElementById('errBox').classList.remove('show');
}

function previewImage(input){
  const box = document.getElementById('previewBox');
  box.innerHTML = '';
  if(input.files && input.files[0]){
    const img = document.createElement('img');
    img.src = URL.createObjectURL(input.files[0]);
    box.appendChild(img);
  }
}

async function runDetect(){
  const modelFile = document.getElementById('modelFile').files[0];
  const imageFile = document.getElementById('imageFile').files[0];
  if(!modelFile){ alert('กรุณาเลือกไฟล์ Model'); return; }
  if(!imageFile){ alert('กรุณาเลือกรูปภาพ');    return; }

  const errBox    = document.getElementById('errBox');
  const spinner   = document.getElementById('spinner');
  const resultCard= document.getElementById('resultCard');
  const runBtn    = document.getElementById('runBtn');

  errBox.classList.remove('show');
  resultCard.classList.remove('show');
  spinner.classList.add('show');
  runBtn.disabled = true;

  const fd = new FormData();
  fd.append('task',           currentTask);
  fd.append('anomalib_model', document.getElementById('anomalib_model').value);
  fd.append('model_file',     modelFile);
  fd.append('image_file',     imageFile);

  try{
    const res  = await fetch('/api/detect', {method:'POST', body:fd});
    const data = await res.json();

    if(!res.ok){
      errBox.textContent = data.detail || JSON.stringify(data);
      errBox.classList.add('show');
      return;
    }

    renderResult(data);
  } catch(e){
    errBox.textContent = 'เกิดข้อผิดพลาด: ' + e.message;
    errBox.classList.add('show');
  } finally{
    spinner.classList.remove('show');
    runBtn.disabled = false;
  }
}

function renderResult(d){
  const card    = document.getElementById('resultCard');
  const sumBox  = document.getElementById('summaryBox');
  const img     = document.getElementById('resultImg');
  const title   = document.getElementById('resultTitle');

  const taskName = {
    detect:'Detection', segment:'Segmentation',
    classify:'Classification', anomalib:'Anomalib'
  }[d.task] || d.task;
  title.textContent = '\u2705 \u0e1c\u0e25\u0e25\u0e31\u0e1e\u0e18\u0e4c  \u2014  ' + taskName;

  let html = '<div>' + (d.summary || '') + '</div>';

  if(d.task === 'anomalib'){
    const cls = d.label ? 'abnormal' : 'normal';
    html += `<div class="score-row">
      <span class="score-badge ${cls}">${d.status}</span>
      <span class="score-badge" style="background:#0f172a;color:#cbd5e1;">
        Score: ${d.score}
      </span>
    </div>`;
  } else if(d.task === 'classify'){
    html += '<div class="score-row">';
    (d.top5 || []).forEach((t,i)=>{
      const pct = (t[1]*100).toFixed(1);
      const col = i===0 ? '#4ade80' : '#93c5fd';
      html += `<span class="score-badge" style="background:#0f172a;color:${col};">${t[0]} ${pct}%</span>`;
    });
    html += '</div>';
  } else {
    if(d.labels && d.labels.length){
      html += '<div class="score-row">';
      d.labels.slice(0,8).forEach(l=>{
        html += `<span class="tag">${l}</span>`;
      });
      if(d.labels.length>8) html += `<span class="tag">+${d.labels.length-8} more</span>`;
      html += '</div>';
    }
  }

  sumBox.innerHTML = html;
  img.src   = d.result_image;
  card.classList.add('show');
  card.scrollIntoView({behavior:'smooth', block:'start'});
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_detection:app", host="0.0.0.0", port=5631, reload=True)
