// label.js — Label Tool P2+P3 (Detect BBox + Segment Polygon)
// =============================================================================

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId  = null;
let images     = [];       // [{name, thumb}]
let currentIdx = -1;
let currentTask = 'detect';

// ── Segment state ─────────────────────────────────────────────────────────────
let polygons    = [];    // [{id, class_id, points:[{x,y}] normalised 0-1}]
let polyIdCnt   = 0;
let curPoly     = null;  // polygon กำลังวาด: {class_id, points:[]}
let selPolyId   = null;
let segUnsaved  = false;

// ── Classify state ─────────────────────────────────────────────────────────────────
let clsLabels      = {};   // { imageIdx: className }  — in-memory cache
let clsUnsaved     = false;
let anomLabels     = {};   // { imageIdx: {label, is_defect, has_mask} } — in-memory cache
let anomCurrentType = 'good'; // 'good' | 'defect'
let segTool    = 'polygon';  // 'polygon' | 'brush'
let brushSize  = 20;          // px in canvas-display coords
let isBrushing = false;
let brushMasks = [];          // dataURL (หรือ null) ต่อรูป index — เก็บ brush ค้างไว้เมื่อเปลี่ยนรูป
const BRUSH_CANVAS = document.getElementById('brushCanvas');
const BRUSH_CTX    = BRUSH_CANVAS.getContext('2d');
const BRUSH_CURSOR = document.getElementById('brushCursor');

// ── Detect state ──────────────────────────────────────────────────────────────
let imgW = 0, imgH = 0;    // ขนาดจริงของรูป
let scale = 1;             // canvas display scale
let offsetX = 0, offsetY = 0; // canvas offset inside wrap
let boxes = [];            // [{id, class_id, cx, cy, w, h}]  normalised YOLO
let selectedBoxId = null;
let activeTool = 'draw';   // 'draw' | 'select'
let unsaved = false;

// drawing drag state
let isDragging = false;
let dragStart  = {x:0, y:0};   // in image px
let dragCur    = {x:0, y:0};

// moving/resizing state
let isMoving   = false;
let moveStart  = {x:0, y:0};
let moveOrigBox = null;

// resize handle
let resizeHandle = null;   // 'tl'|'tr'|'bl'|'br'|null
let resizeOrigBox = null;

// classes
let classes = [];  // [{id, name, color}]
let activeClassId = 0;
let nextClassId = 0;
let boxIdCounter = 0;

// canvas
const CANVAS  = document.getElementById('mainCanvas');
const CTX     = CANVAS.getContext('2d');
const IMG_OBJ = new Image();

// ── Spinner ───────────────────────────────────────────────────────────────────
function showSpinner(msg){
  document.getElementById('spinTxt').textContent = msg||'กำลังประมวลผล...';
  document.getElementById('spinnerOverlay').classList.add('show');
}
function hideSpinner(){ document.getElementById('spinnerOverlay').classList.remove('show'); }

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(msg, ok=false){
  const el = document.getElementById('statusTxt');
  el.textContent = msg;
  el.className   = ok ? 'status-ok' : '';
}

// ── Task Tab ──────────────────────────────────────────────────────────────────
function switchTask(task){
  currentTask = task;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.task === task));
  document.getElementById('detectToolbar').style.display  = task==='detect'  ? 'flex' : 'none';
  document.getElementById('segToolbar').style.display     = task==='segment' ? 'flex' : 'none';
  document.getElementById('clsToolbar').style.display     = task==='classify'? 'flex' : 'none';
  document.getElementById('anomToolbar').style.display     = task==='anomalib' ? 'flex' : 'none';
  // toggle right-panel lists
  const boxHd      = document.querySelector('.box-panel-hd:not(#polyPanelHd):not(#clsSummaryHd):not(#anomSummaryHd)');
  const polyHd     = document.getElementById('polyPanelHd');
  const boxLst     = document.getElementById('boxList');
  const polyLst    = document.getElementById('polyList');
  const clsSumHd   = document.getElementById('clsSummaryHd');
  const clsSumLst  = document.getElementById('clsSummaryList');
  const anomSumHd  = document.getElementById('anomSummaryHd');
  const anomSumLst = document.getElementById('anomSummaryList');
  if(task==='detect'){
    if(boxHd)   boxHd.style.display   = '';
    if(boxLst)  boxLst.style.display  = '';
    if(polyHd)  polyHd.style.display  = 'none';
    if(polyLst) polyLst.style.display = 'none';
    if(clsSumHd)  clsSumHd.style.display  = 'none';
    if(clsSumLst) clsSumLst.style.display = 'none';
    if(anomSumHd)  anomSumHd.style.display  = 'none';
    if(anomSumLst) anomSumLst.style.display = 'none';
    CANVAS.className = '';
    BRUSH_CANVAS.style.display = 'none';
    BRUSH_CURSOR.style.display = 'none';
  } else if(task==='segment'){
    if(boxHd)   boxHd.style.display   = 'none';
    if(boxLst)  boxLst.style.display  = 'none';
    if(polyHd)  polyHd.style.display  = '';
    if(polyLst) polyLst.style.display = '';
    if(clsSumHd)  clsSumHd.style.display  = 'none';
    if(clsSumLst) clsSumLst.style.display = 'none';
    if(anomSumHd)  anomSumHd.style.display  = 'none';
    if(anomSumLst) anomSumLst.style.display = 'none';
    segTool = 'polygon';
    CANVAS.className = 'tool-seg';
    BRUSH_CANVAS.style.display = 'none';
    BRUSH_CURSOR.style.display = 'none';
    document.getElementById('segToolPoly').classList.add('active');
    document.getElementById('segToolBrush').classList.remove('active');
    document.getElementById('brushControls').style.display = 'none';
    document.getElementById('polyControls').style.display  = 'contents';
    document.getElementById('btnConfirmBrush').style.display = 'none';
    document.getElementById('btnClearBrush').style.display   = 'none';
  } else if(task==='classify'){
    if(boxHd)   boxHd.style.display   = 'none';
    if(boxLst)  boxLst.style.display  = 'none';
    if(polyHd)  polyHd.style.display  = 'none';
    if(polyLst) polyLst.style.display = 'none';
    if(clsSumHd)  clsSumHd.style.display  = '';
    if(clsSumLst) clsSumLst.style.display = '';
    if(anomSumHd)  anomSumHd.style.display  = 'none';
    if(anomSumLst) anomSumLst.style.display = 'none';
    CANVAS.className = 'tool-cls';
    BRUSH_CANVAS.style.display = 'none';
    BRUSH_CURSOR.style.display = 'none';
    renderClsQuickGrid();
    renderClsSummary();
    if(currentIdx >= 0) updateClsBadgeUI(currentIdx);
  } else if(task==='anomalib'){
    if(boxHd)   boxHd.style.display   = 'none';
    if(boxLst)  boxLst.style.display  = 'none';
    if(polyHd)  polyHd.style.display  = 'none';
    if(polyLst) polyLst.style.display = 'none';
    if(clsSumHd)  clsSumHd.style.display  = 'none';
    if(clsSumLst) clsSumLst.style.display = 'none';
    if(anomSumHd)  anomSumHd.style.display  = '';
    if(anomSumLst) anomSumLst.style.display = '';
    CANVAS.className = 'tool-anom';
    // ปิด brush ตั้งต้น (จนกว่าผู้ใช้จะติ๊ก Pixel Mask)
    BRUSH_CANVAS.style.display = 'none';
    BRUSH_CURSOR.style.display = 'none';
    BRUSH_CTX.clearRect(0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height);
    const anomMaskChk = document.getElementById('anomUseMask');
    if(anomMaskChk) anomMaskChk.checked = false;
    const anomMaskHint = document.getElementById('anomMaskHint');
    if(anomMaskHint) anomMaskHint.style.display = 'none';
    renderAnomSummary();
    if(currentIdx >= 0) updateAnomBadgeUI(currentIdx);
  }
  if(imgW) drawAll();
  saveSessionToStorage();
  setStatus(task==='detect'  ? 'โหมด Detect BBox' :
            task==='segment' ? 'โหมด Segment Polygon — เลือก Polygon หรือ Brush tool' :
            task==='classify'? 'โหมด Classify — คลิกปุ่ม class เพื่อ assign ให้รูปนี้' :
            task==='anomalib'? 'โหมด Anomalib — เลือก Good / Defect แล้วกด Assign' :
            `โหมด ${task}`);
}

// ── Upload Images ─────────────────────────────────────────────────────────────
async function doUploadImages(){
  const allFiles = document.getElementById('folderInput').files;
  if(!allFiles||allFiles.length===0){ alert('กรุณาเลือก Folder หรือไฟล์รูปภาพก่อน'); return; }
  const IMG_TYPES = ['image/jpeg','image/png','image/bmp','image/webp','image/tiff'];
  const IMG_EXTS  = ['.jpg','.jpeg','.png','.bmp','.webp','.tiff','.tif'];
  const files = Array.from(allFiles).filter(f => {
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
    return IMG_TYPES.includes(f.type)||IMG_EXTS.includes(ext);
  });
  if(files.length===0){ alert('ไม่พบไฟล์รูปภาพใน folder นี้'); return; }
  showSpinner(`กำลังอัปโหลด ${files.length} ไฟล์...`);
  try{
    const fd = new FormData();
    for(const f of files) fd.append('files', f);
    const res  = await fetch('/api/upload-images',{method:'POST',body:fd});
    const data = await res.json();
    if(!res.ok){ alert(data.detail||'อัปโหลดไม่สำเร็จ'); return; }
    loadSession(data);
  } catch(e){ alert('เกิดข้อผิดพลาด: '+e.message);
  } finally { hideSpinner(); }
}

// ── Upload Video ──────────────────────────────────────────────────────────────
async function doUploadVideo(){
  const file = document.getElementById('videoInput').files[0];
  if(!file){ alert('กรุณาเลือกไฟล์วิดีโอก่อน'); return; }
  const fps = parseFloat(document.getElementById('fpsInput').value)||1;
  showSpinner(`กำลัง extract frames (${fps} fps)...`);
  try{
    const fd = new FormData();
    fd.append('file',file); fd.append('fps',fps);
    const res  = await fetch('/api/upload-video',{method:'POST',body:fd});
    const data = await res.json();
    if(!res.ok){ alert(data.detail||'Extract ไม่สำเร็จ'); return; }
    loadSession(data);
  } catch(e){ alert('เกิดข้อผิดพลาด: '+e.message);
  } finally { hideSpinner(); }
}

// ── Load Session ──────────────────────────────────────────────────────────────
function loadSession(data){
  sessionId = data.session_id;
  images    = data.thumbs;

  // render thumbnail sidebar
  const sidebar = document.getElementById('sidebar');
  const notice  = document.getElementById('sidebarEmpty');
  if(notice) notice.style.display = 'none';
  sidebar.innerHTML = '';
  images.forEach((im,i) => {
    const div = document.createElement('div');
    div.className = 'thumb-item';
    div.id = `th-${i}`;
    div.onclick = () => selectImage(i);
    div.innerHTML = `<span class="thumb-idx">${i+1}</span>
      <img src="${im.thumb}" alt="${im.name}" loading="lazy"/>
      <div class="tname">${im.name}</div>`;
    sidebar.appendChild(div);
  });

  const badge = document.getElementById('sessionBadge');
  if(badge) badge.textContent = `Session: ${sessionId.slice(0,8)}\u2026  |  ${images.length} ภาพ`;
  document.getElementById('statusCount').textContent = `รวม ${images.length} ภาพ`;
  setStatus(`โหลดสำเร็จ ${images.length} ภาพ`, true);

  // load saved classes for this session
  fetch(`/api/classes/${sessionId}`)
    .then(r=>r.json()).then(d=>{
      if(d.classes && d.classes.length > 0){
        classes = d.classes;
        nextClassId = Math.max(...classes.map(c=>c.id)) + 1;
        renderClassList();
      }
    });

  // persist session id to localStorage so page refresh can recover
  saveSessionToStorage();

  // load classify labels for all images (if any)
  loadAllClsLabels();

  // load anomalib labels for all images (if any)
  loadAllAnomLabels();

  if(images.length > 0) selectImage(0);
}

// ── Select Image ──────────────────────────────────────────────────────────────
async function selectImage(idx){
  if(!sessionId||images.length===0) return;

  // auto-save before switching
  if(currentIdx >= 0){
    if(unsaved)    await saveLabels(true);
    if(segUnsaved) await saveSegments(true);
    // save brush mask ของรูปเก่าไว้
    saveBrushMask(currentIdx);
  }

  idx = Math.max(0, Math.min(images.length-1, idx));
  currentIdx = idx;

  // highlight sidebar
  document.querySelectorAll('.thumb-item').forEach(el=>el.classList.remove('active'));
  const th = document.getElementById(`th-${idx}`);
  if(th){ th.classList.add('active'); th.scrollIntoView({block:'nearest',behavior:'smooth'}); }

  // update nav UI
  document.getElementById('viewerEmpty').style.display = 'none';
  document.getElementById('imgName').style.display = 'block';
  document.getElementById('imgName').textContent = images[idx].name;
  document.getElementById('canvasNav').style.display = 'flex';
  document.getElementById('idxBadge').textContent = `${idx+1} / ${images.length}`;
  document.getElementById('btnSave').disabled    = false;
  document.getElementById('btnSegSave').disabled = false;
  setStatus(`รูป: ${images[idx].name}  (${idx+1}/${images.length})`);

  // get image size
  const sz = await fetch(`/api/image-size/${sessionId}/${idx}`).then(r=>r.json());
  imgW = sz.width; imgH = sz.height;

  // load detect labels
  const ld = await fetch(`/api/load-labels/${sessionId}/${idx}`).then(r=>r.json());
  boxes = ld.boxes.map(b=>({...b, id: boxIdCounter++}));
  // update classes if server has more
  if(ld.classes && ld.classes.length > classes.length){
    classes = ld.classes;
    nextClassId = Math.max(...classes.map(c=>c.id)) + 1;
    renderClassList();
  }

  // load segment labels
  const sl = await fetch(`/api/load-segments/${sessionId}/${idx}`).then(r=>r.json());
  polygons = sl.polygons.map(p=>({...p, id: polyIdCnt++}));
  curPoly  = null; selPolyId = null;

  // load image into canvas
  IMG_OBJ.onload = () => { resizeCanvas(); restoreBrushMask(idx); drawAll(); };
  IMG_OBJ.src = `/api/image/${sessionId}/${idx}?t=${Date.now()}`;

  markUnsaved(false);
  markSegUnsaved(false);
  renderBoxList();
  renderPolyList();
  // update classify UI for this image
  if(currentTask === 'classify'){
    updateClsBadgeUI(idx);
    renderClsQuickGrid();
  }
  // update anomalib UI for this image
  if(currentTask === 'anomalib'){
    updateAnomBadgeUI(idx);
    // ถ้าเปิด Pixel Mask ไว้ ให้ restore brush mask ของรูปใหม่
    if(isAnomBrushActive()){
      restoreBrushMask(idx);
    } else {
      BRUSH_CTX.clearRect(0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height);
    }
  }
}

function navigate(delta){ selectImage(currentIdx+delta); }

// ── Canvas resize ─────────────────────────────────────────────────────────────
function resizeCanvas(){
  const wrap = document.getElementById('canvasWrap');
  const ww = wrap.clientWidth  - 20;
  const wh = wrap.clientHeight - 20;
  scale   = Math.min(ww/imgW, wh/imgH, 1);
  const cw = Math.round(imgW * scale);
  const ch = Math.round(imgH * scale);
  CANVAS.width  = cw;
  CANVAS.height = ch;
  offsetX = (wrap.clientWidth  - cw) / 2;
  offsetY = (wrap.clientHeight - ch) / 2;
  CANVAS.style.position = 'absolute';
  CANVAS.style.left = offsetX + 'px';
  CANVAS.style.top  = offsetY + 'px';
  // sync brush canvas size & position (assignment clears content)
  BRUSH_CANVAS.width  = cw;
  BRUSH_CANVAS.height = ch;
  BRUSH_CANVAS.style.left = offsetX + 'px';
  BRUSH_CANVAS.style.top  = offsetY + 'px';
  // restore brush mask if any (resize clears canvas)
  restoreBrushMask(currentIdx);
}

window.addEventListener('resize', ()=>{ if(imgW) { resizeCanvas(); drawAll(); } });

// ── Brush mask persistence helpers ───────────────────────────────────────────
function saveBrushMask(idx){
  if(idx < 0) return;
  // เช็คว่ามีอะไรระบายไว้ไหม (alpha channel)
  const data = BRUSH_CTX.getImageData(0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height).data;
  const hasContent = data.some((v, i) => i % 4 === 3 && v > 0);
  brushMasks[idx] = hasContent ? BRUSH_CANVAS.toDataURL() : null;

  // anomalib mode: ส่ง mask ไปเก็บที่ server เพื่อใช้เป็น ground_truth
  if(currentTask === 'anomalib' && isAnomBrushActive() && sessionId && hasContent){
    fetch(`/api/save-brush-mask/${sessionId}/${idx}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ data_url: brushMasks[idx] })
    }).catch(()=>{});
  }
}

function restoreBrushMask(idx){
  BRUSH_CTX.clearRect(0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height);
  const url = brushMasks[idx];
  if(!url) return;
  const img = new Image();
  img.onload = () => {
    BRUSH_CTX.drawImage(img, 0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height);
  };
  img.src = url;
}

// ── Draw ──────────────────────────────────────────────────────────────────────
function drawAll(){
  CTX.clearRect(0,0,CANVAS.width,CANVAS.height);
  CTX.drawImage(IMG_OBJ, 0, 0, CANVAS.width, CANVAS.height);

  if(currentTask === 'detect'){
    boxes.forEach(b => drawBox(b, b.id === selectedBoxId));
    if(isDragging){
      const x = Math.min(dragStart.x, dragCur.x) * scale;
      const y = Math.min(dragStart.y, dragCur.y) * scale;
      const w = Math.abs(dragCur.x - dragStart.x) * scale;
      const h = Math.abs(dragCur.y - dragStart.y) * scale;
      CTX.strokeStyle = getClassColor(activeClassId);
      CTX.lineWidth = 2;
      CTX.setLineDash([5,3]);
      CTX.strokeRect(x,y,w,h);
      CTX.setLineDash([]);
      CTX.fillStyle = hexAlpha(getClassColor(activeClassId), 0.15);
      CTX.fillRect(x,y,w,h);
    }
  } else if(currentTask === 'segment'){
    polygons.forEach(p => drawPolygon(p, p.id === selPolyId));
    if(curPoly && curPoly.points.length > 0) drawInProgressPoly();
  }
}

const HANDLE_SIZE = 7;

function drawBox(b, selected){
  const cls = classes.find(c=>c.id===b.class_id);
  const color = cls ? cls.color : '#ffffff';
  const x = (b.cx - b.w/2) * imgW * scale;
  const y = (b.cy - b.h/2) * imgH * scale;
  const w = b.w * imgW * scale;
  const h = b.h * imgH * scale;

  CTX.strokeStyle = color;
  CTX.lineWidth   = selected ? 2.5 : 1.8;
  CTX.setLineDash(selected ? [6,2] : []);
  CTX.strokeRect(x,y,w,h);
  CTX.setLineDash([]);
  CTX.fillStyle = hexAlpha(color, selected ? 0.18 : 0.08);
  CTX.fillRect(x,y,w,h);

  // label tag
  const label = cls ? `${b.class_id}: ${cls.name}` : `cls ${b.class_id}`;
  CTX.font = 'bold 11px Segoe UI,sans-serif';
  const tw = CTX.measureText(label).width + 8;
  const th = 16;
  const tx = x;
  const ty = y > th ? y - th : y;
  CTX.fillStyle = color;
  CTX.fillRect(tx, ty, tw, th);
  CTX.fillStyle = '#000';
  CTX.fillText(label, tx+4, ty+12);

  // resize handles (only when selected)
  if(selected){
    const hs = HANDLE_SIZE;
    [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([hx,hy])=>{
      CTX.fillStyle = '#fff';
      CTX.strokeStyle = color;
      CTX.lineWidth = 1.5;
      CTX.fillRect(hx-hs/2, hy-hs/2, hs, hs);
      CTX.strokeRect(hx-hs/2, hy-hs/2, hs, hs);
    });
  }
}

// ── Canvas mouse helpers ──────────────────────────────────────────────────────
function canvasToImage(e){
  const rect = CANVAS.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left)  / scale,
    y: (e.clientY - rect.top)   / scale,
  };
}

function getHandleAt(b, px, py){
  const hs = HANDLE_SIZE / scale;
  const x1 = (b.cx - b.w/2) * imgW;
  const y1 = (b.cy - b.h/2) * imgH;
  const x2 = x1 + b.w * imgW;
  const y2 = y1 + b.h * imgH;
  const handles = {tl:[x1,y1], tr:[x2,y1], bl:[x1,y2], br:[x2,y2]};
  for(const [name,[hx,hy]] of Object.entries(handles)){
    if(Math.abs(px-hx)<hs && Math.abs(py-hy)<hs) return name;
  }
  return null;
}

function getBoxAt(px, py){
  for(let i=boxes.length-1; i>=0; i--){
    const b=boxes[i];
    const x1=(b.cx-b.w/2)*imgW, y1=(b.cy-b.h/2)*imgH;
    const x2=x1+b.w*imgW,       y2=y1+b.h*imgH;
    if(px>=x1&&px<=x2&&py>=y1&&py<=y2) return b;
  }
  return null;
}

// ── Tool buttons ──────────────────────────────────────────────────────────────
function setTool(t){
  activeTool = t;
  document.getElementById('toolDraw').classList.toggle('active',   t==='draw');
  document.getElementById('toolSelect').classList.toggle('active', t==='select');
  CANVAS.className = t==='select' ? 'tool-select' : '';
}

// ── Mouse events ──────────────────────────────────────────────────────────────
CANVAS.addEventListener('mousedown', e=>{
  if(!sessionId||currentIdx<0) return;
  if(currentTask !== 'detect') return;  // segment handled separately
  const {x,y} = canvasToImage(e);

  if(activeTool==='draw'){
    isDragging = true;
    dragStart  = {x,y};
    dragCur    = {x,y};

  } else if(activeTool==='select'){
    // check resize handle first
    if(selectedBoxId !== null){
      const sb = boxes.find(b=>b.id===selectedBoxId);
      if(sb){
        const h = getHandleAt(sb,x,y);
        if(h){
          resizeHandle  = h;
          resizeOrigBox = {...sb};
          isMoving      = false;
          return;
        }
      }
    }
    // check box hit
    const hit = getBoxAt(x,y);
    if(hit){
      selectedBoxId = hit.id;
      isMoving      = true;
      moveStart     = {x,y};
      moveOrigBox   = {...hit};
    } else {
      selectedBoxId = null;
    }
    document.getElementById('btnDelBox').disabled = selectedBoxId===null;
    drawAll();
  }
});

CANVAS.addEventListener('mousemove', e=>{
  if(!sessionId||currentIdx<0) return;
  if(currentTask !== 'detect') return;  // segment handled separately
  const {x,y} = canvasToImage(e);

  if(activeTool==='draw' && isDragging){
    dragCur = {x,y};
    drawAll();

  } else if(activeTool==='select'){
    if(resizeHandle && resizeOrigBox){
      const ob = resizeOrigBox;
      let x1=(ob.cx-ob.w/2)*imgW, y1=(ob.cy-ob.h/2)*imgH;
      let x2=x1+ob.w*imgW,        y2=y1+ob.h*imgH;
      if(resizeHandle==='tl'){x1=x;y1=y;}
      if(resizeHandle==='tr'){x2=x;y1=y;}
      if(resizeHandle==='bl'){x1=x;y2=y;}
      if(resizeHandle==='br'){x2=x;y2=y;}
      if(x2>x1 && y2>y1){
        const sb = boxes.find(b=>b.id===selectedBoxId);
        if(sb){
          sb.cx = ((x1+x2)/2)/imgW;
          sb.cy = ((y1+y2)/2)/imgH;
          sb.w  = (x2-x1)/imgW;
          sb.h  = (y2-y1)/imgH;
          markUnsaved(true); drawAll();
        }
      }
    } else if(isMoving && moveOrigBox){
      const dx = (x - moveStart.x) / imgW;
      const dy = (y - moveStart.y) / imgH;
      const sb = boxes.find(b=>b.id===selectedBoxId);
      if(sb){
        sb.cx = Math.max(sb.w/2, Math.min(1-sb.w/2, moveOrigBox.cx+dx));
        sb.cy = Math.max(sb.h/2, Math.min(1-sb.h/2, moveOrigBox.cy+dy));
        markUnsaved(true); drawAll();
      }
    } else {
      // cursor hint
      if(selectedBoxId!==null){
        const sb = boxes.find(b=>b.id===selectedBoxId);
        if(sb && getHandleAt(sb,x,y)) CANVAS.style.cursor='nwse-resize';
        else if(getBoxAt(x,y)) CANVAS.style.cursor='grab';
        else CANVAS.style.cursor='default';
      } else {
        CANVAS.style.cursor = getBoxAt(x,y) ? 'grab' : 'default';
      }
    }
  }
});

CANVAS.addEventListener('mouseup', e=>{
  if(!sessionId||currentIdx<0) return;
  if(currentTask !== 'detect') return;
  const {x,y} = canvasToImage(e);

  if(activeTool==='draw' && isDragging){
    isDragging = false;
    const x1=Math.min(dragStart.x,x), y1=Math.min(dragStart.y,y);
    const x2=Math.max(dragStart.x,x), y2=Math.max(dragStart.y,y);
    const bw=(x2-x1)/imgW, bh=(y2-y1)/imgH;
    if(bw > 0.005 && bh > 0.005){
      boxes.push({
        id: boxIdCounter++,
        class_id: activeClassId,
        cx: ((x1+x2)/2)/imgW,
        cy: ((y1+y2)/2)/imgH,
        w: bw, h: bh
      });
      markUnsaved(true);
      renderBoxList();
    }
    drawAll();

  } else if(activeTool==='select'){
    isMoving=false; resizeHandle=null; resizeOrigBox=null; moveOrigBox=null;
    if(unsaved) renderBoxList();
  }
});

CANVAS.addEventListener('mouseleave', ()=>{
  if(isDragging){ isDragging=false; drawAll(); }
  isMoving=false; resizeHandle=null;
});

// ── Delete / Clear boxes ──────────────────────────────────────────────────────
function deleteSelectedBox(){
  if(selectedBoxId===null) return;
  boxes = boxes.filter(b=>b.id!==selectedBoxId);
  selectedBoxId=null;
  document.getElementById('btnDelBox').disabled=true;
  markUnsaved(true); drawAll(); renderBoxList();
}

function clearAllBoxes(){
  if(!boxes.length) return;
  if(!confirm('ลบ box ทั้งหมดในรูปนี้?')) return;
  boxes=[]; selectedBoxId=null;
  document.getElementById('btnDelBox').disabled=true;
  markUnsaved(true); drawAll(); renderBoxList();
}

// ── Save Labels ───────────────────────────────────────────────────────────────
async function saveLabels(silent=false){
  if(!sessionId||currentIdx<0) return;
  try{
    const payload = {
      boxes:   boxes.map(({class_id,cx,cy,w,h})=>({class_id,cx,cy,w,h})),
      classes: classes
    };
    const res = await fetch(`/api/save-labels/${sessionId}/${currentIdx}`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error((await res.json()).detail||'บันทึกไม่สำเร็จ');
    markUnsaved(false);
    // mark thumbnail
    const th = document.getElementById(`th-${currentIdx}`);
    if(th) th.classList.toggle('has-label', boxes.length>0);
    if(!silent) setStatus(`บันทึก Label สำเร็จ (${boxes.length} boxes)`, true);
  } catch(e){
    if(!silent) alert('บันทึกไม่สำเร็จ: '+e.message);
  }
}

function markUnsaved(v){
  unsaved = v;
  const badge = document.getElementById('saveBadge');
  badge.textContent = v ? 'ยังไม่ได้บันทึก' : 'บันทึกแล้ว';
  badge.className   = 'save-badge ' + (v ? 'unsaved' : 'saved');
}

// ── Class Manager ─────────────────────────────────────────────────────────────
const PRESET_COLORS = ['#22c55e','#3b82f6','#f43f5e','#f59e0b','#a78bfa',
                       '#06b6d4','#ec4899','#84cc16','#fb923c','#e879f9'];

function addClass(){
  const name = document.getElementById('newClsName').value.trim();
  if(!name){ alert('กรุณากรอกชื่อ class'); return; }
  if(classes.find(c=>c.name===name)){ alert('มี class นี้แล้ว'); return; }
  const color = document.getElementById('newClsColor').value;
  classes.push({id: nextClassId, name, color});
  activeClassId = nextClassId;
  nextClassId++;
  document.getElementById('newClsName').value = '';
  // suggest next color
  document.getElementById('newClsColor').value =
    PRESET_COLORS[classes.length % PRESET_COLORS.length];
  renderClassList();
  if(sessionId) saveClassesToServer();
}

function deleteClass(id){
  if(boxes.some(b=>b.class_id===id)){
    if(!confirm('มี box ที่ใช้ class นี้ อยู่ ยืนยันลบ class?')) return;
  }
  classes = classes.filter(c=>c.id!==id);
  if(activeClassId===id) activeClassId = classes[0]?.id ?? 0;
  renderClassList();
  if(sessionId) saveClassesToServer();
}

function selectClass(id){
  activeClassId = id;
  renderClassList();
}

function renderClassList(){
  const list = document.getElementById('clsList');
  list.innerHTML = '';
  classes.forEach(c=>{
    const div = document.createElement('div');
    div.className = 'cls-item' + (c.id===activeClassId?' active':'');
    div.onclick = ()=>selectClass(c.id);
    div.innerHTML = `<span class="cls-swatch" style="background:${c.color}"></span>
      <span class="cls-id">${c.id}</span>
      <span class="cls-name">${c.name}</span>
      <button class="cls-del" onclick="event.stopPropagation();deleteClass(${c.id})" title="ลบ">&#10005;</button>`;
    list.appendChild(div);
  });
  document.getElementById('clsBadge').textContent = `${classes.length} classes`;
}

async function saveClassesToServer(){
  if(!sessionId || currentIdx < 0) return;
  await fetch(`/api/save-labels/${sessionId}/${currentIdx}`,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({boxes: boxes.map(({class_id,cx,cy,w,h})=>({class_id,cx,cy,w,h})), classes})
  });
}

function getClassColor(id){
  return classes.find(c=>c.id===id)?.color || '#ffffff';
}

// ── Box List Panel ────────────────────────────────────────────────────────────
function renderBoxList(){
  const list = document.getElementById('boxList');
  list.innerHTML = '';
  document.getElementById('boxCount').textContent = boxes.length;
  boxes.forEach((b,i)=>{
    const cls = classes.find(c=>c.id===b.class_id);
    const color = cls?.color||'#fff';
    const name  = cls?.name||`cls ${b.class_id}`;
    const div = document.createElement('div');
    div.className = 'box-item'+(b.id===selectedBoxId?' active':'');
    div.onclick = ()=>{
      selectedBoxId = b.id;
      setTool('select');
      document.getElementById('btnDelBox').disabled=false;
      renderBoxList(); drawAll();
    };
    div.innerHTML = `<span class="box-swatch" style="background:${color}"></span>
      <span>${i+1}. ${name}</span>
      <button class="box-del" onclick="event.stopPropagation();deleteBoxById(${b.id})" title="ลบ">&#10005;</button>`;
    list.appendChild(div);
  });
}

function deleteBoxById(id){
  boxes = boxes.filter(b=>b.id!==id);
  if(selectedBoxId===id){ selectedBoxId=null; document.getElementById('btnDelBox').disabled=true; }
  markUnsaved(true); drawAll(); renderBoxList();
}

// ── Hex + alpha helper ────────────────────────────────────────────────────────
function hexAlpha(hex, a){
  const r=parseInt(hex.slice(1,3),16);
  const g=parseInt(hex.slice(3,5),16);
  const b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='ArrowRight'||e.key==='ArrowDown')  navigate(1);
  if(e.key==='ArrowLeft' ||e.key==='ArrowUp')    navigate(-1);
  if(currentTask==='detect'){
    if(e.key==='d'||e.key==='D') setTool('draw');
    if(e.key==='s'||e.key==='S') setTool('select');
    if(e.key==='Delete'||e.key==='Backspace') deleteSelectedBox();
    if((e.ctrlKey||e.metaKey)&&e.key==='s'){ e.preventDefault(); saveLabels(); }
  } else if(currentTask==='segment'){
    if(e.key==='Escape') cancelPoly();
    if(e.key==='Delete'||e.key==='Backspace') deleteSelectedPoly();
    if((e.ctrlKey||e.metaKey)&&e.key==='s'){ e.preventDefault(); saveSegments(); }
  } else if(currentTask==='classify'){
    // ตัวเลข 1-9 เลือก class ช่องทางเร็ว
    const num = parseInt(e.key);
    if(num >= 1 && num <= 9){
      const cls = classes[num-1];
      if(cls) assignClsClass(cls.name);
    }
  } else if(currentTask==='anomalib'){
    // g = เลือก Good, d = เลือก Defect, Enter = Assign รูปนี้
    if(e.key==='g'||e.key==='G'){ setAnomType('good'); }
    if(e.key==='d'||e.key==='D'){ setAnomType('defect'); }
    if(e.key==='Enter'){ assignAnomLabel(); }
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────
document.getElementById('newClsColor').value = PRESET_COLORS[0];
// hide segment, classify, anomalib toolbars on start
document.getElementById('segToolbar').style.display  = 'none';
document.getElementById('clsToolbar').style.display  = 'none';
document.getElementById('anomToolbar').style.display = 'none';

// =============================================================================
// P3 — SEGMENT POLYGON
// =============================================================================

// ── Polygon draw helpers ──────────────────────────────────────────────────────
function imgToCanvas(nx, ny){ return { x: nx * imgW * scale, y: ny * imgH * scale }; }
function canvasToNorm(cx, cy){ return { x: cx / (imgW * scale), y: cy / (imgH * scale) }; }

function drawPolygon(poly, selected){
  if(poly.points.length < 2) return;
  const cls   = classes.find(c=>c.id===poly.class_id);
  const color = cls ? cls.color : '#ffffff';
  CTX.beginPath();
  poly.points.forEach((p,i)=>{
    const {x,y} = imgToCanvas(p.x, p.y);
    i===0 ? CTX.moveTo(x,y) : CTX.lineTo(x,y);
  });
  CTX.closePath();
  CTX.strokeStyle = color;
  CTX.lineWidth   = selected ? 2.5 : 1.8;
  CTX.setLineDash(selected ? [6,2] : []);
  CTX.stroke();
  CTX.setLineDash([]);
  CTX.fillStyle = hexAlpha(color, selected ? 0.22 : 0.10);
  CTX.fill();
  // label
  const label = cls ? `${poly.class_id}: ${cls.name}` : `cls ${poly.class_id}`;
  const {x:lx, y:ly} = imgToCanvas(poly.points[0].x, poly.points[0].y);
  CTX.font = 'bold 11px Segoe UI,sans-serif';
  const tw = CTX.measureText(label).width + 8;
  CTX.fillStyle = color;
  CTX.fillRect(lx, ly-16, tw, 16);
  CTX.fillStyle = '#000';
  CTX.fillText(label, lx+4, ly-4);
  // vertices
  poly.points.forEach(p=>{
    const {x,y} = imgToCanvas(p.x, p.y);
    CTX.beginPath();
    CTX.arc(x, y, selected?5:3, 0, Math.PI*2);
    CTX.fillStyle = selected ? '#fff' : color;
    CTX.fill();
    CTX.strokeStyle = color;
    CTX.lineWidth = 1.5;
    CTX.stroke();
  });
}

function drawInProgressPoly(){
  const color = getClassColor(activeClassId);
  const pts   = curPoly.points;
  if(pts.length === 0) return;
  CTX.beginPath();
  pts.forEach((p,i)=>{
    const {x,y} = imgToCanvas(p.x, p.y);
    i===0 ? CTX.moveTo(x,y) : CTX.lineTo(x,y);
  });
  if(segMousePos){
    CTX.lineTo(segMousePos.x * imgW * scale, segMousePos.y * imgH * scale);
  }
  CTX.strokeStyle = color;
  CTX.lineWidth = 2;
  CTX.setLineDash([5,3]);
  CTX.stroke();
  CTX.setLineDash([]);
  pts.forEach(p=>{
    const {x,y} = imgToCanvas(p.x, p.y);
    CTX.beginPath(); CTX.arc(x,y,5,0,Math.PI*2);
    CTX.fillStyle = color; CTX.fill();
  });
  // first point larger to hint close
  const {x:fx, y:fy} = imgToCanvas(pts[0].x, pts[0].y);
  CTX.beginPath(); CTX.arc(fx, fy, 7, 0, Math.PI*2);
  CTX.strokeStyle = '#fff'; CTX.lineWidth=2; CTX.stroke();
}

let segMousePos = null;

// ── Segment canvas events ─────────────────────────────────────────────────────
CANVAS.addEventListener('mousemove', segMouseMove, true);
function segMouseMove(e){
  // รับทั้ง segment brush และ anomalib pixel mask brush
  const isAnom = isAnomBrushActive();
  if(currentTask !== 'segment' && !isAnom) return;
  const rect = CANVAS.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;

  if(segTool === 'brush' || isAnom){
    // move brush cursor circle
    const wrapEl = document.getElementById('canvasWrap');
    const wrapRect = wrapEl.getBoundingClientRect();
    BRUSH_CURSOR.style.left   = (e.clientX - wrapRect.left) + 'px';
    BRUSH_CURSOR.style.top    = (e.clientY - wrapRect.top)  + 'px';
    BRUSH_CURSOR.style.width  = brushSize * 2 + 'px';
    BRUSH_CURSOR.style.height = brushSize * 2 + 'px';
    BRUSH_CURSOR.style.display = 'block';
    // paint if mouse held
    if(isBrushing){
      const color = isAnom ? 'rgba(239,68,68,0.55)' : getClassColor(activeClassId);
      BRUSH_CTX.beginPath();
      BRUSH_CTX.arc(cx, cy, brushSize, 0, Math.PI * 2);
      BRUSH_CTX.fillStyle = color;
      BRUSH_CTX.fill();
    }
    return;
  }

  // polygon mode
  segMousePos = {
    x: cx / (imgW * scale),
    y: cy / (imgH * scale)
  };
  if(curPoly && curPoly.points.length > 0) drawAll();
}

CANVAS.addEventListener('click', segClick);
function segClick(e){
  if(currentTask !== 'segment') return;
  if(segTool !== 'polygon') return;  // brush mode handles its own events
  const rect = CANVAS.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / (imgW * scale);
  const ny = (e.clientY - rect.top)  / (imgH * scale);
  if(nx<0||nx>1||ny<0||ny>1) return;

  // ถ้ายังไม่มี polygon กำลังวาด → เช็คว่าคลิก polygon เดิมไหม (select mode)
  if(!curPoly){
    // หา polygon ที่ถูกคลิก
    let hit = null;
    for(let i=polygons.length-1; i>=0; i--){
      if(pointInPolygon(nx,ny, polygons[i].points)){ hit=polygons[i]; break; }
    }
    selPolyId = hit ? hit.id : null;
    document.getElementById('btnDelPoly').disabled = selPolyId===null;
    drawAll(); return;
  }

  // ดับเบิลคลิก → ปิด polygon
  if(e.detail >= 2){
    closePoly(); return;
  }

  // เช็คว่าคลิกจุดแรกไหม (ปิด polygon)
  if(curPoly.points.length >= 3){
    const fp = curPoly.points[0];
    const dx = (nx - fp.x)*imgW*scale;
    const dy = (ny - fp.y)*imgH*scale;
    if(Math.sqrt(dx*dx+dy*dy) < 10){ closePoly(); return; }
  }
  curPoly.points.push({x:nx, y:ny});
  drawAll();
}

function startNewPoly(){
  if(!sessionId||currentIdx<0){ alert('กรุณาเลือกรูปก่อน'); return; }
  if(curPoly){ alert('กำลังวาด Polygon อยู่ — ดับเบิลคลิกหรือ Escape เพื่อจบก่อน'); return; }
  curPoly = { class_id: activeClassId, points: [] };
  setStatus('คลิกวางจุด — ดับเบิลคลิก หรือคลิกจุดแรก เพื่อปิด Polygon');
}

function closePoly(){
  if(!curPoly || curPoly.points.length < 3){ alert('ต้องมีอย่างน้อย 3 จุด'); return; }
  polygons.push({...curPoly, id: polyIdCnt++});
  curPoly = null; segMousePos = null;
  markSegUnsaved(true);
  drawAll(); renderPolyList();
  setStatus(`เพิ่ม Polygon สำเร็จ (รวม ${polygons.length})`, true);
}

function cancelPoly(){
  curPoly = null; segMousePos = null;
  drawAll();
  setStatus('ยกเลิกการวาด');
}

function deleteSelectedPoly(){
  if(selPolyId===null) return;
  polygons = polygons.filter(p=>p.id!==selPolyId);
  selPolyId=null;
  document.getElementById('btnDelPoly').disabled=true;
  markSegUnsaved(true); drawAll(); renderPolyList();
}

function clearAllPolygons(){
  if(!polygons.length) return;
  if(!confirm('ลบ Polygon ทั้งหมดในรูปนี้?')) return;
  polygons=[]; selPolyId=null; curPoly=null;
  document.getElementById('btnDelPoly').disabled=true;
  markSegUnsaved(true); drawAll(); renderPolyList();
}

// ── Save / Load Segments ──────────────────────────────────────────────────────
async function saveSegments(silent=false){
  if(!sessionId||currentIdx<0) return;
  try{
    const payload = {
      polygons: polygons.map(({class_id,points})=>({class_id,points})),
      classes
    };
    const res = await fetch(`/api/save-segments/${sessionId}/${currentIdx}`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error((await res.json()).detail||'บันทึกไม่สำเร็จ');
    markSegUnsaved(false);
    const th = document.getElementById(`th-${currentIdx}`);
    if(th) th.classList.toggle('has-label', polygons.length>0||boxes.length>0);
    if(!silent) setStatus(`บันทึก Segment สำเร็จ (${polygons.length} polygons)`, true);
  } catch(e){
    if(!silent) alert('บันทึกไม่สำเร็จ: '+e.message);
  }
}

function markSegUnsaved(v){
  segUnsaved = v;
  const badge = document.getElementById('segSaveBadge');
  if(!badge) return;
  badge.textContent = v ? 'ยังไม่ได้บันทึก' : 'บันทึกแล้ว';
  badge.className   = 'save-badge ' + (v ? 'unsaved' : 'saved');
}

// ── Polygon list panel ────────────────────────────────────────────────────────
function renderPolyList(){
  const list = document.getElementById('polyList');
  if(!list) return;
  list.innerHTML = '';
  document.getElementById('polyCount').textContent = polygons.length;
  polygons.forEach((p,i)=>{
    const cls   = classes.find(c=>c.id===p.class_id);
    const color = cls?.color||'#fff';
    const name  = cls?.name||`cls ${p.class_id}`;
    const div   = document.createElement('div');
    div.className = 'box-item'+(p.id===selPolyId?' active':'');
    div.onclick = ()=>{
      selPolyId = p.id;
      document.getElementById('btnDelPoly').disabled=false;
      renderPolyList(); drawAll();
    };
    div.innerHTML = `<span class="box-swatch" style="background:${color}"></span>
      <span>${i+1}. ${name} (${p.points.length}pts)</span>
      <button class="box-del" onclick="event.stopPropagation();deletePolyById(${p.id})" title="ลบ">&#10005;</button>`;
    list.appendChild(div);
  });
}

function deletePolyById(id){
  polygons = polygons.filter(p=>p.id!==id);
  if(selPolyId===id){ selPolyId=null; document.getElementById('btnDelPoly').disabled=true; }
  markSegUnsaved(true); drawAll(); renderPolyList();
}

// ── Point-in-polygon (ray casting) ───────────────────────────────────────────
function pointInPolygon(nx, ny, pts){
  let inside = false;
  for(let i=0,j=pts.length-1; i<pts.length; j=i++){
    const xi=pts[i].x, yi=pts[i].y, xj=pts[j].x, yj=pts[j].y;
    if(((yi>ny)!==(yj>ny))&&(nx<(xj-xi)*(ny-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}

// =============================================================================
// BRUSH TOOL
// =============================================================================

// ── Switch segment sub-tool ───────────────────────────────────────────────────
function setSegTool(t){
  segTool = t;
  document.getElementById('segToolPoly').classList.toggle('active',  t==='polygon');
  document.getElementById('segToolBrush').classList.toggle('active', t==='brush');
  const brushCtrl  = document.getElementById('brushControls');
  const polyCtrl   = document.getElementById('polyControls');
  const btnConfirm = document.getElementById('btnConfirmBrush');
  const btnClr     = document.getElementById('btnClearBrush');

  if(t === 'brush'){
    brushCtrl.style.display  = 'flex';
    polyCtrl.style.display   = 'none';
    btnConfirm.style.display = '';
    btnClr.style.display     = '';
    BRUSH_CANVAS.style.display = 'block';
    CANVAS.className = 'tool-brush';
    // cancel any in-progress polygon
    if(curPoly){ curPoly=null; segMousePos=null; drawAll(); }
  } else {
    brushCtrl.style.display  = 'none';
    polyCtrl.style.display   = 'contents';
    btnConfirm.style.display = 'none';
    btnClr.style.display     = 'none';
    BRUSH_CANVAS.style.display = 'none';
    BRUSH_CURSOR.style.display = 'none';
    BRUSH_CTX.clearRect(0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height);
    CANVAS.className = 'tool-seg';
  }
}

// ── Brush mouse events ────────────────────────────────────────────────────────
function isAnomBrushActive(){
  // brush เปิดใน anomalib mode เมื่อติ๊ก Pixel Mask
  if(currentTask !== 'anomalib') return false;
  const chk = document.getElementById('anomUseMask');
  return chk && chk.checked;
}

CANVAS.addEventListener('mousedown', e=>{
  const isBrushMode = (currentTask === 'segment' && segTool === 'brush') || isAnomBrushActive();
  if(!isBrushMode) return;
  if(!sessionId || currentIdx < 0) return;
  isBrushing = true;
  const rect = CANVAS.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  // anomalib ใช้สีแดงโปร่งใส, segment ใช้สีของ class
  const color = isAnomBrushActive() ? 'rgba(239,68,68,0.55)' : getClassColor(activeClassId);
  BRUSH_CTX.beginPath();
  BRUSH_CTX.arc(cx, cy, brushSize, 0, Math.PI * 2);
  BRUSH_CTX.fillStyle = color;
  BRUSH_CTX.fill();
}, true);   // capture so detect mousedown doesn't also fire

CANVAS.addEventListener('mouseup', e=>{
  const isBrushMode = (currentTask === 'segment' && segTool === 'brush') || isAnomBrushActive();
  if(!isBrushMode) return;
  isBrushing = false;
  saveBrushMask(currentIdx);
}, true);

CANVAS.addEventListener('mouseleave', e=>{
  if(segTool === 'brush' || isAnomBrushActive()){
    isBrushing = false;
    BRUSH_CURSOR.style.display = 'none';
  }
});

// ── Confirm brush → convert mask to polygon (segment) / keep mask (anomalib) ──
function confirmBrush(){
  const w = BRUSH_CANVAS.width;
  const h = BRUSH_CANVAS.height;
  if(!w || !h) return;

  // ── anomalib mode: เก็บ mask ไว้เฉยๆ ไม่ต้องแปลงเป็น polygon ──
  if(isAnomBrushActive()){
    saveBrushMask(currentIdx);
    // force has_mask = true ใน anomLabels
    if(anomLabels[currentIdx]){
      anomLabels[currentIdx].has_mask = true;
      // อัปเดตที่ server ด้วย
      fetch(`/api/save-anomalib/${sessionId}/${currentIdx}`,{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(anomLabels[currentIdx])
      });
    }
    setStatus('บันทึก Brush Mask สำหรับ Anomalib สำเร็จ ✓', true);
    return;
  }

  // ── segment mode: แปลง mask → polygon เหมือนเดิม ──
  const imgData = BRUSH_CTX.getImageData(0, 0, w, h);
  const pts = maskToPolygon(imgData, w, h);
  if(!pts || pts.length < 3){
    setStatus('วาด Brush ก่อนกดยืนยัน (ต้องมีพื้นที่เพียงพอ)', false);
    return;
  }
  polygons.push({ id: polyIdCnt++, class_id: activeClassId, points: pts });
  markSegUnsaved(true);
  renderPolyList();
  drawAll();
  BRUSH_CTX.clearRect(0, 0, w, h);
  brushMasks[currentIdx] = null;  // mask ถูกยืนยันแล้ว ล้างค้าง
  setStatus(`เพิ่ม Polygon จาก Brush สำเร็จ (รวม ${polygons.length})`, true);
}

function clearBrush(){
  BRUSH_CTX.clearRect(0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height);
  brushMasks[currentIdx] = null;
}

// ── Mask → polygon contour using marching squares (simplified) ────────────────
function maskToPolygon(imgData, w, h){
  // Build binary grid: 1 where alpha > 0
  const alpha = new Uint8Array(w * h);
  for(let i = 0; i < w * h; i++) alpha[i] = imgData.data[i*4+3] > 10 ? 1 : 0;

  // Find starting edge pixel (topmost row with paint)
  let startX = -1, startY = -1;
  outer: for(let y = 0; y < h; y++){
    for(let x = 0; x < w; x++){
      if(alpha[y*w+x]){
        startX = x; startY = y;
        break outer;
      }
    }
  }
  if(startX === -1) return [];

  // Moore neighbourhood boundary tracing
  const dx = [1,1,0,-1,-1,-1, 0, 1];
  const dy = [0,1,1, 1, 0,-1,-1,-1];
  const boundary = [];
  let cx = startX, cy = startY;
  // entry direction: came from left (dir=0)
  let backDir = 4; // opposite of 0
  const maxIter = w * h * 2;
  let iter = 0;
  do {
    boundary.push({x: cx, y: cy});
    // clockwise from backtrack dir
    let found = false;
    for(let k = 1; k <= 8; k++){
      const nd = (backDir + k) % 8;
      const nx = cx + dx[nd];
      const ny = cy + dy[nd];
      if(nx>=0 && nx<w && ny>=0 && ny<h && alpha[ny*w+nx]){
        // entry to (nx,ny) from direction nd, so back is opposite
        backDir = (nd + 4) % 8;
        cx = nx; cy = ny;
        found = true; break;
      }
    }
    if(!found) break;
    if(++iter > maxIter) break;
  } while(!(cx===startX && cy===startY));

  if(boundary.length < 6) return [];

  // Thin boundary (keep every Nth point to reduce density before RDP)
  const step = Math.max(1, Math.floor(boundary.length / 400));
  const thinned = boundary.filter((_,i)=>i%step===0);

  // RDP simplification
  const eps = Math.max(1.5, (w + h) * 0.003);
  const simplified = rdpSimplify(thinned, eps);

  // Normalise to 0-1
  return simplified.map(p => ({ x: p.x / w, y: p.y / h }));
}

// ── Ramer-Douglas-Peucker line simplification ─────────────────────────────────
function rdpSimplify(pts, eps){
  if(pts.length <= 2) return pts;
  let maxDist = 0, maxIdx = 0;
  const p1 = pts[0], p2 = pts[pts.length-1];
  for(let i = 1; i < pts.length-1; i++){
    const d = pointLineDistance(pts[i], p1, p2);
    if(d > maxDist){ maxDist = d; maxIdx = i; }
  }
  if(maxDist > eps){
    const left  = rdpSimplify(pts.slice(0, maxIdx+1), eps);
    const right = rdpSimplify(pts.slice(maxIdx), eps);
    return [...left.slice(0,-1), ...right];
  }
  return [p1, p2];
}

function pointLineDistance(p, a, b){
  const dx = b.x-a.x, dy = b.y-a.y;
  if(dx===0 && dy===0) return Math.hypot(p.x-a.x, p.y-a.y);
  const t = ((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy);
  const px = a.x+t*dx, py = a.y+t*dy;
  return Math.hypot(p.x-px, p.y-py);
}

// =============================================================================
// P4 — CLASSIFY  (1 image = 1 class)
// =============================================================================

// ── Assign class to current image ─────────────────────────────────────────────
async function assignClsClass(className){
  if(!sessionId || currentIdx < 0){ alert('กรุณาเลือกรูปก่อน'); return; }
  clsLabels[currentIdx] = className;
  // save to server
  await fetch(`/api/save-classify/${sessionId}/${currentIdx}`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({class_name: className})
  });
  markClsUnsaved(false);
  updateClsBadgeUI(currentIdx);
  renderClsSummary();
  setStatus(`✓ รูป ${images[currentIdx].name} → "${className}"`, true);
  // update thumbnail badge
  const th = document.getElementById(`th-${currentIdx}`);
  if(th) th.setAttribute('data-cls', className);
  renderClsThumbBadge(currentIdx, className);
}

function updateClsBadgeUI(idx){
  const name = clsLabels[idx] || '';
  const badge = document.getElementById('clsCurrentBadge');
  if(badge){
    badge.textContent = name ? `✓ "${name}"` : '— ยังไม่เลือก —';
    badge.style.color = name ? 'var(--accent)' : 'var(--muted)';
  }
  // highlight active card in quick grid
  document.querySelectorAll('.cls-quick-card').forEach(btn=>{
    btn.classList.toggle('active', btn.dataset.name === name);
  });
}

function markClsUnsaved(v){
  clsUnsaved = v;
  const badge = document.getElementById('clsSaveBadge');
  if(!badge) return;
  badge.textContent = v ? 'ยังไม่บันทึก' : 'บันทึกแล้ว';
  badge.className   = 'save-badge ' + (v ? 'unsaved' : 'saved');
}

// ── Quick-select grid (renders class buttons in toolbar) ──────────────────────
function renderClsQuickGrid(){
  const grid = document.getElementById('clsQuickGrid');
  if(!grid) return;
  grid.innerHTML = '';
  if(classes.length === 0){
    grid.innerHTML = '<span style="font-size:.75rem;color:var(--muted);">ยังไม่มี class — เพิ่มที่แผง Classes ขวามือ</span>';
    return;
  }
  classes.forEach((cls, i) => {
    const btn = document.createElement('button');
    btn.className   = 'cls-quick-card';
    btn.dataset.name = cls.name;
    btn.title       = `[${i+1}] ${cls.name}`;
    btn.style.setProperty('--cls-color', cls.color);
    const current = currentIdx >= 0 ? clsLabels[currentIdx] : '';
    if(current === cls.name) btn.classList.add('active');
    btn.innerHTML = `<span class="cls-quick-dot" style="background:${cls.color}"></span>${cls.name}${i<9?`<span class="cls-quick-key">${i+1}</span>`:''}`;
    btn.onclick = () => assignClsClass(cls.name);
    grid.appendChild(btn);
  });
}

// ── Summary panel (right side) ───────────────────────────────────────────────
function renderClsSummary(){
  const list = document.getElementById('clsSummaryList');
  if(!list) return;
  // count per class
  const counts = {};
  Object.values(clsLabels).forEach(n=>{ counts[n] = (counts[n]||0)+1; });
  const labeled = Object.keys(clsLabels).length;
  document.getElementById('clsLabelCount').textContent = labeled;
  document.getElementById('clsTotalCount').textContent = images.length;

  list.innerHTML = '';
  // sort by class order in classes array
  const seen = new Set();
  [...classes.map(c=>c.name), ...Object.keys(counts)].forEach(name=>{
    if(seen.has(name)) return; seen.add(name);
    const color = classes.find(c=>c.name===name)?.color || '#888';
    const cnt   = counts[name] || 0;
    const div = document.createElement('div');
    div.className = 'box-item';
    div.innerHTML = `<span class="box-swatch" style="background:${color}"></span>
      <span style="flex:1">${name}</span>
      <span style="font-size:.75rem;color:var(--accent2);font-weight:700;">${cnt}</span>`;
    list.appendChild(div);
  });
}

// ── Thumbnail class badge ─────────────────────────────────────────────────────
function renderClsThumbBadge(idx, name){
  const th = document.getElementById(`th-${idx}`);
  if(!th) return;
  let badge = th.querySelector('.cls-thumb-badge');
  if(!badge){
    badge = document.createElement('span');
    badge.className = 'cls-thumb-badge';
    th.appendChild(badge);
  }
  if(name){
    const color = classes.find(c=>c.name===name)?.color || '#888';
    badge.textContent = name;
    badge.style.background = color;
    badge.style.color = '#000';
    badge.style.display = 'block';
  } else {
    badge.style.display = 'none';
  }
}

// ── Load all classify labels for current session ──────────────────────────────
async function loadAllClsLabels(){
  if(!sessionId) return;
  try{
    const data = await fetch(`/api/load-classify-all/${sessionId}`).then(r=>r.json());
    clsLabels = {};
    Object.entries(data.labels).forEach(([k,v])=>{ clsLabels[parseInt(k)] = v; });
    // render badges on all thumbnails
    Object.entries(clsLabels).forEach(([k,v])=> renderClsThumbBadge(parseInt(k), v));
    if(currentTask==='classify'){
      renderClsSummary();
      if(currentIdx >= 0) updateClsBadgeUI(currentIdx);
    }
  } catch(e){ /* ignore */ }
}

// ── Export detect dataset as ZIP ──────────────────────────────────────────────
async function doExportDetect(){
  if(!sessionId){ alert('ไม่มี session'); return; }
  const valPct = parseInt(document.getElementById('detectValSlider')?.value || 20) / 100;
  showSpinner('กำลัง export Detect dataset...');
  try{
    const res = await fetch(`/api/export-detect/${sessionId}?val_split=${valPct}`, {method:'POST'});
    if(!res.ok){ const d=await res.json(); alert(d.detail||'export ไม่สำเร็จ'); return; }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `detect_${sessionId.slice(0,8)}.zip`; a.click();
    URL.revokeObjectURL(url);
    setStatus('Export Detect ZIP สำเร็จ ✓', true);
  } catch(e){ alert('export ไม่สำเร็จ: '+e.message);
  } finally { hideSpinner(); }
}

// ── Export segment dataset as ZIP ─────────────────────────────────────────────
async function doExportSegment(){
  if(!sessionId){ alert('ไม่มี session'); return; }
  const valPct = parseInt(document.getElementById('segValSlider')?.value || 20) / 100;
  showSpinner('กำลัง export Segment dataset...');
  try{
    const res = await fetch(`/api/export-segment/${sessionId}?val_split=${valPct}`, {method:'POST'});
    if(!res.ok){ const d=await res.json(); alert(d.detail||'export ไม่สำเร็จ'); return; }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `segment_${sessionId.slice(0,8)}.zip`; a.click();
    URL.revokeObjectURL(url);
    setStatus('Export Segment ZIP สำเร็จ ✓', true);
  } catch(e){ alert('export ไม่สำเร็จ: '+e.message);
  } finally { hideSpinner(); }
}

// ── Export classify dataset as ZIP ────────────────────────────────────────────
async function doExportClassify(){
  if(!sessionId){ alert('ไม่มี session'); return; }
  const labeled = Object.keys(clsLabels).length;
  if(labeled === 0){ alert('ยังไม่ได้ assign class ให้รูปใดเลย'); return; }
  const valPct = parseInt(document.getElementById('valSplitSlider').value) / 100;
  showSpinner(`กำลัง export ${labeled} รูป...`);
  try{
    const res = await fetch(`/api/export-classify/${sessionId}?val_split=${valPct}`, {method:'POST'});
    if(!res.ok){ const d=await res.json(); alert(d.detail||'export ไม่สำเร็จ'); return; }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `classify_${sessionId.slice(0,8)}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`Export สำเร็จ ${labeled} รูป ✓`, true);
  } catch(e){ alert('export ไม่สำเร็จ: '+e.message);
  } finally { hideSpinner(); }
}


// =============================================================================
// ── Anomalib ─────────────────────────────────────────────────────────────────

function setAnomType(type){
  anomCurrentType = type;
  const isDefect = (type === 'defect');
  document.getElementById('anomBtnGood').classList.toggle('active', !isDefect);
  document.getElementById('anomBtnDefect').classList.toggle('active', isDefect);
  const wrap = document.getElementById('anomDefectTypeWrap');
  if(wrap) wrap.style.display = isDefect ? 'flex' : 'none';
}

function updateAnomMaskHint(){
  const chk  = document.getElementById('anomUseMask');
  const hint = document.getElementById('anomMaskHint');
  const useMask = chk && chk.checked;
  if(hint) hint.style.display = useMask ? 'inline' : 'none';
  // เปิด/ปิด brush canvas เมื่ออยู่ใน anomalib mode
  if(currentTask === 'anomalib'){
    if(useMask){
      BRUSH_CANVAS.style.display = 'block';
      CANVAS.className = 'tool-brush';
      // restore brush mask ของรูปนี้ถ้ามี
      restoreBrushMask(currentIdx);
    } else {
      BRUSH_CTX.clearRect(0, 0, BRUSH_CANVAS.width, BRUSH_CANVAS.height);
      BRUSH_CANVAS.style.display = 'none';
      BRUSH_CURSOR.style.display = 'none';
      CANVAS.className = 'tool-anom';
    }
  }
}

async function assignAnomLabel(){
  if(!sessionId || currentIdx < 0){ alert('กรุณาเลือกรูปก่อน'); return; }
  const isDefect = (anomCurrentType === 'defect');
  let label = 'good';
  if(isDefect){
    const inp = document.getElementById('anomDefectType');
    label = (inp ? inp.value.trim() : '') || 'defect';
  }
  const useMask = document.getElementById('anomUseMask')?.checked || false;
  const payload = { label, is_defect: isDefect, has_mask: useMask };
  try{
    const res = await fetch(`/api/save-anomalib/${sessionId}/${currentIdx}`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
    });
    if(!res.ok){ const d=await res.json(); alert(d.detail||'บันทึกไม่สำเร็จ'); return; }
    anomLabels[currentIdx] = payload;
    updateAnomBadgeUI(currentIdx);
    renderAnomThumbBadge(currentIdx, payload);
    renderAnomSummary();
    setStatus(`Assign "${label}" ให้รูป ${currentIdx+1} สำเร็จ ✓`, true);
  } catch(e){ alert('เกิดข้อผิดพลาด: '+e.message); }
}

function updateAnomBadgeUI(idx){
  const badge = document.getElementById('anomCurrentBadge');
  if(!badge) return;
  const info = anomLabels[idx];
  if(!info){
    badge.textContent = '— ยังไม่เลือก —';
    badge.style.color = 'var(--muted)';
  } else if(!info.is_defect){
    badge.textContent = '✓ Good';
    badge.style.color = '#22c55e';
  } else {
    const mask = info.has_mask ? ' +mask' : '';
    badge.textContent = `✗ ${info.label}${mask}`;
    badge.style.color = '#ef4444';
  }
}

function renderAnomThumbBadge(idx, info){
  const thumb = document.querySelector(`.thumb-card[data-idx="${idx}"]`);
  if(!thumb) return;
  let badge = thumb.querySelector('.anom-thumb-badge');
  if(!badge){
    badge = document.createElement('div');
    badge.className = 'anom-thumb-badge';
    thumb.appendChild(badge);
  }
  if(!info){
    badge.style.display = 'none';
    badge.className = 'anom-thumb-badge';
  } else if(!info.is_defect){
    badge.textContent = 'good';
    badge.className = 'anom-thumb-badge good';
    badge.style.display = '';
  } else {
    badge.textContent = info.label;
    badge.className = 'anom-thumb-badge defect';
    badge.style.display = '';
  }
}

function renderAnomSummary(){
  const hd  = document.getElementById('anomSummaryHd');
  const lst = document.getElementById('anomSummaryList');
  if(!hd||!lst) return;
  const total    = images.length;
  const labeled  = Object.keys(anomLabels).length;
  document.getElementById('anomLabelCount').textContent = labeled;
  document.getElementById('anomTotalCount').textContent = total;

  // count by label
  const counts = {}; // label -> {count, is_defect}
  for(const info of Object.values(anomLabels)){
    if(!counts[info.label]) counts[info.label] = {count:0, is_defect:info.is_defect};
    counts[info.label].count++;
  }
  const unlabeled = total - labeled;
  let html = '';
  for(const [lbl, d] of Object.entries(counts)){
    const cls = d.is_defect ? 'defect-row' : 'good-row';
    const dotCls = d.is_defect ? 'defect' : 'good';
    html += `<div class="anom-row ${cls}">
      <div class="anom-dot ${dotCls}"></div>
      <span class="anom-row-name">${lbl}</span>
      <span class="anom-row-cnt">${d.count}</span>
    </div>`;
  }
  if(unlabeled > 0){
    html += `<div class="anom-row" style="opacity:.5">
      <div class="anom-dot" style="background:#6b7280"></div>
      <span class="anom-row-name">ยังไม่เลือก</span>
      <span class="anom-row-cnt">${unlabeled}</span>
    </div>`;
  }
  lst.innerHTML = html || '<div style="color:var(--muted);font-size:.8rem;padding:8px">ยังไม่มีข้อมูล</div>';
}

async function loadAllAnomLabels(){
  if(!sessionId) return;
  try{
    const res = await fetch(`/api/load-anomalib-all/${sessionId}`);
    if(!res.ok) return;
    const data = await res.json();
    anomLabels = {};
    for(const [idx, info] of Object.entries(data.labels||{})){
      anomLabels[parseInt(idx)] = info;
    }
    // rebuild thumb badges after gallery is ready
    requestAnimationFrame(()=>{
      for(const [idx, info] of Object.entries(anomLabels)){
        renderAnomThumbBadge(parseInt(idx), info);
      }
      if(currentTask==='anomalib'){
        renderAnomSummary();
        if(currentIdx >= 0) updateAnomBadgeUI(currentIdx);
      }
    });
  } catch(e){ /* ignore */ }
}

async function doExportAnomlib(){
  if(!sessionId){ alert('ไม่มี session'); return; }
  const labeled = Object.keys(anomLabels).length;
  if(labeled === 0){ alert('ยังไม่ได้ assign label ให้รูปใดเลย'); return; }
  const product  = (document.getElementById('anomProductName')?.value.trim() || 'product').replace(/[^\w\-]/g,'_');
  const valPct   = parseInt(document.getElementById('anomValSlider')?.value || 20) / 100;
  showSpinner(`กำลัง export ${labeled} รูป...`);
  try{
    const res = await fetch(`/api/export-anomalib/${sessionId}?product=${encodeURIComponent(product)}&val_split=${valPct}`, {method:'POST'});
    if(!res.ok){ const d=await res.json(); alert(d.detail||'export ไม่สำเร็จ'); return; }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `anomalib_${product}_${sessionId.slice(0,8)}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`Export Anomalib "${product}" สำเร็จ ✓`, true);
  } catch(e){ alert('export ไม่สำเร็จ: '+e.message);
  } finally { hideSpinner(); }
}

// =============================================================================

function saveSessionToStorage(){
  if(!sessionId) return;
  try{
    localStorage.setItem('labelSession', JSON.stringify({
      sessionId,
      task: currentTask
    }));
  } catch(e){ /* quota exceeded — ignore */ }
}

async function tryRestoreSession(){
  let stored;
  try{ stored = JSON.parse(localStorage.getItem('labelSession')); } catch(e){ return; }
  if(!stored || !stored.sessionId) return;

  showSpinner('กำลังกู้ข้อมูล session...');
  try{
    const res = await fetch(`/api/restore/${stored.sessionId}`);
    if(!res.ok){
      localStorage.removeItem('labelSession');
      hideSpinner();
      return;
    }
    const data = await res.json();
    loadSession(data);
    // restore task tab after loadSession (which calls switchTask('detect'))
    if(stored.task && stored.task !== 'detect'){
      switchTask(stored.task);
    }
    setStatus('กู้ข้อมูล session สำเร็จ ✓', true);
  } catch(e){
    localStorage.removeItem('labelSession');
  } finally{
    hideSpinner();
  }
}

// Call after DOM is ready — script is at end of <body> so DOM is already ready
tryRestoreSession();
