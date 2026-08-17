"""Detector-ONLY web tester for the V4 FCOS layout model.

No recognizer, no line-split. Upload a page -> see every detection box (class + score) drawn on
the DPI-normalized image, plus a sortable list of coordinates. Tune score threshold / NMS / class-
agnostic NMS / containment suppression LIVE: the model runs once per image at a low threshold and
caches the raw boxes, so moving a slider just re-filters (instant, no GPU re-run).

  python -m src.serve.detector_app \
      --ckpt data/checkpoints/detector_obj_v4_parallel/step_DET/model.pt \
      --profile parallel --size 1280 --device cuda --port 8010
  # then open http://localhost:8010
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image

from src.detect.model.fcos import build_obj_detector
from src.detect.infer_obj import detect_page_obj
from src.detect.data.build_obj_targets import CLASSES_V4
from src.pipeline.run_ocr import normalize_dpi

_S = {}   # det, size, device, raw (last image's raw dets), img_bgr (last normalized image)

_rng = np.random.RandomState(7)
COLORS = {c: tuple(int(x) for x in _rng.randint(60, 256, 3)) for c in CLASSES_V4}
COLORS.update({"table": (0, 0, 255), "image": (255, 90, 0), "chart": (255, 160, 0),
               "text": (0, 200, 0), "title": (255, 0, 200), "heading": (200, 0, 255),
               "formula": (0, 200, 255)})


def _agnostic_nms(dets, iou_thr):
    if not dets:
        return dets
    b = np.array([d[0] for d in dets], np.float32)
    s = np.array([d[2] for d in dets], np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return [dets[i] for i in keep]


def _containment_suppress(dets, frac):
    out = []
    for box, cid, sc in sorted(dets, key=lambda d: -d[2]):
        x1, y1, x2, y2 = box
        area = max(1.0, (x2 - x1) * (y2 - y1))
        drop = False
        for kb, _, _ in out:
            ix1, iy1 = max(x1, kb[0]), max(y1, kb[1])
            ix2, iy2 = min(x2, kb[2]), min(y2, kb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / area > frac:
                drop = True; break
        if not drop:
            out.append((box, cid, sc))
    return out


def _filter(raw, p):
    d = [t for t in raw if t[2] >= p["score"]]
    if p["agnostic"]:
        d = _agnostic_nms(d, p["nms"])
    if p["containment"]:
        d = _containment_suppress(d, p["contain_frac"])
    return d


def _render(img_bgr, dets):
    im = img_bgr.copy()
    dets = sorted(dets, key=lambda d: (round(d[0][1] / 20), d[0][0]))
    rows = []
    for i, (box, cid, sc) in enumerate(dets):
        x1, y1, x2, y2 = (int(v) for v in box)
        name = CLASSES_V4[cid]
        col = COLORS.get(name, (140, 140, 140))
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 3 if name == "table" else 2)
        cv2.putText(im, f"{i}:{name} {sc:.2f}", (x1 + 2, max(16, y1 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        rows.append({"i": i, "cls": name, "score": round(float(sc), 3),
                     "box": [x1, y1, x2, y2], "w": x2 - x1, "h": y2 - y1})
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    overlay = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""
    from collections import Counter
    counts = dict(Counter(r["cls"] for r in rows))
    return {"overlay": overlay, "rows": rows, "counts": counts, "total": len(rows)}


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>V4 Detector Tester</title>
<style>
 *{box-sizing:border-box;font-family:ui-sans-serif,system-ui,sans-serif}
 body{margin:0;background:#0f1115;color:#e6e6e6}
 header{background:#1a1d24;padding:11px 18px;display:flex;align-items:center;gap:14px;border-bottom:1px solid #2a2e37}
 header b{font-weight:600}.muted{color:#8b95a3;font-size:13px}
 .wrap{display:flex;height:calc(100vh - 47px)}
 .left{flex:1;display:flex;flex-direction:column;border-right:1px solid #2a2e37;min-width:0}
 .right{width:430px;display:flex;flex-direction:column}
 .stage{flex:1;overflow:auto;display:flex;align-items:flex-start;justify-content:center;padding:12px;background:#15171d}
 .stage img{max-width:100%;border-radius:6px;box-shadow:0 0 0 1px #2a2e37}
 .drop{margin:14px;flex:1;border:2px dashed #353a44;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#8b95a3;cursor:pointer;text-align:center;padding:20px}
 .ctl{padding:14px 16px;border-bottom:1px solid #2a2e37}
 .ctl label{display:block;font-size:13px;color:#9aa4b2;margin:10px 0 4px}
 .ctl input[type=range]{width:100%}
 .row{display:flex;align-items:center;gap:8px;font-size:13px;margin:8px 0}
 .val{color:#3b82f6;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}
 .list{flex:1;overflow:auto;font-size:12px;font-family:ui-monospace,monospace}
 .lr{display:flex;gap:8px;padding:5px 14px;border-bottom:1px solid #20242c}
 .lr:hover{background:#1a1d24}
 .li{color:#3b82f6;min-width:22px}.lc{min-width:96px;color:#cbd5e1}.ls{min-width:42px;color:#fbbf24}
 .lb{color:#8b95a3}
 .counts{padding:8px 16px;font-size:12px;color:#9aa4b2;border-bottom:1px solid #2a2e37}
 .busy{color:#fbbf24}.ok{color:#34d399}
</style></head><body>
<header><b>V4 Detector Tester</b> <span class="muted" id="meta">detector-only · no recognizer</span>
 <span class="muted" id="status" style="margin-left:auto">idle</span></header>
<div class="wrap">
 <div class="left">
   <label class="drop" id="drop"><span id="hint">Click to upload a document image</span>
     <input id="file" type="file" accept="image/*" style="display:none"></label>
   <div class="stage" id="stage" style="display:none"><img id="prev"></div>
 </div>
 <div class="right">
   <div class="ctl">
     <label>Score threshold: <span class="val" id="vscore">0.40</span></label>
     <input type="range" id="score" min="0.05" max="0.9" step="0.01" value="0.40">
     <label>NMS IoU: <span class="val" id="vnms">0.50</span></label>
     <input type="range" id="nms" min="0.1" max="0.9" step="0.05" value="0.50">
     <div class="row"><input type="checkbox" id="agnostic" checked><span>Class-agnostic NMS (merge duplicate boxes across classes)</span></div>
     <div class="row"><input type="checkbox" id="containment"><span>Containment suppression (drop nested boxes)</span></div>
     <label>Containment fraction: <span class="val" id="vcf">0.70</span></label>
     <input type="range" id="cf" min="0.3" max="0.95" step="0.05" value="0.70">
   </div>
   <div class="counts" id="counts">—</div>
   <div class="list" id="list"></div>
 </div>
</div>
<script>
let FILE=null;
const $=id=>document.getElementById(id);
const status=$('status'),prev=$('prev'),stage=$('stage'),drop=$('drop');
$('drop').onclick=()=>$('file').click();
$('file').onchange=e=>{FILE=e.target.files[0]; if(FILE){drop.style.display='none';stage.style.display='flex';detect();}};
['score','nms','cf'].forEach(id=>$(id).oninput=()=>{ $('v'+(id==='cf'?'cf':id)).textContent=(+$(id).value).toFixed(2); refilter();});
['agnostic','containment'].forEach(id=>$(id).onchange=refilter);
function params(){return {score:+$('score').value,nms:+$('nms').value,agnostic:$('agnostic').checked,
   containment:$('containment').checked,contain_frac:+$('cf').value};}
async function detect(){
  if(!FILE)return; status.textContent='running detector…'; status.className='busy';
  const fd=new FormData(); fd.append('image',FILE); fd.append('params',JSON.stringify(params()));
  const r=await fetch('/detect',{method:'POST',body:fd}); show(await r.json());
}
async function refilter(){
  if(!FILE)return; status.textContent='filtering…'; status.className='busy';
  const r=await fetch('/refilter',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params())});
  show(await r.json());
}
function show(d){
  if(d.error){status.textContent=d.error;return;}
  prev.src=d.overlay; status.textContent=d.total+' boxes'; status.className='ok';
  $('counts').textContent=Object.entries(d.counts).map(([k,v])=>k+':'+v).join('  ');
  $('list').innerHTML=d.rows.map(r=>`<div class="lr"><span class="li">${r.i}</span>`+
    `<span class="lc">${r.cls}</span><span class="ls">${r.score}</span>`+
    `<span class="lb">[${r.box.join(', ')}] ${r.w}×${r.h}</span></div>`).join('');
}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers(); self.wfile.write(_PAGE.encode("utf-8"))

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path == "/refilter":
            if "raw" not in _S:
                self._json({"error": "upload an image first"}); return
            p = json.loads(body)
            self._json(_render(_S["img_bgr"], _filter(_S["raw"], p))); return
        if self.path != "/detect":
            self.send_response(404); self.end_headers(); return
        try:
            boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
            parts = body.split(b"--" + boundary)
            img_payload, params = None, {"score": 0.4, "nms": 0.5, "agnostic": True,
                                        "containment": False, "contain_frac": 0.7}
            for part in parts:
                if b'name="image"' in part:
                    img_payload = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                elif b'name="params"' in part:
                    params = json.loads(part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0])
            pil = normalize_dpi(Image.open(io.BytesIO(img_payload)).convert("RGB"), 2400)
        except Exception as e:
            self._json({"error": f"bad upload: {e}"}, 400); return
        img_bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        # run the model ONCE at a low floor; slider changes just re-filter the cached raw boxes
        cfg = {"score_thresh": 0.05, "nms_iou": params.get("nms", 0.5), "max_det": 400}
        raw = detect_page_obj(_S["det"], pil, _S["size"], _S["device"], cfg)
        _S["raw"], _S["img_bgr"] = raw, img_bgr
        self._json(_render(img_bgr, _filter(raw, params)))


def main():
    ap = argparse.ArgumentParser(description="FCOS detector-only web tester")
    from pathlib import Path as _P
    _steps = sorted(_P("data/checkpoints/detector_obj_v5").glob("step_*"))
    ap.add_argument("--ckpt", default=str(_steps[-1] / "model.pt") if _steps
                    else "data/checkpoints/detector_obj_v5/model.pt")
    ap.add_argument("--profile", default="single")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    det = build_obj_detector(args.profile)
    state = torch.load(args.ckpt, map_location="cpu")
    det.load_state_dict(state.get("model", state))
    det.eval().to(args.device)
    _S.update(det=det, size=args.size, device=args.device)
    print(f"[detector-tester] {args.profile} loaded ({det.num_classes} classes), size={args.size}, "
          f"device={args.device}")
    print(f"[detector-tester] http://localhost:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
