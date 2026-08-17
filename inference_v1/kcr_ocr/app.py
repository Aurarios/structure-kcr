"""KCR OCR Studio - a lightweight web UI for the two-stage Khmer OCR pipeline.

A debugging-first OCR studio. Left panel: the uploaded page with a toggle to overlay the detector's
numbered line boxes (so you can see exactly which box produced which text). Right panel: tabbed
Markdown / HTML / Text output plus a **Lines** tab listing every numbered line's transcription and
predicted region class -> trivial to spot a mis-detected / mis-read line.

Pure Python stdlib (http.server) -> no extra deps, no framework. Runs the pipeline on CPU by default
so it won't fight a training GPU.

  python -m src.serve.app --det-ckpt data/checkpoints/detector_single/step_00207730/model.pt \
      --rec-ckpt data/checkpoints/recognizer_single/step_00040000/model.pt --port 8000
  # then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image

from .pipeline import load_models, run_ocr

try:
    from .build_det_targets import CLASSES
except Exception:
    CLASSES = ["background", "title", "section_header", "text", "list_item",
               "caption", "form_label", "form_value", "table_cell"]

_STATE = {}   # det, rec, enc, det_cfg, device

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>KCR OCR Studio</title>
<style>
 *{box-sizing:border-box;font-family:ui-sans-serif,system-ui,'Khmer OS',sans-serif}
 body{margin:0;background:#0f1115;color:#e6e6e6}
 header{background:#1a1d24;padding:12px 20px;display:flex;align-items:center;gap:14px;
        border-bottom:1px solid #2a2e37}
 header b{font-weight:600;letter-spacing:.3px}
 header .meta{margin-left:auto;font-size:12px;color:#8b95a3}
 .wrap{display:flex;height:calc(100vh - 49px)}
 .left{width:46%;border-right:1px solid #2a2e37;display:flex;flex-direction:column;background:#15171d}
 .right{flex:1;display:flex;flex-direction:column;min-width:0}
 .bar{display:flex;align-items:center;gap:8px;padding:9px 14px;border-bottom:1px solid #2a2e37;background:#1a1d24}
 .tab{padding:6px 13px;border-radius:6px;cursor:pointer;background:#262a33;font-size:13px;user-select:none}
 .tab.active{background:#3b82f6;color:#fff}
 .status{margin-left:auto;font-size:13px;color:#9aa4b2}
 .ok{color:#34d399;font-weight:600}.busy{color:#fbbf24;font-weight:600}.err{color:#f87171;font-weight:600}
 .btn{padding:6px 12px;border-radius:6px;background:#262a33;border:1px solid #353a44;color:#e6e6e6;cursor:pointer;font-size:13px}
 .btn:hover{background:#313641}
 .toggle{display:flex;align-items:center;gap:6px;font-size:13px;color:#9aa4b2;cursor:pointer}
 .stage{flex:1;overflow:auto;display:flex;align-items:flex-start;justify-content:center;padding:14px}
 .stage img{max-width:100%;border-radius:6px;box-shadow:0 0 0 1px #2a2e37}
 .drop{margin:16px;flex:1;border:2px dashed #353a44;border-radius:10px;display:flex;align-items:center;
       justify-content:center;color:#8b95a3;cursor:pointer;text-align:center;padding:20px}
 .out{flex:1;overflow:auto;padding:18px 22px;white-space:pre-wrap;line-height:1.9;font-size:15px}
 .out table{border-collapse:collapse;margin:10px 0;width:100%}
 .out th,.out td{border:1px solid #353a44;padding:6px 10px;text-align:left}
 .out h1{font-size:1.5em;margin:.4em 0}.out h2{font-size:1.2em;margin:.4em 0}
 .out.frameview{padding:0}
 .out iframe{width:100%;height:100%;border:0;border-radius:6px;background:#fff}
 pre.code{font-family:ui-monospace,monospace;font-size:13px;color:#cbd5e1;white-space:pre-wrap}
 .lines{padding:8px 0}
 .lrow{display:flex;gap:10px;padding:7px 16px;border-bottom:1px solid #20242c;font-size:14px}
 .lrow:hover{background:#1a1d24}
 .lidx{color:#3b82f6;font-variant-numeric:tabular-nums;min-width:30px;font-family:ui-monospace,monospace}
 .lcls{color:#8b95a3;font-size:11px;min-width:96px;text-transform:uppercase;letter-spacing:.4px;padding-top:2px}
 .ltxt{flex:1}
</style></head><body>
<header><b>KCR OCR Studio</b> <span style="color:#8b95a3;font-size:13px">Khmer two-stage OCR &mdash; debug</span>
 <span class="meta" id="meta"></span></header>
<div class="wrap">
  <div class="left">
    <div class="bar"><span>Document</span>
      <label class="toggle" style="margin-left:auto"><input type="checkbox" id="ov" checked> Detection overlay</label>
      <button class="btn" id="reset">New</button>
    </div>
    <label class="drop" id="drop"><span id="hint">Click to upload a document image</span>
      <input id="file" type="file" accept="image/*" style="display:none"></label>
    <div class="stage" id="stage" style="display:none"><img id="prev"></div>
  </div>
  <div class="right">
    <div class="bar">
      <div class="tab active" data-v="markdown">Markdown</div>
      <div class="tab" data-v="html">HTML</div>
      <div class="tab" data-v="text">Text</div>
      <div class="tab" data-v="lines">Lines</div>
      <span class="status">Status: <span id="status" class="ok">idle</span></span>
      <button class="btn" id="copy">Copy</button>
    </div>
    <div class="out" id="out">Upload a document to begin.</div>
  </div>
</div>
<script>
let RES=null, view='markdown', ORIG=null;
const out=document.getElementById('out'), status=document.getElementById('status'),
      prev=document.getElementById('prev'), stage=document.getElementById('stage'),
      drop=document.getElementById('drop'), meta=document.getElementById('meta');
drop.onclick=()=>document.getElementById('file').click();
document.getElementById('reset').onclick=()=>{RES=null;ORIG=null;stage.style.display='none';
  drop.style.display='flex';out.textContent='Upload a document to begin.';meta.textContent='';
  status.textContent='idle';status.className='ok';};
document.getElementById('ov').onchange=showImg;
document.getElementById('file').onchange=async e=>{
  const f=e.target.files[0]; if(!f) return;
  const r=new FileReader(); r.onload=()=>{ORIG=r.result;}; r.readAsDataURL(f);
  drop.style.display='none'; stage.style.display='flex';
  status.textContent='Processing...'; status.className='busy'; out.textContent='Running detector + recognizer...';
  const fd=new FormData(); fd.append('image',f);
  try{
    const resp=await fetch('/ocr',{method:'POST',body:fd}); RES=await resp.json();
    if(RES.error){status.textContent='error'; status.className='err'; out.textContent=RES.error; return;}
    status.textContent='done ('+RES.seconds+'s)'; status.className='ok';
    meta.textContent=RES.lines+' lines  ·  '+RES.blocks+' blocks';
    showImg(); render();
  }catch(err){status.textContent='error'; status.className='err'; out.textContent=String(err);}
};
function showImg(){
  if(!RES&&!ORIG) return;
  prev.src=(document.getElementById('ov').checked && RES && RES.overlay)?RES.overlay:ORIG;
}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
const FRAME_CSS="<!doctype html><meta charset='utf-8'><style>"+
  "body{font-family:system-ui,'Khmer OS',sans-serif;margin:18px;color:#111;background:#fff;line-height:1.85}"+
  "table{border-collapse:collapse;margin:12px 0}th,td{border:1px solid #999;padding:6px 11px;text-align:left}"+
  "th{background:#f0f0f0}h1{font-size:1.6em;margin:.4em 0}h2{font-size:1.25em;margin:.4em 0}</style>";
function render(){
  if(!RES){return;}
  out.classList.toggle('lines', view==='lines');
  out.classList.toggle('frameview', view==='html');
  if(view==='html'){                                       // rendered HTML in an isolated iframe
    out.innerHTML='<iframe id="fr"></iframe>';
    document.getElementById('fr').srcdoc=FRAME_CSS+RES.html;
  }
  else if(view==='markdown'){out.innerHTML=RES.html;}      // structured visual render
  else if(view==='text'){out.textContent=RES.text;}
  else { // lines: numbered, class-tagged, matches overlay box numbers
    out.innerHTML='<div class="lines">'+RES.linelist.map(l=>
      '<div class="lrow"><span class="lidx">'+l.i+'</span>'+
      '<span class="lcls">'+l.cls+'</span><span class="ltxt">'+esc(l.text)+'</span></div>').join('')+'</div>';
  }
}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active'); view=t.dataset.v; render();
});
document.getElementById('copy').onclick=()=>{
  if(!RES) return;
  const txt=view==='text'?RES.text:(view==='html'?RES.html:(view==='lines'?
    RES.linelist.map(l=>'['+l.i+'] '+l.text).join('\\n'):RES.markdown));
  navigator.clipboard.writeText(txt);
};
</script></body></html>"""


def _overlay_b64(res) -> str:
    """Draw numbered detection boxes on the DPI-normalized image -> base64 data URL (JPEG)."""
    im = cv2.cvtColor(np.asarray(res["norm_image"].convert("RGB")), cv2.COLOR_RGB2BGR)
    for i, u in enumerate(res["line_units"]):
        x1, y1, x2, y2 = [int(v) for v in u["box"]]
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(im, str(i), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _cls_name(u) -> str:
    cid = u.get("cls", 0)
    return CLASSES[cid] if isinstance(cid, int) and 0 <= cid < len(CLASSES) else str(cid)


def _fig_data_url(crop) -> str:
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _embed_figures(html: str, figures) -> str:
    """Swap the <img data-figure="N"> placeholders for the actual cropped region (data URL)."""
    for f in figures:
        url = _fig_data_url(f["crop"])
        html = html.replace(f'<img data-figure="{f["index"]}" alt="figure">',
                            f'<img src="{url}" alt="figure" style="max-width:100%;border:1px solid #ccc">')
    return html


def _dump_audit(res) -> None:
    """Write the LAST upload's normalized image, numbered overlay, and exact per-line box coords +
    text to data/manifests/studio_audit/ so detection issues can be diagnosed from real numbers."""
    from pathlib import Path
    d = Path("data/manifests/studio_audit"); d.mkdir(parents=True, exist_ok=True)
    norm = cv2.cvtColor(np.asarray(res["norm_image"].convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(d / "normalized.jpg"), norm, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ov = norm.copy()
    for i, u in enumerate(res["line_units"]):
        x1, y1, x2, y2 = [int(v) for v in u["box"]]
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(ov, str(i), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(str(d / "detection.jpg"), ov)
    w, h = res["norm_image"].size
    lines = [f"page: {w}x{h}  lines={len(res['line_units'])}"]
    for i, u in enumerate(res["line_units"]):
        b = [int(v) for v in u["box"]]
        lines.append(f"[{i:02d}] box={b} cls={_cls_name(u)} text={u['text']!r}")
    (d / "lines.txt").write_text("\n".join(lines), encoding="utf-8")
    # save cropped image/formula regions ("crop along the way")
    figs = res.get("figures", [])
    if figs:
        fd = d / "figures"; fd.mkdir(exist_ok=True)
        for f in figs:
            cv2.imwrite(str(fd / f"fig_{f['index']:02d}_{f['class_name']}.jpg"), f["crop"])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path != "/ocr":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        # crude multipart parse: find the image payload between boundaries
        try:
            boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
            part = body.split(b"--" + boundary)[1]
            payload = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
            img = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as e:
            self._json({"error": f"bad upload: {e}"}, 400); return

        t0 = time.time()
        try:
            res = run_ocr(img, _STATE["det"], _STATE["rec"], _STATE["enc"],
                          _STATE["det_cfg"], _STATE["device"])
        except Exception as e:
            self._json({"error": f"ocr failed: {e}"}, 500); return

        _dump_audit(res)
        self._json({
            "markdown": res["markdown"], "html": _embed_figures(res["html"], res.get("figures", [])),
            "text": res["text"],
            "overlay": _overlay_b64(res),
            "linelist": [{"i": i, "text": u["text"], "cls": _cls_name(u)}
                         for i, u in enumerate(res["line_units"])],
            "blocks": len(res["block_units"]), "lines": len(res["line_units"]),
            "seconds": round(time.time() - t0, 1),
        })

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    from pathlib import Path as _P
    _ASSETS = _P(__file__).resolve().parent.parent / "assets" / "checkpoints"
    ap = argparse.ArgumentParser(description="KCR OCR Studio web UI")
    ap.add_argument("--det-ckpt", default=str(_ASSETS / "detector.pt"))
    ap.add_argument("--rec-ckpt", default=str(_ASSETS / "recognizer.pt"))
    ap.add_argument("--det-profile", default="single")
    ap.add_argument("--rec-profile", default="single")
    ap.add_argument("--det-size", type=int, default=1536)
    ap.add_argument("--device", default="cpu")   # cpu so it won't fight a training GPU
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print("[serve] loading models...")
    det, rec, enc = load_models(args.det_ckpt, args.rec_ckpt, args.det_profile,
                                args.rec_profile, args.device)
    _STATE.update(det=det, rec=rec, enc=enc, device=args.device,
                  det_cfg={"image_size": args.det_size, "prob_thresh": 0.3,
                           "box_unclip_ratio": 2.0, "min_box_size": 4,
                           "min_confidence": 0.5, "upscale_width": 2400})
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] KCR OCR Studio on http://localhost:{args.port}  (device={args.device})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
