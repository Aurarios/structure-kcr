"""Bind data_formulation.html to REAL labels: pick one representative page per layout from E:/kcr-v3,
embed its label geometry + a downscaled base64 image, and fill the __LAYOUTS_DATA__ placeholder."""
import base64, glob, json, os
import cv2

BASE = "E:/kcr-v3"
LAYOUTS = ["book_page", "contract_legal", "exam_paper", "id_card", "receipt_invoice"]
HTML = "data_formulation.html"
DISP_W = 720          # downscale embedded image to this width (boxes use original dims -> still align)
MAXTXT = 200

# legacy block_type -> canonical (mirror of build_obj_targets._ALIAS_V4)
ALIAS = {"section_header": "heading", "byline": "caption", "form": "form_value", "list": "list_item",
         "figure": "image", "photo": "image", "logo": "image", "stamp": "image", "seal": "image",
         "table_region": "table"}


def pick_label(layout):
    files = sorted(glob.glob(f"{BASE}/{layout}/labels/*.json"))[:120]
    best, best_score = None, -1
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        blks = [b for b in d.get("blocks", []) if b.get("bbox") and len(b["bbox"]) == 4]
        if not (5 <= len(blks) <= 40):
            continue
        distinct = len({ALIAS.get(b.get("block_type"), b.get("block_type")) for b in blks})
        score = distinct * 100 - abs(len(blks) - 18)   # max variety, ~18 blocks
        if score > best_score:
            best, best_score = (f, d), score
    return best


def img_b64(layout, label_id):
    p = f"{BASE}/{layout}/images/{label_id}.jpg"
    if not os.path.exists(p):
        g = glob.glob(f"{BASE}/{layout}/images/{label_id}.*")
        p = g[0] if g else None
    if not p:
        return ""
    im = cv2.imread(p)
    if im is None:
        return ""
    h, w = im.shape[:2]
    im = cv2.resize(im, (DISP_W, max(1, round(h * DISP_W / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def rb(b):
    return [round(float(v)) for v in b]


out = {}
for lay in LAYOUTS:
    picked = pick_label(lay)
    if not picked:
        print(f"[skip] {lay}: no suitable label"); continue
    f, d = picked
    lid = d.get("id") or os.path.splitext(os.path.basename(f))[0]
    blocks = []
    for b in d.get("blocks", []):
        bb = b.get("bbox")
        if not bb or len(bb) != 4:
            continue
        bt = ALIAS.get(b.get("block_type"), b.get("block_type"))
        lines = [rb(ln) for ln in (b.get("lines") or []) if ln and len(ln) == 4]
        blocks.append({"block_type": bt, "bbox": rb(bb), "lines": lines,
                       "text": (b.get("text") or "")[:MAXTXT],
                       "tbl": b.get("tbl")})
    out[lay] = {"image_width": d.get("image_width"), "image_height": d.get("image_height"),
                "img": img_b64(lay, lid), "blocks": blocks}
    print(f"[ok] {lay}: {os.path.basename(f)}  {len(blocks)} blocks  "
          f"{d.get('image_width')}x{d.get('image_height')}  img={len(out[lay]['img'])//1024}KB")

payload = json.dumps(out, ensure_ascii=False)
html = open(HTML, encoding="utf-8").read()
html = html.replace("__LAYOUTS_DATA__", payload)
open(HTML, "w", encoding="utf-8").write(html)
print(f"\n[done] wrote {HTML}  ({len(html)//1024} KB total, {len(out)} layouts)")
