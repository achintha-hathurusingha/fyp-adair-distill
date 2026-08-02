"""Per-image strips for the real-world demo set.

Two figures per image:
  <stem>_synthetic.png  noisy / AdaIR / B0 FP32 / B0 INT8 / ground truth, per sigma
  <stem>_native.png     original JPEG / AdaIR / B0 -- NO ground truth exists
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path("runs/demo_nb"); REAL = OUT / "real"
STRIPS = OUT / "strips"; STRIPS.mkdir(parents=True, exist_ok=True)
rw = json.loads((OUT / "real_world.json").read_text())
i8 = json.loads((OUT / "int8_real.json").read_text())
SIGMAS = (15, 25, 50)

def font(sz):
    for n in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(n, sz)
        except Exception: pass
    return ImageFont.load_default()
F_H, F_L, F_T = font(19), font(17), font(25)

def get(rows, stem, sg, key):
    for r in rows:
        if r.get("image", "").startswith(stem) and r.get("sigma") == sg and key in r:
            return r[key]
    return None

stems = sorted({p.name.split("_clean")[0] for p in REAL.glob("*_clean.npy")})
TH, PAD, HDR, LAB, TIT = 320, 8, 30, 26, 46

for stem in stems:
    clean = np.load(REAL / f"{stem}_clean.npy")
    def rs(a): return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((TH, TH), Image.LANCZOS))

    # --- synthetic: 5 columns x 3 sigma rows ---
    cols = ["degraded", "AdaIR 28.8M", "B0 FP32 7.4M", "B0 INT8 on S24", "ground truth"]
    W = len(cols)*TH + (len(cols)+1)*PAD
    H = TIT + HDR + len(SIGMAS)*(TH+LAB+PAD) + PAD
    cv = Image.new("RGB", (W, H), (16,17,20)); d = ImageDraw.Draw(cv)
    d.text((PAD, 10), f"{stem} — synthetic noise (our pipeline), ground truth known", font=F_T, fill=(240,240,245))
    for i, c in enumerate(cols):
        col = (150,220,255) if "INT8" in c else ((255,200,140) if "AdaIR" in c else (215,215,220))
        d.text((PAD+i*(TH+PAD)+3, TIT+5), c, font=F_H, fill=col)
    y = TIT+HDR
    for sg in SIGMAS:
        tiles = [np.load(REAL/f"{stem}_s{sg}_noisy.npy"), np.load(REAL/f"{stem}_s{sg}_adair.npy"),
                 np.load(REAL/f"{stem}_s{sg}_b0.npy")]
        p8 = REAL/f"{stem}_s{sg}_int8.npy"
        tiles.append(np.load(p8) if p8.exists() else np.zeros_like(clean))
        tiles.append(clean)
        labs = [f"sigma {sg}  {get(rw,stem,sg,'noisy_psnr') or 0:.2f} dB",
                f"{get(rw,stem,sg,'adair_psnr') or 0:.2f} dB",
                f"{get(rw,stem,sg,'b0_psnr') or 0:.2f} dB",
                f"{get(i8,stem,sg,'int8_psnr') or 0:.2f} dB", "reference"]
        cl = [(200,200,205),(255,200,140),(215,215,220),(150,220,255),(200,200,205)]
        for i,(a,l,c) in enumerate(zip(tiles,labs,cl)):
            x = PAD+i*(TH+PAD); cv.paste(Image.fromarray(rs(a)), (x,y))
            d.text((x+3, y+TH+3), l, font=F_L, fill=c)
        y += TH+LAB+PAD
    cv.save(STRIPS/f"{stem}_synthetic.png")

    # --- native: 3 columns, NO ground truth ---
    cols2 = ["original JPEG (as downloaded)", "AdaIR 28.8M", "B0 FP32 7.4M"]
    W2 = len(cols2)*TH + (len(cols2)+1)*PAD; H2 = TIT+HDR+TH+LAB+2*PAD
    c2 = Image.new("RGB", (W2,H2), (16,17,20)); d2 = ImageDraw.Draw(c2)
    d2.text((PAD,10), f"{stem} — native degradation, NO ground truth (no PSNR possible)",
            font=F_T, fill=(240,240,245))
    for i,c in enumerate(cols2):
        col = (255,200,140) if "AdaIR" in c else (215,215,220)
        d2.text((PAD+i*(TH+PAD)+3, TIT+5), c, font=F_H, fill=col)
    for i,a in enumerate([clean, np.load(REAL/f"{stem}_native_adair.npy"),
                          np.load(REAL/f"{stem}_native_b0.npy")]):
        x = PAD+i*(TH+PAD); c2.paste(Image.fromarray(rs(a)), (x, TIT+HDR))
    c2.save(STRIPS/f"{stem}_native.png")

print(f"wrote {len(list(STRIPS.glob('*.png')))} strips to {STRIPS}")
