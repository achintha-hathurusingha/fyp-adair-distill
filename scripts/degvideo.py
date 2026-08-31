"""720p preview: degradation type CHANGES over time, one at a time, and a
single all-in-one student restores it without ever being told which.

Sources are real held-out pairs (Rain100L rainy/gt, SOTS hazy/clear) so the
model sees in-distribution inputs; only the noise segment is synthesised
(sigma=25), which is how noise is generated in our pipeline anyway.

Source images top out at ~550x978, so panels are upscaled ~1.2x to fill 720p.
Stated rather than hidden -- this is a qualitative demo, not a benchmark.
"""
import sys, os, glob
sys.path.insert(0, ".")
import numpy as np, torch, cv2, yaml
from src.models.nafnet import build_nafnet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, H, FPS, SEG = 1280, 720, 24, 4.0          # 3 segments x 4s = 12s
PW, PH = 620, 462                             # 4:3 panels fill the 720p frame
CFG = dict(width=16, enc_blk_nums=[2,2,4,8], middle_blk_num=12, dec_blk_nums=[2,2,2,2],
           norm_type="layernorm2d", full_res_norm_type="affine_clamp",
           clamp_bound=8.0, enc_clamp_stages=[3], deep_clamp_bound=32.0)
DATA = yaml.safe_load(open("configs/paths.local.yaml"))["data_root"]
OUT = "reports/student_v3/degradation_timeline_720p.mp4"

ck = sorted(glob.glob("runs/b0v2_kd_feat/B0V2-KD-FEAT/B0V2-KD-FEAT_seed0_*/best.pth"))[-1]
m = build_nafnet(CFG)
sd = torch.load(ck, map_location="cpu", weights_only=False)
sd = sd.get("model", sd.get("state_dict", sd))
m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
m = m.eval().to(DEV)
print("model:", os.path.basename(os.path.dirname(ck)), flush=True)

def rd(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

def crop169(a, z, n=3):
    """Ken Burns: crop a panel-aspect window that slowly zooms in."""
    h, w = a.shape[:2]
    AR = PW / PH
    ch = int(w / AR)
    if ch > h: ch = h; cw = int(h * AR)
    else: cw = w
    s = 1.0 - 0.12 * z                                   # zoom 100% -> 88%
    cw2, ch2 = int(cw * s), int(ch * s)
    x = int((w - cw2) * 0.5); y = int((h - ch2) * (0.35 + 0.15 * z))
    return cv2.resize(a[y:y+ch2, x:x+cw2], (PW, PH), interpolation=cv2.INTER_AREA
                      if cw2 > PW else cv2.INTER_CUBIC)

# ---- real held-out pairs, one list per segment -----------------------------
segs = []
r_in = sorted(glob.glob(f"{DATA}/test/derain/demo/input/*"))[:6]
segs.append(("RAIN", "Rain100L (real pairs)",
             [(rd(p), rd(p.replace("/input/", "/target/"))) for p in r_in]))
h_in = sorted(glob.glob(f"{DATA}/test/dehaze/demo/input/*"))[:6]
h_gt = sorted(glob.glob(f"{DATA}/test/dehaze/demo/target/*"))[:6]
segs.append(("HAZE", "RESIDE SOTS-outdoor (real pairs)",
             [(rd(a), rd(b)) for a, b in zip(h_in, h_gt)]))
n_gt = sorted(glob.glob(f"{DATA}/test/denoise/bsd68/*"))[:6]
rng = np.random.default_rng(0)
segs.append(("NOISE", "BSD68 + Gaussian sigma=25",
             [(np.clip(rd(p) + rng.standard_normal(rd(p).shape) * 25, 0, 255).astype(np.uint8),
               rd(p)) for p in n_gt]))
print("segments:", [(s[0], len(s[2])) for s in segs], flush=True)

def restore(img):
    t = torch.from_numpy(img.astype(np.float32) / 255.).permute(2,0,1)[None].to(DEV)
    with torch.no_grad(): o = m(t)
    return (o[0].clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

def psnr(a, b):
    e = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if e <= 1e-12 else 10 * np.log10(255.0**2 / e)

INK, DIMC, ACC = (28,20,16), (114,100,90), (122,111,31)
OKC, BADC = (91,125,46), (58,66,166)                      # BGR
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
F = cv2.FONT_HERSHEY_SIMPLEX
nseg = int(SEG * FPS)

for si, (tag, src, pairs) in enumerate(segs):
    per = max(1, nseg // len(pairs))
    for fi in range(nseg):
        idx = min(fi // per, len(pairs) - 1)
        z = (fi % per) / max(per - 1, 1)
        deg_full, gt_full = pairs[idx]
        deg = crop169(deg_full, z); gt = crop169(gt_full, z)
        out = restore(deg)

        fr = np.full((H, W, 3), 247, np.uint8)
        cv2.rectangle(fr, (0,0), (W,74), (250,248,246), -1)
        cv2.line(fr, (0,74), (W,74), (232,226,221), 1)
        cv2.putText(fr, "ONE all-in-one student, degradation changing over time",
                    (28,32), F, 0.62, INK, 1, cv2.LINE_AA)
        cv2.putText(fr, "the model is never told which degradation is present",
                    (28,58), F, 0.46, DIMC, 1, cv2.LINE_AA)

        # segment timeline
        for k,(t2,_,_) in enumerate(segs):
            x0 = 700 + k*190
            on = (k == si)
            cv2.rectangle(fr, (x0,20), (x0+178,52), ACC if on else (238,233,229), -1)
            cv2.putText(fr, t2, (x0+14,42), F, 0.52,
                        (255,255,255) if on else DIMC, 2 if on else 1, cv2.LINE_AA)

        y0 = 130
        fr[y0:y0+PH, 12:12+PW] = deg[:, :, ::-1]
        fr[y0:y0+PH, 12+PW+24:12+2*PW+24] = out[:, :, ::-1]
        for x,lab,col in ((12,"DEGRADED INPUT",BADC), (12+PW+24,"RESTORED",OKC)):
            cv2.rectangle(fr, (x,y0-30), (x+PW,y0-2), (240,236,232), -1)
            cv2.putText(fr, lab, (x+10,y0-10), F, 0.5, col, 1, cv2.LINE_AA)
            cv2.rectangle(fr, (x,y0), (x+PW,y0+PH), (225,219,214), 1)

        pi, po = psnr(gt, deg), psnr(gt, out)
        yb = y0 + PH + 34
        cv2.putText(fr, f"input {pi:5.2f} dB", (12,yb), F, 0.6, BADC, 1, cv2.LINE_AA)
        cv2.putText(fr, f"restored {po:5.2f} dB", (12+PW+24,yb), F, 0.6, OKC, 2, cv2.LINE_AA)
        cv2.putText(fr, f"+{po-pi:.2f} dB", (12+PW+24+210,yb), F, 0.6, ACC, 2, cv2.LINE_AA)
        cv2.putText(fr, f"source: {src}", (12,yb+28), F, 0.42, DIMC, 1, cv2.LINE_AA)
        cv2.putText(fr, "B0V2-KD-FEAT  7.37M params  |  panels upscaled ~1.2x from source",
                    (12,yb+52), F, 0.42, DIMC, 1, cv2.LINE_AA)
        vw.write(fr)
    print(f"  {tag} done", flush=True)

vw.release()
print("wrote", OUT, os.path.getsize(OUT)//1024, "KB")
