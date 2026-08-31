"""Real inference: degraded -> student(s) -> teacher -> GT, per degradation,
with amplified error maps. Every pixel here is a real forward pass; nothing
is illustrative.

Runs on CPU by default so it cannot disturb the training occupying the GPU.
"""
import sys, os, glob
sys.path.insert(0, ".")
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.nafnet import build_nafnet
from src.models.student_v3 import build_student_v3
from src.models.teacher_wrapper import FrozenTeacher
import yaml

DEV = "cpu"
PATCH = 256
CFG = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2],
           norm_type="layernorm2d", full_res_norm_type="affine_clamp",
           clamp_bound=8.0, enc_clamp_stages=[3], deep_clamp_bound=32.0)
paths = yaml.safe_load(open("configs/paths.local.yaml"))
DATA = paths["data_root"]
W = paths["adair_weights_root"]
OUT = "reports/student_v3/visual_gaps.png"


def crop(a, n=PATCH):
    h, w = a.shape[:2]
    y, x = max(0, (h - n) // 2), max(0, (w - n) // 2)
    a = a[y:y + n, x:x + n]
    if a.shape[0] < n or a.shape[1] < n:
        a = np.array(Image.fromarray(a).resize((n, n)))
    return a


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def to_t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.).permute(2, 0, 1)[None].to(DEV)


def to_np(t):
    return (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m <= 1e-12 else 10 * np.log10(255.0 ** 2 / m)


def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval()


print("loading models...", flush=True)
kd_ck = sorted(glob.glob("runs/b0v2_kd_feat/B0V2-KD-FEAT/B0V2-KD-FEAT_seed0_*/best.pth"))[-1]
v3_ck = sorted(glob.glob("runs/b0v3/B0V3/B0V3_seed1_*/best.pth"))[-1]
m_kd = load_ckpt(build_nafnet(CFG), kd_ck)
m_v3 = load_ckpt(build_student_v3(CFG), v3_ck)
teacher = FrozenTeacher(os.path.join(W, "adair3d.ckpt"), device=DEV)
print(f"  KD student : {os.path.basename(os.path.dirname(kd_ck))}")
print(f"  v3 student : {os.path.basename(os.path.dirname(v3_ck))}")

# --- one real held-out sample per degradation ------------------------------
cases = []

r = sorted(glob.glob(f"{DATA}/test/derain/demo/input/*"))
if r:
    gtp = r[0].replace("/input/", "/target/")
    cases.append(("derain", crop(load(r[0])), crop(load(gtp))))

h = sorted(glob.glob(f"{DATA}/test/dehaze/demo/input/*"))
if h:
    hg = sorted(glob.glob(f"{DATA}/test/dehaze/demo/target/*"))
    cases.append(("dehaze", crop(load(h[0])), crop(load(hg[0]))))

n = sorted(glob.glob(f"{DATA}/test/denoise/bsd68/*"))
if n:
    cl = crop(load(n[0]))
    rng = np.random.default_rng(0)
    noisy = np.clip(cl + rng.standard_normal(cl.shape) * 25, 0, 255).astype(np.uint8)
    cases.append(("denoise σ=25", noisy, cl))

print(f"{len(cases)} cases: {[c[0] for c in cases]}", flush=True)

MODELS = [("KD student", m_kd), ("Student v3", m_v3), ("AdaIR teacher", teacher)]
rows = []
for name, deg, gt in cases:
    outs = []
    for mn, m in MODELS:
        with torch.no_grad():
            o = to_np(m(to_t(deg)))
        outs.append((mn, o, psnr(gt, o)))
    rows.append((name, deg, gt, outs, psnr(gt, deg)))
    print(f"  {name}: input {psnr(gt,deg):.2f} | " +
          " | ".join(f"{mn} {p:.2f}" for mn, _, p in outs), flush=True)

# --- figure: images on top, amplified error maps below ---------------------
ncol = 2 + len(MODELS)
fig, axes = plt.subplots(len(rows) * 2, ncol, figsize=(2.35 * ncol, 2.55 * len(rows) * 2), dpi=170)
if len(rows) == 1:
    axes = axes.reshape(2, ncol)

for ri, (name, deg, gt, outs, pin) in enumerate(rows):
    top, bot = axes[ri * 2], axes[ri * 2 + 1]
    panels = [("degraded input", deg, pin)] + [(mn, o, p) for mn, o, p in outs] + [("ground truth", gt, None)]
    for ci, (t, img, p) in enumerate(panels):
        top[ci].imshow(img)
        ttl = t if p is None else f"{t}\n{p:.2f} dB"
        top[ci].set_title(ttl, fontsize=9, fontweight="bold" if p is None else "normal",
                          color="#10141C")
        top[ci].axis("off")
        if ci == 0:
            top[ci].text(-0.08, 0.5, name, transform=top[ci].transAxes, rotation=90,
                         va="center", ha="center", fontsize=11, fontweight="bold",
                         color="#1F6F7A")
        # error map row
        if p is None:
            bot[ci].axis("off"); continue
        err = np.abs(img.astype(np.float32) - gt.astype(np.float32)).mean(2)
        im = bot[ci].imshow(err, cmap="inferno", vmin=0, vmax=40)
        bot[ci].set_title("|error| x amplified", fontsize=7.5, color="#5A6472")
        bot[ci].axis("off")
    bot[-1].axis("off")

fig.suptitle("Real inference on held-out images — same crop, same harness.  "
             "Error maps share one scale (0-40), so panels are directly comparable.",
             fontsize=11, fontweight="bold", y=0.995, color="#10141C")
fig.tight_layout(rect=[0.01, 0, 1, 0.985])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print(f"\nwrote {OUT}")
