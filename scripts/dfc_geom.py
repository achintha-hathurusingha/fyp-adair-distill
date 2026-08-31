"""Does the band-mask GEOMETRY matter? Radial (DFC) vs rectangular (AdaIR).

Same DFC ratio (Eq.4-5), same data, same classifier -- only the band mask
changes:

  RADIAL      rho = sqrt(u^2+v^2)     circular annuli   (DFC / our own work)
  SQUARE      rho = max(|u|,|v|)      square annuli     (AdaIR's box geometry)
  SEPARABLE   independent u and v bands, concatenated   (AdaIR's alpha,beta
                                                         degree of freedom)

A square passes alpha on the axes but alpha*sqrt(2) on the diagonal, and a
non-square rectangle is orientation-selective along the cardinal axes -- a
degree of freedom the radial form discards. Rain is anisotropic, so if that
freedom is worth anything it should show up on rain specifically.
"""
import sys, os, glob
sys.path.insert(0, ".")
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = yaml.safe_load(open("configs/paths.local.yaml"))["data_root"]
OUT = "reports/dfc"; os.makedirs(OUT, exist_ok=True)
B, SIZE, SIG = 24, 256, 0.035
rng = np.random.default_rng(0)

def load(p, n=SIZE):
    a = np.asarray(Image.open(p).convert("RGB"), np.float64)/255.
    h, w = a.shape[:2]; y, x = max(0,(h-n)//2), max(0,(w-n)//2)
    a = a[y:y+n, x:x+n]
    return a if a.shape[:2]==(n,n) else np.asarray(
        Image.fromarray((a*255).astype(np.uint8)).resize((n,n)), np.float64)/255.

fy = np.fft.fftshift(np.fft.fftfreq(SIZE))[:, None]
fx = np.fft.fftshift(np.fft.fftfreq(SIZE))[None, :]
U, V = np.abs(fy)/np.abs(fy).max(), np.abs(fx)/np.abs(fx).max()
mu = np.linspace(0, 1, B)

def ring(rho):                       # Gaussian annuli over any radius field
    return np.stack([np.exp(-(rho-m)**2/(2*SIG**2)) for m in mu])

RAD  = ring(np.sqrt((U*np.ones_like(V))**2 + (V*np.ones_like(U))**2)/np.sqrt(2))
SQ   = ring(np.maximum(U*np.ones_like(V), V*np.ones_like(U)))          # L_inf
SEP_U = ring(U*np.ones_like(V)); SEP_V = ring(V*np.ones_like(U))       # separable

def power(img):
    P = 0
    for c in range(3):
        P = P + np.abs(np.fft.fftshift(np.fft.fft2(img[:,:,c])))**2
    return P

def dfc_with(masks, clean, deg, eps=1e-12):
    Pr, Py = power(deg-clean), power(deg)
    Rt = (masks*Pr[None]).sum((1,2)) / ((masks*Py[None]).sum((1,2))+eps)
    return Rt/(Rt.sum()+eps)

cases = {}
r_in = sorted(glob.glob(f"{DATA}/test/derain/demo/input/*"))[:40]
cases["rain"] = [(load(p.replace("/input/","/target/")), load(p)) for p in r_in]
h_in = sorted(glob.glob(f"{DATA}/test/dehaze/demo/input/*"))[:40]
h_gt = sorted(glob.glob(f"{DATA}/test/dehaze/demo/target/*"))[:40]
cases["haze"] = [(load(b), load(a)) for a,b in zip(h_in,h_gt)]
cases["noise"] = [(lambda c:(c,np.clip(c+rng.standard_normal(c.shape)*(25/255.),0,1)))(load(p))
                  for p in sorted(glob.glob(f"{DATA}/test/denoise/bsd68/*"))[:40]]

feats = {"radial":[], "square":[], "separable":[]}; y=[]
for i,(k,pairs) in enumerate(cases.items()):
    for clean,deg in pairs:
        feats["radial"].append(dfc_with(RAD, clean, deg))
        feats["square"].append(dfc_with(SQ, clean, deg))
        feats["separable"].append(np.concatenate([dfc_with(SEP_U,clean,deg),
                                                  dfc_with(SEP_V,clean,deg)]))
        y.append(i)
    print(f"  {k} done", flush=True)
y=np.array(y); feats={k:np.array(v) for k,v in feats.items()}

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

def evaluate(X):
    cv=StratifiedKFold(5,shuffle=True,random_state=0); acc=[]; C=np.zeros((3,3))
    for tr,te in cv.split(X,y):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000))
        m.fit(X[tr],y[tr]); p=m.predict(X[te])
        acc.append((p==y[te]).mean()); C+=confusion_matrix(y[te],p,labels=[0,1,2])
    return float(np.mean(acc)), float(np.std(acc)), C

names=list(cases); res={}
print(f"\n{'geometry':11s} {'overall':>16s} | per-class recall")
for g in ("radial","square","separable"):
    a,s,C = evaluate(feats[g]); res[g]=(a,s,C)
    rec = C.diagonal()/C.sum(1)
    print(f"{g:11s} {a*100:8.1f}% +/-{s*100:4.1f} | " +
          "  ".join(f"{n} {r*100:5.1f}%" for n,r in zip(names,rec)))

fig,ax=plt.subplots(1,2,figsize=(12.5,4.4),dpi=170)
COL={"radial":"#4A5A6B","square":"#A6423A","separable":"#2E7D5B"}
gs=list(res); xs=np.arange(len(gs))
ax[0].bar(xs,[res[g][0]*100 for g in gs],yerr=[res[g][1]*100 for g in gs],
          color=[COL[g] for g in gs],width=.55,capsize=4)
for i,g in enumerate(gs):
    ax[0].text(i,res[g][0]*100+1.4,f"{res[g][0]*100:.1f}%",ha="center",fontweight="bold")
ax[0].set_xticks(xs); ax[0].set_xticklabels(["radial\n(DFC)","square\n(AdaIR box)","separable\n(AdaIR α,β)"])
ax[0].axhline(33.3,color="#8A7E5C",ls="--",lw=1); ax[0].text(2.35,34.5,"chance",fontsize=8,color="#8A7E5C")
ax[0].set_ylim(0,108); ax[0].set_ylabel("degradation ID accuracy (%)")
ax[0].set_title("Band-mask geometry, same DFC ratio and data",fontsize=11,fontweight="bold")
ax[0].grid(axis="y",alpha=.25)

w=0.26
for j,g in enumerate(gs):
    rec=res[g][2].diagonal()/res[g][2].sum(1)
    ax[1].bar(np.arange(3)+(j-1)*w,rec*100,w,color=COL[g],
              label=["radial","square","separable"][j])
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names)
ax[1].set_ylim(0,112); ax[1].set_ylabel("per-class recall (%)")
ax[1].set_title("Where the geometry matters",fontsize=11,fontweight="bold")
ax[1].legend(fontsize=8.5); ax[1].grid(axis="y",alpha=.25)
fig.suptitle("Radial vs rectangular frequency bands — DFC's geometry vs AdaIR's",
             fontsize=12.5,fontweight="bold",y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/dfc_geometry.png",facecolor="white",bbox_inches="tight")
print(f"\nwrote {OUT}/dfc_geometry.png")
