import json

arms = {
    "B0V2-KD-FEAT": "runs/b0v2_kd_feat/B0V2-KD-FEAT/B0V2-KD-FEAT_seed0_20260828_193951/history.json",
    "B0V3-s0":      "runs/b0v3/B0V3/B0V3_seed0_20260830_141103/history.json",
    "B0V3-s1":      "runs/b0v3/B0V3/B0V3_seed1_20260830_180851/history.json",
    "B0V3-KD-FEAT": "runs/b0v3_kd_feat/B0V3-KD-FEAT/B0V3-KD-FEAT_seed0_20260831_083259/history.json",
}

data = {}
for a, f in arms.items():
    h = json.load(open(f))
    data[a] = {e["iteration"]: e for e in h}
    last = h[-1]
    print("%-14s %2d entries, last it=%6d  combined=%.3f" % (a, len(h), last["iteration"], last["psnr"]))

print()
print("COMBINED PSNR by iteration")
print("%7s %14s %14s %14s %14s" % ("it", *arms.keys()))
its = sorted(set().union(*[set(d) for d in data.values()]))
for it in its:
    cells = []
    for a in arms:
        e = data[a].get(it)
        cells.append("%.3f" % e["psnr"] if e else "-")
    print("%7d %14s %14s %14s %14s" % (it, *cells))

print()
print("PER-TASK at matched iterations")
for it in [69000, 75000, 90000]:
    print("\n--- it %d ---" % it)
    print("%-14s %9s %9s %9s %9s" % ("arm", "combined", "denoise", "derain", "dehaze"))
    for a in arms:
        e = data[a].get(it)
        if not e:
            print("%-14s %9s" % (a, "(none)"))
            continue
        print("%-14s %9.3f %9.3f %9.3f %9.3f" % (
            a, e["psnr"], e["psnr_denoise"], e["psnr_derain"], e["psnr_dehaze"]))
    base = data["B0V3-KD-FEAT"].get(it)
    if base:
        for a in arms:
            if a == "B0V3-KD-FEAT":
                continue
            e = data[a].get(it)
            if not e:
                continue
            print("  delta KD-FEAT - %-13s %+7.3f  (dn %+.3f  rn %+.3f  hz %+.3f)" % (
                a, base["psnr"] - e["psnr"],
                base["psnr_denoise"] - e["psnr_denoise"],
                base["psnr_derain"] - e["psnr_derain"],
                base["psnr_dehaze"] - e["psnr_dehaze"]))

print()
print("B0V3 seed0 vs seed1 (noise floor check, shared iterations)")
common = sorted(set(data["B0V3-s0"]) & set(data["B0V3-s1"]))
diffs = []
for it in common:
    d = abs(data["B0V3-s0"][it]["psnr"] - data["B0V3-s1"][it]["psnr"])
    diffs.append((it, d))
    print("  it %6d  s0=%.3f  s1=%.3f  |diff|=%.3f" % (
        it, data["B0V3-s0"][it]["psnr"], data["B0V3-s1"][it]["psnr"], d))
late = [d for it, d in diffs if it >= 51000]
if late:
    print("  mean |diff| over it>=51000: %.4f dB  (n=%d)" % (sum(late)/len(late), len(late)))
