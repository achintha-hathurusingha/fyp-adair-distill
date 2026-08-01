"""Mann-Kendall trend test on a run's clamp diagnostics.

PRE-COMMITTED. Written and committed BEFORE the data it judges had accumulated,
because this investigation twice read a trend out of sparse noisy data and twice
had to withdraw it ("doubling every ~5k", and the Q-A "escalation"). Fixing the
criteria in advance is the only defence against doing it a third time.

    python scripts/trend_test.py <run_dir>

Two series, same test:

  clamp_max_preclamp  the largest PRE-clamp magnitude in each interval
  clamp_engage_rate   how often the clamp fired in each interval

Reading (only the third row is evidence of a spreading problem):

  premax flat,   engage flat    -> stationary heavy tail; matches Q-A's own swings
  premax rising, engage flat    -> larger events, no more frequent; note it
  premax rising, engage rising  -> bigger AND more frequent; genuine intensification
  premax flat,   engage rising  -> unexpected; investigate before concluding

WHY MANN-KENDALL. The series is heavy-tailed, not normal — Q-A's comparable
`maxgn` swings three orders of magnitude between adjacent intervals — so an
ordinary least-squares slope is dominated by single large values. Mann-Kendall
is rank-based and only looks at the sign of pairwise differences.

NOTE ON THE LOG TRANSFORM. The brief asks for the test on log(premax). Because
Mann-Kendall depends solely on the SIGN of pairwise differences, and log is
strictly monotonic on positive values, MK(log x) and MK(x) are *identical* —
same S, same tau, same p. The log is applied for reporting readability; it
cannot change the verdict, and claiming it made the test more robust would be
wrong.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: Two-sided significance level, fixed in advance.
ALPHA = 0.05


def mann_kendall(x: list[float]) -> dict[str, float]:
    """Mann-Kendall trend test with tie correction.

    Returns S, Kendall's tau, the normal statistic Z, and a two-sided p-value.
    """
    n = len(x)
    if n < 4:
        raise ValueError(f"need at least 4 points for a meaningful test, got {n}")

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += (x[j] > x[i]) - (x[j] < x[i])

    # Variance, corrected for ties (clamp_engage_rate is often exactly 0).
    counts: dict[float, int] = {}
    for v in x:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0

    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    denom = n * (n - 1) / 2.0
    return {"n": n, "S": float(s), "tau": s / denom, "Z": z, "p": p}


def verdict(res: dict[str, float]) -> str:
    if res["p"] >= ALPHA:
        return "no significant trend"
    return "RISING (significant)" if res["S"] > 0 else "FALLING (significant)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--min-points", type=int, default=8,
                    help="refuse to judge on fewer intervals than this")
    args = ap.parse_args()

    hist = json.loads((Path(args.run_dir) / "history.json").read_text(encoding="utf-8"))
    rows = [r for r in hist if "clamp_max_preclamp" in r]
    if not rows:
        raise SystemExit("no clamp diagnostics in history.json — was tracking enabled?")

    iters = [r["iteration"] for r in rows]
    premax = [r["clamp_max_preclamp"] for r in rows]
    engage = [r["clamp_engage_rate"] for r in rows]

    print(f"{'iteration':>10}{'premax':>14}{'log premax':>14}{'engage %':>12}")
    for i, pm, en in zip(iters, premax, engage):
        lg = math.log(pm) if pm > 0 else float("-inf")
        print(f"{i:>10}{pm:>14.6g}{lg:>14.4f}{en * 100:>12.4f}")
    print()

    if len(rows) < args.min_points:
        print(f"ONLY {len(rows)} INTERVALS — need {args.min_points}. No verdict.")
        print("Reporting a trend from fewer points is exactly the error this "
              "script exists to prevent.")
        return

    log_premax = [math.log(v) if v > 0 else math.log(1e-12) for v in premax]
    r_pm = mann_kendall(log_premax)
    r_en = mann_kendall(engage)

    print(f"{'series':<22}{'n':>4}{'S':>8}{'tau':>9}{'Z':>9}{'p':>10}   verdict")
    for name, r in (("log(premax)", r_pm), ("clamp_engage_rate", r_en)):
        print(f"{name:<22}{int(r['n']):>4}{r['S']:>8.0f}{r['tau']:>9.3f}"
              f"{r['Z']:>9.3f}{r['p']:>10.4f}   {verdict(r)}")

    pm_up = r_pm["p"] < ALPHA and r_pm["S"] > 0
    en_up = r_en["p"] < ALPHA and r_en["S"] > 0
    print()
    if pm_up and en_up:
        print("READING: BOTH RISING — genuine evidence of intensifying pathology.")
        print("Re-run the AGC diagnostic on the current state to locate it. Do "
              "NOT switch fixes on this alone.")
    elif pm_up:
        print("READING: events larger but not more frequent. Worth noting; not "
              "on its own evidence of a spreading problem.")
    elif en_up:
        print("READING: engagement rising with flat magnitude — unexpected "
              "combination. Investigate before concluding.")
    else:
        print("READING: STATIONARY heavy tail, consistent with Q-A's own "
              "comparable swings. A single large premax value is NOT itself "
              "meaningful — clamp backward gradient is zero outside the bound "
              "regardless of magnitude, so 8500->8 and 9->8 are mechanically "
              "identical. Extended-horizon validation complete.")


if __name__ == "__main__":
    main()
