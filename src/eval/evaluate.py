"""THE evaluation harness (engineering rule 3).

Every experiment, baseline and reproduction computes its numbers here. No script
computes metrics inline, ever — that is how two numbers in the same report end
up not being comparable.

Aggregation matches AdaIR's `AverageMeter` (`utils/val_utils.py:8-26`): metrics
are computed **per image** and then averaged over the set. This is *not* the same
as computing a global MSE and converting once, and the two differ by a few tenths
of a dB on heterogeneous sets.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.eval.metrics import ADAIR_DEFAULT, MetricConfig, psnr_ssim

#: One evaluation sample: (name, degraded, clean). Tensors are CHW or NCHW.
Sample = tuple[str, torch.Tensor, torch.Tensor]
#: A model maps a degraded batch to a restored batch.
ModelFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class EvalResult:
    """Aggregate and per-image metrics for one (model, dataset, config) triple."""

    name: str
    n_images: int = 0
    psnr: float = 0.0
    ssim: float = 0.0
    per_image: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "name": self.name,
            "n_images": self.n_images,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "config": self.config,
            "per_image": self.per_image,
        }


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """Accept CHW or NCHW; return NCHW."""
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"expected a 3-D or 4-D tensor, got shape {tuple(x.shape)}")


@torch.no_grad()
def evaluate(model_fn: ModelFn, samples: Iterable[Sample], *,
             name: str = "eval",
             config: MetricConfig = ADAIR_DEFAULT,
             device: str | torch.device = "cpu",
             keep_per_image: bool = True,
             progress: bool = False) -> EvalResult:
    """Run ``model_fn`` over ``samples`` and aggregate metrics.

    Args:
        model_fn: maps a degraded NCHW batch to a restored NCHW batch. Must
            already be in eval mode with gradients disabled — this harness does
            not silently fix a model that was left in training mode.
        samples: iterable of ``(name, degraded, clean)``.
        name: label recorded in the result and in ``metrics.json``.
        config: metric conventions. Defaults reproduce AdaIR.
        device: device to move inputs to.
        keep_per_image: retain per-image rows (needed to diagnose outliers).
        progress: show a tqdm bar.

    Returns:
        :class:`EvalResult` with per-image-averaged PSNR/SSIM.

    Raises:
        ValueError: if a restored batch does not match the clean batch shape —
            never silently cropped or resized to fit.
    """
    iterator: Iterator[Sample] = iter(samples)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=name)

    result = EvalResult(name=name, config=_config_dict(config))
    psnr_sum = ssim_sum = 0.0
    count = 0

    for sample_name, degraded, clean in iterator:
        deg = _to_nchw(degraded).to(device)
        gt = _to_nchw(clean).to(device)

        restored = model_fn(deg)
        if restored.shape != gt.shape:
            raise ValueError(
                f"{sample_name}: model output {tuple(restored.shape)} does not "
                f"match ground truth {tuple(gt.shape)}. Refusing to resize — fix "
                "the padding/cropping convention instead.")

        r = restored.detach().cpu().numpy()
        c = gt.detach().cpu().numpy()
        for i in range(r.shape[0]):
            p, s = psnr_ssim(np.transpose(r[i], (1, 2, 0)),
                             np.transpose(c[i], (1, 2, 0)), config)
            psnr_sum += p
            ssim_sum += s
            count += 1
            if keep_per_image:
                result.per_image.append(
                    {"name": sample_name, "psnr": p, "ssim": s})

    if count == 0:
        raise ValueError(f"{name}: no samples evaluated — empty dataset?")

    result.n_images = count
    result.psnr = psnr_sum / count
    result.ssim = ssim_sum / count
    return result


def _config_dict(config: MetricConfig) -> dict[str, Any]:
    """Serialise the metric conventions alongside every number they produced."""
    return {f: getattr(config, f) for f in MetricConfig.__dataclass_fields__}


def write_results(results: list[EvalResult], out_path: str | Path) -> Path:
    """Write ``metrics.json``. Never overwrites silently — raises if present."""
    path = Path(out_path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; results are never overwritten (rule 6)")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [r.as_dict() for r in results],
        "summary": {r.name: {"psnr": r.psnr, "ssim": r.ssim,
                             "n_images": r.n_images} for r in results},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def format_table(results: list[EvalResult],
                 reference: dict[str, tuple[float, float]] | None = None) -> str:
    """Render results as Markdown, optionally against published reference values.

    ``reference`` maps result name -> (published_psnr, published_ssim); the delta
    columns are what Gates G2/G3 are judged on.
    """
    has_ref = bool(reference)
    header = "| set | n | PSNR | SSIM |"
    sep = "|---|---|---|---|"
    if has_ref:
        header += " published PSNR | ΔPSNR | published SSIM | ΔSSIM |"
        sep += "---|---|---|---|"
    lines = [header, sep]

    for r in results:
        line = f"| {r.name} | {r.n_images} | {r.psnr:.2f} | {r.ssim:.4f} |"
        if has_ref:
            ref = (reference or {}).get(r.name)
            if ref:
                dp, ds = r.psnr - ref[0], r.ssim - ref[1]
                flag = "" if abs(dp) <= 0.10 else " ⚠"
                line += (f" {ref[0]:.2f} | **{dp:+.2f}**{flag} | "
                         f"{ref[1]:.4f} | {ds:+.4f} |")
            else:
                line += " — | — | — | — |"
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(
        "src/eval/evaluate.py is a library. Task 2.6 adds the reproduction CLI "
        "once datasets (2.3) and the teacher wrapper (2.5) exist.")
