"""Frozen AdaIR teacher — inference and caching ONLY (Phase 01).

Guarantees enforced here, because each failure is silent and catastrophic:

* the checkpoint loads with **zero missing and zero unexpected keys**, verified
  after stripping the Lightning ``net.`` prefix
* the module is in ``eval()`` mode and every parameter has ``requires_grad=False``
* eval mode is re-asserted on **every** forward — a teacher accidentally left in
  training mode produces plausible-but-wrong outputs (BatchNorm/dropout drift)
  with no error

Explicitly **out of scope for Phase 01**: feature-extraction hooks into AdaIR's
internals, adapters/projectors, and any distillation plumbing. This class does
inference and nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from src.utils.config import REPO_ROOT

_STRIP_PREFIXES = ("net.", "module.", "model.")
#: AdaIR's parameter count; a load that produces a different count is wrong.
EXPECTED_PARAMS = 28_784_824


def _load_adair_class():
    """Import the AdaIR architecture from the vendored repository."""
    repo = REPO_ROOT / "third_party" / "AdaIR"
    if not repo.exists():
        raise FileNotFoundError(
            f"AdaIR not vendored at {repo}. Clone it first:\n"
            f"    git clone --depth 1 https://github.com/c-yn/AdaIR.git {repo}")
    sys.path.insert(0, str(repo))
    try:
        from net.model import AdaIR
        return AdaIR
    finally:
        sys.path.remove(str(repo))


def _extract_state_dict(ckpt: dict) -> dict:
    """Pull the weights out of a Lightning or raw checkpoint and un-prefix them."""
    sd = None
    for key in ("state_dict", "params", "model", "net"):
        if isinstance(ckpt.get(key), dict):
            sd = ckpt[key]
            break
    if sd is None:
        sd = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    if not sd:
        raise ValueError("no state dict found in checkpoint")

    for prefix in _STRIP_PREFIXES:
        if all(k.startswith(prefix) for k in sd):
            return {k[len(prefix):]: v for k, v in sd.items()}
    return dict(sd)


class FrozenTeacher(nn.Module):
    """A frozen AdaIR model. Inference only.

    Args:
        checkpoint: path to a released ``.ckpt`` or a stripped ``.pth``.
        device: device to place the model on.
        expected_params: parameter count to assert after loading.
    """

    def __init__(self, checkpoint: str | Path, *, device: str | torch.device = "cpu",
                 expected_params: int = EXPECTED_PARAMS) -> None:
        super().__init__()
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"teacher checkpoint not found: {path}")

        adair_cls = _load_adair_class()
        model = adair_cls(decoder=True)  # matches AdaIRModel (test.py:21)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = _extract_state_dict(ckpt)

        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{path.name}: partial state-dict load — "
                f"{len(incompatible.missing_keys)} missing, "
                f"{len(incompatible.unexpected_keys)} unexpected. "
                f"missing[:5]={incompatible.missing_keys[:5]} "
                f"unexpected[:5]={incompatible.unexpected_keys[:5]}")

        n_params = sum(p.numel() for p in model.parameters())
        if expected_params and n_params != expected_params:
            raise RuntimeError(
                f"{path.name}: loaded {n_params:,} parameters, expected "
                f"{expected_params:,} — architecture mismatch")

        model.eval()
        model.requires_grad_(False)
        self.net = model.to(device)
        self.device = torch.device(device)
        self.checkpoint = path
        self.n_params = n_params
        self.epoch = ckpt.get("epoch")
        self.global_step = ckpt.get("global_step")

    def train(self, mode: bool = True):  # type: ignore[override]
        """Refuse to enter training mode. The teacher is frozen, permanently."""
        if mode:
            raise RuntimeError(
                "FrozenTeacher cannot be put in training mode. It is an "
                "inference-only wrapper; if you need a trainable AdaIR, build "
                "it directly.")
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore a batch. Asserts eval mode and no-grad on every call."""
        if self.net.training:
            raise RuntimeError(
                "teacher is in training mode — refusing to run inference")
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}")
        out = self.net(x.to(self.device))
        if out.shape != x.shape:
            raise ValueError(
                f"teacher output {tuple(out.shape)} != input {tuple(x.shape)}")
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FrozenTeacher({self.checkpoint.name}, "
                f"params={self.n_params:,}, epoch={self.epoch}, "
                f"device={self.device})")


def load_teacher(checkpoint: str | Path, *,
                 device: str | torch.device = "cpu") -> FrozenTeacher:
    """Load the frozen AdaIR teacher."""
    return FrozenTeacher(checkpoint, device=device)
