"""Frozen AdaIR teacher — inference and caching ONLY (Phase 01).

Guarantees enforced here, because each failure is silent and catastrophic:

* the checkpoint loads with **zero missing and zero unexpected keys**, verified
  after stripping the Lightning ``net.`` prefix
* the module is in ``eval()`` mode and every parameter has ``requires_grad=False``
* eval mode is re-asserted on **every** forward — a teacher accidentally left in
  training mode produces plausible-but-wrong outputs (BatchNorm/dropout drift)
  with no error

Explicitly **out of scope for Phase 01**: adapters/projectors and any
distillation plumbing. This class does inference, plus (as of the kd_feature
experiment, see reports/kd_feature/plan.md) exposing one specific internal
tensor — ``latent_pre`` — via a non-invasive forward hook. Nothing else.
``forward()``'s existing contract is unchanged; ``forward_with_latent`` is
additive only, so kd_freq and the plain response-KD path are untouched.
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

        # Himeth's frequency-mask fix (FYP/Workspace/Himeth/mask_fix.md):
        # released AdaIR's FreModule mask is empty at every resolution this
        # project trains/evaluates at (128px patches), degenerating the
        # entire frequency-mining path to torch.abs() -- see that report and
        # src/models/adair_freq_fix.py for the full diagnosis. Fine-tuned
        # checkpoints record which mask mode they were trained with in their
        # own top-level dict; applying the fix for those is not optional --
        # loading them without it is the SAME weights with a dead frequency
        # path, silently. Checkpoints without a `mode` key (every checkpoint
        # used anywhere in this project before this) are entirely
        # unaffected -- this is purely additive.
        self.freq_fix_mode = ckpt.get("mode")
        if self.freq_fix_mode is not None:
            from src.models.adair_freq_fix import apply_freq_fix
            apply_freq_fix(self.net, mode=self.freq_fix_mode, tau=ckpt.get("tau", 0.05))

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

    @torch.no_grad()
    def forward_with_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore a batch AND return AdaIR's bottleneck tensor, ``latent_pre``.

        ``latent_pre`` is ``self.net.latent``'s output — the same tensor
        teacher-experiments/test05_5 calls ``latent_pre`` (``net/model.py:441``,
        captured *before* ``self.fre1``'s frequency modulation is applied).
        TEST05.5's causal audit found this representation, not the frequency
        pathway, is the well-supported distillation signal — see
        reports/kd_feature/plan.md.

        Captured via a forward hook on the real submodule (never a forward-
        pass reimplementation — the safe pattern established in
        teacher-experiments/test18, after an earlier reimplementation attempt
        there called methods that don't exist on the real module). The hook
        is registered and removed within this single call, so it never
        persists onto a model shared with plain ``forward()`` calls
        elsewhere in the trainer.

        Returns:
            ``(restored_output, latent_pre)`` — shapes ``(B, 3, H, W)`` and
            ``(B, 384, H/8, W/8)`` for AdaIR's default ``dim=48``.

        Raises:
            RuntimeError: if the hook never fires — would mean
                ``self.net.latent`` was not called during this forward, which
                should be impossible for AdaIR's own architecture and would
                indicate a wrong model class or a broken checkpoint load.
        """
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, inputs, output):
            captured["latent_pre"] = output

        handle = self.net.latent.register_forward_hook(_hook)
        try:
            out = self.forward(x)
        finally:
            handle.remove()

        if "latent_pre" not in captured:
            raise RuntimeError(
                "forward_with_latent: hook on self.net.latent did not fire — "
                "latent_pre was not captured")
        return out, captured["latent_pre"]

    @torch.no_grad()
    def forward_tiled(self, x: torch.Tensor, *, tile: int = 256,
                      overlap: int = 32) -> torch.Tensor:
        """Restore a large image in overlapping tiles.

        AdaIR is a transformer and its attention memory scales badly with
        resolution, so full-resolution RESIDE images can exhaust GPU memory.
        Tiles are blended with a cosine-tapered weight map rather than hard-cut,
        because hard tile boundaries leave visible seams that would then be
        baked into every cached teacher output.

        Falls back to a single forward when the image already fits the tile.

        Args:
            x: NCHW input.
            tile: tile side length.
            overlap: overlap between adjacent tiles, in pixels.
        """
        if self.net.training:
            raise RuntimeError(
                "teacher is in training mode — refusing to run inference")
        _, _, h, w = x.shape
        if h <= tile and w <= tile:
            return self.forward(x)
        if overlap >= tile:
            raise ValueError(f"overlap {overlap} must be smaller than tile {tile}")

        stride = tile - overlap
        device = self.device
        out = torch.zeros_like(x, device=device)
        weight = torch.zeros((1, 1, h, w), device=device)

        # Cosine taper: full weight in the tile interior, falling smoothly to ~0
        # at any edge that abuts a neighbouring tile. Interior edges of the
        # image are NOT tapered — nothing overlaps there to make up the weight.
        # A Hann window of length 2*overlap rises 0->1 over its first half and
        # falls 1->0 over its second, giving the two ramps directly.
        hann = torch.hann_window(2 * overlap, periodic=False, device=device)
        rise, fall = hann[:overlap], hann[overlap:]

        def taper(length: int, lead: bool, trail: bool) -> torch.Tensor:
            t = torch.ones(length, device=device)
            k = min(overlap, length // 2)
            if lead and k > 0:
                t[:k] = rise[:k]
            if trail and k > 0:
                t[-k:] = fall[-k:]
            return t

        ys = list(range(0, max(1, h - tile + 1), stride))
        xs = list(range(0, max(1, w - tile + 1), stride))
        if ys[-1] != h - tile:
            ys.append(max(0, h - tile))
        if xs[-1] != w - tile:
            xs.append(max(0, w - tile))

        for top in ys:
            for left in xs:
                patch = x[:, :, top:top + tile, left:left + tile].to(device)
                pred = self.forward(patch)
                ph, pw = pred.shape[-2:]
                wy = taper(ph, top > 0, top + ph < h)
                wx = taper(pw, left > 0, left + pw < w)
                wmap = (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)
                out[:, :, top:top + ph, left:left + pw] += pred * wmap
                weight[:, :, top:top + ph, left:left + pw] += wmap

        if float(weight.min()) <= 0:
            raise RuntimeError("tiling left uncovered pixels — check tile/overlap")
        return out / weight

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FrozenTeacher({self.checkpoint.name}, "
                f"params={self.n_params:,}, epoch={self.epoch}, "
                f"device={self.device})")


def load_teacher(checkpoint: str | Path, *,
                 device: str | torch.device = "cpu") -> FrozenTeacher:
    """Load the frozen AdaIR teacher."""
    return FrozenTeacher(checkpoint, device=device)
