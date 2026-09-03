"""HaMeR 3D MANO regression → MediaPipe-indexed kp3d for LEAP Hand W1.

Wraps hamer (fp16) so a MediaPipe-detected hand crop is turned into a real
MANO 3D hand model: 21 keypoints (order 1:1 with MediaPipe) + 778 verts.

Design: docs/design/2026-08-05-hamer-3d-integration-w1.md
"""

import io
import contextlib
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class HaMeR3DResult:
    """Container for one hamer regression output (single hand)."""

    def __init__(self, kp3d, verts, cam_t, kp2d_patch, box_center, box_size):
        self.kp3d = np.asarray(kp3d, dtype=np.float64)          # (21,3) metric, MP order
        self.verts = np.asarray(verts, dtype=np.float64)        # (778,3) metric
        self.cam_t = np.asarray(cam_t, dtype=np.float64)        # (3,) weak-persp translation
        self.kp2d_patch = np.asarray(kp2d_patch, dtype=np.float64)  # (21,2) patch pixels
        self.box_center = np.asarray(box_center, dtype=np.float64)  # (2,) full-frame px
        self.box_size = float(box_size)                          # full-frame px extent


class HaMeR3D:
    """Lazy-loaded hamer (fp16) regression from a full frame + hand bbox.

    available is False when torch/hamer are missing or no CUDA GPU — callers
    must fall back to MediaPipe pseudo-3D in that case.
    """

    def __init__(self, checkpoint: Optional[str] = None,
                 device: Optional[str] = None, fp16: bool = True):
        self.available = False
        self.fp16 = fp16
        self.model = None
        self.model_cfg = None
        self.image_size = None
        self.device = None
        try:
            import torch
        except Exception:
            return
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            return
        try:
            from hamer.configs import get_config
            from hamer.models import HAMER, DEFAULT_CHECKPOINT
        except Exception:
            return
        try:
            self._load(checkpoint or DEFAULT_CHECKPOINT, get_config, HAMER)
            self.available = True
        except Exception:
            self.model = None

    def _hamer_data_dir(self) -> Path:
        """Locate hamer's _DATA dir.

        hamer's CACHE_DIR_HAMER is the cwd-relative './_DATA', so it only
        resolves when the process runs from the hamer repo. Prefer a ./_DATA
        in the current working dir, else anchor to the hamer package root so
        the module works from any cwd (tests run from python/).
        """
        if Path("_DATA").is_dir():
            return Path("_DATA")
        import hamer as hamer_pkg
        return Path(hamer_pkg.__file__).resolve().parent.parent / "_DATA"

    def _load(self, checkpoint, get_config, HAMER):
        ckpt_path = str(Path(checkpoint))
        if not Path(ckpt_path).exists():
            # DEFAULT_CHECKPOINT is the relative './_DATA/...' path; anchor it
            # to the package root when the current cwd cannot resolve it.
            ckpt_path = str(self._hamer_data_dir() / "hamer_ckpts" / "checkpoints" / "hamer.ckpt")
        model_cfg = get_config(str(Path(ckpt_path).parent.parent / "model_config.yaml"),
                               update_cachedir=False)
        if not Path(model_cfg.MANO.MODEL_PATH).exists():
            # cfg MANO paths are relative to _DATA; make them absolute.
            data_dir = self._hamer_data_dir()
            model_cfg.defrost()
            model_cfg.MANO.MODEL_PATH = str(data_dir / "data" / "mano")
            model_cfg.MANO.MEAN_PARAMS = str(data_dir / "data" / "mano_mean_params.npz")
            model_cfg.freeze()
        if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
            model_cfg.defrost()
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            model_cfg.freeze()
        model = HAMER.load_from_checkpoint(ckpt_path, strict=False, cfg=model_cfg,
                                           map_location="cpu")
        self.model = model.half().to(self.device) if self.fp16 else model.to(self.device)
        self.model.eval()
        self.model_cfg = model_cfg
        self.image_size = int(model_cfg.MODEL.IMAGE_SIZE)

    def regress(self, frame_bgr: np.ndarray,
                bbox_xyxy: Tuple[int, int, int, int]) -> Optional[HaMeR3DResult]:
        """Run hamer on a hand crop. Returns None on failure.

        Cropping/resizing is handled inside by hamer's ViTDetDataset (the same
        path smoke_test uses). bbox_xyxy is full-frame pixel coords.
        """
        if not self.available:
            return None
        import torch
        try:
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.utils import recursive_to

            boxes = np.array([bbox_xyxy], dtype=np.float32)
            right = np.array([1], dtype=np.int32)  # LEAP is a right hand
            dataset = ViTDetDataset(self.model_cfg, frame_bgr, boxes, right,
                                    rescale_factor=2.0)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0)
            with torch.no_grad(), torch.autocast(device_type="cuda",
                                                 dtype=torch.float16):
                # ViTDetDataset prints a debug line every item — suppress it.
                with contextlib.redirect_stdout(io.StringIO()):
                    batch = recursive_to(next(iter(loader)), self.device)
                if self.fp16:
                    for k, v in batch.items():
                        if torch.is_tensor(v) and v.is_floating_point():
                            batch[k] = v.half()
                out = self.model(batch)
        except Exception:
            return None

        kp3d = to_mediapipe_order(out["pred_keypoints_3d"][0].float().cpu().numpy())
        if not np.isfinite(kp3d).all():
            return None
        verts = out["pred_vertices"][0].float().cpu().numpy()
        cam_t = out["pred_cam_t"][0].float().cpu().numpy()
        kp2d_patch = out["pred_keypoints_2d"][0].float().cpu().numpy()
        box_center = batch["box_center"][0].float().cpu().numpy()
        box_size = float(batch["box_size"][0].item())
        return HaMeR3DResult(kp3d, verts, cam_t, kp2d_patch, box_center, box_size)

    def project_to_frame(self, result: HaMeR3DResult, pts3d: np.ndarray) -> np.ndarray:
        """Project 3D points (metric) to full-frame pixels via hamer's weak perspective.

        p_patch = (focal/IMAGE_SIZE) * (xy + cam_t.xy) / (z + cam_t.z) is the
        network's normalized patch coords (origin at patch center, same as
        pred_keypoints_2d). Full-frame px = box_center + p_patch * box_size —
        the convention hamer's own eval uses (hamer/utils/pose_utils.py).
        """
        pts = np.asarray(pts3d, dtype=np.float64)
        focal = float(self.model_cfg.EXTRA.FOCAL_LENGTH) / self.image_size
        z = pts[:, 2] + result.cam_t[2]
        p_patch = np.zeros((len(pts), 2), dtype=np.float64)
        nz = np.abs(z) > 1e-6
        p_patch[nz] = focal * (pts[nz, :2] + result.cam_t[:2]) / z[nz, None]
        return result.box_center + p_patch * result.box_size


def to_mediapipe_order(kp3d: np.ndarray) -> np.ndarray:
    """Return kp3d as-is.

    hamer's MANO joint order (wrist, thumb mcp/pip/dip/tip, index, middle,
    ring, pinky) is positionally 1:1 with MediaPipe indices, so no reordering
    is required. Kept explicit so a future MANO change is patched here only.
    """
    return np.asarray(kp3d, dtype=np.float64)


def hand_bbox_from_landmarks(
    pts_xy: np.ndarray,
    image_shape: Tuple[int, int],
    margin: float = 1.5,
    square: bool = True,
    min_size: int = 32,
) -> Optional[Tuple[int, int, int, int]]:
    """Square crop around a hand's 21 landmarks (pixel xy).

    Returns (x0, y0, x1, y1) clamped to the frame, or None if degenerate.
    """
    h, w = image_shape
    xs, ys = pts_xy[:, 0], pts_xy[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())

    if square:
        side = max(x1 - x0, y1 - y0) * margin
        if side < min_size:
            return None
        half = side / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    else:
        padx = (x1 - x0) * (margin - 1) / 2
        pady = (y1 - y0) * (margin - 1) / 2
        x0, y0, x1, y1 = x0 - padx, y0 - pady, x1 + padx, y1 + pady

    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return (x0, y0, x1, y1)


# ─── Demo / Test ─────────────────────────────────────────────────

def smoke_on_image(image_path: str):
    """Run MediaPipe → hamer on a single image and print kp3d stats."""
    import cv2
    try:
        from .hand_tracker import HandTracker
    except (ImportError, ValueError):
        from gesture_mapping import HandTracker

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] cannot read {image_path}")
        return
    h, w = img.shape[:2]
    tracker = HandTracker(max_num_hands=1)
    results = tracker.detect(img)
    if not results:
        print("  (no hand detected)")
        return
    hand = results[0]
    mp_pts = tracker.landmark_xy(hand, (h, w))
    bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
    if bbox is None:
        print("  (bbox too small)")
        return
    h3d = HaMeR3D()
    if not h3d.available:
        print("  (hamer unavailable)")
        return
    res = h3d.regress(img, bbox)
    if res is None:
        print("  (hamer regression failed)")
        return
    print(f"  kp3d shape={res.kp3d.shape} finite={bool(np.isfinite(res.kp3d).all())}")
    print(f"  verts shape={res.verts.shape}")
    print(f"  kp3d range=[{res.kp3d.min():.3f}, {res.kp3d.max():.3f}] (meters)")
    proj = h3d.project_to_frame(res, res.kp3d)
    dist = np.linalg.norm(proj - mp_pts, axis=1)
    print(f"  projection median dist vs MediaPipe: {np.median(dist):.1f} px")
    tracker.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        smoke_on_image(sys.argv[1])
    else:
        print("usage: python hamer_3d.py <image_path>")
