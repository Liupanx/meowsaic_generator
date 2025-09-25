from __future__ import annotations
import os, glob
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import gradio as gr
import cv2
from pathlib import Path
from datasets import load_dataset

try:
    from scipy.spatial import cKDTree
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False

DEFAULT_BLOCK = 32   # cell size in pixels
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==================== image utils ====================
def img_to_nparr(x) -> np.ndarray:
    """Accept PIL or np, return uint8 RGB np array (H,W,3)."""
    if hasattr(x, "convert"):  
        arr = np.array(x.convert("RGB"))
    else:
        arr = np.array(x)
        if arr.ndim == 2:                       
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:                   
            arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def resize_np(arr: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Resize keeping aspect so max(H,W)=max_side."""
    H, W = arr.shape[:2]
    s = max(H, W)
    if s <= max_side:
        return arr
    scale = max_side / s
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    try:
        return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    except Exception:
        return np.array(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS), np.uint8)

def trim_to_multiple_np(arr: np.ndarray, block: int) -> np.ndarray:
    H, W = arr.shape[:2]
    H2 = (H // block) * block
    W2 = (W // block) * block
    if H2 == 0 or W2 == 0:
        raise gr.Error(f"Image is too small for block={block}.")
    return arr[:H2, :W2]

def center_crop_square(arr: np.ndarray) -> np.ndarray:
    """Center crop to square (min side)."""
    H, W = arr.shape[:2]
    s = min(H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    return arr[y0:y0+s, x0:x0+s]

def adaptive_cells_mean(arr: np.ndarray, block: int, detail_thresh: float = 18.0, min_block: int = 8):
    """
    Return a list of rectangles and their mean RGBs.
    Rect = (y0, x0, size). For high-detail cells, splits once into 4 subcells.
    """
    H, W, C = arr.shape
    H2 = (H // block) * block
    W2 = (W // block) * block
    base = arr[:H2, :W2]

    rects = []              # [(y0, x0, size), ...]
    means = []              # [(r,g,b) float32, ...]
    half = max(min_block, block // 2)

    def cell_detail(patch: np.ndarray) -> float:
        # max per-channel std is a good, cheap detail score
        s = patch.reshape(-1, C).astype(np.float32).std(axis=0)
        return float(s.max())

    for y0 in range(0, H2, block):
        for x0 in range(0, W2, block):
            patch = base[y0:y0+block, x0:x0+block]
            d = cell_detail(patch)

            if d > detail_thresh and block > min_block:
                # split once into 4 subcells of size=half
                y1, x1 = y0 + half, x0 + half
                quads = [
                    (y0, x0, half), (y0, x1, half),
                    (y1, x0, half), (y1, x1, half),
                ]
                for (yy, xx, sz) in quads:
                    p = base[yy:yy+sz, xx:xx+sz]
                    m = p.reshape(-1, C).mean(axis=0).astype(np.float32)
                    rects.append((yy, xx, sz))
                    means.append(m)
            else:
                # keep whole cell
                m = patch.reshape(-1, C).mean(axis=0).astype(np.float32)
                rects.append((y0, x0, block))
                means.append(m)

    return rects, np.stack(means, axis=0)  # means shape: (N,3)


# ==================== dataset tiles ====================


def list_dataset_images(folder: str, limit: int) -> List[str]:
    if not os.path.isdir(folder):
        raise gr.Error(f"Folder not found: {folder}")
    paths = []
    for ext in ALLOWED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    if not paths:
        raise gr.Error(f"No images found in {folder} (allowed: {sorted(ALLOWED_EXTS)})")
    return paths[:max(1, int(limit))]

def load_tiles_from_hf(repo_id: str, split="train", limit=256) -> list[np.ndarray]:
    """Load images, center-crop to square, keep as uint8 RGB np arrays."""
    ds = load_dataset(repo_id, split=split, streaming=False)
    tiles = []
    for ex in ds.select(range(min(limit, len(ds)))):
        img = ex["image"]        
        arr = img_to_nparr(img)  
        tiles.append(center_crop_square(arr))
    if not tiles:
        raise gr.Error("No images found in Hugging Face dataset")
    return tiles

def compute_features(tiles: List[np.ndarray], feature_space: str) -> np.ndarray:
    """
    Return (T,3) features per tile.
    - 'Lab'  : mean in CIE Lab (requires skimage)
    - 'RGB'  : mean in RGB
    """
    T = len(tiles)
    feats = np.zeros((T, 3), dtype=np.float32)
    for i, t in enumerate(tiles):
        feats[i] = t.reshape(-1, 3).mean(axis=0).astype(np.float32)
    return feats

def build_index(feats: np.ndarray):
    """Return a KD-Tree if available, else None (we’ll brute-force)."""
    if HAVE_KDTREE:
        return cKDTree(feats)
    return None

def nearest_index(query_vec: np.ndarray, feats: np.ndarray, tree=None) -> int:
    if tree is not None:
        _, idx = tree.query(query_vec, k=1)
        return int(idx)
    # brute force
    d2 = ((feats - query_vec[None, :]) ** 2).sum(axis=1)
    return int(np.argmin(d2))

# ==================== mosaic pipeline ====================
def build_photomosaic(img, repo_id: str, block: int, max_side: int,
                      feature_space: str, tile_limit: int):

    if img is None:
        raise gr.Error("Please upload an image first.")

    base = img_to_nparr(img)
    base = resize_np(base, max_side=max_side)

    # 1) Get adaptive cells + their mean RGBs
    rects, target_feats = adaptive_cells_mean(base, block=block,
                                              detail_thresh=18.0,  # tweak in UI if you want
                                              min_block=max(4, block // 4))

    # 2) Load tiles once (same as before)
    tiles_raw = load_tiles_from_hf(repo_id, limit=tile_limit)

    # Pillow 10 compatibility for LANCZOS
    LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

    # Keep one copy at max size; we’ll resize per-rect
    tiles_rgb = [np.asarray(Image.fromarray(t), dtype=np.uint8) for t in tiles_raw]

    feats = compute_features(tiles_raw, feature_space)   # (T,3)
    tree = build_index(feats)

    # 3) Paste matched tiles per rectangle (adaptive sizes)
    out = np.zeros_like(base, dtype=np.uint8)
    used_idx = set()

    for (y0, x0, sz), q in zip(rects, target_feats):
        idx = nearest_index(q, feats, tree)
        used_idx.add(idx)
        tile = tiles_rgb[idx]
        # resize tile to the rect size
        tile_resized = np.array(Image.fromarray(tile).resize((sz, sz), LANCZOS), dtype=np.uint8)
        out[y0:y0+sz, x0:x0+sz] = tile_resized

    used_swatches = [np.array(Image.fromarray(tiles_rgb[i]).resize((block, block), LANCZOS), dtype=np.uint8)
                     for i in list(used_idx)[:100]]

    return out, used_swatches


# ==================== Gradio UI ====================
with gr.Blocks() as demo:
    gr.Markdown("## Meowsaic Generator🐱🪄")

    with gr.Row():
        img_in     = gr.Image(type="numpy", label="Upload image")
        mosaic_out = gr.Image(type="numpy", label="Photomosaic", format="jpeg")

    with gr.Row():
        block      = gr.Slider(4, 64, value=DEFAULT_BLOCK, step=1, label="Grid size")
        max_side   = gr.Slider(256, 2048, value=800, step=32, label="Image size")
    with gr.Row():
        feature    = gr.Textbox(value="RGB", label="Color space")
        tile_limit = gr.Slider(16, 1000, value=256, step=16, label="Image variance")
    used_gallery  = gr.Gallery(label="Tiles used", columns=10, height=140)

    gr.Button("Build Photomosaic").click(
        build_photomosaic,
        [img_in, gr.State("GalaxyRaccon/cat_images"), block, max_side, feature, tile_limit],
        [mosaic_out, used_gallery],
    )

if __name__ == "__main__":
    demo.launch(share=True, show_api=True)