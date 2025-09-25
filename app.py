from __future__ import annotations
import os, glob
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import gradio as gr
import cv2
from pathlib import Path

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

def per_cell_mean(arr: np.ndarray, block: int) -> np.ndarray:
    """Mean RGB per cell → (rows, cols, 3) float32 in [0,255]."""
    H, W, C = arr.shape
    rows, cols = H // block, W // block
    tiles = arr.reshape(rows, block, cols, block, C).swapaxes(1, 2) 
    return tiles.mean(axis=(2, 3), dtype=np.float32)


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

def load_and_prepare_tiles(folder: str, limit: int) -> List[np.ndarray]:
    """Load images, center-crop to square, keep as uint8 RGB np arrays."""
    paths = list_dataset_images(folder, limit)
    tiles = []
    for p in paths:
        try:
            arr = img_to_nparr(Image.open(p))
            tiles.append(center_crop_square(arr))
        except Exception:
            # skip unreadable files silently
            continue
    if not tiles:
        raise gr.Error(f"Could not load any valid images from {folder}")
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

def resolve_folder(folder: str) -> Path:
    base = Path(__file__).resolve().parent
    return (base / folder).resolve()

# ==================== mosaic ====================
def build_photomosaic(
    img,                 # uploaded image (PIL or np)
    dataset_folder: str, # e.g., "datasets"
    block: int,          # cell size in px
    max_side: int,       # pre-resize max side
    feature_space: str,  # "Lab" or "RGB"
    tile_limit: int,     # max dataset images to use
):
    # --- prep target image ---
    base = img_to_nparr(img)
    base = resize_np(base, max_side=max_side)
    base = trim_to_multiple_np(base, block)
    H, W = base.shape[:2]
    rows, cols = H // block, W // block

    # --- get absolute path of the image dataset folder ---
    resolve_folder(dataset_folder)
    # --- load dataset tiles, compute features, build index ---
    tiles_raw = load_and_prepare_tiles(dataset_folder, tile_limit)  # list[np.uint8]
    # resize all tiles once to block size (avoid resizing per cell)
    tiles_resized = [
        np.array(Image.fromarray(t).resize((block, block), Image.LANCZOS), dtype=np.uint8)
        for t in tiles_raw
    ]
    feats = compute_features(tiles_raw, feature_space)
    tree = build_index(feats)

    # --- compute per-cell feature for target ---

    rgb_cells = per_cell_mean(base, block)  # (rows, cols, 3)
    target_feats = rgb_cells.reshape(-1, 3)

    # --- choose tiles + paste ---
    out = np.zeros_like(base)
    idx_map = np.zeros((rows, cols), dtype=np.int32)
    n = 0
    for r in range(rows):
        for c in range(cols):
            q = target_feats[n]
            idx = nearest_index(q, feats, tree)
            idx_map[r, c] = idx
            tile = tiles_resized[idx]
            y0, x0 = r * block, c * block
            out[y0:y0+block, x0:x0+block] = tile
            n += 1

    # return which tiles were used as thumbnails
    used = sorted(set(idx_map.reshape(-1).tolist()))
    used_swatches = [tiles_resized[i] for i in used[:100]]  # cap gallery size
    return out, used_swatches, f"Cells: {rows}×{cols} = {rows*cols}, Tiles loaded: {len(tiles_resized)}, KDTree: {HAVE_KDTREE}"

# ==================== Gradio UI ====================
with gr.Blocks() as demo:
    gr.Markdown("## Meowsaic Generator🐱🪄")

    with gr.Row():
        img_in     = gr.Image(type="numpy", label="Upload image")
        mosaic_out = gr.Image(type="numpy", label="Photomosaic", format="jpeg")

    with gr.Row():
        dataset    = gr.Textbox(value="datasets", label="Dataset folder path")
        block      = gr.Slider(4, 64, value=DEFAULT_BLOCK, step=1, label="Cell size (pixels)")
        max_side   = gr.Slider(256, 2048, value=800, step=32, label="Max side (pre-resize)")
    with gr.Row():
        feature    = gr.Textbox(value="RGB", label="Color space for matching")
        tile_limit = gr.Slider(16, 1000, value=256, step=16, label="Max tiles to load")
    used_gallery  = gr.Gallery(label="Tiles used (subset)", columns=10, height=140)

    gr.Button("Build Photomosaic").click(
        build_photomosaic,
        [img_in, dataset, block, max_side, feature, tile_limit],
        [mosaic_out, used_gallery],
    )

if __name__ == "__main__":
    demo.launch(share=False, show_api=False)