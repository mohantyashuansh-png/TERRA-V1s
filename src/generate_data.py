"""
Desert Scene Synthetic Data Generator
Generates realistic desert environment images with semantic segmentation masks.
"""

import numpy as np
import cv2
import os
import random
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ─── Class Config ────────────────────────────────────────────────────────────
CLASS_NAMES = ['Sky', 'Sand', 'Rock', 'Vegetation', 'Shadow', 'Distant Terrain']
CLASS_COLORS = [
    (135, 206, 235),   # Sky - light blue
    (210, 180, 140),   # Sand - sandy tan
    (105,  90,  75),   # Rock - brownish gray
    ( 85, 120,  60),   # Vegetation - muted green
    ( 90,  80,  65),   # Shadow - dark sand
    (160, 140, 115),   # Distant Terrain - hazy tan
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (512, 512)

# ─── Noise Utilities ─────────────────────────────────────────────────────────

def smooth_noise(shape, scale=80, seed=None):
    """Gaussian-blurred random noise for organic terrain."""
    if seed is not None:
        np.random.seed(seed)
    n = np.random.rand(*shape).astype(np.float32)
    k = max(1, int(scale / 10)) * 2 + 1
    n = cv2.GaussianBlur(n, (k, k), scale / 5)
    n = (n - n.min()) / (n.max() - n.min() + 1e-8)
    return n

# ─── Scene Generation ─────────────────────────────────────────────────────────

def make_horizon_line(width, seed=None):
    """Generate a wavy horizon line y-position for each column."""
    if seed is not None:
        np.random.seed(seed)
    
    # 35-55% from top = sky
    base_h = random.uniform(0.35, 0.55)
    freq   = random.uniform(0.003, 0.008)
    amp    = random.uniform(10, 30)
    xs     = np.arange(width)
    
    horizon = (base_h * IMG_SIZE[0]
               + amp * np.sin(2 * np.pi * freq * xs + random.uniform(0, 2*np.pi))
               + amp * 0.5 * np.sin(4 * np.pi * freq * xs + random.uniform(0, 2*np.pi)))
    return horizon.astype(np.int32)


def make_dune_ridges(horizon, width, height, seed=None):
    """Return (mask_sand, mask_shadow) arrays."""
    if seed is not None:
        np.random.seed(seed)
    
    mask_sand   = np.zeros((height, width), dtype=np.uint8)
    mask_shadow = np.zeros((height, width), dtype=np.uint8)

    for col in range(width):
        start = horizon[col]
        # everything below horizon is sand initially
        if start < height:
            mask_sand[start:, col] = 1

    # Add dune ridges via noise-displaced rows
    noise = smooth_noise((height, width), scale=120, seed=seed)
    ridge_thresh = random.uniform(0.60, 0.75)

    for col in range(width):
        hor = horizon[col]
        for row in range(hor, height):
            rel_y = (row - hor) / max(1, height - hor)
            nval  = noise[row, col]
            if nval > ridge_thresh and rel_y < 0.35:
                # shadow on the lee side of the dune
                mask_shadow[row, col] = 1
                mask_sand[row, col]   = 0

    return mask_sand, mask_shadow


def place_rocks(height, width, horizon, seed=None):
    """Return boolean mask of rock regions."""
    if seed is not None:
        np.random.seed(seed)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    n_rocks = random.randint(2, 8)
    
    for _ in range(n_rocks):
        col = random.randint(0, width - 1)
        row_min = horizon[col]
        row_max = min(height - 1, row_min + int(0.4 * (height - row_min)))
        
        if row_max <= row_min:
            continue
            
        cy = random.randint(row_min, row_max)
        cx = col
        ry = random.randint(15, 55)
        rx = random.randint(20, 70)
        cv2.ellipse(mask, (cx, cy), (rx, ry), random.uniform(0, 360), 0, 360, 1, -1)
    return mask


def place_vegetation(height, width, horizon, seed=None):
    """Return boolean mask of sparse vegetation."""
    if seed is not None:
        np.random.seed(seed)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    n_plants = random.randint(3, 12)
    
    for _ in range(n_plants):
        cx = random.randint(10, width - 10)
        base_row = horizon[cx]
        
        # Ensure we don't go out of bounds
        if base_row >= height - 10: 
            continue
            
        cy = random.randint(base_row, min(height - 10, base_row + int(0.5 * (height - base_row))))
        plant_type = random.choice(['shrub', 'cactus'])
        
        if plant_type == 'shrub':
            r = random.randint(8, 22)
            cv2.circle(mask, (cx, cy), r, 1, -1)
        else:  # cactus
            h_body = random.randint(20, 50)
            w_body = random.randint(6, 12)
            # Main body
            cv2.rectangle(mask, (cx - w_body, cy - h_body), (cx + w_body, cy), 1, -1)
            # Arms
            arm_len = random.randint(10, 25)
            arm_y = cy - h_body // 2
            if arm_y > 0 and arm_y < height:
                cv2.rectangle(mask, (cx - w_body - arm_len, arm_y),
                              (cx - w_body, arm_y + 8), 1, -1)
                cv2.rectangle(mask, (cx + w_body, arm_y),
                              (cx + w_body + arm_len, arm_y + 8), 1, -1)
    return mask


def make_distant_terrain(horizon, width, seed=None):
    """Hazy mountain/hill silhouette just above horizon."""
    if seed is not None:
        np.random.seed(seed)
    
    band_h = random.randint(20, 60)
    mask = np.zeros((IMG_SIZE[0], width), dtype=np.uint8)
    
    for col in range(width):
        top = horizon[col] - band_h + int(15 * np.sin(col * 0.04 + random.uniform(0, 6)))
        bot = horizon[col]
        top = max(0, top)
        if bot > top:
            mask[top:bot, col] = 1
    return mask

# ─── Image Rendering ──────────────────────────────────────────────────────────

def render_image(seg_mask, seed=None):
    """Convert a segmentation mask into a photorealistic-ish desert image."""
    if seed is not None:
        np.random.seed(seed)
    
    h, w = seg_mask.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    time_of_day = random.choice(['day', 'golden_hour', 'dusk'])

    sky_colors = {
        'day':         [(135, 180, 220), (180, 210, 240)],
        'golden_hour': [(220, 140,  60), (240, 190, 100)],
        'dusk':        [( 80,  60, 120), (200, 130,  80)],
    }
    sand_colors = {
        'day':         (210, 175, 125),
        'golden_hour': (230, 160,  80),
        'dusk':        (160, 120,  90),
    }

    # 1. Sky gradient
    sky_bot, sky_top = sky_colors[time_of_day]
    # Identify sky rows to apply gradient (approximate)
    sky_mask = (seg_mask == 0)
    if np.any(sky_mask):
        rows, cols = np.where(sky_mask)
        sky_min, sky_max = rows.min(), rows.max()
        for row in range(sky_min, sky_max + 1):
            t = (row - sky_min) / max(1, sky_max - sky_min)
            # Linear interpolate
            c = tuple(int(sky_top[i] * t + sky_bot[i] * (1 - t)) for i in range(3))
            # Apply to pixels in this row that are sky
            row_mask = sky_mask[row, :]
            img[row, row_mask] = c

    # 2. Sand
    base_sand = np.array(sand_colors[time_of_day], dtype=np.float32)
    noise_sand = smooth_noise((h, w), scale=60, seed=seed) * 30 - 15
    
    # Broadcast noise to RGB
    sand_tex = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        sand_tex[:, :, c] = np.clip(base_sand[c] + noise_sand, 0, 255)
    
    # Apply sand texture where mask == 1
    img = np.where(seg_mask[:, :, None] == 1, sand_tex, img)

    # 3. Rock (Class 2)
    rock_noise = smooth_noise((h, w), scale=25, seed=(seed or 0) + 1) * 40 - 20
    rock_base = [105, 90, 75]
    rock_tex = np.zeros((h, w, 3), dtype=np.float32)
    for c, base in enumerate(rock_base):
        rock_tex[:, :, c] = np.clip(base + rock_noise, 0, 255)
    img = np.where(seg_mask[:, :, None] == 2, rock_tex, img)

    # 4. Vegetation (Class 3)
    veg_noise = smooth_noise((h, w), scale=15, seed=(seed or 0) + 2) * 25 - 12
    veg_base = [85, 120, 60]
    veg_tex = np.zeros((h, w, 3), dtype=np.float32)
    for c, base in enumerate(veg_base):
        veg_tex[:, :, c] = np.clip(base + veg_noise, 0, 255)
    img = np.where(seg_mask[:, :, None] == 3, veg_tex, img)

    # 5. Shadow (Class 4)
    shadow_base = [70, 62, 50]
    shadow_tex = np.zeros((h, w, 3), dtype=np.float32)
    for c, base in enumerate(shadow_base):
        shadow_tex[:, :, c] = np.clip(base + rock_noise * 0.5, 0, 255)
    img = np.where(seg_mask[:, :, None] == 4, shadow_tex, img)

    # 6. Distant Terrain (Class 5)
    dist_noise = smooth_noise((h, w), scale=40) * 20 - 10
    dist_base = [160, 140, 115]
    dist_tex = np.zeros((h, w, 3), dtype=np.float32)
    for c, base in enumerate(dist_base):
        dist_tex[:, :, c] = np.clip(base + dist_noise, 0, 255)
    img = np.where(seg_mask[:, :, None] == 5, dist_tex, img)

    # 7. Global Atmosphere / Haze
    atm_noise = smooth_noise((h, w), scale=200, seed=(seed or 0) + 99) * 12 - 6
    img = np.clip(img + atm_noise[:, :, np.newaxis], 0, 255)

    # Add haze to distant terrain
    haze = (seg_mask == 5).astype(np.float32) * 0.35
    img = img * (1 - haze[:, :, np.newaxis]) + 220 * haze[:, :, np.newaxis]

    return img.astype(np.uint8)

# ─── Main Generator ──────────────────────────────────────────────────────────

def generate_scene(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    h, w = IMG_SIZE
    seg_mask = np.zeros((h, w), dtype=np.uint8)  # default = sky

    horizon = make_horizon_line(w, seed=seed)

    # Distant terrain
    dist_mask = make_distant_terrain(horizon, w, seed=seed)
    seg_mask[dist_mask == 1] = 5

    # Sand & Shadow
    sand_mask, shadow_mask = make_dune_ridges(horizon, w, h, seed=seed)
    seg_mask[sand_mask == 1] = 1
    seg_mask[shadow_mask == 1] = 4

    # Rocks
    rock_mask = place_rocks(h, w, horizon, seed=seed)
    seg_mask[(rock_mask == 1) & (seg_mask >= 1)] = 2

    # Vegetation
    veg_mask = place_vegetation(h, w, horizon, seed=seed)
    seg_mask[(veg_mask == 1) & (seg_mask >= 1)] = 3

    img = render_image(seg_mask, seed=seed)
    
    # Post-processing
    img = cv2.GaussianBlur(img, (3, 3), 0.7)

    return img, seg_mask

def generate_split(split, n, base_dir, seed_offset=0):
    img_dir  = Path(base_dir) / split / 'images'
    mask_dir = Path(base_dir) / split / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n), desc=f'Generating {split}'):
        seed = seed_offset + i
        img, mask = generate_scene(seed=seed)
        # Save as BGR for OpenCV
        cv2.imwrite(str(img_dir / f'{split}_{i:04d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f'{split}_{i:04d}.png'), mask)
    
    print(f'✓ {split}: {n} samples → {img_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=600)
    parser.add_argument('--val',   type=int, default=150)
    parser.add_argument('--test',  type=int, default=100)
    parser.add_argument('--out',   type=str, default='data')
    args = parser.parse_args()

    print(f'\n🏜  Desert Synthetic Data Generator')
    generate_split('train', args.train, args.out, seed_offset=0)
    generate_split('val',   args.val,   args.out, seed_offset=10000)
    generate_split('test',  args.test,  args.out, seed_offset=20000)