# gpu_adaptive_median.py
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

from numba import cuda, int32, uint8, float32

# -----------------------
# Kernel parameters
# -----------------------
# max window radius (matching original): 5 => max kernel size = 11x11 => 121 elements
MAX_WINDOW_RADIUS = 5
MAX_VLENGTH = (2 * MAX_WINDOW_RADIUS + 1) ** 2  # 121

# -----------------------
# Helper: insertion sort (on fixed-size local array)
# -----------------------
@cuda.jit(device=True)
def insertion_sort(arr, n):
    # arr is a 1D local array of length >= n
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# -----------------------
# Kernel: adaptive median single-channel
# Input:
#   d_in : H x W uint8 input image (flattened)
#   d_out: H x W uint8 output image (flattened)
#   width, height: image dims
#   window_radius: integer (1..MAX_WINDOW_RADIUS)
#   threshold: float threshold (0 => pure median replacement)
# Notes:
#   index mapping uses row-major indexing: idx = y*width + x
# -----------------------
@cuda.jit
def adaptive_median_kernel(d_in, d_out, width, height, window_radius, threshold):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    r = window_radius
    # skip border pixels we cannot fully process (same as original)
    if x < r or x >= (width - (r + 1)) or y < r or y >= (height - (r + 1)):
        # copy input to output for border
        d_out[y * width + x] = d_in[y * width + x]
        return

    # Build local window vector
    vlength = (2 * r + 1) * (2 * r + 1)

    # local fixed-size array allocated in registers/local memory
    # Note: local arrays must have compile-time size, use MAX_VLENGTH
    local = cuda.local.array(shape=MAX_VLENGTH, dtype=uint8)

    idx = 0
    for dy in range(-r, r + 1):
        yy = y + dy
        base = yy * width
        for dx in range(-r, r + 1):
            xx = x + dx
            local[idx] = d_in[base + xx]
            idx += 1

    # sort the first vlength elements
    # convert to int for comparisons is implicit with uint8
    insertion_sort(local, vlength)

    # median
    mid = vlength // 2
    median = local[mid]  # uint8

    # threshold check
    if threshold <= 0.0:
        d_out[y * width + x] = median
        return

    # compute absolute deviations into a small local float array (use float32)
    # we'll reuse part of local: but easier to copy to float local array
    # create local float array
    localf = cuda.local.array(shape=MAX_VLENGTH, dtype=float32)
    for i in range(vlength):
        # abs(int(local[i]) - int(median))
        localf[i] = float32(abs(int(local[i]) - int(median)))

    # sort localf (need insertion sort for floats) - reuse same function via cast? implement here:
    for i in range(1, vlength):
        key = localf[i]
        j = i - 1
        while j >= 0 and localf[j] > key:
            localf[j + 1] = localf[j]
            j -= 1
        localf[j + 1] = key

    # median absolute deviation (MAD)
    mad = localf[vlength // 2]
    Sk = 1.4826 * mad  # float32

    pix = float32(int(d_in[y * width + x]))
    if abs(pix - float32(int(median))) > threshold * Sk:
        d_out[y * width + x] = median
    else:
        d_out[y * width + x] = d_in[y * width + x]


# -----------------------
# Python wrapper
# -----------------------
def adaptive_median_gpu_single_channel(img_uint8, window_radius=1, threshold=0.0):
    """
    img_uint8 : 2D numpy array dtype=uint8
    window_radius: 1..MAX_WINDOW_RADIUS
    Returns filtered 2D uint8 array
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA not available. Install CUDA and Numba or use CPU fallback.")

    if window_radius < 1 or window_radius > MAX_WINDOW_RADIUS:
        raise ValueError(f"window_radius must be 1..{MAX_WINDOW_RADIUS}")

    h, w = img_uint8.shape
    # flatten arrays
    hwl = h * w
    d_in = cuda.to_device(img_uint8.ravel())
    d_out = cuda.device_array(hwl, dtype=np.uint8)

    # choose block/grid sizes
    threadsperblock = (16, 16)
    blocks_x = (w + threadsperblock[0] - 1) // threadsperblock[0]
    blocks_y = (h + threadsperblock[1] - 1) // threadsperblock[1]
    grid = (blocks_x, blocks_y)

    adaptive_median_kernel[grid, threadsperblock](d_in, d_out, w, h, window_radius, float(threshold))

    out_flat = d_out.copy_to_host()
    out = out_flat.reshape((h, w))
    return out.astype(np.uint8)


# -----------------------
# RGB wrapper: process each channel
# -----------------------
def adaptive_median_gpu_rgb(rgb_img, window_radius=1, threshold=0.0):
    """
    rgb_img: HxWx3 uint8
    Returns filtered HxWx3 uint8
    """
    b, g, r = cv2.split(rgb_img)
    b_f = adaptive_median_gpu_single_channel(b, window_radius, threshold)
    g_f = adaptive_median_gpu_single_channel(g, window_radius, threshold)
    r_f = adaptive_median_gpu_single_channel(r, window_radius, threshold)
    return cv2.merge((b_f, g_f, r_f))


# -----------------------
# CPU fallback (pure numpy implementation)
# -----------------------
def adaptive_median_cpu_single_channel(img, window_radius=1, threshold=0.0):
    h, w = img.shape
    out = img.copy().astype(np.uint8)
    r = window_radius
    vlength = (2 * r + 1) ** 2

    for y in range(r, h - (r + 1)):
        for x in range(r, w - (r + 1)):
            local = img[y - r:y + r + 1, x - r:x + r + 1].ravel()
            sorted_local = np.sort(local)
            median = sorted_local[vlength // 2]

            if threshold <= 0.0:
                out[y, x] = median
            else:
                scale = np.abs(local.astype(np.int32) - int(median))
                scale_sorted = np.sort(scale)
                Sk = 1.4826 * scale_sorted[vlength // 2]
                if abs(int(img[y, x]) - int(median)) > threshold * Sk:
                    out[y, x] = median
    return out.astype(np.uint8)


# -----------------------
# Integration: process_images wrapper (fits your pipeline)
# -----------------------
def process_images_gpu(input_csv, output_csv, output_base_dir,
                       window_radius=1, threshold=0.0, use_gpu=True):
    df = pd.read_csv(input_csv)

    new_rgb_paths = []
    new_ir_paths = []

    rgb_out_dir = os.path.join(output_base_dir, '254p RGB Images')
    ir_out_dir = os.path.join(output_base_dir, '254p Thermal Images')
    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(ir_out_dir, exist_ok=True)

    print(f"Processing {len(df)} images (Adaptive Median Filter, GPU={use_gpu}) ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        csv_dir = os.path.dirname(input_csv)
        rgb_path = os.path.normpath(os.path.join(csv_dir, row['rgb_frame']))
        ir_path = os.path.normpath(os.path.join(csv_dir, row['ir_frame']))

        # RGB
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"Warning: Could not read RGB image at {rgb_path}")
            new_rgb_paths.append(None)
        else:
            if use_gpu and cuda.is_available():
                try:
                    rgb_filtered = adaptive_median_gpu_rgb(rgb_img, window_radius, threshold)
                except Exception as e:
                    print("GPU processing failed, falling back to CPU for this image. Error:", e)
                    # fallback
                    b, g, rch = cv2.split(rgb_img)
                    b_f = adaptive_median_cpu_single_channel(b, window_radius, threshold)
                    g_f = adaptive_median_cpu_single_channel(g, window_radius, threshold)
                    r_f = adaptive_median_cpu_single_channel(rch, window_radius, threshold)
                    rgb_filtered = cv2.merge((b_f, g_f, r_f))
            else:
                # CPU fallback
                b, g, rch = cv2.split(rgb_img)
                b_f = adaptive_median_cpu_single_channel(b, window_radius, threshold)
                g_f = adaptive_median_cpu_single_channel(g, window_radius, threshold)
                r_f = adaptive_median_cpu_single_channel(rch, window_radius, threshold)
                rgb_filtered = cv2.merge((b_f, g_f, r_f))

            rgb_filename = os.path.basename(rgb_path)
            save_path_rgb = os.path.join(rgb_out_dir, rgb_filename)
            cv2.imwrite(save_path_rgb, rgb_filtered)
            rel_path_rgb = os.path.join('..', output_base_dir, '254p RGB Images', rgb_filename)
            new_rgb_paths.append(rel_path_rgb)

        # IR
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir_img is None:
            print(f"Warning: Could not read IR image at {ir_path}")
            new_ir_paths.append(None)
        else:
            if use_gpu and cuda.is_available():
                try:
                    ir_filtered = adaptive_median_gpu_single_channel(ir_img, window_radius, threshold)
                except Exception as e:
                    print("GPU processing failed for IR, falling back to CPU. Error:", e)
                    ir_filtered = adaptive_median_cpu_single_channel(ir_img, window_radius, threshold)
            else:
                ir_filtered = adaptive_median_cpu_single_channel(ir_img, window_radius, threshold)

            ir_filename = os.path.basename(ir_path)
            save_path_ir = os.path.join(ir_out_dir, ir_filename)
            cv2.imwrite(save_path_ir, ir_filtered)
            rel_path_ir = os.path.join('..', output_base_dir, '254p Thermal Images', ir_filename)
            new_ir_paths.append(rel_path_ir)

    # update csv
    df['rgb_frame'] = new_rgb_paths
    df['ir_frame'] = new_ir_paths
    df = df.dropna(subset=['rgb_frame', 'ir_frame'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved processed images to '{output_base_dir}' and CSV to '{output_csv}'.")

# -----------------------
# Example CLI usage
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU-accelerated Adaptive Median Filter (Numba CUDA)")
    parser.add_argument("--input-csv", default="/mnt/c/Users/T2430451/data/dataframes/clahe.csv")
    parser.add_argument("--output-csv", default="/mnt/c/Users/T2430451/data/dataframes/adaptive_median_gpu.csv")
    parser.add_argument("--output-base", default="/mnt/c/Users/T2430451/data/datasets/adaptive-median-gpu")
    parser.add_argument("--window-radius", type=int, default=1, choices=range(1, MAX_WINDOW_RADIUS+1))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = parser.parse_args()

    process_images_gpu(args.input_csv, args.output_csv, args.output_base,
                       window_radius=args.window_radius, threshold=args.threshold, use_gpu=args.use_gpu)
