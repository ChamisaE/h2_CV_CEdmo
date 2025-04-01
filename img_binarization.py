#Chamisa Edmo
#CV Spring 2025
#KUID: 2209458

import sys
import numpy as np
from PIL import Image
from collections import deque
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_closing, generate_binary_structure


# --------------------------------------------------------------------------
# 1) Image I/O Functions
# --------------------------------------------------------------------------
def load_as_grayscale(image_path):
    """
    Load an image in grayscale (8-bit) using Pillow.
    Returns a NumPy 2D array of shape (H, W).
    """
    img = Image.open(image_path).convert('L')  # 'L' => 8-bit grayscale
    return np.array(img, dtype=np.uint8)

def save_grayscale(array, out_path):
    """
    Save a 2D NumPy array as an 8-bit grayscale PNG.
    """
    img = Image.fromarray(array, mode='L')
    img.save(out_path)

def save_color(array, out_path):
    """
    Save a 3D NumPy array (H, W, 3) as an RGB PNG.
    """
    img = Image.fromarray(array, mode='RGB')
    img.save(out_path)

# --------------------------------------------------------------------------
# 2) Simple Manual Threshold (alternative to Otsu)
# --------------------------------------------------------------------------

def simple_threshold(gray_img):
    """
    Example manual thresholding:
    - Compute mean intensity, optionally add a margin (like +20).
    - Convert to 255 (foreground) if below threshold, else 0 (background).
    """
    mean_val = gray_img.mean()
    threshold = mean_val + 20
    #tools should be white (255) if gray_img < threshold
    binary_img = np.where(gray_img < threshold, 255, 0).astype(np.uint8)
    return binary_img

# --------------------------------------------------------------------------
# 3) Flood-Fill Labeling (4-Connectivity)
# --------------------------------------------------------------------------

def flood_fill_label(binary_img):
    """
    Label each connected foreground region (pixel=255) with a unique integer.
    Returns: (label_img, number_of_labels).
    Uses BFS for 4-connectivity.
    """
    rows, cols = binary_img.shape
    label_img = np.zeros_like(binary_img, dtype=int)
    current_label = 0

    for r in range(rows):
        for c in range(cols):
            #iff we find a foreground pixel (255) that hasn't been labeled yet
            if binary_img[r, c] == 255 and label_img[r, c] == 0:
                # Start a new connected-component label
                current_label += 1

                #initialize BFS queue with the starting pixel
                queue = deque()
                queue.append((r, c))
                #assign this pixel our new label
                label_img[r, c] = current_label

                #breadth-first search (bfs) to label all connected pixels
                while queue:
                    rr, cc = queue.popleft()
                    #check the four neighbors (4-connectivity)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        #ensure we're within image bounds
                        if 0 <= nr < rows and 0 <= nc < cols:
                            #if neighbor is also foreground (255) and not labeled yet
                            if binary_img[nr, nc] == 255 and label_img[nr, nc] == 0:
                                label_img[nr, nc] = current_label
                                queue.append((nr, nc))

    return label_img, current_label

# --------------------------------------------------------------------------
# 4) Convert Labels to Color Image
# --------------------------------------------------------------------------
def label_to_color(label_img, num_labels):
    """
    Convert integer labels (1..num_labels) into distinct colors in an RGB array.
    """
    height, width = label_img.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)

    #several distinct colors (R, G, B). We can cycle if more than 7 objects.
    color_map = [
        (255,   0,   0),   # Red
        (  0, 255,   0),   # Green
        (  0,   0, 255),   # Blue
        (255, 255,   0),   # Yellow
        (255,   0, 255),   # Magenta
        (  0, 255, 255),   # Cyan
        (127, 127, 127),   # Gray (fallback)
    ]

    #assign each label a color
    for label in range(1, num_labels + 1):
        color_idx = (label - 1) % len(color_map)
        mask = (label_img == label)
        color_img[mask] = color_map[color_idx]

    return color_img

# --------------------------------------------------------------------------
# 5) Main 
# --------------------------------------------------------------------------
def main(image_path):
    # 1) Load image in grayscale
    gray_img = load_as_grayscale(image_path)

    # 2) Threshold
    # --- Option A: Otsu’s threshold
    otsu_val = threshold_otsu(gray_img)
    binary_img = np.where(gray_img < otsu_val, 255, 0).astype(np.uint8)

    # --- Option B: Manual threshold
    #binary_img = simple_threshold(gray_img)

    # 3) (Optional) Morphological closing to merge small gaps
    bin_bool = (binary_img == 255)

    #se = generate_binary_structure(2, 2)  # 2D, 8-connectivity
    # Use a custom structuring element (e.g., 3x3 or 5x5)
    # A 3x3 kernel:
    # se = np.ones((3, 3), dtype=bool)
    # A 5x5 kernel:
    se = np.ones((5, 5), dtype=bool)
    bin_closed = binary_closing(bin_bool, structure=se)
    binary_cleaned = np.where(bin_closed, 255, 0).astype(np.uint8)

    # 4) Save the binary result (for Part I)
    save_grayscale(binary_cleaned, "binary_output.png")
    print("[INFO] Saved binary_output.png")

    # 5) Label connected components (flood fill) on the cleaned image
    label_img, num_objects = flood_fill_label(binary_cleaned)

    # 6) Convert labels to color
    color_img = label_to_color(label_img, num_objects)

    # 7) Save the colored image
    save_color(color_img, "color_output.png")
    print(f"[INFO] Saved color_output.png with {num_objects} distinct object(s).")

# --------------------------------------------------------------------------
# Run the main() if script is called directly
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python img_binarization.py <image_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)
