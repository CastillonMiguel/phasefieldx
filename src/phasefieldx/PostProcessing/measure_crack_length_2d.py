"""
Measure crack length in 2D phase-field simulations
==================================================

This script provides a post-processing workflow for measuring crack length in 2D phase-field fracture
simulations, following the procedure presented in :footcite:t:`Castillon2025_arxiv`. It is designed to
work with simulation outputs in VTU format (e.g., from dolfinx or FEniCSx), where the phase-field
variable ('phi') indicates the presence of a crack.

Workflow overview
-----------------
1. VTU to PNG conversion: For each .vtu file in the simulation output folder, the mesh is visualized,
    and the crack region (where phi > 0.95) is highlighted in red. The resulting image is saved as a PNG
    for further processing.

2. Crack skeletonization and length measurement: Each PNG image is processed to extract the red crack
    region, skeletonize it (reducing it to a 1-pixel-wide centerline), and count the number of pixels
    in the skeleton. The pixel count is converted to a physical length using the known specimen size and
    image resolution.

3. Saving results: The measured crack lengths for all time steps are saved to a text file
    (`crack_lengths.txt`) with a column header 'a' for easy loading into pandas or other analysis tools.

4. Visualization: Optionally, a GIF is generated showing the crack evolution, with the measured crack
    length overlaid on each frame.

Requirements
------------
- pyvista
- matplotlib
- scikit-image
- numpy
- imageio

Usage example
-------------
Set your simulation folder and specimen size, then call:

     folder_simulation = "path/to/your/simulation/folder"
     measure_crack(folder_simulation, physical_horizontal=1.0, physical_vertical=1.0)

This will generate PNG images, measure crack lengths, and create a GIF in the specified folder.

Functions
---------
- get_black_bbox(image): Finds the bounding box of the specimen in an image.
- generate_crack_images(input_folder, output_folder): Converts VTU files to PNG images with cracks highlighted.
- measure_crack_lengths(output_folder, dx, dy): Measures crack lengths from PNG images and saves results.
- generate_crack_gif(output_folder, crack_lengths, gif_name): Creates a GIF showing crack evolution.
- measure_crack(folder_simulation, physical_horizontal, physical_vertical): Main workflow function.
"""

import os
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except Exception:
    plt = None
    mpimg = None

# Optional imports: only required if user calls functions that need them.
try:
    import pyvista as pv
except Exception:
    pv = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from skimage.morphology import skeletonize
    from skimage.color import rgb2hsv
except Exception:
    skeletonize = None
    rgb2hsv = None


def get_black_bbox(image):
    """
    Find the bounding box of the black (background/specimen) region in an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB).

    Returns
    -------
    min_row, max_row, min_col, max_col : int
        Indices defining the bounding box of the black region.
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = np.mean(image[..., :3], axis=2)
    else:
        gray = image
    # Threshold for black
    black_mask = gray < 0.2
    rows = np.any(black_mask, axis=1)
    cols = np.any(black_mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    return min_row, max_row, min_col, max_col


def generate_crack_images(input_folder, output_folder):
    """
    Generate PNG images from .vtu files, highlighting the crack region in red.

    For each .vtu file in the input folder, this function:
    - Loads the mesh and phase-field data.
    - Thresholds the 'phi' field to extract the crack region.
    - Plots the mesh in black and the crack in red.
    - Saves the result as a PNG image in the output folder.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing .vtu files.
    output_folder : str
        Path to the folder where PNG images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    pv.global_theme.allow_empty_mesh = True

    # List all .vtu files (skip the first if needed)
    pvtu_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.vtu')])[1:]

    for fname in pvtu_files:
        file_path = os.path.join(input_folder, fname)
        file_vtu = pv.read(file_path)
        crack = file_vtu.threshold(value=0.95, scalars='phi', invert=False)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(file_vtu, color='black', opacity=1.0)
        plotter.add_mesh(crack, color='red', opacity=1.0)
        plotter.camera_position = 'xy'
        img_path = os.path.join(output_folder, fname.replace('.vtu', '.png'))
        plotter.show(screenshot=img_path, window_size=(1024, 1024))
        plotter.close()

    print(f"Saved images to {output_folder}")


def measure_crack_lengths(output_folder, dx, dy):
    """
    Measure crack lengths from PNG images and save results to a text file.

    For each PNG image in the output folder:
    - Converts the image to HSV and thresholds for red pixels (crack).
    - Skeletonizes the crack region to obtain a 1-pixel wide centerline.
    - Counts the number of skeleton pixels and converts to physical length.

    Parameters
    ----------
    output_folder : str
        Path to the folder containing PNG images.
    dx : float
        Physical size of a pixel in the horizontal direction.
    dy : float
        Physical size of a pixel in the vertical direction (not used, but available).

    Returns
    -------
    crack_lengths : np.ndarray
        Array of measured crack lengths for each image.
    """
    # Ensure optional dependencies are available
    if skeletonize is None or rgb2hsv is None:
        raise ImportError(
            "measure_crack_lengths requires scikit-image. "
            "Install it with `pip install scikit-image` or avoid calling this function."
        )

    image_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])

    crack_lengths = []
    for fname in image_files:
        img_path = os.path.join(output_folder, fname)
        img = mpimg.imread(img_path)

        pixel_length = dx  # Use dx for pixel-to-physical conversion
        # Convert to HSV and threshold for red pixels (crack)
        hsv = rgb2hsv(img[..., :3])
        red_mask = ((hsv[..., 0] < 0.05) | (hsv[..., 0] > 0.95)) & (hsv[..., 1] > 0.5) & (hsv[..., 2] > 0.2)

        # Skeletonize to get the crack centerline
        skeleton = skeletonize(red_mask)
        crack_lengths.append(skeleton.sum())

    # Convert pixel count to physical length
    crack_lengths = np.array(crack_lengths) * pixel_length

    # Save crack lengths to a text file with column name 'a'
    np.savetxt(os.path.join(output_folder, "crack_lengths.txt"), crack_lengths, header="a", comments='')

    print("Crack lengths:", crack_lengths)
    return crack_lengths


def generate_crack_gif(output_folder, crack_lengths, gif_name="crack_evolution.gif"):
    """
    Generate a GIF showing crack evolution, overlaying the measured crack length on each frame.

    Parameters
    ----------
    output_folder : str
        Path to the folder containing PNG images.
    crack_lengths : array-like
        Crack lengths to overlay on each frame.
    gif_name : str, optional
        Name of the output GIF file (default: "crack_evolution.gif").
    """
    if imageio is None:
        raise ImportError(
            "generate_crack_gif requires imageio. "
            "Install it with `pip install imageio` or avoid calling this function."
        )
        
    image_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
    images = []

    for fname, crack_length in zip(image_files, crack_lengths):
        img_path = os.path.join(output_folder, fname)
        img = imageio.imread(img_path)

        # Overlay crack length using matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')
        ax.text(0.05, 0.95, f"Crack length: {crack_length:.3f}", color='yellow',
                fontsize=18, fontweight='bold', ha='left', va='top', transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
        fig.canvas.draw()

        # Get RGBA buffer and convert to numpy array
        frame = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB if needed
        if frame.shape[2] == 4:
            frame = frame[..., :3]
        images.append(frame)
        plt.close(fig)

    gif_path = os.path.join(output_folder, gif_name)
    imageio.mimsave(gif_path, images, duration=0.7)
    print(f"GIF saved to {gif_path}")


def measure_crack(folder_simulation, physical_horizontal, physical_vertical):
    """
    Main workflow to measure crack lengths and generate images and GIF.

    This function:
    - Generates PNG images from VTU files, highlighting cracks.
    - Detects the specimen bounding box and computes pixel size.
    - Measures crack lengths from images and saves results.
    - Generates a GIF showing crack evolution.

    Parameters
    ----------
    folder_simulation : str
        Path to the simulation folder containing 'paraview-solutions_vtu'.
    physical_horizontal : float
        Physical width of the specimen (used to compute pixel size).
    physical_vertical : float
        Physical height of the specimen (used to compute pixel size).
    """
    input_folder = os.path.join(folder_simulation, "paraview-solutions_vtu")
    output_folder = os.path.join(folder_simulation, "paraview_images")

    generate_crack_images(input_folder, output_folder)

    # Use the first image to determine pixel size
    image_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
    img_path = os.path.join(output_folder, image_files[0])
    img = mpimg.imread(img_path)

    min_row, max_row, min_col, max_col = get_black_bbox(img)
    horizontal_pixels = max_col - min_col + 1
    vertical_pixels = max_row - min_row + 1

    # Compute pixel size in physical units
    dx = physical_horizontal / horizontal_pixels
    dy = physical_vertical / vertical_pixels

    print(f"Detected specimen bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
    print(f"Horizontal pixels: {horizontal_pixels}, Vertical pixels: {vertical_pixels}")
    print(f"Pixel size: dx={dx:.6f}, dy={dy:.6f}")

    crack_lengths = measure_crack_lengths(output_folder, dx, dy)
    generate_crack_gif(output_folder, crack_lengths)
