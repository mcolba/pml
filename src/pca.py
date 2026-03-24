import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
from sklearn.decomposition import PCA

"""
https://www.demographic-research.org/volumes/vol44/19/44-19.pdf
    
"""


def pca_cps_2(x: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Apply PCA and compute cos² values for each observation.

    Cos² represents the quality of representation of each observation
    on the principal components. It measures how well each observation
    is represented by each principal component.

    Parameters:
        x: Input data array of shape (n_samples, n_features)
        k: Number of principal components to compute cos² for

    Returns:
        Array of shape (n_samples, k) containing cos² values
    """
    # Fit PCA with k components
    pca = PCA(n_components=k)

    # Transform data to get coordinates in PC space
    scores = pca.fit_transform(x)

    # Compute squared distance of each observation from the origin in original centered space
    x_centered = np.asarray(x - np.mean(x, axis=0))
    squared_distances = np.sum(x_centered**2, axis=1).reshape(-1, 1)

    # Compute cos² = (score on PC)² / (total squared distance)
    cos2 = (scores**2) / squared_distances

    return cos2


def blend_3_colors(w: np.ndarray) -> str:
    """
    Blend 3 colors based on 3 input weights and return a hex color code.

    Uses complementary color scheme:
    - High w1 → Cyan (low red)
    - High w2 → Magenta (low green)
    - High w3 → Yellow (low blue)

    Parameters:
        w: Array-like of 3 weights (w1, w2, w3)

    Returns:
        Hex color code string (e.g., '#FF5500')
    """
    # Normalize weights to sum to 1
    total = sum(w)
    if total == 0:
        w1, w2, w3 = 1 / 3, 1 / 3, 1 / 3
    else:
        w1, w2, w3 = w[0] / total, w[1] / total, w[2] / total

    # Complementary color scheme: each weight reduces one color channel
    r = int(255 * (1 - w1))  # High w1 → low red → Cyan
    g = int(255 * (1 - w2))  # High w2 → low green → Magenta
    b = int(255 * (1 - w3))  # High w3 → low blue → Yellow

    # Clamp values to valid RGB range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Return hex color code
    return f"#{r:02X}{g:02X}{b:02X}"


def create_triangle_legend_ternary(
    ax=None,
    labels: tuple = ("PC1", "PC2", "PC3"),
    scale: int = 20,
):
    """
    Create a triangular color legend using python-ternary library.

    Coordinate convention (python-ternary): (left, right, top)

    Uses complementary color scheme matching blend_3_colors:
    - Top vertex (PC1 high)  -> Cyan    (low red)
    - Left vertex (PC2 high) -> Magenta (low green)
    - Right vertex (PC3 high)-> Yellow  (low blue)
    """
    from ternary.helpers import simplex_iterator

    # Color function using complementary color scheme (matches blend_3_colors)
    def color_point(x, y, z, scale):
        # Complementary colors: each coordinate reduces one channel
        r = 1 - (y / float(scale))  # High y (top) → low red → Cyan (PC1)
        g = 1 - (z / float(scale))  # High z (left) → low green → Magenta (PC2)
        b = 1 - (x / float(scale))  # High x (right) → low blue → Yellow (PC3)
        return (r, g, b, 1.0)

    # Pre-compute heatmap data as dictionary
    def generate_heatmap_data(scale):
        d = dict()
        for i, j, k in simplex_iterator(scale):
            d[(i, j, k)] = color_point(i, j, k, scale)
        return d

    # Create ternary figure
    if ax is None:
        figure, tax = ternary.figure(scale=scale)
        figure.set_size_inches(5, 8)
    else:
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
        figure = ax.get_figure()

    # Generate color data and fill triangle using heatmap
    data = generate_heatmap_data(scale)
    tax.heatmap(data, style="triangular", colorbar=False, use_rgba=True)

    # Set labels at corners with offset to avoid overlap
    fontsize = 10
    tax.top_corner_label(labels[0], fontsize=fontsize, fontweight="bold")
    tax.left_corner_label(labels[1], fontsize=fontsize, fontweight="bold", offset=-0.1)
    tax.right_corner_label(labels[2], fontsize=fontsize, fontweight="bold", offset=-0.1)

    # Remove axis
    tax.boundary()
    tax.get_axes().axis("off")

    return figure, tax


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_yield_curves.csv")
    df.diff().dropna(inplace=True)
    w = pca_cps_2(df, 3)
    col = np.apply_along_axis(blend_3_colors, 1, w)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df.index, df.index, c=col)

    # Add triangle legend as inset in corner
    # [left, bottom, width, height] in figure coordinates (0-1)
    ax_inset = fig.add_axes([0.2, 0.65, 0.25, 0.25])
    create_triangle_legend_ternary(ax=ax_inset, scale=20)

    plt.show()
