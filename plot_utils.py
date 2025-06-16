import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import hashlib
import colorsys

def get_unique_color(label):
    """
    Returns a consistent hex color code based on a SHA-1 hash of `label`.
    Two different labels *will almost never* share the same color,
    barring a cryptographic hash collision.

    Parameters
    ----------
    label : str (or something convertible to str)
        The label to map into a color.

    Returns
    -------
    color_hex : str
        A hex color string (e.g. '#7f3a65').
    """
    # Ensure we're dealing with a string
    label_str = str(label)

    # Compute SHA-1 hash of the label
    label_hash = hashlib.sha1(label_str.encode('utf-8')).hexdigest()

    # Convert the hex digest into an integer
    hash_int = int(label_hash, 16)

    # Use part of that integer to define our hue in [0,1]
    # (modulo by 360 first, then divide by 360)
    hue = (hash_int % 360) / 360.0

    # Fix saturation and lightness for a visually distinct color
    saturation = 0.7
    lightness  = 0.5

    # Convert from HLS to RGB in [0,1]
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

    # Scale to [0,255] and format as hex
    r_int, g_int, b_int = int(r*255), int(g*255), int(b*255)
    color_hex = f"#{r_int:02x}{g_int:02x}{b_int:02x}"

    return color_hex



def plot_value_counts(
    value_counts,
    legend_title="Corruption Type",
    y_label="Proportion",
    figsize=(6, 3),
    palette=None,
    save_filename: str = "proportions"
):
    """
    Plot a bar chart from a pandas Series of value counts (or proportions),
    placing the legend outside, removing x-axis labels, and keeping y-axis labels.

    Uses a hash-based method to assign the same color to identical labels
    without storing them in a dictionary.

    Parameters
    ----------
    value_counts : pd.Series
        Series containing counts (or proportions) indexed by category.
    legend_title : str, optional
        Title of the legend.
    y_label : str, optional
        Label for the y-axis.
    figsize : tuple of (float, float), optional
        Figure size in inches (width, height).

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Dynamically compute colors for each label
    labels = value_counts.index
    colors = [get_unique_color(lbl) for lbl in labels]

    # Plot the bar chart with computed colors
    value_counts.plot(kind="bar", color=colors, ax=ax)

    # Build legend handles for each label in the same order
    handles = [
        mpl.patches.Patch(color=c, label=lbl)
        for c, lbl in zip(colors, labels)
    ]

    # Place the legend to the right
    ax.legend(
        handles=handles,
        title=legend_title,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Remove x-axis label and tick labels
    ax.set(xlabel="", ylabel=y_label)
    ax.set_xticklabels([])

    plt.tight_layout()

    # Save the figure

    # Create the output directory if it doesn't exist
    os.makedirs("la_output/plots", exist_ok=True)

    plt.savefig(f"la_output/plots/{save_filename}.png", bbox_inches='tight', dpi=300)

    plt.show()
