import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import seaborn as sns
from pathlib import Path
import sys

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_FILE = OUTPUT_DIR / "eda_report.txt"
DATA_DIR = Path(__file__).parent.parent

classes = [
    "Background",
    "Broccoli_green_weeds_1",
    "Broccoli_green_weeds_2",
    "Fallow",
    "Fallow_rough_plow",
    "Fallow_smooth",
    "Stubble",
    "Celery",
    "Grapes_untrained",
    "Soil_vineyard_develop",
    "Corn_senesced_green_weeds",
    "Lettuce_romaine_4wk",
    "Lettuce_romaine_5wk",
    "Lettuce_romaine_6wk",
    "Lettuce_romaine_7wk",
    "Vineyard_untrained",
    "Vineyard_vertical_trellis"
]


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main():
    salinas_image = np.load(DATA_DIR / "salinas_image.npy")
    salinas_labels = np.load(DATA_DIR / "salinas_labels.npy")

    print("Exploratory Data Analysis - Salinas Dataset")
    print("-" * 50)
    print(f"\nDataset loaded")
    print(f"  Image: {salinas_image.shape}")
    print(f"  Labels: {salinas_labels.shape}")

    height, width, num_channels = salinas_image.shape
    print(f"\nImage dimensions:")
    print(f"  Height: {height}, Width: {width}, Hyperspectral Channels: {num_channels}")

    # Sample channels
    print("\nGenerating sample channels visualization...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(salinas_image[:, :, 2], cmap='viridis')
    plt.title('Channel 3')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(salinas_image[:, :, 64], cmap='viridis')
    plt.title('Channel 65')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(salinas_image[:, :, 94], cmap='viridis')
    plt.title('Channel 95')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_channels.png", dpi=100, bbox_inches='tight')
    print("  Saved: sample_channels.png")
    plt.close()

    # Label statistics
    unique_labels = np.unique(salinas_labels)
    print(f"\nLabel statistics:")
    print(f"  Unique labels: {len(unique_labels)}")
    print(f"  Classes defined: {len(classes)}")

    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:len(classes)]

    # Class distribution
    print("Generating class distribution visualization...")
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(
        np.arange(len(classes)).reshape(-1, 1),
        cmap=plt.matplotlib.colors.ListedColormap(colors)
    )
    plt.title('Color-Label Assignments')
    plt.yticks(range(len(classes)), classes, fontsize=8)
    plt.xticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(salinas_labels, cmap=plt.matplotlib.colors.ListedColormap(colors))
    plt.title('Ground Truth')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=100, bbox_inches='tight')
    print("  Saved: class_distribution.png")
    plt.close()

    # Pixel counts
    salinas_image_r = salinas_image.reshape(-1, salinas_image.shape[2])
    salinas_labels_r = salinas_labels.reshape(-1)
    pixel_counts = np.bincount(salinas_labels_r)

    print("\nPixel counts per class:")
    for i in range(len(pixel_counts)):
        if pixel_counts[i] > 0:
            print(f"  {pixel_counts[i]:6d} pixels in category {i:2d} ({classes[i]})")

    print(f"\n  Reshaped image: {salinas_image_r.shape}")
    print(f"  Reshaped labels: {salinas_labels_r.shape}")

    # Spectral signatures
    print("Generating spectral signatures plot...")
    random.seed(7)
    class_indices = [np.where(salinas_labels_r == i)[0] for i in range(len(classes))]
    random_pixels = [random.choice(indices) for indices in class_indices if len(indices) > 0]
    spectral_signatures = salinas_image_r[random_pixels]

    plt.figure(figsize=(12, 6))
    for i, signature in enumerate(spectral_signatures):
        plt.plot(signature, label=classes[i], color=colors[i])

    plt.xlabel('Spectral Band')
    plt.ylabel('Intensity')
    plt.title('Hyperspectral Signatures per Class')
    plt.legend(fontsize=8, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectral_signatures.png", dpi=100, bbox_inches='tight')
    print("  Saved: spectral_signatures.png")
    plt.close()

    # Correlation matrix
    print("Generating correlation matrix heatmap...")
    correlation_matrix = np.corrcoef(salinas_image_r, rowvar=False)

    plt.figure(figsize=(8, 7))
    sns.heatmap(correlation_matrix, cmap='viridis', square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Band Correlation Matrix')
    plt.xlabel('Spectral Band')
    plt.ylabel('Spectral Band')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=100, bbox_inches='tight')
    print("  Saved: correlation_matrix.png")
    plt.close()

    # Summary statistics
    print(f"\nSpectral statistics:")
    print(f"  Mean intensity: {salinas_image_r.mean():.4f}")
    print(f"  Std deviation: {salinas_image_r.std():.4f}")
    print(f"  Min intensity: {salinas_image_r.min():.4f}")
    print(f"  Max intensity: {salinas_image_r.max():.4f}")

    print("\nDone")
    print(f"Report: {REPORT_FILE}")
    print(f"Plots: {OUTPUT_DIR}")


if __name__ == "__main__":
    report_handle = open(REPORT_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, report_handle)
    try:
        main()
    finally:
        sys.stdout = original_stdout
        report_handle.close()

    print("All outputs saved to /output directory.")