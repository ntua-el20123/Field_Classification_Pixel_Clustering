import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from fcmeans import FCM


OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_FILE = OUTPUT_DIR / "default_test_report.txt"
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


def print_pixel_counts(labels, label_name):
    print(f"\n{label_name} pixel counts:")
    pixel_counts = np.bincount(labels)
    for i, count in enumerate(pixel_counts):
        if count > 0:
            class_name = classes[i] if i < len(classes) else f"Cluster_{i}"
            print(f"  {count:6d} pixels in category {i:2d} ({class_name})")


def save_comparison_plot(pred_labels, title, file_name, salinas_labels, colors, height, width):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(
        np.arange(len(classes)).reshape(-1, 1),
        cmap=plt.matplotlib.colors.ListedColormap(colors)
    )
    plt.title('Color-Label Assignments')
    plt.yticks(range(len(classes)), classes, fontsize=8)
    plt.xticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(pred_labels.reshape(height, width), cmap=plt.matplotlib.colors.ListedColormap(colors))
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(salinas_labels, cmap=plt.matplotlib.colors.ListedColormap(colors))
    plt.title('Ground Truth')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=100, bbox_inches='tight')
    print(f"Saved: {file_name}")
    plt.close()


def main():
    salinas_image = np.load(DATA_DIR / "salinas_image.npy")
    salinas_labels = np.load(DATA_DIR / "salinas_labels.npy")

    height, width, _ = salinas_image.shape
    salinas_image_r = salinas_image.reshape(-1, salinas_image.shape[2])
    salinas_labels_r = salinas_labels.reshape(-1)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:len(classes)]

    print("Clustering Test - Salinas Dataset")
    print("-" * 50)
    print(f"Image shape: {salinas_image.shape}")
    print(f"Labels shape: {salinas_labels.shape}")
    print(f"Reshaped image: {salinas_image_r.shape}")

    kmeans = KMeans(n_clusters=17, random_state=7)
    kmeans_labels = kmeans.fit_predict(salinas_image_r)
    print_pixel_counts(kmeans_labels, "K-Means")
    print(f"\nK-Means ARI: {adjusted_rand_score(salinas_labels_r, kmeans_labels):.4f}")
    print(f"K-Means Silhouette: {silhouette_score(salinas_image_r, kmeans_labels):.4f}")
    save_comparison_plot(
        kmeans_labels,
        "K-Means Clustering",
        "kmeans_clustering.png",
        salinas_labels,
        colors,
        height,
        width
    )

    fcm = FCM(n_clusters=17, random_state=7)
    fcm.fit(salinas_image_r)
    fcm_labels = fcm.predict(salinas_image_r)
    print_pixel_counts(fcm_labels, "Fuzzy C-Means")
    print(f"\nFuzzy C-Means ARI: {adjusted_rand_score(salinas_labels_r, fcm_labels):.4f}")
    print(f"Fuzzy C-Means Silhouette: {silhouette_score(salinas_image_r, fcm_labels):.4f}")
    save_comparison_plot(
        fcm_labels,
        "Fuzzy C-Means Clustering",
        "fcm_clustering.png",
        salinas_labels,
        colors,
        height,
        width
    )

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