import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_FILE = OUTPUT_DIR / "pca_test_report.txt"
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

    print("PCA Clustering Test - Salinas Dataset")
    print("-" * 50)
    print(f"Original image shape: {salinas_image_r.shape}")

    pca = PCA(n_components=3)
    salinas_image_3 = pca.fit_transform(salinas_image_r)
    print(f"Reduced image shape: {salinas_image_3.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.6f}")

    kmeans_3 = KMeans(n_clusters=17, random_state=7)
    kmeans_labels_3 = kmeans_3.fit_predict(salinas_image_3)
    print_pixel_counts(kmeans_labels_3, "K-Means (3 PCA components)")
    print(f"\nK-Means ARI: {adjusted_rand_score(salinas_labels_r, kmeans_labels_3):.4f}")
    print(f"K-Means Silhouette: {silhouette_score(salinas_image_3, kmeans_labels_3):.4f}")
    save_comparison_plot(
        kmeans_labels_3,
        "K-Means with 3 PCA Components",
        "pca_kmeans_clustering.png",
        salinas_labels,
        colors,
        height,
        width
    )

    components_range = range(1, 6)
    ari_scores = []
    silhouette_scores = []

    print("\nMetrics across component counts:")
    for n_comp in components_range:
        pca_i = PCA(n_components=n_comp)
        reduced_salinas_i = pca_i.fit_transform(salinas_image_r)

        kmeans_i = KMeans(n_clusters=17, random_state=7)
        kmeans_labels_i = kmeans_i.fit_predict(reduced_salinas_i)

        ari_i = adjusted_rand_score(salinas_labels_r, kmeans_labels_i)
        silhouette_i = silhouette_score(reduced_salinas_i, kmeans_labels_i)
        ari_scores.append(ari_i)
        silhouette_scores.append(silhouette_i)
        print(f"  {n_comp:d} components: ARI={ari_i:.4f}, Silhouette={silhouette_i:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(components_range, ari_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('PCA Components')
    plt.ylabel('Adjusted Rand Index')
    plt.title('Clustering Quality vs Dimensionality')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(components_range, silhouette_scores, marker='s', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('PCA Components')
    plt.ylabel('Silhouette Score')
    plt.title('Cluster Cohesion vs Dimensionality')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_metrics_analysis.png", dpi=100, bbox_inches='tight')
    print(f"\nSaved: pca_metrics_analysis.png")
    plt.close()

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