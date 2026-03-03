# Hyperspectral Image Classification via Pixel Clustering

Unsupervised classification of the Salinas hyperspectral scene using K-Means and Fuzzy C-Means clustering, with optional PCA-based dimensionality reduction.

## Background

The Salinas dataset is a 512x217 hyperspectral image captured by the AVIRIS sensor over the Salinas Valley, California. Each pixel is described by 204 spectral bands and belongs to one of 17 land-cover classes (including background). The goal is to assign each pixel to a class using unsupervised clustering and compare the result against ground truth labels.

## Project Structure

```
source/
    eda.py            Exploratory data analysis and visualization
    default_test.py   K-Means and Fuzzy C-Means on raw spectral data
    pca_test.py       K-Means on PCA-reduced data with component sweep
output/               Generated plots and text reports
salinas_image.npy     Hyperspectral image array (512, 217, 204)
salinas_labels.npy    Ground truth label map (512, 217)
```

## Pipeline

The project is organized into three standalone scripts, each covering a distinct stage:

### Exploratory Analysis (`eda.py`)

Loads the dataset and produces a visual and statistical summary: sample spectral channels, ground truth class distribution, per-class pixel counts, spectral signatures of representative pixels, and a band correlation matrix.

### Baseline Clustering (`default_test.py`)

Applies K-Means and Fuzzy C-Means (17 clusters, matching the number of ground truth classes) directly on the full 204-dimensional spectral data. Reports per-cluster pixel counts, Adjusted Rand Index (ARI), and Silhouette Score. Saves side-by-side comparison plots against ground truth.

### PCA-Reduced Clustering (`pca_test.py`)

Reduces the spectral data to 3 principal components via PCA, then applies K-Means. Additionally sweeps PCA components from 1 to 5, recording ARI and Silhouette at each level, and plots the resulting metric curves.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the scripts in order from the project root:

```bash
python source/eda.py
python source/default_test.py
python source/pca_test.py
```

Each script writes its console output to a report file in `output/` and saves all plots there as well.

## Output

After a full run, the `output/` directory contains:

| File | Source | Description |
|---|---|---|
| `eda_report.txt` | eda.py | Full EDA console log |
| `sample_channels.png` | eda.py | Three sample spectral channels |
| `class_distribution.png` | eda.py | Ground truth color map |
| `spectral_signatures.png` | eda.py | Per-class spectral curves |
| `correlation_matrix.png` | eda.py | Band-to-band correlation heatmap |
| `default_test_report.txt` | default_test.py | Clustering metrics and pixel counts |
| `kmeans_clustering.png` | default_test.py | K-Means vs ground truth |
| `fcm_clustering.png` | default_test.py | Fuzzy C-Means vs ground truth |
| `pca_test_report.txt` | pca_test.py | PCA clustering metrics |
| `pca_kmeans_clustering.png` | pca_test.py | PCA K-Means vs ground truth |
| `pca_metrics_analysis.png` | pca_test.py | ARI and Silhouette vs component count |

## Evaluation Metrics

- **Adjusted Rand Index (ARI)**: Measures agreement between predicted clusters and ground truth labels, corrected for chance. Ranges from -1 to 1; higher is better.
- **Silhouette Score**: Measures how similar each point is to its own cluster versus neighboring clusters. Ranges from -1 to 1; higher indicates better-defined clusters.

## Results

### Dataset Characteristics

The image contains 111,104 pixels across 204 spectral bands. Class sizes are heavily imbalanced: Background dominates with 56,975 pixels (~51%), while the smallest class (Lettuce_romaine_6wk) has only 916 pixels. Band intensities range from -11 to 9,207 (mean 1,196, std 1,098). The correlation matrix reveals strong inter-band correlation, which motivates the use of PCA.

### Baseline Clustering on Raw Data

| Method | ARI | Silhouette |
|---|---|---|
| K-Means | 0.1394 | 0.3874 |
| Fuzzy C-Means | 0.1311 | 0.3462 |

Both methods achieve low ARI values, indicating poor agreement with the ground truth when clustering directly on the full 204-dimensional data. K-Means slightly outperforms Fuzzy C-Means on both metrics. The moderate Silhouette scores suggest internally coherent clusters that simply do not align well with the labeled classes.

### PCA-Reduced Clustering

PCA with 3 components captures 99.14% of the total variance (74.5% + 23.5% + 1.1%).

| Components | ARI | Silhouette |
|---|---|---|
| 1 | 0.1266 | 0.5346 |
| 2 | 0.1648 | 0.4725 |
| 3 | 0.1675 | 0.4329 |
| 4 | 0.1495 | 0.4320 |
| 5 | 0.1449 | 0.4166 |

K-Means on 3 PCA components yields an ARI of 0.1675, a noticeable improvement over the raw-data baseline (0.1394). Silhouette also increases from 0.3874 to 0.4329. Performance peaks at 3 components and degrades slightly beyond that, consistent with the variance distribution concentrated in the first three principal components.

### Observations

- **Dimensionality reduction helps.** PCA removes redundant spectral correlation and improves both clustering quality and cluster separation. Three components are sufficient to retain over 99% of variance.
- **ARI remains low overall.** Unsupervised clustering with 17 classes on spectrally overlapping land-cover types is inherently difficult. The low ARI across all configurations reflects the fundamental limitation of purely spectral, pixel-level clustering without spatial context.
- **Silhouette and ARI diverge.** Higher Silhouette at 1 component (0.5346) but lower ARI (0.1266) shows that fewer clusters in a lower-dimensional space are geometrically tighter, but this does not translate to correct label assignment. The trade-off suggests that spectral similarity alone is insufficient to distinguish all 17 classes.
- **K-Means vs Fuzzy C-Means.** On this dataset, K-Means consistently outperforms Fuzzy C-Means. The soft assignments of FCM do not provide an advantage when cluster boundaries in spectral space are ambiguous across many classes.
- **Class imbalance matters.** The dominant Background class (51% of pixels) biases clustering toward large, spectrally homogeneous regions, leaving smaller classes underrepresented in the cluster assignments.

## Dataset

The Salinas scene is publicly available from the [GIC - Universidad del Pais Vasco](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas).

Place `salinas_image.npy` and `salinas_labels.npy` in the project root before running.
