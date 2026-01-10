# Scientific Pipelines: Multi-Domain Embedding-Based Analysis

A unified framework for unsupervised scientific discovery using self-supervised vision transformers (DINOv3) across planetary science and astronomy domains.

## Project Overview

This project implements embedding-based analysis pipelines for diverse scientific domains, leveraging state-of-the-art self-supervised learning to enable unsupervised classification, clustering, and anomaly detection without requiring labeled training data.

### Core Philosophy

- **Self-supervised embeddings**: Use DINOv3 (768-dim) to extract rich visual representations
- **Unsupervised discovery**: Cluster, rank, and identify novel patterns without labels
- **Domain-agnostic architecture**: Shared infrastructure adapted to domain-specific physics
- **Reproducible science**: YAML-configured pipelines with full provenance tracking

## Current Domains

### 1. Astronomy: Backyard Worlds Planet 9

**Objective**: Detect moving objects (brown dwarfs, planets, asteroids) in WISE infrared survey data.

**Data**:
- 4-frame flipbook sequences from Zooniverse Backyard Worlds project
- 256×256 pixel WISE infrared images
- ~1,000+ subjects from citizen science observations

**Pipeline**:
```
Flipbook Sequences (4 frames)
    ↓
DINOv3 Embedding Extraction (768-dim per frame)
    ↓
Temporal Aggregation: [mean, std, delta] → 2304-dim sequence vectors
    ↓
HDBSCAN Clustering (behavior categories)
    ↓
Multi-Factor Scoring (6 components)
    ↓
Ranked Candidate List
```

**Scoring Components** (MovingObjectScorer):
- **Motion Magnitude (30%)**: Non-zero displacement detection via delta embeddings
- **Motion Consistency (25%)**: Linear vs erratic motion from std embeddings
- **Temporal Coherence (20%)**: Object presence across all frames
- **Novelty Score (15%)**: kNN-based outlier detection in embedding space
- **Dipole Signature (10%)**: WISE-specific bright/dark pair patterns
- **Artifact Penalty (-10%)**: Edge effects, saturation, known flags

**Key Innovation**:
Temporal sequence encoder captures motion in embedding space without explicit optical flow or point matching.

**Outputs**:
- `moving_object_ranking.csv`: Top-N candidates with component scores
- `subject_clusters.csv`: Behavioral groupings (fast movers, slow movers, stationary, artifacts)
- `embeddings.parquet`: 2304-dim sequence embeddings for analysis

**Configuration**: `configs/pipelines/backyard_worlds_movers.yaml`

**Usage**:
```bash
pixi run python scripts/run_backyard_worlds.py \
  --config configs/pipelines/backyard_worlds_movers.yaml
```

---

### 2. Planetary Science: Mars CTX Terrain Classification

**Objective**: Identify and classify geological terrain types from Mars Reconnaissance Orbiter CTX imagery.

**Data**:
- Context Camera (CTX) grayscale imagery (~6 m/pixel resolution)
- Processed through ISIS3 (map projection, calibration)
- Large-scale mosaics tiled into 256×256 patches

**Pipeline**:
```
CTX GeoTIFF Mosaics
    ↓
Intelligent Tiling (quality filtering, overlap)
    ↓
DINOv3 Embedding Extraction (768-dim per tile)
    ↓
HDBSCAN Clustering (terrain categories)
    ↓
Novelty Detection (unusual geological features)
    ↓
Spatial Analysis & Visualization
```

**Key Features**:
- **Quality filtering**: Excludes low-information tiles (uniform, edge artifacts)
- **Spatial awareness**: Preserves geographic coordinates for terrain mapping
- **Scalability**: Processes multi-gigabyte CTX mosaics efficiently
- **Geological interpretation**: Clusters correspond to dunes, bedrock, craters, etc.

**Outputs**:
- `embeddings.parquet`: Tile embeddings with spatial metadata
- `terrain_clusters.csv`: Cluster assignments per tile
- `novelty_scores.csv`: Anomaly rankings for discovery

**Configuration**: `configs/pipelines/ctx_terrain.yaml`

**ISIS3 Integration**: Automated download and processing via USGS ISIS3 tools (see `ISIS3_SETUP.md`)

---

## Core Infrastructure

### Embedding Extraction

**DINOv3Extractor** (`src/scientific_pipelines/core/embeddings/dinov3.py`)

- **Models supported**:
  - `dinov3_vitb14` (768-dim, recommended) - 86M parameters
  - `dinov3_vits14` (384-dim, lightweight) - 21M parameters
  - `dinov3_vitl14` (1024-dim, high capacity) - 300M parameters
  - `dinov3_vitg14` (1536-dim, maximum quality) - 1.1B parameters

- **Features**:
  - GPU/CPU support with CUDA acceleration
  - Optional FP16 half-precision for speed
  - Batch processing with configurable workers
  - PyTorch Hub integration (facebookresearch/dinov3)

- **Preprocessing**: ImageNet normalization, 224×224 resize, RGB conversion

### Clustering

**HDBSCANClusterer** (`src/scientific_pipelines/core/clustering/hdbscan_clusterer.py`)

- **Algorithm**: Hierarchical Density-Based Spatial Clustering
- **Key advantages**:
  - Automatic cluster number determination
  - Handles outliers (assigns label -1)
  - Soft cluster membership probabilities
  - Robust to noise and varying densities

- **Tunable parameters**:
  - `min_cluster_size`: 0.5-1% of dataset (e.g., 50 for 10k samples)
  - `min_samples`: 1/5 of min_cluster_size for robustness
  - `metric`: euclidean (default), cosine, manhattan

### Novelty Detection

**NoveltyDetector** (`src/scientific_pipelines/core/clustering/novelty.py`)

- **Methods**:
  1. **kNN Distance** (recommended): Average distance to k-nearest neighbors
  2. **Local Outlier Factor (LOF)**: Density-based anomaly scoring
  3. **Isolation Forest**: Tree-based outlier detection
  4. **Centroid Distance**: Distance to cluster centers

- **FAISS acceleration**: Optional GPU-accelerated kNN for large datasets

### Sequence Encoding (Backyard Worlds)

**FlipbookSequenceEncoder** (`src/scientific_pipelines/astronomy/backyard_worlds/sequence_encoder.py`)

- **Purpose**: Encode multi-frame temporal sequences into single vectors
- **Strategy**:
  1. Extract DINOv3 embeddings for each frame independently
  2. Compute temporal statistics:
     - `mean_embed`: Average appearance across frames
     - `std_embed`: Appearance variability (coherence proxy)
     - `mean_delta`: Frame-to-frame changes (motion proxy)
  3. Concatenate: `[mean, std, delta]` → 3×D dimensional vector

- **Output**: 2304-dim vectors (768 × 3) capturing both appearance and motion

---

## Project Structure

```
mars_astrobio/
├── configs/
│   └── pipelines/
│       ├── backyard_worlds.yaml          # Brown dwarf detection
│       ├── backyard_worlds_movers.yaml   # Moving object detection
│       └── ctx_terrain.yaml              # Mars terrain classification
├── src/
│   └── scientific_pipelines/
│       ├── core/
│       │   ├── embeddings/
│       │   │   ├── base.py               # Abstract base class
│       │   │   ├── dinov3.py             # DINOv3 extractor
│       │   │   └── pipeline.py           # Batch processing
│       │   └── clustering/
│       │       ├── hdbscan_clusterer.py  # Clustering
│       │       └── novelty.py            # Anomaly detection
│       ├── astronomy/
│       │   └── backyard_worlds/
│       │       ├── downloader.py         # Zooniverse Panoptes API
│       │       ├── sequence_encoder.py   # Temporal aggregation
│       │       ├── brown_dwarf_scorer.py # Brown dwarf heuristics
│       │       ├── moving_object_scorer.py # General mover scoring
│       │       └── pipeline.py           # End-to-end pipeline
│       └── planetary/
│           └── mars/
│               └── ctx/
│                   ├── downloader.py     # USGS ODE API
│                   ├── tiling.py         # Intelligent tile extraction
│                   └── pipeline.py       # Terrain classification
├── scripts/
│   ├── download_backyard_worlds.py
│   ├── process_zooniverse_classifications.py
│   └── setup_isis3.sh                    # ISIS3 installation
├── data/                                 # .gitignored
│   ├── backyard_worlds/
│   └── raw/ctx/
├── outputs/                              # .gitignored
│   ├── backyard_worlds_movers/
│   └── ctx_terrain/
└── pyproject.toml                        # Pixi dependencies
```

---

## Dependencies

**Managed via Pixi** (see `pyproject.toml`):

### Core ML Stack
- `torch>=2.5.0`, `torchvision>=0.20.0` - Deep learning
- `hdbscan>=0.8.33` - Clustering
- `faiss-cpu>=1.7.4` - Fast kNN search
- `scikit-learn>=1.4.0` - ML utilities
- `timm>=0.9.0` - Vision model loading

### Scientific Computing
- `numpy>=1.26.0`, `pandas>=2.2.0` - Data manipulation
- `scipy>=1.12.0` - Scientific algorithms
- `pyarrow>=14.0.0` - Efficient columnar storage

### Domain-Specific
- `panoptes-client>=1.6.0` - Zooniverse API
- `astropy>=6.0.0` - Astronomical data formats (FITS)
- `gdal>=3.8.0` - Geospatial raster processing
- `pillow>=10.2.0` - Image I/O

### Visualization (optional)
- `matplotlib>=3.8.0` - Plotting
- `seaborn>=0.13.0` - Statistical visualization
- `umap-learn>=0.5.5` - Dimensionality reduction

---

## Configuration System

All pipelines use YAML configuration for reproducibility:

```yaml
# Example: configs/pipelines/backyard_worlds_movers.yaml

pipeline_name: "backyard_worlds_moving_objects"
output_dir: "outputs/backyard_worlds_movers"

scorer_type: "moving_object"  # or "brown_dwarf"

embedding:
  model_name: "dinov3_vitb14"
  device: "cuda"
  use_half_precision: false

clustering:
  min_cluster_size: 50
  min_samples: 10
  metric: "euclidean"

moving_object_scoring:
  weights:
    motion_magnitude: 0.30
    motion_consistency: 0.25
    temporal_coherence: 0.20
    novelty: 0.15
    dipole_signature: 0.10
    artifact_penalty: -0.10
  optimal_motion_magnitude: 0.15

novelty:
  method: "knn"
  k: 10
  use_faiss: true
```

---

## Key Publications & References

**DINOv2/DINOv3**:
- Oquab et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision"
- Meta AI Research (2024). "DINOv3: Scaling Vision Transformers"

**HDBSCAN**:
- McInnes et al. (2017). "hdbscan: Hierarchical density based clustering"
- Campello et al. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates"

**Backyard Worlds**:
- Kuchner et al. (2017). "Backyard Worlds: Planet 9" - Zooniverse citizen science project
- WISE Survey: Wright et al. (2010). "The Wide-field Infrared Survey Explorer"

**CTX**:
- Malin et al. (2007). "Context Camera Investigation on Mars Reconnaissance Orbiter"
- ISIS3: USGS Astrogeology Science Center processing system

---

## Validation Strategies

### Unsupervised Validation (No Labels Required)

1. **Distribution Analysis**
   - Plot embedding distributions (UMAP/t-SNE)
   - Verify expected physical patterns (e.g., motion distribution bimodality)

2. **Cluster Coherence**
   - Silhouette scores for cluster quality
   - Visual inspection of cluster examples
   - Physics-based interpretation (e.g., terrain types, motion patterns)

3. **Novelty Calibration**
   - Outlier score distributions
   - Top-K visual inspection for true anomalies
   - Cross-validation with domain experts

4. **Component Correlation**
   - Score component independence (correlation matrix)
   - Ablation studies (remove one component, measure impact)

5. **Cross-Pipeline Validation**
   - Compare brown dwarf vs moving object scorers
   - Overlap analysis (subset relationship verification)

### Semi-Supervised Validation (When Available)

1. **Zooniverse Classifications**
   - Use volunteer annotations as weak labels
   - Compute precision@K, recall@K metrics
   - Calibrate scoring weights based on agreement

2. **Expert Labeling**
   - Sample top-N candidates for expert review
   - Iteratively refine scoring heuristics
   - Active learning feedback loop

---

## Future Directions

### Near-Term Enhancements

1. **Supervised Fine-Tuning**
   - If labels become available, fine-tune DINOv3 on domain data
   - Train lightweight classification heads on frozen embeddings
   - Multi-task learning (e.g., motion + morphology)

2. **Enhanced Motion Modeling**
   - Save per-frame deltas for true consistency scoring
   - Optical flow integration for motion verification
   - Trajectory prediction for fast movers

3. **Multi-Modal Fusion**
   - Combine visual embeddings with spectral data (e.g., WISE photometry)
   - Text embeddings for metadata (observation logs, classifications)
   - Graph neural networks for spatial relationships (terrain mapping)

4. **Active Learning**
   - Use top-ranked candidates to bootstrap supervised models
   - Uncertainty sampling for optimal labeling
   - Human-in-the-loop validation workflows

### Long-Term Vision

1. **Cross-Domain Transfer**
   - Train meta-learners across astronomy + planetary science
   - Few-shot adaptation to new domains (e.g., exoplanet transits)
   - Universal scientific embedding space

2. **Real-Time Processing**
   - Deploy pipelines for live data streams (e.g., MRO downlinks)
   - Automated alert systems for novel discoveries
   - Integration with observational scheduling

3. **Foundation Models**
   - Domain-specific vision transformers (AstroViT, PlanetViT)
   - Pre-training on massive unlabeled scientific archives
   - Zero-shot classification for new categories

4. **Explainability**
   - Attention visualization for discovery interpretation
   - Concept activation vectors for physics-grounded features
   - Counterfactual explanations for scoring decisions

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{scientific_pipelines_2025,
  author = {Ackerman, David},
  title = {Scientific Pipelines: Multi-Domain Embedding-Based Analysis},
  year = {2025},
  url = {https://github.com/yourusername/mars_astrobio},
  note = {DINOv3-based unsupervised discovery for planetary science and astronomy}
}
```

---

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mars_astrobio.git
cd mars_astrobio

# Install dependencies with Pixi
pixi install

# Activate environment
pixi shell
```

### Quick Start: Backyard Worlds

```bash
# Download sample data (1000 subjects)
pixi run python scripts/download_backyard_worlds.py --limit 1000

# Run moving object detection pipeline
pixi run python -m scientific_pipelines.astronomy.backyard_worlds.pipeline \
  --config configs/pipelines/backyard_worlds_movers.yaml

# Results in: outputs/backyard_worlds_movers/moving_object_ranking.csv
```

### Quick Start: CTX Terrain

```bash
# Set up ISIS3 (required for CTX processing)
./setup_isis3.sh

# Download CTX image
pixi run python src/scientific_pipelines/planetary/mars/ctx/downloader.py \
  --product-id P01_001334_1823_XI_02N282W

# Run terrain classification
pixi run python -m scientific_pipelines.planetary.mars.ctx.pipeline \
  --config configs/pipelines/ctx_terrain.yaml

# Results in: outputs/ctx_terrain/terrain_clusters.csv
```

---

## Documentation

- **[ISIS3_SETUP.md](ISIS3_SETUP.md)**: Setting up USGS ISIS3 for CTX processing
- **[QUICKSTART_ISIS3.md](QUICKSTART_ISIS3.md)**: CTX pipeline tutorial
- **[CTX_ISIS3_EXAMPLE.md](CTX_ISIS3_EXAMPLE.md)**: End-to-end CTX workflow

---

## Contributing

This is a research codebase under active development. Contributions are welcome:

1. **Bug Reports**: Open GitHub issues with reproducible examples
2. **Feature Requests**: Propose new domains, scorers, or analysis methods
3. **Pull Requests**: Add new pipelines following existing architecture patterns

### Code Style
- Follow existing patterns (see `BrownDwarfScorer` or `CTXTerrainPipeline`)
- YAML configs for all tunable parameters
- Comprehensive docstrings (Google style)
- Type hints for public APIs

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **DINOv3**: Meta AI Research (facebookresearch/dinov3)
- **Backyard Worlds**: Marc Kuchner (NASA GSFC), Zooniverse team, citizen scientists
- **CTX Data**: NASA/JPL, Mars Reconnaissance Orbiter team
- **ISIS3**: USGS Astrogeology Science Center
- **Compute**: Janelia Research Campus Scientific Computing

---

## Contact

David Ackerman
Janelia Research Campus
ackermand@janelia.hhmi.org

**Project Repository**: https://github.com/yourusername/mars_astrobio
**Issues**: https://github.com/yourusername/mars_astrobio/issues
