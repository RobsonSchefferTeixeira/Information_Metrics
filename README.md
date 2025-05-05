# Information_Metrics

**Information_Metrics** is a Python package designed to compute spatial information metrics from neural data, supporting both calcium imaging signals and spikes. It offers tools for analyzing spatial coding properties of neurons, such as place cells, in one-dimensional (1D) and two-dimensional (2D) environments.

## Features

- **Versatile Data Support**: Analyze both calcium imaging and spike train data.
- **Spatial Metrics Computation**: Calculate various spatial information metrics tailored for different data types.
- **Data Processing Utilities**: Utilize a suite of helper functions for data validation, normalization, smoothing, and more.
- **Comprehensive Tutorials**: Access Jupyter notebooks demonstrating usage scenarios and analysis pipelines.

## Upcoming Features

I`m actively working on implementing more advanced analyses to further enhance the capabilities of the package:

- **Decoding Approaches**: Methods to decode behavioral or environmental variables from neural activity.
- **Dimensionality Reduction**: Techniques like PCA, t-SNE, or UMAP to uncover low-dimensional structures in high-dimensional neural data.
- **Assembly Detection**: Tools to identify and analyze co-active neuronal assemblies that may underlie specific cognitive functions.
- **Representational Drift Analysis**: Methods to examine changes in neural representations over time to study stability and plasticity.
- **Place Field Remapping**: Analyses to investigate how place cells alter their firing fields in response to environmental changes.


## Getting Started

### Prerequisites

- Python 3.7 or higher
- Recommended: Create a virtual environment to manage dependencies.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/RobsonSchefferTeixeira/Information_Metrics.git
   cd Information_Metrics
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running Tutorials

Navigate to the `notebooks/` directory and open any of the Jupyter notebooks to explore various analysis examples:

- `tutorial_place_cell_imaging_binarized.ipynb`
- `tutorial_place_cell_imaging_continuous_1D.ipynb`
- `tutorial_place_cell_imaging_continuous_1D_trials.ipynb`
- `tutorial_place_cell_imaging_continuous_2D.ipynb`
- `tutorial_place_cell_spikes.ipynb`
- `signal_simulation.ipynb`

These notebooks provide step-by-step guides on processing data, computing metrics, and interpreting results.
