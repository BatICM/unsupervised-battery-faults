# Early Battery Fault Detection using Multi-Feature Clustering and Unsupervised Scoring

[![License](https://img.shields.io/github/license/BatICM/unsupervised-battery-faults)](https://github.com/BatICM/unsupervised-battery-faults/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

This repository implements the early fault detection algorithm for lithium-ion battery packs described in the paper: "An early fault detection method of series battery packs based on multi-feature clustering and unsupervised scoring" by Wenhao Nie, Zhongwei Deng, et al.

## Overview

This project presents a novel approach for early detection of lithium-ion battery faults in electric vehicles (EVs), addressing a critical safety challenge in the rapidly growing EV market. Using cloud-based battery data, our method employs multi-feature clustering and unsupervised scoring to detect potential faults significantly earlier than traditional Battery Management Systems (BMS).

### Key Features

- **Multi-feature extraction** to characterize battery faults from various perspectives
- **Unsupervised clustering** for real-time anomaly detection without labeled training data
- **Hierarchical warning strategy** to minimize false alarms while providing early detection
- **Validated on real-world EV data** with over 10 days advance warning compared to BMS

## Method

Our algorithm consists of four main components:

1. **Data Preprocessing**: Clean and prepare cloud-based battery data
2. **Feature Extraction**: Extract three key features from battery voltage data:
   - Shannon entropy
   - Cell state value (based on State Representation Methodology)
   - Extended RMSE (Root Mean Square Error)
3. **Unsupervised Scoring**: 
   - Feature normalization
   - DBSCAN clustering to identify outliers
   - Sliding window technique for iterative scoring
4. **Hierarchical Warning**: Two-level warning strategy using thresholds and cumulative sum

## Repository Structure

```
unsupervised-battery-faults/
│
├── data/                                # Raw data directory
│   ├── car1.csv                         # Battery data for car type 1
│   ├── car2.csv                         # Battery data for car type 2
│   ├── car3.csv                         # Battery data for car type 3
│   └── car4.csv                         # Battery data for car type 4
│
├── code_new/                            # Main code directory
│   ├── preprocessing.py                 # Data preprocessing module with vehicle-specific functions
│   │   ├── preprocess_car1()            # Car1-specific preprocessing
│   │   ├── preprocess_car2()            # Car2-specific preprocessing 
│   │   ├── preprocess_car3()            # Car3-specific preprocessing
│   │   └── preprocess_car4()            # Car4-specific preprocessing
│   │
│   ├── feature_extraction/              # Feature extraction modules
│   │   ├── shannon_entropy.py           # Shannon entropy feature extraction
│   │   │   ├── ShannonEn()              # Shannon entropy calculation for single window
│   │   │   └── ShannonEn_K_Ensemble()   # Ensemble Shannon entropy across sliding windows
│   │   │
│   │   ├── srm.py                       # State Relation Method implementation
│   │   │   └── SRM()                    # State Relation Method algorithm
│   │   │
│   │   └── rmse.py                      # Root Mean Square Error feature extraction
│   │       └── calculate_window_mean()  # RMSE calculation with sliding windows
│   │
│   ├── clustering/                      # Clustering and anomaly detection
│   │   ├── dbscan_clustering.py         # DBSCAN clustering implementation
│   │   └── cusum_alarm.py               # CUSUM alarm detection algorithm
│   │       └── cusum_alarm_detection()  # Cumulative sum alarm detection function
│   │
│   ├── visualization/                   # Data visualization utilities
│   │   ├── plot_scores.py               # Score plotting functionality
│   │   └── plot_alarms.py               # Alarm visualization
│   │
│   └── main.py                          # Main script that orchestrates the entire workflow
│
├── output/                              # Output directory for results and plots
│   ├── alarm_plots/                     # Generated alarm plots
│   └── feature_plots/                   # Feature visualization plots
│
├── notebooks/                           # Jupyter notebooks for analysis and exploration
│   ├── data_exploration.ipynb           # Data exploration notebook
│   └── model_evaluation.ipynb           # Model evaluation notebook
│
└── README.md                            # Project documentation

## Installation

Clone this repository and install the required packages:

```bash
# Clone the repository
git clone https://github.com/BatICM/unsupervised-battery-faults.git
cd unsupervised-battery-faults

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```

### Example Notebook

See the `notebooks/demo.ipynb` for a complete example of using the algorithm on sample data.

## Parameters

The algorithm uses the following key parameters:

- **DBSCAN parameters**:
  - `eps` (default: 0.6): The maximum distance between two samples for them to be considered as part of the same neighborhood
  - `min_pts` (default: 3): Minimum number of samples in a neighborhood for a point to be considered a core point

- **Hierarchical warning thresholds**:
  - `threshold_1` (default: 0.5): First-level warning threshold for unsupervised scores
  - `threshold_2` (default: 100): Second-level warning threshold for cumulative sum values

- **Window length**:
  - `window_length` (default: "auto"): Length of sliding window for iterative scoring (defaults to number of cells in the battery pack)

## Results

When tested on data from three different vehicles with battery faults, our method:

1. Successfully detected all faulty cells with 100% accuracy (compared to 50-75% for single-feature methods)
2. Produced zero false alarms (compared to 2500-8000+ for single-feature methods)
3. Provided warnings over 10 days earlier than traditional BMS alerts

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{nie2025early,
  title={An early fault detection method of series battery packs based on multi-feature clustering and unsupervised scoring},
  author={Nie, Wenhao and Deng, Zhongwei and Li, Jinwen and Zhang, Kai and Zhou, Jingjing and Xiang, Fei},
  journal={Energy},
  volume={323},
  pages={135754},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (Grant No. 52472401), Sichuan Science and Technology Program (Grant No. 2024NSFSC0938), and China Postdoctoral Science Foundation (Grant No. 2023T160085).

## Contact

For questions or support, please open an issue on GitHub or contact the authors:
- Wenhao Nie（niewh1817@163.com）
- Zhongwei Deng (dengzw1127@uestc.edu.cn)
