# Logo Similarity Grouping Project

A Flask-based application that scrapes logos from domains, preprocesses the images, extracts visual features, and groups similar logos. The project utilizes multiple modules to handle scraping, preprocessing, feature extraction (e.g., color palettes, pHash, CNN-based features), and grouping using various clustering algorithms.

## Features

- **Logo Scraping:** Retrieve logos from provided domains.
- **Preprocessing:** Eliminate outliers and standardize logo images.
- **Feature Extraction:**
  - **Color Palette Extraction:** Adjustable number of dominant colors.
  - **Perceptual Hash (pHash):** Compute image pHash for similarity comparison.
  - **CNN-based Features:** Extract features using pretrained models (e.g., ResNet, EfficientNet, MobileNet).
- **Grouping:** Group logos based on criteria (overall color, top2 colors, etc.) using algorithms such as DBSCAN, KMeans, or graph-based methods.
- **API Endpoints:** RESTful endpoints to trigger scraping, grouping, and feature extraction.
- **Visualization:** Generate visualizations of grouped logos from CSV files.
- **Test Scripts:** Command-line scripts to test the API endpoints and visualize results.

## Requirements

- Python 3.11+
- Required packages are listed in [`requirements.txt`](requirements.txt)

## Installation and Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```
2. **Create and activate a virtual enviironment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the application:**
    ```bash
    python app.py
    ```
5. **Use the API endpoints:**
    - **Scrape and extract features from logos:** `http://localhost:5151/api/scrape`
    - **Group logos:** `http://localhost:5151/api/group_by`
    - **Get results of logo grouping:** `http://localhost:5151/api/get_groups_by`

6. **Better test, use and run the application using the provided scripts in tests folder (they also have instructions on how to use them):**
    ```bash
    python tests/tester.py -h
    ```

## Data Visualization

Some visualizations of the grouped logos are provided in the `visualizations` folder. The visualizations are generated using the scripts in utils folder.

## Project Structure
```.
├── app.py                # Flask application entry point
├── data                  # CSV files of features and groups
├── logos                 # Scraped and preprocessed logo images
├── logs                  # Log files for various modules
├── modules
│   ├── __init__.py
│   ├── feature_extraction.py  # Extracts color palettes, pHash, CNN features
│   ├── grouping.py            # Groups logos based on visual features
│   ├── scraping.py            # Scrapes logos from domains
│   └── preprocessing.py       # Eliminates outliers and standardizes images
├── requirements.txt      # Required packages
└── tests
    ├── domain_list.txt   # List of unique domains to scrape logos from
    └── tester.py  # Test script for API endpoints
```

## Result visualization example

Below is an example of a visualization of grouped logos based on features extracted using a CNN model. The logos are grouped based on the graph-based clustering algorithm. In the image there are only 30 of the logo groups.

![Grouped Logos](visualizations/grouped_conv_features_graph_based_clusters_visualization.png)

