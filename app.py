from flask import Flask, request, jsonify
import csv
import os
import pandas as pd
from modules import scraping, feature_extraction, grouping, preprocessing

app = Flask(__name__)
NUM_WORKERS = 10

@app.route('/api/scrape', methods=['POST'])
def api_scrape():
    """
    Expects a JSON payload with:
      - 'domains': a list of domains,
      - 'features': a dictionary specifying which features to extract, for example:
            {
                "color_palette": 4,    # extract a 4-color palette
                "overall_color": true, # extract overall color
                "phash": true,         # extract perceptual hash
                "convnet": true        # extract convnet features
            }
    
    Calls the scraping module, preprocesses the scraped logos, and then extracts features based
    on the requested options.
    """
    data = request.get_json()
    domains = data.get('domains')
    features = data.get('features', {})

    if not domains or not isinstance(domains, list):
        return jsonify({'error': 'Please provide a list of domains'}), 400

    # Scrape logos for the provided domains
    scraper = scraping.LogoScraper(output_dir='logos', log_filepath='./logs/scraper.log', num_workers=NUM_WORKERS)
    scraper.scrape_domains(domains)
    
    # Preprocess the scraped logos
    preprocessor = preprocessing.Preprocessor(output_dir='logos', num_workers=NUM_WORKERS, log_filepath='./logs/preprocessor.log')
    preprocessor.eliminate_outliers()
    preprocessor.standardize_images()

    # Extract features from the preprocessed logos
    feature_extractor = feature_extraction.FeatureExtractor(
        image_folder='logos', num_workers=NUM_WORKERS, log_filepath='./logs/feature_extraction.log', device='mps'
    )
    
    # Extract color palette if requested
    if "color_palette" in features:
        try:
            num_colors = int(features.get("color_palette", 4))
        except Exception:
            num_colors = 4
        output_csv = f'./data/color_palette_{num_colors}color.csv'
        feature_extractor.extract_color_palette(output_file=output_csv, num_colors=num_colors)

    # Extract overall color if requested and the color palette option was not exactly 1
    if features.get("overall_color", False) and features.get("color_palette") != 1:
        feature_extractor.extract_color_palette(output_file='./data/color_palette_1color.csv', num_colors=1)    
    # Extract perceptual hash if requested
    if features.get("phash", False):
        feature_extractor.extract_phash(output_file='./data/phash.csv')
    
    # Extract convnet features if requested
    if features.get("convnet", False):
        model_name = 'resnet18'
        feature_extractor.extract_convnet_features(output_file='./data/convnet_features.csv', model_name=model_name)
    
    return jsonify({'message': 'Scraping and feature extraction completed successfully'}), 200


@app.route('/api/group_by', methods=['POST'])
def api_group_by():
    """
    Expects a JSON payload with:
      - criteria: a string indicating the grouping criteria ('overall', 'top2', 'all', 'phash', or 'conv_features')
      - algorithm: a string specifying the grouping algorithm ('dbscan', 'kmeans', or 'graph_based')
      - num_clusters: (optional; required if algorithm is 'kmeans') the number of clusters to use
      
    Groups logos based on the provided criteria and algorithm, writes the results to a file,
    and returns the grouping result as JSON.
    """
    data = request.get_json()
    criteria = data.get('criteria')
    algorithm = data.get('algorithm')
    num_clusters = data.get('num_clusters', None)

    # Validate criteria
    valid_criteria = ['overall', 'top2', 'all', 'phash', 'conv_features']
    if criteria not in valid_criteria:
        return jsonify({'error': f"Invalid criteria. Must be one of {', '.join(valid_criteria)}."}), 400

    # Validate algorithm
    valid_algorithms = ['dbscan', 'kmeans', 'graph_based']
    if algorithm not in valid_algorithms:
        return jsonify({'error': f"Invalid algorithm. Must be one of {', '.join(valid_algorithms)}."}), 400

    # For kmeans, ensure num_clusters is provided and valid.
    if algorithm == 'kmeans':
        if num_clusters is None:
            return jsonify({'error': "For kmeans algorithm, 'num_clusters' parameter is required."}), 400
        try:
            num_clusters = int(num_clusters)
        except ValueError:
            return jsonify({'error': "'num_clusters' must be an integer."}), 400

    try:
        # Instantiate the Grouper with your data.
        grouper = grouping.Grouper()
        grouper.similarity_matrix_dir = "./data/similarity_matrices/"

        grouper.group_by(criteria=criteria, algorithm=algorithm, num_clusters=num_clusters, results_filepath=f"./data/grouped_{criteria}_{algorithm}.csv")
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify({'message': 'Grouping completed successfully'}), 200


@app.route('/api/get_groups_by', methods=['GET'])
def get_groups_by():
    """
    GET endpoint that returns grouped domains based on provided criteria and algorithm.

    Expects query parameters:
      - criteria: e.g. "phash", "overall", etc.
      - algorithm: e.g. "graph_based", "kmeans", "dbscan", etc.

    It builds the filename as:
        grouped_{criteria}_{algorithm}.csv
    from the data folder, reads the CSV (with two columns: filename and group),
    removes the file extensions from the filenames, and responds with a JSON mapping
    each group to a list of domain names (filenames without extension).
    """
    criteria = request.args.get('criteria')
    algorithm = request.args.get('algorithm')

    if not criteria or not algorithm:
        return jsonify({"error": "Both 'criteria' and 'algorithm' query parameters are required."}), 400

    # Build the grouping file path
    filename = f"grouped_{criteria.lower()}_{algorithm.lower()}.csv"
    data_folder = "./data"
    file_path = os.path.join(data_folder, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"Grouping file {filename} not found."}), 404

    try:
        # Assume CSV has no header; first column is filename and second column is group.
        df = pd.read_csv(file_path)
        df.columns = ["File", "Group"]
    except Exception as e:
        return jsonify({"error": f"Error reading grouping file: {e}"}), 500

    groups = {}
    for group, subdf in df.groupby("Group"):
        # Remove file extension to get just the domain name
        domains = [os.path.splitext(f)[0] for f in subdf["File"].tolist()]
        groups[str(group)] = domains

    return jsonify({"groups": groups}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5151, debug=True)
