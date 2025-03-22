import argparse
import json
import requests

def test_scrape(args):
    # Read domains from the file, ignoring empty lines
    domains = []
    with open(args.domains_file, "r") as f:
        for line in f:
            domain = line.strip()
            if domain:
                domains.append(domain)
    
    # Build the features dictionary based on provided arguments
    features = {}
    if args.color_palette is not None:
        features["color_palette"] = args.color_palette
    if args.phash:
        features["phash"] = True
    if args.convnet is not None:
        features["convnet"] = args.convnet
    if args.overall_color:
        features["overall_color"] = True
    
    # Construct JSON payload for scraping
    payload = {
        "domains": domains,
        "features": features
    }
    
    print("Sending the following payload to scrape endpoint:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(f"{args.url}/api/scrape", json=payload)
        print("Response Status Code:", response.status_code)
        result_json = response.json()
        print("Response JSON:", json.dumps(result_json, indent=2))
        if args.save_response:
            with open(args.save_response, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"Response JSON saved to {args.save_response}")
    except Exception as e:
        print("An error occurred during scraping:", str(e))

def test_group(args):
    # Construct JSON payload for grouping (POST request)
    payload = {
        "criteria": args.criteria,
        "algorithm": args.algorithm
    }
    # Include num_clusters if algorithm is kmeans.
    if args.algorithm == "kmeans":
        if args.num_clusters is None:
            print("Error: For kmeans algorithm, --num_clusters is required.")
            return
        payload["num_clusters"] = args.num_clusters

    print("Sending the following payload to group endpoint:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(f"{args.url}/api/group_by", json=payload)
        print("Response Status Code:", response.status_code)
        result_json = response.json()
        print("Response JSON:", json.dumps(result_json, indent=2))
        if args.save_response:
            with open(args.save_response, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"Response JSON saved to {args.save_response}")
    except Exception as e:
        print("An error occurred during grouping:", str(e))

def test_get_groups(args):
    # Construct query parameters for GET request.
    if not args.criteria or not args.algorithm:
        print("Error: --criteria and --algorithm are required for get_groups mode.")
        return
    
    query_params = {
        "criteria": args.criteria,
        "algorithm": args.algorithm
    }
    print("Sending GET request with the following query parameters:")
    print(json.dumps(query_params, indent=2))
    
    try:
        response = requests.get(f"{args.url}/api/get_groups_by", params=query_params)
        print("Response Status Code:", response.status_code)
        result_json = response.json()
        print("Response JSON:", json.dumps(result_json, indent=2))
        if args.save_response:
            with open(args.save_response, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"Response JSON saved to {args.save_response}")
    except Exception as e:
        print("An error occurred during get_groups request:", str(e))

def main():
    parser = argparse.ArgumentParser(
        description="Test API endpoints for scraping, grouping, and getting groups."
    )
    parser.add_argument("--mode", choices=["scrape", "group", "get_groups"], default="scrape",
                        help="Select mode: 'scrape' for scraping, 'group' for grouping (POST), or 'get_groups' for getting groups (GET).")
    parser.add_argument("--url", type=str, default="http://localhost:5151",
                        help="Base URL of the API (default: http://localhost:5151)")
    parser.add_argument("--save_response", type=str, default=None,
                        help="Path to file to store the response JSON")
    
    # Arguments for scraping mode.
    parser.add_argument("domains_file", nargs="?", help="Path to the file containing domains (required for scrape mode)")
    parser.add_argument("--color_palette", type=int, default=None,
                        help="Number of colors for color palette extraction (e.g., 4)")
    parser.add_argument("--phash", action="store_true",
                        help="Flag to extract perceptual hash (pHash)")
    parser.add_argument("--convnet", type=str, nargs='?', const='resnet18', default='resnet18',
                        help="Pretrained model for convnet feature extraction (e.g., 'resnet18', 'efficientnet_b0', 'mobilenet_v2')")
    parser.add_argument("--overall_color", action="store_true",
                        help="Flag to extract the overall dominant color")
    
    # Arguments for grouping and get_groups modes.
    parser.add_argument("--criteria", type=str,
                        choices=['overall', 'top2', 'all', 'phash', 'conv_features'],
                        help="Grouping criteria: 'overall', 'top2', 'all', 'phash', or 'conv_features'")
    parser.add_argument("--algorithm", type=str,
                        choices=['dbscan', 'kmeans', 'graph_based'],
                        help="Grouping algorithm: 'dbscan', 'kmeans', or 'graph_based'")
    parser.add_argument("--num_clusters", type=int, default=None,
                        help="Number of clusters to use (required if algorithm is 'kmeans')")
    
    args = parser.parse_args()
    
    if args.mode == "scrape":
        if not args.domains_file:
            print("Error: domains_file is required for scrape mode.")
            return
        test_scrape(args)
    elif args.mode == "group":
        if not args.criteria or not args.algorithm:
            print("Error: --criteria and --algorithm are required for group mode.")
            return
        test_group(args)
    elif args.mode == "get_groups":
        if not args.criteria or not args.algorithm:
            print("Error: --criteria and --algorithm are required for get_groups mode.")
            return
        test_get_groups(args)
    else:
        print("Unsupported mode.")

if __name__ == "__main__":
    main()
