import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import imagehash
from torchvision import models, transforms
import torch

class FeatureExtractor:
    """
    A class for extracting visual features from images, such as the dominant color palette.
    """

    def __init__(self, image_folder, num_workers=1, log_filepath=None, device='cpu'):
        """
        Initialize the FeatureExtractor.

        Parameters:
            image_folder (str): Directory containing the images to process.
            num_workers (int, optional): Number of threads to use for parallel processing.
                Defaults to min(8, os.cpu_count()).
            log_filepath (str or None): File path where log messages should be appended.
                If None, logs will only be printed to the console.
        """
        self.image_folder = image_folder
        self.num_workers = num_workers
        self.log_filepath = log_filepath
        self.device = device


    def log_message(self, message, status="INFO"):
        """
        Logs a message with a timestamp. The message is printed to the console and, if a log_filepath
        is provided, also appended to the log file.

        Parameters:
            message (str): The message to log.
            status (str): Status label (e.g., "INFO", "SUCCESS", "WARNING", "ERROR").
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emojis = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌"
        }
        full_message = f"[{timestamp}] {emojis.get(status, 'ℹ️')} {message}"
        print(full_message)
        if self.log_filepath:
            try:
                with open(self.log_filepath, "a") as log_file:
                    log_file.write(full_message + "\n")
            except Exception as e:
                print(f"Failed to write to log file: {e}")


    def extract_dominant_colors(self, image_path, k=5):
        """
        Extracts k dominant colors from an image and computes their confidence levels.

        Parameters:
            image_path (str): Path to the image file.
            k (int): Number of dominant colors to extract.

        Returns:
            tuple or None: A tuple (image_path, color_palette) where color_palette is a list of tuples
                           (color, confidence), or None if the image cannot be processed.
        """
        image = cv2.imread(image_path)
        if image is None:
            self.log_message(f"Could not read image: {image_path}", "ERROR")
            return None

        # Convert from BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)

        # Remove very dark pixels (to avoid black/transparent backgrounds).
        pixels = np.array([p for p in pixels if np.any(p > 10)])
        if len(pixels) == 0:
            self.log_message(f"Image mostly dark: {image_path}", "WARNING")
            return None

        # Apply K-Means clustering.
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pixels)
        cluster_centers = kmeans.cluster_centers_

        # Count occurrences of each cluster.
        label_counts = Counter(labels)
        total_pixels = sum(label_counts.values())

        # Build the color palette list with confidence levels.
        color_palette = []
        for i, color in enumerate(cluster_centers):
            confidence = label_counts[i] / total_pixels
            color_palette.append((tuple(map(int, color)), round(confidence, 4)))

        self.log_message(f"Extracted {k} dominant colors for image: {image_path}", "SUCCESS")
        return image_path, color_palette


    def process_image_color_palette(self, image_path, k=5):
        """
        Wrapper function to process a single image and extract its dominant colors.

        Parameters:
            image_path (str): Path to the image file.
            k (int): Number of dominant colors to extract.

        Returns:
            tuple or None: The result from extract_dominant_colors, or None if processing failed.
        """
        result = self.extract_dominant_colors(image_path, k=k)
        return result if result else None


    def extract_color_palette(self, output_file, num_colors=4):
        """
        Processes all images in self.image_folder in parallel to extract their dominant color palettes
        and saves the results to a CSV file.

        Parameters:
            output_file (str): Path to the output CSV file.
            num_colors (int): Number of dominant colors to extract from each image.
        """
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith(valid_extensions)
        ]
        self.log_message(f"Found {len(image_files)} images in {self.image_folder}.", "INFO")
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.process_image_color_palette, img, k=num_colors): img for img in image_files}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        if not results:
            self.log_message("No images processed successfully.", "WARNING")
            return

        # Build DataFrame columns dynamically.
        columns = ["Filepath"]
        for i in range(num_colors):
            columns.extend([f"Color_{i+1}", f"Confidence_{i+1}"])

        data = []
        for filepath, color_palette in results:
            row = [filepath]
            for color, confidence in color_palette:
                row.extend([color, confidence])
            data.append(row)

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file, index=False)
        self.log_message(f"Color palette data saved to {output_file}", "SUCCESS")


    def process_image_phash(self, image_path):
        """
        Computes the perceptual hash (pHash) for a single image.

        Parameters:
            image_path (str): Path to the image file.
        """
        img_path = os.path.join(self.image_folder, image_path)
        try:
            img = Image.open(img_path)
            return image_path, str(imagehash.phash(img))
        except Exception as e:
            self.log_message(f"Error processing {image_path}: {e}", "ERROR")
            return None


    def extract_phash(self, output_file):
        """
        Extracts the perceptual hash (pHash) for each image in self.image_folder and saves the results to a CSV file.

        Parameters:
            output_file (str): Path to the CSV output file.
        """
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(valid_extensions)]
        self.log_message(f"Found {len(image_files)} images for pHash extraction.", "INFO")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.process_image_phash, f): f for f in image_files}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

        if results:
            df = pd.DataFrame(results, columns=["Filepath", "pHash"])
            df.to_csv(output_file, index=False)
            self.log_message(f"pHash data saved to {output_file}", "SUCCESS")
        else:
            self.log_message("No pHash data extracted.", "WARNING")

    
    def process_image_convnet(self, image_path, model, transform):
        """
        Processes a single image using a pretrained CNN model to extract its feature vector.
        
        Parameters:
            image_path (str): The image image_path in self.image_folder.
            model (torch.nn.Module): The pretrained CNN model (with classification layer removed).
            transform (torchvision.transforms.Compose): Transformations to apply to the image.
        
        Returns:
            tuple or None: (image_path, normalized_feature_vector) if successful; otherwise, None.
        """
        import os
        import numpy as np
        from PIL import Image

        img_path = os.path.join(self.image_folder, image_path)
        try:
            with Image.open(img_path).convert('RGB') as img:
                img_tensor = transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = model(img_tensor).squeeze().cpu().numpy()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            return image_path, feat
        except Exception as e:
            self.log_message(f"Error processing {image_path}: {e}", "ERROR")
            return None


    def extract_convnet_features(self, output_file, model_name='resnet18'):
        """
        Extracts image features using a specified pretrained CNN model and saves the results to a CSV file.

        Supported models include:
        - 'resnet18'
        - 'efficientnet_b0'
        - 'mobilenet_v2'

        Parameters:
            output_file (str): Path to the CSV output file.
            model_name (str): Name of the pretrained model to use.
        """
        model_dict = {
            'resnet18': models.resnet18,
            'efficientnet_b0': models.efficientnet_b0,
            'mobilenet_v2': models.mobilenet_v2,
        }
        
        model_key = model_name.lower()
        if model_key not in model_dict:
            self.log_message(f"Model '{model_name}' not supported.", "ERROR")
            return

        # Load and prepare the model
        model = model_dict[model_key](pretrained=True)
        # Remove the classification layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        model.to(self.device)

        # Define image transformations matching the network's input requirements
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(valid_extensions)]
        self.log_message(f"Found {len(image_files)} images for feature extraction using {model_name}.", "INFO")

        features = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self.process_image_convnet, f, model, transform): f for f in image_files}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    features.append(result)

        if features:
            feat_length = len(features[0][1])
            columns = ["Filepath"] + [f"Feature_{i}" for i in range(feat_length)]
            data = [[fname] + feat.tolist() for fname, feat in features]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(output_file, index=False)
            self.log_message(f"Pretrained features saved to {output_file}.", "SUCCESS")
        else:
            self.log_message("No features extracted.", "WARNING")

