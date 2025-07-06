# src/modules/feature_extraction.py (Adapted for DB)

import io
from PIL import Image
import imagehash
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from datetime import datetime

class FeatureExtractor:
    def __init__(self, log_filepath=None):
        """
        The extractor no longer needs image_folder or device settings for this scope.
        """
        self.log_filepath = log_filepath

    def get_color_palette_from_data(self, image_data: bytes, num_colors=4) -> list | None:
        """
        Extracts dominant color palette from image byte data.
        Returns a list of tuples: [( (r,g,b), confidence ), ...].
        """
        try:
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            # Resizing speeds up KMeans
            img.thumbnail((150, 150))
            
            pixels = np.array(img).reshape(-1, 3)
            if len(pixels) == 0:
                self.log_message("Image contains no pixels to analyze.", "WARNING")
                return None

            kmeans = KMeans(n_clusters=num_colors, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(pixels)
            
            label_counts = Counter(labels)
            total_pixels = sum(label_counts.values())
            
            color_palette = []
            for i, color in enumerate(kmeans.cluster_centers_):
                confidence = label_counts[i] / total_pixels
                color_palette.append((tuple(map(int, color)), round(confidence, 4)))

            self.log_message(f"Extracted {num_colors} dominant colors.", "SUCCESS")
            return color_palette
        except Exception as e:
            self.log_message(f"Could not extract color palette: {e}", "ERROR")
            return None

    def get_phash_from_data(self, image_data: bytes) -> str | None:
        """
        Computes the perceptual hash (pHash) from image byte data.
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            return str(imagehash.phash(img))
        except Exception as e:
            self.log_message(f"Error computing pHash: {e}", "ERROR")
            return None
            
    def log_message(self, message, status="INFO"):
        # This method's logic remains exactly the same.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emojis = {"INFO": "ℹ️","SUCCESS": "✅","WARNING": "⚠️","ERROR": "❌"}
        print(f"[{timestamp}] {emojis.get(status, 'ℹ️')} {message}")