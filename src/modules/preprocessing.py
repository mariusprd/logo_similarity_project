# src/modules/preprocessing.py (Adapted for DB)

import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import shannon_entropy
from datetime import datetime
from gridfs import GridFS
import io

class Preprocessor:
    def __init__(self, db_client, log_filepath=None):
        """
        Initialize the Preprocessor with a database client.
        """
        self.db = db_client.logo_db
        self.fs = GridFS(self.db)  # Use GridFS for storing images
        self.log_filepath = log_filepath

    def eliminate_outliers_from_db(self, threshold=0.957):
        """
        Finds and deletes outlier images from the database based on a computed score.
        This is a batch process intended to be run as a maintenance task.
        """
        all_logos = list(self.db.logos.find({}))
        if not all_logos:
            self.log_message("No logos in the database to process.", "INFO")
            return

        file_data = []
        for logo_doc in all_logos:
            try:
                image_data = self.fs.get(logo_doc['image_file_id']).read()
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                img_np = np.array(img)
                color_entropy = shannon_entropy(img_np)
                width, height = img.size
                area = width * height
                # Keep track of the document ID
                file_data.append((logo_doc['_id'], logo_doc['domain'], color_entropy, area))
            except Exception as e:
                self.log_message(f"Error analyzing logo for {logo_doc['domain']} (ID: {logo_doc['_id']}): {e}", "ERROR")

        if not file_data:
            self.log_message("No valid image data could be analyzed.", "WARNING")
            return

        df = pd.DataFrame(file_data, columns=["MongoID", "Domain", "ColorEntropy", "Area"])
        # (The rest of your outlier score calculation logic is preserved)
        df_numeric = df.drop(columns=["MongoID", "Domain"]).dropna()
        if df_numeric.empty: return

        lower_percentile, upper_percentile, epsilon = 0.01, 0.98, 1e-8
        low_thresholds = df_numeric.quantile(lower_percentile)
        high_thresholds = df_numeric.quantile(upper_percentile)

        def compute_outlier_score(row):
            entropy_score = (row["ColorEntropy"] - low_thresholds["ColorEntropy"]) / (high_thresholds["ColorEntropy"] - low_thresholds["ColorEntropy"] + epsilon)
            area_score = (row["Area"] - low_thresholds["Area"]) / (high_thresholds["Area"] - low_thresholds["Area"] + epsilon)
            return entropy_score * 0.9 + area_score * 0.1

        df["OutlierScore"] = df.apply(compute_outlier_score, axis=1)
        outliers = df[df["OutlierScore"] > threshold]

        if outliers.empty:
            self.log_message("No outliers found in the database.", "INFO")
            return

        for _, row in outliers.iterrows():
            logo_id_to_delete = row['MongoID']
            domain = row['Domain']
            score = row['OutlierScore']
            try:
                # Find the logo document to get the image_file_id
                logo_to_delete = self.db.logos.find_one({"_id": logo_id_to_delete})
                if logo_to_delete:
                    # Delete from GridFS, logos collection, and similarities collection
                    self.fs.delete(logo_to_delete['image_file_id'])
                    self.db.logos.delete_one({"_id": logo_id_to_delete})
                    self.db.similarities.delete_many({"$or": [{"logo_id_1": logo_id_to_delete}, {"logo_id_2": logo_id_to_delete}]})
                    self.log_message(f"Deleted outlier logo from DB: {domain} (ID: {logo_id_to_delete}, Score: {score:.3f})", "SUCCESS")
            except Exception as e:
                self.log_message(f"Failed to delete outlier {domain} (ID: {logo_id_to_delete}): {e}", "ERROR")

    def log_message(self, message, status="INFO"):
        # This method's logic remains exactly the same.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emojis = {"INFO": "ℹ️","SUCCESS": "✅","WARNING": "⚠️","ERROR": "❌","PROCESS": "🔍","COMPLETE": "🎉"}
        print(f"[{timestamp}] {emojis.get(status, 'ℹ️')} {message}")