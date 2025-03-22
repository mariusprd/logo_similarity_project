import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy


class Preprocessor:
    """
    A class to preprocess images, including eliminating outliers based on a computed outlier score
    and standardizing image dimensions.
    """

    def __init__(self, output_dir, num_workers=1, log_filepath=None):
        """
        Initialize the Preprocessor.

        Parameters:
            output_dir (str): Directory containing images to process.
            num_workers (int): Number of worker threads to use for concurrent processing.
            log_filepath (str or None): File path where log messages should be appended. If None, logs
                                        will only be printed to the console.
        """
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.log_filepath = log_filepath


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
            "ERROR": "❌",
            "PROCESS": "🔍",
            "COMPLETE": "🎉"
        }
        full_message = f"[{timestamp}] {emojis.get(status, 'ℹ️')} {message}"
        print(full_message)
        if self.log_filepath:
            try:
                with open(self.log_filepath, "a") as log_file:
                    log_file.write(full_message + "\n")
            except Exception as e:
                print(f"Failed to write to log file: {e}")


    def eliminate_outliers(self, threshold=0.957):
        """
        Eliminates outlier images from self.output_dir by deleting files whose computed outlier
        score exceeds the given threshold.

        The outlier score is computed as a weighted combination of Shannon entropy and image area.
        Higher scores indicate a higher likelihood that the image is an outlier.

        Parameters:
            threshold (float): Outlier score threshold above which images will be deleted.
        """
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.ico', '.svg'}
        file_data = []
        failed_files = []
        lock = threading.Lock()

        def load_and_analyze(file_name):
            file_path = os.path.join(self.output_dir, file_name)
            try:
                img = Image.open(file_path)
                # Convert image mode for compatibility
                if img.mode in ["P", "LA"]:
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")
            except Exception as e:
                self.log_message(f"Error loading {file_path}: {e}", "ERROR")
                with lock:
                    failed_files.append(file_name)
                return

            try:
                img_np = np.array(img)
                color_entropy = shannon_entropy(img_np)
                width, height = img.size
                area = width * height
                with lock:
                    file_data.append((file_path, color_entropy, area))
            except Exception as e:
                self.log_message(f"Error analyzing {file_path}: {e}", "ERROR")
                with lock:
                    failed_files.append(file_name)

        try:
            files = [
                f for f in os.listdir(self.output_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
        except Exception as e:
            self.log_message(f"Error listing folder {self.output_dir}: {e}", "ERROR")
            return

        total_files = len(files)
        self.log_message(f"Total files found: {total_files}", "INFO")

        # Process image analysis concurrently
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(load_and_analyze, files)

        if not file_data:
            self.log_message("No valid images found for outlier elimination.", "WARNING")
            return

        # Create a DataFrame with the collected image data
        df = pd.DataFrame(file_data, columns=["FilePath", "ColorEntropy", "Area"])
        df_numeric = df.drop(columns=["FilePath"]).dropna()
        if df_numeric.empty:
            self.log_message("No numeric data available for outlier calculation.", "WARNING")
            return

        # Compute normalization thresholds using percentiles
        lower_percentile = 0.01
        upper_percentile = 0.98
        low_thresholds = df_numeric.quantile(lower_percentile)
        high_thresholds = df_numeric.quantile(upper_percentile)
        epsilon = 1e-8  # To prevent division by zero

        def compute_outlier_score(row):
            entropy_score = (row["ColorEntropy"] - low_thresholds["ColorEntropy"]) / (
                high_thresholds["ColorEntropy"] - low_thresholds["ColorEntropy"] + epsilon
            )
            area_score = (row["Area"] - low_thresholds["Area"]) / (
                high_thresholds["Area"] - low_thresholds["Area"] + epsilon
            )
            return entropy_score * 0.9 + area_score * 0.1

        df["OutlierScore"] = df.apply(compute_outlier_score, axis=1)
        outliers = df[df["OutlierScore"] > threshold]

        if outliers.empty:
            self.log_message("No outliers found above the threshold.", "INFO")
            return

        def delete_file(file_path, score):
            try:
                os.remove(file_path)
                self.log_message(
                    f"Deleted outlier file: {os.path.basename(file_path)} (Score: {score:.3f})",
                    "SUCCESS"
                )
            except Exception as e:
                self.log_message(f"Failed to delete {file_path}: {e}", "ERROR")

        # Delete outlier files concurrently using the class's thread count
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for _, row in outliers.iterrows():
                executor.submit(delete_file, row["FilePath"], row["OutlierScore"])

        if failed_files:
            self.log_message("Some files could not be processed:", "WARNING")
            for f in failed_files:
                self.log_message(f, "WARNING")


    def standardize_images(self, target_size=(128, 128)):
        """
        Standardizes images in self.output_dir by resizing them to the specified target dimensions 
        while preserving aspect ratio. The standardized images are saved as PNG files in the same directory.

        Parameters:
            target_size (tuple): Desired dimensions (width, height) for the standardized images.
                                 Default is (128, 128).
        """
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.ico', '.svg'}

        def process_file(file_name):
            file_path = os.path.join(self.output_dir, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()

            if os.path.isfile(file_path) and file_ext in valid_extensions:
                try:
                    with Image.open(file_path) as img:
                        # Ensure image supports transparency
                        img = img.convert("RGBA")
                        
                        # Calculate new dimensions while preserving aspect ratio
                        width, height = img.size
                        if width > height:
                            new_width = target_size[0]
                            new_height = int((height / width) * target_size[0])
                        else:
                            new_height = target_size[1]
                            new_width = int((width / height) * target_size[1])
                        
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        # Create a new image with a transparent background
                        new_img = Image.new("RGBA", target_size, (0, 0, 0, 0))
                        paste_position = (
                            (target_size[0] - new_width) // 2,
                            (target_size[1] - new_height) // 2
                        )
                        new_img.paste(img, paste_position, img if img.mode == "RGBA" else None)
                        
                        new_file_name = os.path.splitext(file_name)[0] + ".png"
                        new_file_path = os.path.join(self.output_dir, new_file_name)
                        new_img.save(new_file_path, "PNG")
                        self.log_message(f"Processed: {file_name} -> {new_file_name}", "SUCCESS")
                except Exception as e:
                    self.log_message(f"Failed to process {file_name}: {e}", "ERROR")

        # Process files concurrently using the class's thread count
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(process_file, os.listdir(self.output_dir))