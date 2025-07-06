# src/modules/scraping.py (Adapted for DB)

import requests
import os
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from gridfs import GridFS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

class LogoScraper:
    def __init__(self, db_client, log_filepath="../scraper.log"):
        """
        Initialize the LogoScraper with a database client.
        The db_client should be a connected pymongo.MongoClient instance.
        """
        # The scraper now works with a database, not an output directory
        self.db = db_client.logo_db # Or your chosen DB name
        self.fs = GridFS(self.db)  # Use GridFS for storing images
        self.log_filepath = log_filepath

    # The scrape_domains and process_domain methods are no longer needed here.
    # The Celery task will handle the looping and calling scrape_logo_for_domain.

    def scrape_logo_for_domain(self, domain, use_clearbit_api=True):
        """
        Attempts to scrape a logo and returns the GridFS file_id if successful.
        """
        if use_clearbit_api:
            file_id = self.fetch_logo_from_clearbit(domain)
            if file_id:
                return file_id # Return the ObjectId of the file in GridFS

        prefixes = ["https://", "http://", "https://www.", "http://www."]
        for prefix in prefixes:
            url = f"{prefix}{domain}"
            logo_url = self.extract_logo_url(url)
            if logo_url:
                file_id = self.save_logo_file(logo_url, domain)
                if file_id:
                    return file_id
        return None

    def save_logo_file(self, image_url, domain):
        """
        Saves the logo from a URL into MongoDB GridFS and returns the file ID.
        """
        try:
            response = requests.get(image_url, headers=HEADERS, timeout=10, verify=False)
            response.raise_for_status()
            
            # Save the image content directly to GridFS
            file_id = self.fs.put(response.content, filename=domain)
            self.log_message(f"Logo manually saved to DB for {domain} with file_id {file_id}.", "SUCCESS")
            return file_id
        except Exception as e:
            self.log_message(f"Failed to save logo to DB for {domain}. Error: {str(e)}", "ERROR")
            return None

    def extract_logo_url(self, url):
        # This method's logic remains exactly the same as your original.
        html = self.fetch_page_html(url)
        if not html:
            self.log_message(f"Failed to retrieve content from {url}.", "WARNING")
            return None
        # ... (rest of your original extract_logo_url logic)
        soup = BeautifulSoup(html, "html.parser")
        selectors = [
            'meta[property="og:image"]','meta[name="twitter:image"]','link[rel~="icon"]',
            'link[rel~="shortcut icon"]','img[src*="logo"]','img[class*="logo"]',
            'img[id*="logo"]','img[alt*="logo"]','header img','nav img'
        ]
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                element_url = element.get("content") or element.get("href") or element.get("src")
                if element_url:
                    return urljoin(url, element_url)
        return None

    def fetch_page_html(self, url):
        # This method's logic remains exactly the same.
        try:
            response = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def fetch_logo_from_clearbit(self, domain):
        """
        Fetches logo from Clearbit, saves it to GridFS, and returns the file ID.
        """
        url = f"https://logo.clearbit.com/{domain}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=8)
            response.raise_for_status()
            
            # Check for a valid image response before saving
            if 'image' in response.headers.get('content-type', ''):
                file_id = self.fs.put(response.content, filename=domain)
                self.log_message(f"Logo fetched from Clearbit for {domain} with file_id {file_id}.", "SUCCESS")
                return file_id
        except requests.RequestException as e:
            self.log_message(f"Clearbit did not return a logo for {domain}. Error: {str(e)}", "WARNING")
        return None
        
    def log_message(self, message, status="INFO"):
        # This method's logic remains exactly the same.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emojis = {"INFO": "ℹ️","SUCCESS": "✅","WARNING": "⚠️","ERROR": "❌","PROCESS": "🔍","COMPLETE": "🎉"}
        print(f"[{timestamp}] {emojis.get(status, 'ℹ️')} {message}")