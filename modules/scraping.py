import os
import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Default headers for HTTP requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


class LogoScraper:
    """
    A class to scrape logos from a list of domains using both the Clearbit API and manual extraction.
    """

    def __init__(self, output_dir="logos", log_filepath="../scraper.log", num_workers=1):
        """
        Initialize the LogoScraper.

        Parameters:
            output_dir (str): Directory where logos will be saved.
            num_workers (int): Maximum number of threads for concurrent scraping.
        """
        self.output_dir = output_dir
        self.log_filepath = log_filepath
        self.num_workers = num_workers
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def scrape_domains(self, domain_list):
        """
        Scrape logos for the provided list of domains using multithreading.

        Parameters:
            domain_list (list): List of domain names as strings.
        """
        success_count = 0
        with open(self.log_filepath, "w") as log_file:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(self.process_domain, domain, idx, len(domain_list), log_file)
                    for idx, domain in enumerate(domain_list, start=1)
                ]
                for future in futures:
                    success_count += future.result()

        success_rate = (success_count / len(domain_list)) * 100 if domain_list else 0
        self.log_message(
            f"Completed logo extraction for {len(domain_list)} domains with a success rate of "
            f"{success_rate:.2f}% ({success_count}/{len(domain_list)}).", "COMPLETE"
        )


    def process_domain(self, domain, idx, total, log_file):
        """
        Process a single domain: search for and save its logo.

        Parameters:
            domain (str): Domain name to process.
            idx (int): Current index in the domain list.
            total (int): Total number of domains.
            log_file (file object): File object for logging the results.

        Returns:
            int: 1 if the logo was found and saved; otherwise, 0.
        """
        self.log_message(f"Processing ({idx}/{total}): Searching logo for {domain}...", "PROCESS")
        if self.scrape_logo_for_domain(domain, use_clearbit_api=True):
            log_file.write(f"{domain}: Logo found and saved successfully. ✅\n")
            return 1
        else:
            log_file.write(f"{domain}: Logo not found. ❌\n")
            return 0


    def scrape_logo_for_domain(self, domain, use_clearbit_api=True):
        """
        Attempt to scrape a logo for the given domain using the Clearbit API first, 
        then manual extraction if necessary.

        Parameters:
            domain (str): Domain name to search for.
            use_clearbit_api (bool): Whether to try fetching from Clearbit API first.

        Returns:
            bool: True if a logo is found and saved; otherwise, False.
        """
        if use_clearbit_api and self.fetch_logo_from_clearbit(domain):
            return True

        # Try multiple URL prefixes for manual extraction
        prefixes = ["https://", "http://", "https://www.", "http://www."]
        for prefix in prefixes:
            url = f"{prefix}{domain}"
            logo_url = self.extract_logo_url(url)
            if logo_url and self.save_logo_file(logo_url, domain):
                return True
        return False


    def save_logo_file(self, image_url, domain):
        """
        Save the logo image from the provided URL to the output directory.

        Parameters:
            image_url (str): URL of the logo image.
            domain (str): Domain name to use for naming the saved file.

        Returns:
            bool: True if the image is saved successfully; otherwise, False.
        """
        try:
            response = requests.get(image_url, headers=HEADERS, timeout=10, verify=False)
            response.raise_for_status()
            ext = image_url.split('.')[-1].split('?')[0].lower()
            if ext not in ['png', 'jpg', 'jpeg', 'svg', 'webp', 'ico']:
                ext = 'png'
            filepath = os.path.join(self.output_dir, f"{domain}.{ext}")
            with open(filepath, "wb") as f:
                f.write(response.content)
            self.log_message(f"Logo manually saved for {domain}.", "SUCCESS")
            return True
        except Exception as e:
            self.log_message(f"Failed to save logo for {domain}. Error: {str(e)}", "ERROR")
            return False


    def extract_logo_url(self, url):
        """
        Extract the logo URL from the webpage at the given URL.

        Parameters:
            url (str): URL of the webpage to parse.

        Returns:
            str or None: Resolved URL of the logo if found; otherwise, None.
        """
        html = self.fetch_page_html(url)
        if not html:
            self.log_message(f"Failed to retrieve content from {url}.", "WARNING")
            return None

        soup = BeautifulSoup(html, "html.parser")
        selectors = [
            'meta[property="og:image"]',
            'meta[name="twitter:image"]',
            'link[rel~="icon"]',
            'link[rel~="shortcut icon"]',
            'img[src*="logo"]',
            'img[class*="logo"]',
            'img[id*="logo"]',
            'img[alt*="logo"]',
            'header img',
            'nav img'
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                element_url = element.get("content") or element.get("href") or element.get("src")
                if element_url:
                    return urljoin(url, element_url)

        # Fallback to favicon
        favicon_url = urljoin(url, "/favicon.ico")
        try:
            response = requests.get(favicon_url, headers=HEADERS, timeout=8, verify=False)
            if response.status_code == 200:
                return favicon_url
        except requests.RequestException:
            pass

        return None


    def fetch_page_html(self, url):
        """
        Fetch the HTML content of the given URL.

        Parameters:
            url (str): URL to fetch.

        Returns:
            str or None: HTML content if the request is successful; otherwise, None.
        """
        try:
            response = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None


    def fetch_logo_from_clearbit(self, domain):
        """
        Attempt to fetch the logo for a domain using the Clearbit API.

        Parameters:
            domain (str): Domain name to fetch the logo for.

        Returns:
            bool: True if the logo is fetched and saved successfully; otherwise, False.
        """
        url = f"https://logo.clearbit.com/{domain}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=8)
            response.raise_for_status()
            ext = response.headers.get("content-type", "").split("/")[-1]
            if ext not in ['png', 'jpg', 'jpeg', 'svg', 'webp', 'ico']:
                ext = "png"
            filepath = os.path.join(self.output_dir, f"{domain}.{ext}")
            with open(filepath, "wb") as f:
                f.write(response.content)
            self.log_message(f"Logo fetched from Clearbit for {domain}.", "SUCCESS")
            return True
        except requests.RequestException as e:
            self.log_message(f"Clearbit API did not return a logo for {domain}. Error: {str(e)}", "WARNING")
            return False


    def log_message(self, message, status="INFO"):
        """
        Display a formatted log message with a timestamp and a status indicator.

        Parameters:
            message (str): The message to display.
            status (str): The status level (e.g., INFO, SUCCESS, WARNING, ERROR, PROCESS, COMPLETE).
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
        print(f"[{timestamp}] {emojis.get(status, 'ℹ️')} {message}")
