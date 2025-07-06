# src/tasks.py (Refactored)

from datetime import datetime
from celery import group
from pymongo import MongoClient
import os

from src.celery_config import celery_app
from src.modules import scraping, feature_extraction, grouping, preprocessing, utils

# --- Database Connection (managed within the task file) ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)

# --- Main Tasks ---

@celery_app.task(name='tasks.start_domain_processing')
def start_domain_processing(domains: list):
    """
    Dispatcher task. Checks for existing domains and creates sub-tasks for new ones.
    """
    db = mongo_client.logo_db
    existing_domains = {doc['domain'] for doc in db.logos.find({}, {'domain': 1})}
    new_domains = [d for d in domains if d not in existing_domains]

    if not new_domains:
        return "No new domains to process. All already exist in the database."

    # Create a group of parallel sub-tasks, one for each new domain.
    job = group(process_one_domain.s(domain) for domain in new_domains)
    job.apply_async()

    return f"Queued processing for {len(new_domains)} new domains."

@celery_app.task(name='tasks.process_one_domain')
def process_one_domain(domain: str):
    """
    The main worker task. Processes a single domain from scraping to similarity calculation.
    """
    db = mongo_client.logo_db
    scraper = scraping.LogoScraper(db_client=mongo_client)
    extractor = feature_extraction.FeatureExtractor()

    # 1. Scrape Logo and get its GridFS file ID
    file_id = scraper.scrape_logo_for_domain(domain)
    if not file_id:
        return f"Failed to scrape logo for {domain}."

    # 2. Extract Features
    image_data = scraper.fs.get(file_id).read()
    phash = extractor.get_phash_from_data(image_data)
    color_palette = extractor.get_color_palette_from_data(image_data, num_colors=4)

    if not phash or not color_palette:
        # Cleanup failed processing
        scraper.fs.delete(file_id)
        return f"Failed to extract features for {domain}."

    # 3. Save Logo Metadata to DB
    logo_doc = {
        "domain": domain,
        "image_file_id": file_id,
        "phash": phash,
        "color_palette": color_palette,
    }
    result = db.logos.insert_one(logo_doc)
    new_logo_id = result.inserted_id

    # 4. Incremental Similarity Calculation
    other_logos = list(db.logos.find({"_id": {"$ne": new_logo_id}}))
    if not other_logos:
        return f"Successfully processed {domain}. No other logos to compare against."
    
    new_similarities = []
    for other_logo in other_logos:
        phash_sim = utils.calculate_phash_similarity(phash, other_logo.get('phash'))
        color_sim = utils.calculate_palette_similarity(color_palette, other_logo.get('color_palette'))

        similarity_doc = {
            "logo_id_1": new_logo_id,
            "logo_id_2": other_logo['_id'],
            "phash_similarity": phash_sim,
            "color_similarity": color_sim,
        }
        new_similarities.append(similarity_doc)

    if new_similarities:
        db.similarities.insert_many(new_similarities)

    return f"Successfully processed {domain} and calculated {len(new_similarities)} new similarities."

@celery_app.task(name='tasks.run_graph_grouping')
def run_graph_grouping(similarity_threshold: float = 0.9):
    """
    Performs graph-based grouping using data from MongoDB and saves the result.
    """
    grouper = grouping.Grouper(db_client=mongo_client)
    grouped_results = grouper.group_by_graph_from_db(similarity_threshold)
    
    # Save results to a dedicated collection
    mongo_client.logo_db.grouping_results.replace_one(
        {"_id": "latest_graph_grouping"},
        {"groups": grouped_results, "updated_at": datetime.utcnow()},
        upsert=True
    )
    return "Graph-based grouping complete and results have been saved."

@celery_app.task(name='tasks.run_outlier_detection')
def run_outlier_detection():
    """
    Runs the batch outlier detection process on the entire database.
    """
    preprocessor = preprocessing.Preprocessor(db_client=mongo_client)
    preprocessor.eliminate_outliers_from_db()
    return "Outlier detection process completed."