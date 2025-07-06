# src/app.py (Refactored)

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
import os

from src.tasks import start_domain_processing, run_graph_grouping, run_outlier_detection

# --- App and DB Connection ---
app = FastAPI(
    title="Logo Similarity API",
    description="An API to process and group company logos using a database-driven backend.",
    version="3.0.0"
)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.logo_db

# --- Pydantic Models ---
class ScrapeRequest(BaseModel):
    domains: List[str]

class GroupRequest(BaseModel):
    similarity_threshold: float = 0.90

class TaskResponse(BaseModel):
    message: str

# --- API Endpoints ---

@app.post("/api/v1/logos", status_code=202, response_model=TaskResponse)
async def scrape_logos(payload: ScrapeRequest = Body(...)):
    """
    Accepts a list of domains and queues a background task to process them.
    This process is idempotent; domains already in the database will be skipped.
    """
    start_domain_processing.delay(payload.domains)
    return {"message": "Domain processing task has been queued."}

@app.post("/api/v1/groups", status_code=202, response_model=TaskResponse)
async def create_groups(payload: GroupRequest = Body(...)):
    """
    Triggers a background task to perform graph-based grouping on all existing data.
    The results will be saved in the database.
    """
    run_graph_grouping.delay(payload.similarity_threshold)
    return {"message": "Grouping task has been queued."}

@app.get("/api/v1/groups")
async def get_groups():
    """
    Retrieves the latest grouping results from the database.
    """
    latest_grouping = db.grouping_results.find_one({"_id": "latest_graph_grouping"})
    if not latest_grouping:
        raise HTTPException(status_code=404, detail="No grouping results found. Please trigger the grouping task first via POST /api/v1/groups.")
    
    # Return the groups, removing the internal _id
    return latest_grouping.get("groups", {})

@app.post("/api/v1/maintenance/run-outlier-detection", status_code=202, response_model=TaskResponse)
async def trigger_outlier_detection():
    """
    Triggers a background maintenance task to find and remove outlier images
    from the entire database.
    """
    run_outlier_detection.delay()
    return {"message": "Outlier detection maintenance task has been queued."}