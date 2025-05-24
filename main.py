import os
import uuid
import shutil
import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from crawler import crawl_recursive_batch, clean_text_for_llm  # assuming you extracted these from earlier

app = FastAPI()

BASE_DIR = "jobs"
os.makedirs(BASE_DIR, exist_ok=True)

status_dict = {}  # In-memory status store

class ScrapeRequest(BaseModel):
    urls: list[str]
    max_depth: int = 2
    tabs: int = 10
    cleaning: bool = True


@app.get("/")
def root():
    return {"status": "Crawl API is live!"}

@app.post("/scrape")
async def start_scrape(req: ScrapeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_DIR, job_id)
    os.makedirs(job_dir)

    raw_path = os.path.join(job_dir, "dataset.txt")
    #cleaned_path = os.path.join(job_dir, "cleaned_dataset.txt")
    log_path = os.path.join(job_dir, "log.txt")

    status_dict[job_id] = {"status": "in_progress"}

    def write_log(msg: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    async def task():
        try:
            write_log("Starting scrape...")
            await crawl_recursive_batch(req.urls, max_depth=req.max_depth, max_concurrent=req.tabs, output_path=raw_path, log_fn=write_log)
            write_log("Scrape complete.")
            if req.cleaning:
                write_log("Cleaning text...")
                clean_text_for_llm(raw_path)
            status_dict[job_id] = {"status": "completed"}
            write_log("✅ Done.")
        except Exception as e:
            write_log(f"❌ Error: {e}")
            status_dict[job_id] = {"status": "failed"}

    background_tasks.add_task(task)
    return {"job_id": job_id}

@app.get("/scrape/{job_id}/status")
def get_status(job_id: str):
    status = status_dict.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return status

@app.get("/scrape/{job_id}/logs")
def get_logs(job_id: str):
    log_path = os.path.join(BASE_DIR, job_id, "log.txt")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Logs not found")
    with open(log_path, "r", encoding="utf-8") as f:
        return {"logs": f.read()}

@app.get("/scrape/{job_id}/dataset")
def get_dataset(job_id: str):
    dataset_path = os.path.join(BASE_DIR, job_id, "dataset.txt")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found or still processing")
    return FileResponse(dataset_path, filename=f"cleaned_{job_id}.txt")

@app.delete("/scrape/{job_id}")
def delete_job(job_id: str):
    job_dir = os.path.join(BASE_DIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    shutil.rmtree(job_dir)
    status_dict.pop(job_id, None)
    return {"message": f"Job {job_id} deleted."}
