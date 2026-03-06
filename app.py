# app.py
import io
import json
import os
import boto3
import pandas as pd
import requests
from datetime import datetime, timezone
from fastapi import FastAPI, BackgroundTasks, Request
from baseline import BaselineManager
from processor import process_file
from zoneinfo import ZoneInfo

import logging

eastern_tz = ZoneInfo("America/New_York")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anomaly_app.log"),
        logging.StreamHandler()
    ]
    
)
# Here converting log timestamps to eastern time zone
logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).astimezone(EASTERN_TZ).timetuple()
logger = logging.getLogger(__name__)


app = FastAPI(title="Anomaly Detection Pipeline")

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        msg_type = request.headers.get("x-amz-sns-message-type")

        # SNS sends a SubscriptionConfirmation before it will deliver any messages.
        # Visiting the SubscribeURL confirms the subscription.
        if msg_type == "SubscriptionConfirmation":
            confirm_url = body["SubscribeURL"]
            requests.get(confirm_url)
            logger.info(f"SNS Subscription Confirmation: Visited {confirm_url}")
            return {"status": "confirmed"}

        if msg_type == "Notification":
            # The SNS message body contains the S3 event as a JSON string
            s3_event = json.loads(body["Message"])
            for record in s3_event.get("Records", []):
                key = record["s3"]["object"]["key"]
                if key.startswith("raw/") and key.endswith(".csv"):
                    logger.info(f"SNS Notification: New file detected: {key}")
                    background_tasks.add_task(process_file, BUCKET_NAME, key)
                else:
                    logger.warning(f"SNS Notification: Ignored file (wrong prefix/suffix): {key}")

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing SNS notification: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    logger.info(f"Query: Fetching {limit} recent anomalies.")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

    keys = sorted(
        [
            obj["Key"]
            for page in pages
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".csv")
        ],
        reverse=True,
    )[:10]

    all_anomalies = []
    for key in keys:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        if "anomaly" in df.columns:
            flagged = df[df["anomaly"] == True].copy()
            flagged["source_file"] = key
            all_anomalies.append(flagged)

    if not all_anomalies:
        return {"count": 0, "anomalies": []}

    combined = pd.concat(all_anomalies).head(limit)
    return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}


@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    logger.info("Query: Fetching anomaly summary.")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

    summaries = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("_summary.json"):
                response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                summaries.append(json.loads(response["Body"].read()))

    if not summaries:
        return {"message": "No processed files yet."}

    total_rows = sum(s["total_rows"] for s in summaries)
    total_anomalies = sum(s["anomaly_count"] for s in summaries)

    return {
        "files_processed": len(summaries),
        "total_rows_scored": total_rows,
        "total_anomalies": total_anomalies,
        "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
        "most_recent": sorted(summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
    }


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    logger.info("Query: Fetching current baseline. ")
    baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
    baseline = baseline_mgr.load()

    channels = {}
    for channel, stats in baseline.items():
        if channel == "last_updated":
            continue
        channels[channel] = {
            "observations": stats["count"],
            "mean": round(stats["mean"], 4),
            "std": round(stats.get("std", 0.0), 4),
            "baseline_mature": stats["count"] >= 30,
        }

    return {
        "last_updated": baseline.get("last_updated"),
        "channels": channels,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "bucket": BUCKET_NAME, 
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
