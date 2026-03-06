#!/usr/bin/env python3
import json
import io
import boto3
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from baseline import BaselineManager
from detector import AnomalyDetector


import logging

eastern_tz = ZoneInfo("America/New_York")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/anomaly-detection/anomaly_app.log"),
        logging.StreamHandler()
    ]
    
)
# Here converting log timestamps to eastern time zone
logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).astimezone(EASTERN_TZ).timetuple()
logger = logging.getLogger(__name__)


s3 = boto3.client("s3")


NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]  # students configure this

def process_file(bucket: str, key: str):
    try:
        start_time = datetime.now(timezone.utc)
        logger.info(f"Processing: s3://{bucket}/{key}")

        # 1. Download raw file
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))

        print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

        # 2. Load current baseline
        baseline_mgr = BaselineManager(bucket=bucket)
        baseline = baseline_mgr.load()

        # 3. Update baseline with values from this batch BEFORE scoring
        #    (use only non-null values for each channel)
        for col in NUMERIC_COLS:
            if col in df.columns:
                clean_values = df[col].dropna().tolist()
                if clean_values:
                    baseline = baseline_mgr.update(baseline, col, clean_values)

        # 4. Run detection
        detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
        scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")

        # 5. Write scored file to processed/ prefix
        output_key = key.replace("raw/", "processed/")
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )

        # 6. Save updated baseline back to S3
        baseline_mgr.save(baseline)

        #having the logger write to disk before uploading to s3
        for handler in logger.handlers:
            handler.flush()

        try:
            s3.upload_file("/opt/anomaly-detection/anomaly_app.log", bucket, "logs/anomaly_app.log")
            logger.info("Successfully synced log file to S3.")
        except Exception as e:
            logger.error(f"Failed to sync logs to S3: {e}")

        # 7. Build and return a processing summary
        anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df else 0
        summary = {
            "source_key": key,
            "output_key": output_key,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "baseline_observation_counts": {
                col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
            }
        }

        # Write summary JSON alongside the processed file
        summary_key = output_key.replace(".csv", "_summary.json")
        s3.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json"
        )

        print(f"  Done: {anomaly_count}/{len(df)} anomalies flagged")
        return summary
    except Exception as e:
        logger.error(f"Critical Error processing {key}: {str(e)}", exc_info=True)
        return None
