import csv
import os
from datetime import datetime

def init_log_file(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Keep consistent with EEG log format
            writer.writerow(["timestamp", "rmssd", "average", "action"])

def log_hrv_to_csv(user_id, rmssd, average, action):
    log_file = f"data/logs/{user_id}_hrv_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize the file (writes header if needed)
    init_log_file(log_file)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, round(rmssd, 2), round(average, 2), action])
