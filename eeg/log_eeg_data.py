import csv

def log_to_csv(log_file, timestamp, attention, stress, action):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, attention, stress, action])

def init_log_file(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "attention", "stress", "action"])
