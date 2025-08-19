import numpy as np
import json
import os

def parse_rr_data(data: bytearray):
    rr_intervals = []
    flags = data[0]
    hr_value_format = flags & 0x01
    energy_expended_status = (flags >> 3) & 0x01
    rr_interval_status = (flags >> 4) & 0x01

    index = 1
    # Read heart rate value: 1 or 2 bytes
    if hr_value_format == 0:
        hr = data[index]
        index += 1
    else:
        hr = int.from_bytes(data[index:index+2], byteorder='little')
        index += 2

    # If energy expended field exists, skip 2 bytes
    if energy_expended_status == 1:
        index += 2

    # Read RR-interval data
    if rr_interval_status == 1:
        while index + 1 < len(data):
            rr_raw = int.from_bytes(data[index:index+2], byteorder='little')
            # RR interval unit is 1/1024 seconds, convert to milliseconds
            rr_ms = rr_raw / 1024 * 1000
            rr_intervals.append(rr_ms)
            index += 2

    return rr_intervals


def calculate_rmssd(rr_intervals):
    if len(rr_intervals) < 2:
        return 0
    diff_sq = [(rr_intervals[i] - rr_intervals[i-1])**2 for i in range(1, len(rr_intervals))]
    rmssd = np.sqrt(np.mean(diff_sq))
    return rmssd


def calibrate_hrv(rr_intervals, k=1.0):
    window_size = 10
    rmssd_values = []

    for i in range(0, len(rr_intervals) - window_size, window_size):
        window = rr_intervals[i:i + window_size]
        if len(window) >= 2:
            rmssd = calculate_rmssd(window)
            rmssd_values.append(rmssd)

    if not rmssd_values:
        return 0, 0

    arr = np.array(rmssd_values)
    mean = np.mean(arr)
    std = np.std(arr)
    low = max(0, mean - k * std)
    high = mean + k * std
    return round(low, 2), round(high, 2)


def save_user_range(user_id, low, high):
    path = f"data/users/{user_id}_range.json"
    data = {"hrv_low": low, "hrv_high": high}
    os.makedirs("data/users", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_user_range(user_id):
    path = f"data/users/{user_id}_range.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
        low = data.get("hrv_low")
        high = data.get("hrv_high")
        if low is None or high is None:
            return None
        return low, high
