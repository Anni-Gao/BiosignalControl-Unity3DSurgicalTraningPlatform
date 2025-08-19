import asyncio
import time
import os
import json
from bleak import BleakScanner, BleakClient
from hrv_utils import parse_rr_data, calculate_rmssd, calibrate_hrv, load_user_range, save_user_range
from log_hrv_data import log_hrv_to_csv, init_log_file
from unity.unity_server import send_scalpel_command, get_current_mode


# Global variables
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
rr_data_buffer = []
rmssd_history = []
MAX_HISTORY = 10

calibrated = False
hrv_low = None
hrv_high = None
start_time = None

user_id = input("Please enter User ID (e.g. user0): ").strip() or "user0"
user_file = os.path.join("data", "users", f"{user_id}_range.json")
log_file = f"data/logs/{user_id}_hrv_log.csv"

# 1. Initialize data directories
os.makedirs(os.path.dirname(user_file), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# 2. Check if user calibration file exists, decide whether calibration is required
if not os.path.exists(user_file):
    with open(user_file, "w") as f:
        json.dump({"hrv_low": None, "hrv_high": None}, f)
    print(f"🆕 Created new user data file: {user_file}")

user_range = None
try:
    user_range = load_user_range(user_id)
except Exception:
    user_range = None

calibrating = user_range is None or user_range == (None, None)

if calibrating:
    print("🧘 Starting 5-minute calibration...")
    start_time = time.time()
    init_log_file(log_file)  # Initialize file only during calibration (consistent with EEG)
else:
    hrv_low, hrv_high = user_range
    calibrated = True
    print(f"🗂️ Loaded calibration range: {hrv_low:.2f} ~ {hrv_high:.2f} ms")

async def main():
    global start_time, calibrated, hrv_low, hrv_high, rr_data_buffer, rmssd_history

    print("🔍 Scanning for devices...")

    devices = await BleakScanner.discover()
    polar_device = next((d for d in devices if d.name and "Polar H10" in d.name), None)

    if not polar_device:
        print("❌ Polar H10 not found")
        return

    print(f"✅ Device found: {polar_device.name} [{polar_device.address}]")
    await asyncio.sleep(2)

    # Calibration and real-time HRV monitoring
    def handle_hr_notification(sender, data):
        global calibrated, hrv_low, hrv_high, start_time, rr_data_buffer, rmssd_history

        rr_intervals = parse_rr_data(data)
        rr_data_buffer.extend(rr_intervals)

        # Calibration phase
        if not calibrated:
            elapsed = time.time() - start_time
            if len(rr_data_buffer) >= 10:
                recent_rr = rr_data_buffer[-10:]
                current_rmssd = calculate_rmssd(recent_rr)
                print(f"🧘 Calibrating: {int(elapsed)} seconds elapsed, real-time HRV: {current_rmssd:.2f} ms")
            else:
                print(f"🧘 Calibrating: {int(elapsed)} seconds elapsed, real-time HRV: calculating...")

            if elapsed >= 300:
                print("🧘 Calibration completed...")
                hrv_low, hrv_high = calibrate_hrv(rr_data_buffer)
                save_user_range(user_id, hrv_low, hrv_high)
                print(f"✅ Saved HRV range: {hrv_low:.2f} ~ {hrv_high:.2f} ms")
                calibrated = True
                rr_data_buffer.clear()
                init_log_file(log_file)  # Reinitialize log file after calibration (consistent with EEG)
            return

        # Monitoring phase
        if len(rr_data_buffer) >= 10:
            recent_rr = rr_data_buffer[-10:]
            rmssd = calculate_rmssd(recent_rr)

            rmssd_history.append(rmssd)
            if len(rmssd_history) > MAX_HISTORY:
                rmssd_history.pop(0)
            avg_rmssd = sum(rmssd_history) / len(rmssd_history)

            alert_flag = rmssd < hrv_low or rmssd > hrv_high
            action = "stop" if alert_flag else "move"

            if alert_flag:
                print(f"⚠️ ALERT! HRV: {rmssd:.2f} out of range, action: {action}")
            else:
                print(f"✅ HRV: {rmssd:.2f}, action: {action}")

            if get_current_mode() == 2:
                send_scalpel_command(
                    state=action,
                    source_id=2,
                    extras={
                        "hrv": round(rmssd, 2),
                        "average": round(avg_rmssd, 2),
                        "low": round(hrv_low, 2),
                        "high": round(hrv_high, 2),
                        "user": user_id
                    }
                )
            else:
                print(f"⚠️ Current mode is {get_current_mode()}, HRV command ignored.")
            log_hrv_to_csv(user_id, rmssd, avg_rmssd, action)

    async with BleakClient(polar_device.address, timeout=30.0) as client:
        print("🔗 Connected")
        await client.start_notify(HR_UUID, handle_hr_notification)

        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
