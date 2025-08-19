import websocket
import time
import threading
import os
import numpy as np
from datetime import datetime

from eeg_utils import send_json, calculate_ranges_with_std, save_user_ranges, load_user_ranges
from log_eeg_data import log_to_csv, init_log_file
from unity.unity_server import send_scalpel_command, get_current_mode

# === Register from https://account.emotiv.com/ and get license ===
CORTEX_URL = "wss://localhost:6868"
CLIENT_ID = "9ydi97uIqo4PwAvsrkSSjFB7XuudKa0sj6vOK4Jl"
CLIENT_SECRET = "vaFevxurynGgjUcWD8I0etEZnsHde7kMlfIkQUwUwxPv5noaSeG8hZXCxVVfMWdMioPfbwX7wH2JOqIlTUePimhtJWgQkne3dDYfoNJqsgdOzD6wvLHGybQ6SNXBZTEj"

user_id = input("Enter EEG User ID (e.g. user0): ").strip() or "user0"
os.makedirs("data/users", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

range_file = f"data/users/{user_id}_range.json"
log_file = f"data/logs/{user_id}_eeg_log.csv"

calibrating = not os.path.exists(range_file)
attention_values, stress_values = [], []
start_time = None

AUTH_TOKEN = None
HEADSET_ID = None
SESSION_ID = None
WS = None


# === WebSocket Message Handling ===

def on_message(ws, message):
    global calibrating, attention_values, stress_values, start_time

    import json
    try:
        msg = json.loads(message)
    except Exception as e:
        print(f"❌ JSON decode error: {e}")
        return

    if isinstance(msg, list):
        for item in msg:
            if isinstance(item, dict):
                handle_message(ws, item)
    elif isinstance(msg, dict):
        handle_message(ws, msg)
    else:
        print(f"⚠️ Unknown message type: {type(msg)}")


def handle_message(ws, msg):
    global AUTH_TOKEN, HEADSET_ID, SESSION_ID, calibrating, start_time

    if not isinstance(msg, dict):
        return

    # Cortex error handling
    if "id" in msg and "error" in msg:
        print(f"❌ Cortex error: {msg['error'].get('message', 'Unknown error')}")
        return

    # Handle authorization, device, session, etc.
    if "id" in msg:
        if msg["id"] == 2:
            AUTH_TOKEN = msg["result"].get("cortexToken")
        elif msg["id"] == 3:
            headsets = msg["result"]
            if isinstance(headsets, list) and len(headsets) > 0:
                HEADSET_ID = headsets[0].get("id")
            else:
                print("❌ No headset found")
        elif msg["id"] == 4:
            SESSION_ID = msg["result"].get("id")
        elif msg["id"] == 5:
            print("✅ Subscribed to MET stream")
        return

    # Handle MET (Mental Effort) data stream
    if "met" in msg:
        met = msg["met"]
        try:
            attention = met[1]
            stress = met[3]
        except Exception as e:
            print(f"❌ Error parsing MET data: {e}")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calibration phase: collect sample data
        if calibrating:
            attention_values.append(attention)
            stress_values.append(stress)
            elapsed = int(time.time() - start_time)
            print(f"🧘 Calibrating... {elapsed}s, attention={attention:.3f}, stress={stress:.3f}")
            return

        try:
            ranges = load_user_ranges(range_file)
            attn_low, attn_high = ranges["attention"]
            stress_max = ranges["stress_max"]

            attention_ok = attention >= attn_low

            action = "move" if attention_ok else "stop"

            status_emoji = "✅" if action == "move" else "⚠️"
            print(
                f"{status_emoji} {timestamp} | Attention: {attention:.3f} (≥{attn_low:.3f}), Stress: {stress:.3f} (≤{stress_max:.3f}) => {action}")

            # Only send command if current mode is EEG (3)
            if get_current_mode() == 3:
                send_scalpel_command(action, source_id=3, extras={
                    "attention": round(attention, 3),
                    "stress": round(stress, 3)
                })
            else:
                print(f"⚠️ Current mode is {get_current_mode()}, EEG command ignored.")

            log_to_csv(log_file, timestamp, attention, stress, action)
        except Exception as e:
            print(f"❌ Error processing MET data: {e}")


def on_open(ws):
    def setup():
        global start_time, calibrating, AUTH_TOKEN, HEADSET_ID, SESSION_ID

        send_json(ws, "requestAccess", {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET
        }, id=1)
        time.sleep(1)

        send_json(ws, "authorize", {
            "clientId": CLIENT_ID,
            "clientSecret": CLIENT_SECRET,
            "debit": 1
        }, id=2)
        time.sleep(1)

        send_json(ws, "queryHeadsets", id=3)
        time.sleep(1)

        while not (AUTH_TOKEN and HEADSET_ID):
            time.sleep(0.5)

        send_json(ws, "createSession", {
            "cortexToken": AUTH_TOKEN,
            "headset": HEADSET_ID,
            "status": "active"
        }, id=4)
        time.sleep(1)

        while not SESSION_ID:
            time.sleep(0.5)

        send_json(ws, "subscribe", {
            "cortexToken": AUTH_TOKEN,
            "session": SESSION_ID,
            "streams": ["met"]
        }, id=5)

        if calibrating:
            print("🧘 Starting 5-minute calibration...")
            start_time = time.time()

            def finish_calibration():
                global calibrating
                attn_low, attn_high = calculate_ranges_with_std(attention_values, k=1.0)

                stress_mean = np.mean(stress_values)
                stress_std = np.std(stress_values)
                stress_max = max(0, stress_mean + stress_std)

                save_user_ranges(range_file, {
                    "attention": [attn_low, attn_high],
                    "stress_max": round(stress_max, 2)
                })
                print(
                    f"✅ Calibration complete:\n  Attention range=[{attn_low:.3f}, {attn_high:.3f}]\n  Stress max ≤ {stress_max:.3f}")
                calibrating = False

            threading.Timer(300, finish_calibration).start()

    threading.Thread(target=setup).start()


def on_error(ws, error):
    print(f"❌ WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed")


def main():
    print(f"📡 Starting EEG monitoring for user: {user_id}")
    if calibrating:
        print("🧘 Entering calibration mode...")
        init_log_file(log_file)

    global WS
    WS = websocket.WebSocketApp(CORTEX_URL,
                                on_message=on_message,
                                on_open=on_open,
                                on_error=on_error,
                                on_close=on_close)
    WS.run_forever(sslopt={"cert_reqs": 0})


if __name__ == "__main__":
    main()
