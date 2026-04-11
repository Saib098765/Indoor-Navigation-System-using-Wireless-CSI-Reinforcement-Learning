import argparse
import csv
import os
import time
import threading
from collections import deque

import numpy as np

NUM_FEATURES      = 128
COLLECTION_SECONDS = 5        # seconds to hold still at each location
OUTPUT_FIELDS     = (
    ["label", "building", "floor", "latitude", "longitude"]
    + [f"CSI_DATA{i}" for i in range(1, NUM_FEATURES + 1)]
)

try:
    from picoscenes import Picoscenes
    PICOSCENES_AVAILABLE = True
except ImportError:
    PICOSCENES_AVAILABLE = False

def extract_amplitude(frame):
    try:
        csi_matrix = frame.CSISegment.CSI
        return np.abs(csi_matrix).flatten().astype(np.float32)
    except AttributeError:
        return np.array([], dtype=np.float32)


def buffer_to_feature(buffer):
    arrays = [a for a in buffer if len(a) > 0]
    if not arrays:
        return np.full(NUM_FEATURES, -100.0, dtype=np.float32)
    max_len = max(len(a) for a in arrays)
    padded  = [np.pad(a, (0, max_len - len(a)), constant_values=0) for a in arrays]
    stacked = np.vstack(padded)
    median_vec = np.median(stacked, axis=0)
    if len(median_vec) >= NUM_FEATURES:
        feat = median_vec[:NUM_FEATURES]
    else:
        feat = np.pad(median_vec, (0, NUM_FEATURES - len(median_vec)), constant_values=0)
    feat = np.log1p(np.maximum(feat, 0))
    mn, mx = feat.min(), feat.max()
    if mx - mn > 1e-6:
        feat = 2.0 * (feat - mn) / (mx - mn) - 1.0
    return feat.astype(np.float32)


def collect_one_location(ps, duration=COLLECTION_SECONDS):
    buf = deque()
    lock = threading.Lock()

    def on_frame(frame):
        amp = extract_amplitude(frame)
        if amp.size > 0:
            with lock:
                buf.append(amp)

    ps.set_rx_callback(on_frame)
    ps.start()
    print(f"  Collecting for {duration}s")
    time.sleep(duration)
    ps.stop()

    with lock:
        feat = buffer_to_feature(buf)
    print(f" Collected {len(buf)} frames → feature shape: {feat.shape}")
    return feat


def rssi_fallback_collect(duration=COLLECTION_SECONDS):
    # fallback rssi
    import subprocess, re
    all_rssi = []
    end = time.time() + duration
    while time.time() < end:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            m = re.search(r"Signal\s*:\s*(\d+)%", line)
            if m:
                all_rssi.append((int(m.group(1)) / 2) - 100)
        time.sleep(1)

    vec = np.array(all_rssi if all_rssi else [-100.0], dtype=np.float32)
    if len(vec) >= NUM_FEATURES:
        vec = vec[:NUM_FEATURES]
    else:
        vec = np.pad(vec, (0, NUM_FEATURES - len(vec)), constant_values=-100.0)
    mn, mx = vec.min(), vec.max()
    if mx - mn > 1e-6:
        vec = 2.0 * (vec - mn) / (mx - mn) - 1.0
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="Wi-Fi")
    parser.add_argument("--out", default="csi_training_data.csv")
    parser.add_argument("--force-rssi", action="store_true")
    args = parser.parse_args()

    rows = []
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"Resuming — {len(rows)} samples already collected.")

    ps = None
    if PICOSCENES_AVAILABLE and not args.force_rssi:
        ps = Picoscenes(args.interface)

    print("Type 'done' when finished.\n")

    location_id = len(rows)  

    while True:
        print(f"\n Location #{location_id} ---")
        label_in = input("  location_id (or 'done'): ").strip()
        if label_in.lower() == "done":
            break

        building = input("  building index (0/1/2): ").strip()
        floor    = input("  floor index (0/1/2/3/4): ").strip()
        lat      = input("  latitude (e.g. 39.99328): ").strip()
        lon      = input("  longitude (e.g. -0.06864): ").strip()

        # Collect CSI / RSSI
        if ps is not None:
            feat = collect_one_location(ps)
        else:
            print("  [RSSI fallback mode]")
            feat = rssi_fallback_collect()

        row = {
            "label":     label_in,
            "building":  building,
            "floor":     floor,
            "latitude":  lat,
            "longitude": lon,
        }
        for i, val in enumerate(feat):
            row[f"CSI_DATA{i+1}"] = f"{val:.6f}"

        rows.append(row)
        location_id += 1

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  Saved. Total samples: {len(rows)}")

    print(f"\nDone. {len(rows)} samples saved to '{args.out}'.")
    print("Next step: run  python train_part1_hierarchical_v2.py")


if __name__ == "__main__":
    main()
