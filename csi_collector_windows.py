"""
Runs on an Intel AX200 / AX210 Windows machine.
Collects real WiFi Channel State Information (CSI) using PicoScenes,
processes it into a 128-dimensional feature vector, and serves it
over a local HTTP server so the Flutter Android app can fetch it.

Requirements
------------
  pip install picoscenes flask flask-cors numpy scipy

Hardware
--------
  Intel AX200 or AX210 NIC (must support monitor mode via PicoScenes)

Usage
-----
  python csi_collector_windows.py --interface Wi-Fi --port 8765

The Flutter app connects to http://<this-machine-ip>:8765/csi and receives a JSON array of 128 floats.
"""

import argparse
import json
import logging
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS

try:
    from picoscenes import Picoscenes
    PICOSCENES_AVAILABLE = True
except ImportError:
    PICOSCENES_AVAILABLE = False
    print("picoscenes not installed.")

# Config 
NUM_FEATURES     = 128          
NUM_SUBCARRIERS  = 64           # AX200 typically gives 64 subcarriers per stream
CACHE_SECONDS    = 3            # aggregate this many seconds before computing feature
FALLBACK_RSSI    = -100.0       # value for missing subcarriers

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 
_lock            = threading.Lock()
_feature_cache   = np.full(NUM_FEATURES, FALLBACK_RSSI, dtype=np.float32)
_raw_csi_buffer  = deque(maxlen=300)   # ring buffer of raw frames
_last_update     = 0.0


def extract_amplitude(csi_frame) -> np.ndarray:
    """
    Pull subcarrier amplitudes from one PicoScenes CSI frame.
    Returns a flat float32 array of length (num_subcarriers * num_streams).
    AX200 in 20 MHz HE mode: 64 subcarriers, up to 2 spatial streams.
    """
    try:
        csi_matrix = csi_frame.CSISegment.CSI   # shape: (subcarriers, rx, tx)
        amplitude  = np.abs(csi_matrix) 
        return amplitude.flatten().astype(np.float32)
    except AttributeError:
        return np.array([], dtype=np.float32)


def buffer_to_feature(buffer: deque) -> np.ndarray:
    """
    Plan
    --------
    1. Stack all amplitude arrays from the buffer.
    2. Take the median across time (robust to transient noise).
    3. Pad or truncate to NUM_FEATURES (128).
    4. Apply log-scale compression and min-max normalisation.
    """
    if not buffer:
        return np.full(NUM_FEATURES, FALLBACK_RSSI, dtype=np.float32)

    arrays = [a for a in buffer if len(a) > 0]
    if not arrays:
        return np.full(NUM_FEATURES, FALLBACK_RSSI, dtype=np.float32)

    max_len = max(len(a) for a in arrays)
    padded  = [np.pad(a, (0, max_len - len(a)), constant_values=0) for a in arrays]
    stacked = np.vstack(padded) # shape: (frames, features)

    # Temporal median 
    median_vec = np.median(stacked, axis=0)   # shape: (max_len,)

    if len(median_vec) >= NUM_FEATURES:
        feat = median_vec[:NUM_FEATURES]
    else:
        feat = np.pad(median_vec, (0, NUM_FEATURES - len(median_vec)),
                      constant_values=0)

    feat = np.log1p(np.maximum(feat, 0))

    # Min-max normalise to [-1, 1]
    mn, mx = feat.min(), feat.max()
    if mx - mn > 1e-6:
        feat = 2.0 * (feat - mn) / (mx - mn) - 1.0
    else:
        feat = np.zeros(NUM_FEATURES, dtype=np.float32)

    return feat.astype(np.float32)

def picoscenes_loop(interface: str):
    """Background thread: continuously collect CSI frames via PicoScenes."""
    log.info(f"Starting PicoScenes capture on interface: {interface}")
    ps = Picoscenes(interface)

    def on_frame(frame):
        amp = extract_amplitude(frame)
        if amp.size > 0:
            with _lock:
                _raw_csi_buffer.append(amp)

    ps.set_rx_callback(on_frame)

    try:
        ps.start()
        log.info("PicoScenes capture started. Receiving CSI frames...")
        while True:
            time.sleep(CACHE_SECONDS)
            with _lock:
                feat = buffer_to_feature(_raw_csi_buffer)
                _raw_csi_buffer.clear()
                global _feature_cache, _last_update
                _feature_cache  = feat
                _last_update    = time.time()
            log.info(f"Feature updated — mean amplitude: {feat.mean():.4f}")
    except KeyboardInterrupt:
        ps.stop()
        log.info("PicoScenes capture stopped.")

def rssi_fallback_loop():
    """
    Fallback: use Windows netsh to collect RSSI from nearby APs and
    build a pseudo-CSI feature vector from RSSI values.
    This gives RSSI-based fingerprinting, not true CSI, but keeps
    the rest of the pipeline working for testing.
    """
    import subprocess
    import re

    log.warning("Using fallback mode — not real CSI.")
    log.warning("For real CSI, install PicoScenes + Intel AX200/AX210.")

    while True:
        try:
            result = subprocess.run(
                ["netsh", "wlan", "show", "networks", "mode=bssid"],
                capture_output=True, text=True, timeout=10
            )
            lines  = result.stdout.splitlines()
            rssi_values = []

            for line in lines:
                match = re.search(r"Signal\s*:\s*(\d+)%", line)
                if match:
                    pct  = int(match.group(1))
                    dbm  = (pct / 2) - 100
                    rssi_values.append(float(dbm))

            if rssi_values:
                vec = np.array(rssi_values, dtype=np.float32)
                if len(vec) >= NUM_FEATURES:
                    vec = vec[:NUM_FEATURES]
                else:
                    vec = np.pad(vec, (0, NUM_FEATURES - len(vec)),
                                 constant_values=FALLBACK_RSSI)

                # Normalise to [-1, 1]
                mn, mx = vec.min(), vec.max()
                if mx - mn > 1e-6:
                    vec = 2.0 * (vec - mn) / (mx - mn) - 1.0

                with _lock:
                    global _feature_cache, _last_update
                    _feature_cache = vec
                    _last_update   = time.time()

                log.info(f"RSSI fallback: {len(rssi_values)} APs scanned.")

        except Exception as e:
            log.error(f"RSSI scan error: {e}")

        time.sleep(CACHE_SECONDS)


@app.route("/csi", methods=["GET"])
def get_csi():
    """
    GET /csi
    Returns a JSON object:
    {
        "features": [f0, f1, ..., f127],   // 128 floats
        "timestamp": 1712345678.123,        // unix time of last update
        "stale": false                      // true if data is older than 5s
    }
    """
    with _lock:
        feat      = _feature_cache.tolist()
        ts        = _last_update
        stale     = (time.time() - ts) > 5.0

    return jsonify({
        "features":  feat,
        "timestamp": ts,
        "stale":     stale,
        "num_features": len(feat),
    })


@app.route("/status", methods=["GET"])
def status():
    with _lock:
        ts    = _last_update
        stale = (time.time() - ts) > 5.0

    return jsonify({
        "ok":              True,
        "mode":            "csi" if PICOSCENES_AVAILABLE else "rssi_fallback",
        "last_update":     ts,
        "stale":           stale,
        "feature_length":  NUM_FEATURES,
    })


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"pong": True})


def main():
    parser = argparse.ArgumentParser(
        description="CSI collector + HTTP server for Flutter indoor nav app"
    )
    parser.add_argument("--interface", default="Wi-Fi",
                        help="NIC name as shown in Device Manager (default: Wi-Fi)")
    parser.add_argument("--port", type=int, default=8765,
                        help="HTTP port the Flutter app connects to (default: 8765)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind Flask server (default: 0.0.0.0)")
    parser.add_argument("--force-rssi", action="store_true",
                        help="Force RSSI fallback mode even if PicoScenes is available")
    args = parser.parse_args()

    if PICOSCENES_AVAILABLE and not args.force_rssi:
        capture_thread = threading.Thread(
            target=picoscenes_loop, args=(args.interface,), daemon=True
        )
    else:
        capture_thread = threading.Thread(
            target=rssi_fallback_loop, daemon=True
        )
    capture_thread.start()

    time.sleep(1)

    log.info(f"HTTP server starting on http://{args.host}:{args.port}")
    log.info(f"Flutter app should connect to: http://<this-PC-IP>:{args.port}/csi")
    log.info("Find this PC's IP with: ipconfig  (look for IPv4 Address)")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()