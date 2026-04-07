# 📡 Indoor Navigation System using Wireless CSI & Reinforcement Learning

> **Author:** Sai Sleghana Bala  
> A thesis project implementing a two-stage AI pipeline for GPS-free indoor navigation — combining a WiFi fingerprint classifier with a Proximal Policy Optimization (PPO) reinforcement learning agent — deployed as a real-time AR mobile app.

---

## 🧭 Overview

GPS fails indoors. This system solves that by turning ambient **WiFi Channel State Information (CSI)** — the signal "fingerprints" already present in any building — into a precise positioning and navigation system.

The pipeline works in two stages:

1. **Localization** — A hierarchical deep neural network reads 128 WiFi RSSI features and predicts the user's current location node on a map graph (building → floor → room).
2. **Navigation** — A PPO reinforcement learning agent plans the optimal path across the map graph from the current location to a user-selected destination.

The result is a **Flutter-based AR mobile app** that overlays turn-by-turn directions on a live camera feed using a compass-aligned arrow, all without any GPS or internet connection at runtime.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                         │
│                                                                  │
│  UJIIndoorLoc Dataset (UCI)                                      │
│         │                                                        │
│         ▼                                                        │
│  download_and_prepare_uji.py  ──►  uji_data/                    │
│         │                          ├── uji_train.csv            │
│         │                          ├── uji_test.csv             │
│         │                          ├── location_info_*.pkl      │
│         │                          └── topology_graph_*.pkl     │
│         │                                                        │
│         ├──► train_part1_hierarchical_v2.py                      │
│         │         Hierarchical CNN Classifier                    │
│         │         (Building → Floor → Room)                      │
│         │         Output: hierarchical_classifier_best.pth       │
│         │                                                        │
│         └──► train_part2_hierarchical.py                         │
│                   PPO RL Navigation Agent                        │
│                   Env: Graph-based MDP                           │
│                   Output: ppo_agent.pth                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      MODEL EXPORT                                │
│                                                                  │
│  converter-flutter.py  →  wifi_tracker.onnx                     │
│  test.py               →  map_graph.json                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                   FLUTTER MOBILE APP (thesis_ar_nav)             │
│                                                                  │
│  WiFi Scan (128 RSSI features)                                   │
│         │                                                        │
│         ▼                                                        │
│  ai_engine.dart  ──►  TFLite Interpreter  ──►  Node ID          │
│                            │                                     │
│                            ▼                                     │
│  navigation_controller.dart  ──►  BFS on map_graph.json         │
│                            │                                     │
│                            ▼                                     │
│  main.dart  ──►  AR Overlay (Camera + Compass Arrow)            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File(s) | Role |
|---|---|---|
| Config | `config.py` | Central PPO hyperparameters & reward shaping |
| Data Prep | `download_and_prepare_uji.py` | Downloads dataset, builds graph topology |
| Graph Rebuild | `rebuild-graph.py`, `prune_graph.py` | k-NN graph construction, edge pruning |
| Classifier | `train_part1_hierarchical_v2.py` | Hierarchical WiFi fingerprint model |
| RL Agent | `train_part2_hierarchical.py` | PPO agent trained on topology graph |
| Export | `converter-flutter.py`, `test.py` | ONNX + JSON export for mobile |
| Mobile App | `thesis_ar_nav/` | Flutter AR navigation UI |

---

## 📊 Dataset — UJIIndoorLoc

- **Source:** [UCI ML Repository — UJIIndoorLoc (ID 310)](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
- **Coverage:** 3 university buildings, 5 floors, 748 unique location nodes
- **Features:** 520 WAP (WiFi Access Point) RSSI readings per sample, compressed to 128 CSI features in this system
- **Size:** ~19,000 training samples, ~1,100 validation samples

The `download_and_prepare_uji.py` script handles everything — download, extraction, format conversion, graph topology construction, and train/test splitting — automatically.

---

## ⚙️ Reinforcement Learning Design

| Parameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Learning Rate | 3e-4 |
| Discount Factor (γ) | 0.99 |
| Clip Epsilon | 0.2 |
| Epochs per Update | 4 |
| Batch Size | 64 |

**Reward shaping:**

| Event | Reward |
|---|---|
| Reaching the goal | +1000 |
| Progress toward goal | +50 |
| Regressing away | −20 |
| Invalid action | −100 |
| Step penalty | −1 |
| Timeout | −500 |

---

## 🖼️ Screenshots

> *(Place your screenshots in this section)*

| Venue Selection | AR Navigation View | Trajectory Plot |
|---|---|---|
| `[screenshot]` | `[screenshot]` | `trajectory_plot.png` |

---

## 🚀 Execution Guide

### Prerequisites

- Python 3.9+
- Flutter SDK 3.x (`sdk: ^3.11.4`)
- Android device or emulator (API 21+) with WiFi scanning support

---

### Part 1 — Python: Training the Models

#### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2 — Download and Prepare the Dataset

This downloads the UJIIndoorLoc dataset from UCI and prepares all graph/topology files.

```bash
python download_and_prepare_uji.py
```

Expected output directory: `uji_data/` containing CSVs and `.pkl` graph files.

#### Step 3 — (Optional) Rebuild / Fix the Navigation Graph

If the graph has disconnected components or bad edges, regenerate it:

```bash
# Rebuild from scratch using k-NN (recommended)
python rebuild-graph.py

# Or prune long/invalid edges from existing graph
python prune_graph.py
```

> Run `python diagnose_graph.py` first to check graph health.

#### Step 4 — Train the WiFi Location Classifier (Part 1)

Trains the hierarchical deep neural network on WiFi fingerprints.

```bash
python train_part1_hierarchical_v2.py
```

Output: `models/hierarchical_classifier_best.pth`, `models/scaler.pkl`

#### Step 5 — Train the RL Navigation Agent (Part 2)

Trains the PPO agent on the topology graph using the classifier's feature embeddings.

```bash
python train_part2_hierarchical.py
```

Output: `models/ppo_agent.pth`, `models/training_history.png`

#### Step 6 — Visualize a Navigation Trajectory (Optional)

```bash
python trajectory.py
```

Output: `trajectory_plot.png` — a figure showing ground truth path vs. AI-predicted locations.

---

### Part 2 — Export Models for Mobile

#### Step 7 — Export Classifier to ONNX

```bash
python converter-flutter.py
```

Output: `wifi_tracker.onnx` + `wifi_tracker.onnx.data`

#### Step 8 — Export Map Graph to JSON

```bash
python test.py
```

Output: `map_graph.json`

#### Step 9 — Copy Assets to Flutter App

```bash
cp wifi_tracker.onnx.data thesis_ar_nav/assets/  # if using custom model
cp map_graph.json thesis_ar_nav/assets/
```

> **Note:** Pre-built `.tflite` models and `map_graph.json` are already included in `thesis_ar_nav/assets/`. You only need this step if you retrained the models.

---

### Part 3 — Running the Flutter App

#### Step 10 — Install Flutter Dependencies

```bash
cd thesis_ar_nav
flutter pub get
```

#### Step 11 — Connect a Device and Run

```bash
flutter run
```

> The app requires a **physical Android device** for real WiFi scanning. Emulators will not return real scan results.

#### App Flow

1. Launch → **Select Venue** (University Building or Railway Station)
2. Select a **Destination** from the available zones
3. The app scans WiFi, feeds 128 RSSI features into the TFLite model
4. Your current node is predicted → BFS path is calculated → **AR compass arrow** points toward the next waypoint

---

## 📁 Project Structure

```
THESIS/
├── config.py                        # Central hyperparameter config
├── requirements.txt                 # Python dependencies
├── download_and_prepare_uji.py      # Dataset download & graph builder
├── train_part1_hierarchical_v2.py   # WiFi classifier training
├── train_part2_hierarchical.py      # PPO RL agent training
├── rebuild-graph.py                 # Graph reconstruction (k-NN)
├── prune_graph.py                   # Edge pruning utility
├── fix_graph_topology.py            # Topology repair helper
├── diagnose_graph.py                # Graph health diagnostics
├── converter-flutter.py             # PyTorch → ONNX export
├── test.py                          # Graph → JSON export
├── trajectory.py                    # Trajectory visualization
├── visualizer.py                    # Map visualizer
├── uji_data/                        # Processed dataset & graph files
├── models/                          # Saved model checkpoints
└── thesis_ar_nav/                   # Flutter mobile application
    ├── lib/
    │   ├── main.dart                # UI, AR overlay, venue selection
    │   ├── ai_engine.dart           # TFLite inference wrapper
    │   └── navigation_controller.dart  # WiFi scan + BFS routing
    └── assets/
        ├── wifi_model.tflite        # University building model
        ├── railway_model.tflite     # Railway station model
        ├── map_graph.json           # University building graph
        └── railway_map.json         # Railway station graph
```

---

## 🛠️ Troubleshooting

**`topology_graph_building0.pkl` not found**
> Run `python download_and_prepare_uji.py` first, then `python rebuild-graph.py` if the file is still missing.

**Graph has disconnected components**
> Run `python diagnose_graph.py` to identify components, then `python rebuild-graph.py` to reconnect them using k-NN bridging.

**Flutter: TFLite model fails to load**
> Ensure `wifi_model.tflite` and `map_graph.json` exist in `thesis_ar_nav/assets/` and are correctly listed in `pubspec.yaml` under `flutter > assets`.

**Flutter: `No Routers Found!` on device**
> Grant **Location Permission** to the app. Android requires location access for WiFi scanning. Also ensure WiFi is enabled and GPS is turned on.

**`ModuleNotFoundError: train_part1_hierarchical_v2`** (during Part 2 training)
> Run both scripts from the root `THESIS/` directory, not from a subdirectory.

**ONNX export fails with shape mismatch**
> Confirm your trained model has `num_rooms = 748`. Check with:
> ```python
> import torch; ckpt = torch.load('models/hierarchical_classifier_best.pth'); print(ckpt['num_rooms'])
> ```

---

## 🧰 Dependencies

### Python
| Package | Purpose |
|---|---|
| `torch` | Model training (classifier + PPO) |
| `numpy`, `pandas` | Data processing |
| `scikit-learn` | Feature scaling, preprocessing |
| `networkx` | Graph construction & pathfinding |
| `matplotlib`, `seaborn` | Visualization |
| `onnx`, `onnxruntime` | Mobile model export |
| `tqdm` | Training progress bars |

### Flutter
| Package | Purpose |
|---|---|
| `tflite_flutter` | On-device AI inference |
| `wifi_scan` | WiFi RSSI scanning |
| `camera` | AR camera feed |
| `flutter_compass` | Device orientation for AR arrow |
| `vector_math` | 3D bearing calculations |

---

## 📜 License

This project was developed as part of an academic thesis. All dataset usage is subject to the [UJIIndoorLoc dataset terms](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).
