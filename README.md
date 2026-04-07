# Indoor Navigation System using Wireless CSI & Reinforcement Learning

> **Author:** Sai Sleghana Bala  
> This is my M.tech thesis project implementing a two-stage AI pipeline for GPS-free indoor navigation, combining a WiFi fingerprint classifier with a Proximal Policy Optimization (PPO) reinforcement learning agent - deployed as a real-time AR mobile app.

---

## Overview

Navigating indoor environments such as offices, hospitals, and shopping complexes poses challenges due to the lack of GPS signals and the limitations of physical maps. Traditional GPS-based systems are ineffective indoors, while static maps need to be established at multiple locations and referenced numerous times by the user in order to reach the destination. This project presents an indoor navigation system that uses WiFi Channel State Information (CSI) and reinforcement learning to provide accurate, real-time guidance without relying on GPS.
The system generates an indoor map by collecting WiFi CSI data from available networks within the environment. WiFi CSI captures the signal characteristics that vary with location, enabling the creation of a unique spatial fingerprint for the indoor space. This network-based map enables precise user localization by matching live CSI readings against the mapped data, thus determining the exact location of the user in the indoor space.
For navigation, users input their destination, and the system employs Proximal Policy Optimization (PPO): a reinforcement learning algorithm, to compute an optimal path from the user location to the entered destination.
Finally, navigation instructions are displayed through an AR interface, overlaying directional cues onto the user’s real-world view via a mobile device. This immersive approach reduces reliance on physical maps and simplifies wayfinding by providing clear, intuitive guidance. This will also help geriatric and young users who might find existing methods intricate to understand.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                         │
│                                                                  │
│  UJIIndoorLoc Dataset (UCI)                                      │
│         │                                                        │
│         ▼                                                        │
│  download_and_prepare_uji.py  ──►  uji_data/                     │
│         │                          ├── uji_train.csv             │
│         │                          ├── uji_test.csv              │
│         │                          ├── location_info_*.pkl       │
│         │                          └── topology_graph_*.pkl      │
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
│  converter-flutter.py  →  wifi_tracker.onnx                      │
│  test.py               →  map_graph.json                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                   FLUTTER MOBILE APP (thesis_ar_nav)             │
│                                                                  │
│  WiFi Scan (128 RSSI features)                                   │
│         │                                                        │
│         ▼                                                        │
│  ai_engine.dart  ──►  TFLite Interpreter  ──►  Node ID           │
│                            │                                     │
│                            ▼                                     │
│  navigation_controller.dart  ──►  BFS on map_graph.json          │
│                            │                                     │
│                            ▼                                     │
│  main.dart  ──►  AR Overlay (Camera + Compass Arrow)             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
---

## Dataset — UJIIndoorLoc

- **Source:** [UCI ML Repository — UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
- **Coverage:** 3 university buildings, 5 floors, 748 unique location nodes
- **Features:** 520 WAP (WiFi Access Point) RSSI readings per sample, compressed to 128 CSI features in this system
- **Size:** ~19,000 training samples, ~1,100 validation samples

The `download_and_prepare_uji.py` script handles everything — download, extraction, format conversion, graph topology construction, and train/test splitting.

---

## RL parameters:

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


## Execution Steps:

> Prereq: Python 3.9+, Flutter SDK 3.x (`sdk: ^3.11.4`), Mobile phone with with WiFi scanning support

### Required Dependencies: 

#### Python
| Package | Purpose |
|---|---|
| `torch` | Model training (classifier + PPO) |
| `numpy`, `pandas` | Data processing |
| `scikit-learn` | Feature scaling, preprocessing |
| `networkx` | Graph construction & pathfinding |
| `matplotlib`, `seaborn` | Visualization |
| `onnx`, `onnxruntime` | Mobile model export |
| `tqdm` | Training progress bars |

#### Flutter
| Package | Purpose |
|---|---|
| `tflite_flutter` | On-device AI inference |
| `wifi_scan` | WiFi RSSI scanning |
| `camera` | AR camera feed |
| `flutter_compass` | Device orientation for AR arrow |
| `vector_math` | 3D bearing calculations |

---

### Scripts:

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

### Part 1 — Python: Model training:

#### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2 — Download and prep the data

```bash
python download_and_prepare_uji.py
```

Expected output directory: `uji_data/` containing CSVs and `.pkl` graph files.

#### Step 3 — Rebuild / Fix the Navigation Graph (Optional)

If the graph has disconnected components or bad edges, regenerate it:

```bash
# Rebuild from scratch using k-NN 
python rebuild-graph.py

# Or prune long/invalid edges from existing graph
python prune_graph.py
```

> Run `python diagnose_graph.py` first to check graph health.

#### Step 4 — Locator: Train the WiFi Location Classifier (Part 1)

Trains the hierarchical deep neural network on WiFi fingerprints.

```bash
python train_part1_hierarchical_v2.py
```

Output: `models/hierarchical_classifier_best.pth`, `models/scaler.pkl`

#### Step 5 — Navigator: Train the RL Agent (Part 2)

Trains the PPO agent on the topology graph using the classifier's feature embeddings.

```bash
python train_part2_hierarchical.py
```

Output: `models/ppo_agent.pth`, `models/training_history.png`

#### Step 6 — Visualize a Navigation Trajectory (Optional) --> generates a navigation map (visual)

```bash
python trajectory.py
```

Output: `trajectory_plot.png` — a figure showing ground truth path vs. AI-predicted locations.

---

### Part 2 — Export Models for Mobile

#### Step 7 — Export Classifier to ONNX 
Note: I used google collab for this, since my numpy version(2.4.4) was higher than the one supported for this function (1.21-1.26)

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

> **Note:** Pre-built `.tflite` models and `map_graph.json` are already included in `thesis_ar_nav/assets/`. You only need this step if models were retrained.

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
