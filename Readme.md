# Smart Building Surveillance System using Computer Vision

## Overview

The **Smart Building Surveillance System** is an AI-powered multi-camera surveillance platform that tracks individuals across an entire building using Computer Vision and Deep Learning. Unlike traditional CCTV systems where each camera operates independently, this project connects all cameras into a unified network represented as a graph. This allows the system to continuously identify, track, and predict the movement of individuals as they move from one camera's field of view to another.

The system combines **YOLO-based person detection**, **face recognition**, **person re-identification (ReID)**, **multi-object tracking**, and **graph-based trajectory prediction** to maintain a persistent identity for every person throughout the building.

This project is designed for applications such as:

* Smart Buildings
* Airports
* Shopping Malls
* Universities
* Hospitals
* Corporate Offices
* Military Facilities
* Smart Cities

---

# Features

* Real-time multi-camera surveillance
* Person detection using YOLO
* Multi-object tracking using ByteTrack
* Face detection inside detected person regions
* Face recognition using ArcFace
* Person Re-Identification (ReID) across cameras
* Global identity assignment across all cameras
* Graph-based camera network
* Real-time movement visualization
* Person trajectory storage
* Future camera prediction
* Live dashboard for monitoring
* Event logging and analytics

---

# Project Architecture

```text
                        Building

           Camera 1 ------- Camera 2
               |                |
               |                |
           Camera 3 ------- Camera 4
               |                |
               |                |
           Camera 5 ------- Camera 6


            Camera Graph Representation

              C1 ----- C2
              |        |
              |        |
              C3 ----- C4
              |        |
              C5 ----- C6
```

Each camera acts as a **graph node**, while walkable paths between cameras are represented as **graph edges**.

---

# System Pipeline

```text
Video Stream
      │
      ▼
YOLO Person Detection
      │
      ▼
ByteTrack Multi-Object Tracking
      │
      ▼
Face Detection
      │
      ▼
ArcFace Face Recognition
      │
      ▼
Person Re-Identification
      │
      ▼
Global Identity Assignment
      │
      ▼
Camera Graph Network
      │
      ▼
Movement Tracking
      │
      ▼
Trajectory Prediction
      │
      ▼
Dashboard Visualization
```

---

# Workflow

## 1. Multi-Camera Video Collection

Video streams are collected simultaneously from multiple CCTV cameras installed throughout the building.

Each camera continuously sends frames to the central processing server.

---

## 2. Person Detection

Each frame is processed using **YOLO** to detect people.

Output:

* Bounding Box
* Confidence Score
* Class Label

Example:

```text
Person

Bounding Box:
(x1, y1, x2, y2)

Confidence:
0.97
```

---

## 3. Multi-Object Tracking

YOLO only detects objects in individual frames.

To maintain identity across consecutive frames, **ByteTrack** assigns a unique tracking ID.

Example:

```text
Frame 1

Person
ID = 12

↓

Frame 2

Person
ID = 12

↓

Frame 3

Person
ID = 12
```

---

## 4. Face Detection

For every detected person, another model searches for a visible face.

Possible detectors:

* RetinaFace
* SCRFD
* MTCNN

Output:

```text
Person

↓

Face Bounding Box
```

---

## 5. Face Recognition

The detected face is processed using **ArcFace**.

ArcFace converts every face into a **512-dimensional embedding vector**.

Example:

```text
Face

↓

ArcFace

↓

[0.21, -0.43, 0.18, ..., 512 values]
```

These embeddings are used to compare identities across different cameras.

---

## 6. Person Re-Identification

Faces are not always visible.

Therefore, the system also extracts body features using Person ReID models.

Possible models:

* FastReID
* OSNet
* StrongSORT
* TransReID

Features learned include:

* Clothing
* Shoes
* Backpack
* Height
* Body Shape
* Walking Style

This produces another embedding representing the person's appearance.

---

## 7. Identity Fusion

The final identity is obtained by combining multiple sources of information:

* Face Embedding
* Body Embedding
* Tracking History
* Clothing Features
* Color Histogram
* Camera Location
* Timestamp
* Walking Direction

These features are fused to compute a similarity score.

```text
Similarity Score

0.96

↓

Same Person
```

---

## 8. Global Identity Assignment

Instead of assigning IDs independently in every camera,

```text
Camera 1

ID = 5

Camera 3

ID = 2

Camera 5

ID = 18
```

The system creates one global identity.

```text
Global Person

ID = 101
```

Now every camera refers to the same person using the same global ID.

---

## 9. Camera Graph

The building layout is modeled as a graph.

Example:

```text
Entrance

↓

Lobby

↓

Hallway

↓

Elevator

↓

Floor 2
```

If a person disappears from one camera,

the graph limits the search to physically connected neighboring cameras instead of searching the entire building.

---

## 10. Movement Database

Every observation is stored.

Example:

```text
Person ID

Camera

Timestamp

Position

Confidence
```

Sample:

```text
101

Camera 1

09:00:12

(320,180)

0.98
```

---

## 11. Trajectory Generation

The system reconstructs the person's path.

Example:

```text
09:00

Entrance

↓

09:01

Lobby

↓

09:02

Hallway

↓

09:04

Elevator

↓

09:06

Floor 2
```

---

## 12. Movement Prediction

Using previous movement history, the system predicts the next likely camera.

Possible prediction models:

* Markov Chains
* LSTM Networks
* Transformers
* Graph Neural Networks (Recommended)

Output:

```text
Current Camera

↓

Camera 4

↓

Prediction

Camera 5

Probability = 92%
```

---

# Technology Stack

| Component            | Technology               |
| -------------------- | ------------------------ |
| Programming Language | Python                   |
| Person Detection     | YOLOv11 / YOLOv8         |
| Object Tracking      | ByteTrack                |
| Face Detection       | RetinaFace / SCRFD       |
| Face Recognition     | ArcFace                  |
| Person ReID          | FastReID / OSNet         |
| Graph Processing     | NetworkX                 |
| Database             | PostgreSQL / MongoDB     |
| Backend              | FastAPI                  |
| Dashboard            | Streamlit / React        |
| Prediction           | LSTM / Transformer / GNN |
| Deep Learning        | PyTorch                  |

---

# Folder Structure

```text
smart-surveillance-system/

│

├── cameras/
│   ├── stream_reader.py
│   ├── camera_manager.py
│   └── graph.py
│
├── detection/
│   ├── yolo_detector.py
│   ├── tracker.py
│   ├── face_detector.py
│   ├── face_recognition.py
│   └── reid.py
│
├── prediction/
│   ├── markov.py
│   ├── lstm_predictor.py
│   └── gnn_predictor.py
│
├── dashboard/
│   ├── app.py
│   ├── map.py
│   └── analytics.py
│
├── database/
│   ├── database.py
│   ├── models.py
│   └── queries.py
│
├── models/
│
├── data/
│
├── utils/
│
├── requirements.txt
│
└── main.py
```

---

# Future Improvements

* Suspicious activity detection
* Abandoned object detection
* Crowd density estimation
* Violence detection
* Fall detection
* Weapon detection
* Restricted area intrusion detection
* Missing person search
* Real-time alert system
* Mobile monitoring application
* Heatmap generation
* Anonymous privacy mode
* Edge-device deployment
* Federated learning for privacy-preserving model updates

---

# Applications

* Smart City Surveillance
* Airport Security
* Railway Stations
* Shopping Malls
* University Campuses
* Hospitals
* Warehouses
* Corporate Buildings
* Residential Communities
* Government Buildings

---

# Research Contribution

This project integrates multiple state-of-the-art Computer Vision techniques into a unified surveillance framework:

* Object Detection
* Multi-Object Tracking
* Face Recognition
* Person Re-Identification
* Graph-Based Camera Modeling
* Trajectory Prediction

By leveraging the building's camera topology and combining visual identity cues with graph-based reasoning, the system can maintain persistent identities across non-overlapping cameras and accurately predict future movements. The architecture is modular, scalable, and suitable for extension with Graph Neural Networks or advanced spatiotemporal models, making it a strong foundation for research in intelligent surveillance and smart infrastructure.

---

# License

This project is intended for **educational and research purposes**. Any deployment in real-world environments should comply with applicable privacy, surveillance, and data protection laws and should be designed with appropriate safeguards for responsible use.
