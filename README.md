# AI-Powered Spatiotemporal Disease Outbreak Predictor

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

> **A production-ready deep learning system that forecasts daily disease spread across 13,000+ locations using a hybrid LSTM-GNN architecture, deployed as a microservice on Kubernetes with GPU acceleration.**

---

## Project Overview

Predicting disease outbreaks requires understanding two critical dimensions: **Time** (historical trends) and **Space** (geographic spread). Traditional models often treat regions in isolation, missing the vital context of how outbreaks spill over from neighboring areas.

This project solves that problem by building a **Hybrid Spatiotemporal Model**:
1.  **LSTM (Long Short-Term Memory):** Captures temporal dependencies from 30-day historical sequences.
2.  **GNN (Graph Neural Network):** Captures spatial dependencies using a K-Nearest Neighbors graph to aggregate infection risks from neighboring regions.

The result is a robust forecasting engine deployed via a modern MLOps pipeline, allowing users to simulate scenarios in real-time.

---

## Key Features

* **Hybrid Deep Learning:** Custom PyTorch architecture fusing LSTM temporal embeddings with Graph Convolutional Network (GCN) spatial embeddings.
* **High-Performance Forecasting:** Achieved an **$R^2$ score of 0.882** and **MAE of 10.28 cases**, significantly outperforming baseline regression models.
* **Robust Data Engineering:** Pipeline integrates 7+ multi-modal datasets (epidemiology, mobility, weather, policy), utilizing **Parquet** for storage optimization and **Log-Transformation** to handle extreme data skew.
* **Scalable Deployment:** Containerized with **Docker** (Multi-stage builds) and orchestrated on **Kubernetes** with **NVIDIA GPU passthrough** for sub-100ms inference latency.
* **Interactive Dashboard:** A user-friendly **Streamlit** UI connected to a **FastAPI** backend, enabling real-time "what-if" analysis for policy planning.

---

## System Architecture

The system follows a microservice architecture pattern:

```mermaid
graph LR
    User[User / Epidemiologist] -- Uses --> UI[Streamlit Dashboard]
    UI -- JSON Request --> API[FastAPI Backend]
    subgraph Kubernetes Cluster
        API -- GPU Inference --> Model[LSTM-GNN Model]
    end
    Model -- Prediction --> API
    API -- Response --> UI

