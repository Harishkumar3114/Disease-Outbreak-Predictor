import torch
import torch.nn as nn
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
from typing import List
from torch_geometric.nn import GraphConv
import os

# ==============================================================================
# 1. MODEL DEFINITION
# ==============================================================================
class LSTM_GNN_Optimized(nn.Module):
    def __init__(self, input_dim, num_nodes, lstm_hidden=256, gnn_hidden=128, embedding_dim=64):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True, num_layers=2, dropout=0.3)
        self.gnn = GraphConv(embedding_dim, gnn_hidden)
        self.fc_head = nn.Sequential(
            nn.Linear(lstm_hidden + gnn_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, seq_batch, node_idx_batch, edge_index):
        lstm_out, _ = self.lstm(seq_batch)
        temporal_features = lstm_out[:, -1, :]
        all_node_embeds = self.node_embedding.weight
        all_spatial_features = torch.relu(self.gnn(all_node_embeds, edge_index))
        batch_spatial_features = all_spatial_features[node_idx_batch]
        combined_features = torch.cat([temporal_features, batch_spatial_features], dim=1)
        prediction = self.fc_head(combined_features).squeeze(-1)
        return prediction

# ==============================================================================
# 2. LOAD GLOBAL ASSETS
# ==============================================================================
print("INFO: Loading all required assets...")
device = torch.device("cpu")

# Define the path to your models folder
MODELS_DIR = r"C:\Users\Deepak\OneDrive\Desktop\ObjectRegon\Deepak_AML_Project_Code\models"

try:
    # --- Load Feature List and Parameters ---
    feature_cols_path = os.path.join(MODELS_DIR, "feature_cols_no_lag.pkl")
    feature_cols = joblib.load(feature_cols_path)
    INPUT_DIM = len(feature_cols)
    NUM_NODES = 13202

    # --- Load Model State ---
    model_path = os.path.join(MODELS_DIR, "best_model_no_lag.pth")
    model = LSTM_GNN_Optimized(input_dim=INPUT_DIM, num_nodes=NUM_NODES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Load Scalers, Graph, and Location Map ---
    feature_scaler_path = os.path.join(MODELS_DIR, "feature_scaler_no_lag.pkl")
    target_scaler_path = os.path.join(MODELS_DIR, "target_scaler_no_lag.pkl")
    edge_index_path = os.path.join(MODELS_DIR, "edge_index_no_lag.pt")
    loc2idx_path = os.path.join(MODELS_DIR, "loc2idx_no_lag.pkl")

    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    edge_index = torch.load(edge_index_path).to(device)
    loc2idx = joblib.load(loc2idx_path)
    
    print(f"✅ INFO: All assets loaded successfully from '{MODELS_DIR}'.")
except FileNotFoundError as e:
    print(f"❌ FATAL ERROR: Missing asset file: {e}. The API cannot start.")
    exit()

# ==============================================================================
# 3. FASTAPI APPLICATION
# ==============================================================================
app = FastAPI(title="Disease Outbreak Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Data Models ---
class FeatureSet(BaseModel):
    features: List[float]

class SequenceInput(BaseModel):
    sequence: conlist(FeatureSet, min_length=30, max_length=30)
    location_key: str

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "API is online and ready."}

@app.post("/predict", tags=["Prediction"])
def predict(data: SequenceInput):
    try:
        # 1. Validate Input
        if len(data.sequence[0].features) != INPUT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Incorrect number of features. API expected {INPUT_DIM}, but request contained {len(data.sequence[0].features)}."
            )

        input_array = np.array([day.features for day in data.sequence])
        if not np.all(np.isfinite(input_array)):
            raise HTTPException(status_code=400, detail="Input data contains invalid numbers (NaN or infinity).")

        # 2. Preprocess
        scaled_features = feature_scaler.transform(input_array)
        seq_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        node_idx = loc2idx.get(data.location_key)
        if node_idx is None:
            raise HTTPException(status_code=404, detail=f"Location key '{data.location_key}' not found.")
        
        node_idx_tensor = torch.tensor([node_idx], dtype=torch.long).to(device)

        # 3. Predict
        with torch.no_grad():
            prediction_log_scaled = model(seq_tensor, node_idx_tensor, edge_index).item()
        
        # 4. Post-process
        prediction_log = target_scaler.inverse_transform([[prediction_log_scaled]])
        final_prediction = np.expm1(prediction_log).item()

        if not np.isfinite(final_prediction):
            raise HTTPException(status_code=500, detail="Model produced a non-finite prediction (NaN or infinity).")

        return { "predicted_new_cases": round(final_prediction, 2) }
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"ERROR: An unhandled exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")