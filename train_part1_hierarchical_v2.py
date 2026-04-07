import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from config import PATHS, TRAINING_CONFIG

DATA_DIR = PATHS['data_processed']
OUTPUT_DIR = PATHS['models']
DEVICE = torch.device(TRAINING_CONFIG['device'])

os.makedirs(OUTPUT_DIR, exist_ok=True)

class UJIDataset(Dataset):
    def __init__(self, csv_path, location_info, augment=False):
        self.df = pd.read_csv(csv_path)
        self.location_info = {loc['location_id']: loc for loc in location_info}
        self.augment = augment
        self.csi_cols = [f'CSI_DATA{i}' for i in range(1, 129)]
        self.X = self.df[self.csi_cols].values.astype(np.float32)
        self.buildings = []
        self.floors = []
        self.rooms = []
        for label in self.df['label'].values:
            loc = self.location_info[label]
            self.buildings.append(loc['building'])
            self.floors.append(loc['floor'])
            self.rooms.append(label)
        self.buildings = np.array(self.buildings)
        self.floors = np.array(self.floors)
        self.rooms = np.array(self.rooms)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            noise = np.random.normal(0, 0.1, x.shape)
            x = x + noise
        return (
            torch.FloatTensor(x),
            torch.LongTensor([self.buildings[idx], self.floors[idx], self.rooms[idx]])
        )

# model
class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim=128, num_buildings=3, num_floors=5, num_rooms=500):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.building_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_buildings)
        )
        self.floor_head = nn.Sequential(
            nn.Linear(128 + num_buildings, 64), nn.ReLU(), nn.Linear(64, num_floors)
        )
        self.room_head = nn.Sequential(
            nn.Linear(128 + num_buildings + num_floors, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_rooms)
        )
    
    def forward(self, x, return_intermediates=False):
        features = self.features(x)
        building_logits = self.building_head(features)
        building_probs = torch.softmax(building_logits, dim=1)
        
        floor_input = torch.cat([features, building_probs], dim=1)
        floor_logits = self.floor_head(floor_input)
        floor_probs = torch.softmax(floor_logits, dim=1)
        
        room_input = torch.cat([features, building_probs, floor_probs], dim=1)
        room_logits = self.room_head(room_input)
        
        if return_intermediates:
            return building_logits, floor_logits, room_logits
        return room_logits

def calculate_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def train_enhanced():
    print("\n" + "="*80)
    print("="*80)
    
    location_info_path = os.path.join(DATA_DIR, "location_info.pkl")
    if not os.path.exists(location_info_path):
        raise FileNotFoundError(f"Could not find {location_info_path}. run the setup script")
        
    with open(location_info_path, 'rb') as f:
        location_info = pickle.load(f)
    
    train_path = os.path.join(DATA_DIR, "uji_train.csv")
    test_path = os.path.join(DATA_DIR, "uji_test.csv")
    train_dataset = UJIDataset(train_path, location_info, augment=True)
    test_dataset = UJIDataset(test_path, location_info, augment=False)
    
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    
    all_buildings = set(loc['building'] for loc in location_info)
    all_floors = set(loc['floor'] for loc in location_info)
    all_room_ids = set(loc['location_id'] for loc in location_info)
    
    dims = (max(all_buildings)+1, max(all_floors)+1, max(all_room_ids)+1)
    print(f"Dimensions: Buildings={dims[0]}, Floors={dims[1]}, Rooms={dims[2]}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    model = HierarchicalClassifier(
        input_dim=128,
        num_buildings=dims[0],
        num_floors=dims[1],
        num_rooms=dims[2]
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc3 = 0.0 
    
    for epoch in range(50): 
        model.train()
        avg_loss = 0
        
        for x, y in train_loader:
            x, y_b, y_f, y_r = x.to(DEVICE), y[:,0].to(DEVICE), y[:,1].to(DEVICE), y[:,2].to(DEVICE)
            optimizer.zero_grad()
            out_b, out_f, out_r = model(x, return_intermediates=True)
            loss = criterion(out_b, y_b)*0.2 + criterion(out_f, y_f)*0.3 + criterion(out_r, y_r)*0.5