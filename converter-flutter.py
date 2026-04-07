import torch
import torch.onnx
from train_part1_hierarchical_v2 import HierarchicalClassifier

checkpoint_path = "models/hierarchical_classifier_best.pth"
ckpt = torch.load(checkpoint_path, map_location='cpu')
num_buildings = ckpt['num_buildings'] 
num_floors = ckpt['num_floors']       
num_rooms = ckpt['num_rooms']         
print(f"Model Config: {num_buildings} Buildings, {num_floors} Floors, {num_rooms} Rooms/Locations")

model = HierarchicalClassifier(128, num_buildings, num_floors, num_rooms)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
dummy_input = torch.randn(1, 128)

torch.onnx.export(
    model, 
    dummy_input, 
    "wifi_tracker.onnx", 
    export_params=True,
    opset_version=11,  
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output']
)
