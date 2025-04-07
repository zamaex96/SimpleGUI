import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
import matplotlib.font_manager as fm
from PIL import Image
import serial # <--- Import serial
import joblib # <--- Import joblib for model loading
import queue  # <--- Import queue for thread communication
import collections # <--- For deque (efficient rolling window)

import torch # <--- Import PyTorch
import torch.nn as nn # <--- Import base class for models

# --- Configuration ---
MODEL_PATH = r'MLassistedObjectMonitoring.pth'
SERIAL_TIMEOUT = 1
PLOT_UPDATE_INTERVAL = 0.1
MAX_PLOT_POINTS = 100
NUM_FEATURES = 5 # <--- Define number of input features
CLASS_LABELS = ['Low', 'Medium', 'Steady', 'High'] # --- MUST MATCH MODEL OUTPUT ORDER ---

# --- !!! Placeholder Model Architecture !!! ---
# --- !!! IMPORTANT: Replace this with your ACTUAL model definition !!! ---
class PlaceholderObjectMonitorNet(nn.Module):
    def __init__(self, input_size=NUM_FEATURES, num_classes=len(CLASS_LABELS)): # <--- Use NUM_FEATURES
        super().__init__()
        # Example: A simple 2-layer linear network adjusted for input size
        self.layer_1 = nn.Linear(input_size, 64) # Increased hidden size maybe
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(64, num_classes)
        print("WARNING: Using Placeholder PyTorch Model ARCHITECTURE.")
        print(f"         Expecting {input_size} input features.")
        print("         Replace PlaceholderObjectMonitorNet with your actual model class.")

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x
    
# --- !!! Function to Load PyTorch Model !!! ---
def load_pytorch_model(model_path, model_definition_class, device='cpu'):
    """Loads a PyTorch model state_dict."""
    try:
        # Instantiate the ACTUAL model architecture - Pass NUM_FEATURES
        model = model_definition_class(input_size=NUM_FEATURES, num_classes=len(CLASS_LABELS)) # <--- Use NUM_FEATURES

        state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)

        # Handle potential 'module.' prefix if saved with DataParallel
        if isinstance(state_dict, nn.DataParallel):
             state_dict = state_dict.module.state_dict()
        elif isinstance(state_dict, dict) and all(key.startswith('module.') for key in state_dict):
             state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded PyTorch model from {model_path}")
        return model.to(device)

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        print("Ensure the PlaceholderObjectMonitorNet class is replaced with your actual model's Python class definition.")
        return None