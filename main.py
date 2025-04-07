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
    
# --- Load the Model ---
# Replace PlaceholderObjectMonitorNet with the actual class name of your network
model = load_pytorch_model(MODEL_PATH, PlaceholderObjectMonitorNet)

# --- Fallback Predictor (adjust to accept 5 args, but maybe still use Acc mainly) ---
class FallbackPredictor:
    def __init__(self):
         print("WARNING: PyTorch model failed to load. Using fallback predictor.")

    def predict(self, acc_val, sp_val, roll_val, pitch_val, yaw_val): # <--- Accept all 5
        # Simple logic, still based on acceleration for this placeholder
        if acc_val < 0.1: return CLASS_LABELS[0]
        elif acc_val < 0.5: return CLASS_LABELS[1]
        elif acc_val < 1.5: return CLASS_LABELS[2]
        else: return CLASS_LABELS[3]

if model is None:
    predictor = FallbackPredictor()
else:
    # Predictor function now takes 5 arguments
    def predictor_func(acc_val, sp_val, roll_val, pitch_val, yaw_val): # <--- Accept all 5
        model.eval()
        with torch.no_grad():
            # 1. Convert inputs to PyTorch Tensor (shape [1, 5])
            input_data = [acc_val, sp_val, roll_val, pitch_val, yaw_val]
            input_tensor = torch.tensor([input_data], dtype=torch.float32) # <--- Shape [1, 5]

            # 2. Get model output (logits)
            outputs = model(input_tensor)

            # 3. Get predicted class index
            predicted_index = torch.argmax(outputs, dim=1).item()

            # 4. Map index to label
            if 0 <= predicted_index < len(CLASS_LABELS):
                return CLASS_LABELS[predicted_index]
            else:
                print(f"Warning: Predicted index {predicted_index} out of range.")
                return "Unknown"
    predictor = predictor_func # Use the function


# --- Thread Control ---
stop_event = threading.Event()
data_queue = queue.Queue()

# --- GUI Definition (make_window) ---
def make_window(theme):
    # ... (make_window code remains exactly the same as before) ...
    sg.theme(theme)
    menu_def = [['&Application', ['E&xit']],
                ['&Help', ['&About']]]
    right_click_menu_def = [[], ['Edit Me', 'Versions', 'Exit']]
    graph_right_click_menu_def = [[], ['Erase', 'Draw Line', 'Draw', ['Circle', 'Rectangle', 'Image'], 'Exit']]

    # Table Data (can be updated later if needed)
    data = [["Acc", 0.0], ["Sp", 0.0]]
    headings = ["Name", "Score"]

    input_layout = [
        [sg.Text('Port', size=(5, 1)), sg.Combo(values=get_serial_ports(), default_value=get_serial_ports()[0] if get_serial_ports() else '', readonly=False, k='-COMBO-'),
         sg.Text('Baud', size=(5,1)), sg.Input('9600', size=(8,1), k='-BAUD-'), # Added Baud Rate
         sg.Button('Connect', k='-CONNECT-', button_color=('white', 'green')), # Added Connect Button
         sg.Button('Disconnect', k='-DISCONNECT-', button_color=('white', 'red'), disabled=True) # Added Disconnect
        ],
        # Only displaying Acc and Sp for now
        [sg.Checkbox('Acceleration', default=True, k='-ACC-', disabled=True), # Display only
         sg.InputText('0.0', size=(7, 1), justification='center', k='-TEXT2-', readonly=True, tooltip='Live Acceleration Reading'),
         sg.Checkbox('Speed', default=True, k='-SP-', disabled=True), # Display only
         sg.InputText('0.0', size=(7, 1), justification='center', k='-TEXT3-', readonly=True, tooltip='Live Speed Reading')],
        # You could add more Text/Input fields here for Roll, Pitch, Yaw if desired
        # [sg.Text('Roll:', size=(5,1)), sg.InputText('0.0', size=(7, 1), k='-ROLL-', readonly=True), ...]
        [sg.Text('Predicted State:', size=(12, 1)),
         sg.Text('N/A', size=(15, 1), font=("Helvetica", 10, 'bold'), k='-STATE-', relief=sg.RELIEF_SUNKEN)], # Display for ML prediction
        [sg.HorizontalSeparator()],
        [sg.Text('Live Data Plot', font=("Helvetica", 10, 'bold'))],
        [sg.Canvas(size=(300, 200), key='-ANIMATED_PLOT-')], # Adjusted size
        [sg.HorizontalSeparator()],
        [sg.Text('Acceleration State Distribution', font=("Helvetica", 10, 'bold'))],
        [sg.Canvas(size=(300, 200), key='-CANVAS-')], # Adjusted size
        [sg.Push(), sg.Text('© AIoT Laboratory', size=(15, 1), font=("Helvetica", 8))]
    ]