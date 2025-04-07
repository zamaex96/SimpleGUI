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

    current_layout = [[sg.T('Anything that you would use for aesthetics is in this tab!')],
                      [sg.ProgressBar(100, orientation='h', size=(20, 20), key='-PROGRESS BAR-'),
                       sg.Button('Test Progress bar')]]

    note_layout = [[sg.Text("Output and errors will display here!")],
                   [sg.Multiline(size=(60, 15), font='Courier 8', expand_x=True, expand_y=True, write_only=True,
                                 reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True, autoscroll=True,
                                 auto_refresh=True, k='-MLOG-')]] # Added key

    logging_layout = [[sg.Text("Anything you would use to graph will display here!")],
                      [sg.Graph((200, 200), (0, 0), (200, 200), background_color="black", key='-GRAPH-',
                                enable_events=True,
                                right_click_menu=graph_right_click_menu_def)],
                      [sg.T('Click anywhere on graph to draw a circle')],
                      [sg.Table(values=data, headings=headings, max_col_width=25,
                                background_color='lightgrey', text_color='black', # Adjusted colors
                                auto_size_columns=True,
                                display_row_numbers=True,
                                justification='right',
                                num_rows=2,
                                alternating_row_color='white', # Adjusted colors
                                key='-TABLE-',
                                row_height=25)]]

    popup_layout = [[sg.Text("Popup Testing")],
                    [sg.Button("Open Folder")],
                    [sg.Button("Open File")]]

    theme_layout = [[sg.Text("See how elements look under different themes!")],
                    [sg.Listbox(values=sg.theme_list(),
                                size=(20, 12),
                                key='-THEME LISTBOX-',
                                enable_events=True)],
                    [sg.Button("Set Theme")]]
    
    # --- Load and Resize Logos ---
    try:
        logo_path = 'logo.png'
        logo_image = Image.open(logo_path)
        resized_logo = logo_image.resize((70, 50))
        resized_logo_path = 'resized_logo.png'
        resized_logo.save(resized_logo_path)

        logo_path2 = 'logo2.png'
        logo_image2 = Image.open(logo_path2)
        resized_logo2 = logo_image2.resize((80, 60))
        resized_logo_path2 = 'resized_logo2.png'
        resized_logo2.save(resized_logo_path2)
        img1 = sg.Image(filename=resized_logo_path2)
        img2 = sg.Image(filename=resized_logo_path)
    except FileNotFoundError:
        print("Warning: Logo files not found. Skipping images.")
        img1 = sg.Text('') # Placeholder if images fail
        img2 = sg.Text('')
    except Exception as e:
        print(f"Error loading images: {e}")
        img1 = sg.Text('')
        img2 = sg.Text('')


    layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
              [img1,
               sg.Text('Aqua-Aware', size=(25, 1), justification='center', font=("Calibri", 16, 'underline'),
                       relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True),
               img2
               ]
              ]
    layout += [[sg.TabGroup([[sg.Tab('Main', input_layout),
                              #sg.Tab('Current Readings', current_layout), # Maybe remove? Main has live data now
                              sg.Tab('Logging', logging_layout),
                              sg.Tab('Import', popup_layout),
                              sg.Tab('Theming', theme_layout),
                              sg.Tab('Output Log', note_layout)]], key='-TAB GROUP-', expand_x=True, expand_y=True)]] # Renamed Note tab
    layout[-1].append(sg.Sizegrip())

    # Make sure finalize=True is set
    window = sg.Window('AIoT Laboratory - Aqua-Aware', layout, right_click_menu=right_click_menu_def,
                       right_click_menu_tearoff=True, grab_anywhere=True, resizable=True, margins=(0, 0),
                       use_custom_titlebar=True, finalize=True, keep_on_top=False) # Keep_on_top=False usually better
    window.set_min_size(window.size)
    return window

# --- Plotting Functions (draw_figure, setup_live_plot, etc.) ---
# Global references for plot figures and axes to update them
fig_agg_live = None
fig_live = None
ax_live = None
acc_line = None
sp_line = None
# Use deque for efficient rolling data storage
time_data = collections.deque(maxlen=MAX_PLOT_POINTS)
acc_data = collections.deque(maxlen=MAX_PLOT_POINTS)
sp_data = collections.deque(maxlen=MAX_PLOT_POINTS)

fig_agg_pie = None
fig_pie = None
ax_pie = None
state_counts = collections.Counter() # To store counts for pie chart