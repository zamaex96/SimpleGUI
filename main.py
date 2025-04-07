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

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def setup_live_plot(window):
    global fig_agg_live, fig_live, ax_live, acc_line, sp_line
    canvas_elem = window['-ANIMATED_PLOT-']
    canvas = canvas_elem.TKCanvas
    plt.style.use('seaborn-v0_8-darkgrid') # Example style
    plt.rcParams.update({'font.size': 7}) # Smaller font for plot

    fig_live, ax_live = plt.subplots(figsize=(4, 2.5)) # Adjusted size
    fig_live.patch.set_alpha(0) # Make figure background transparent if needed
    ax_live.set_xlabel('Time (samples)', fontsize=7, fontweight='bold')
    ax_live.set_ylabel('Value', fontsize=7, fontweight='bold')
    ax_live.tick_params(axis='both', which='major', labelsize=6)

    # Initialize empty plots
    acc_line, = ax_live.plot([], [], 'r-', label='Acc (g)', linewidth=1.5)
    sp_line, = ax_live.plot([], [], 'b-', label='Speed (m/s)', linewidth=1.5)
    ax_live.legend(loc='upper left', fontsize=6)
    ax_live.grid(True, linestyle='--', alpha=0.6)
    fig_live.tight_layout(pad=0.5) # Adjust padding
    fig_agg_live = draw_figure(canvas, fig_live)

def update_live_plot_data():
    """Updates the data of the existing live plot lines."""
    if not ax_live or not fig_agg_live: return # Ensure plot is initialized

    current_time = list(time_data)
    current_acc = list(acc_data)
    current_sp = list(sp_data)

    acc_line.set_data(current_time, current_acc)
    sp_line.set_data(current_time, current_sp)

    # Adjust plot limits dynamically
    ax_live.relim()
    ax_live.autoscale_view(True,True,True)

    # Redraw the canvas
    try:
        fig_agg_live.draw()
    except Exception as e:
        print(f"Error drawing live plot: {e}")

def setup_pie_chart(window):
    global fig_pie, ax_pie, fig_agg_pie
    canvas_elem = window['-CANVAS-']
    canvas = canvas_elem.TKCanvas
    plt.style.use('seaborn-v0_8-pastel') # Different style for pie?
    fig_pie, ax_pie = plt.subplots(figsize=(3.5, 2.5)) # Adjusted size
    fig_pie.patch.set_alpha(0)
    ax_pie.set_title('Acceleration State', fontsize=8, fontweight='bold')
    fig_pie.tight_layout(pad=0.5)
    fig_agg_pie = draw_figure(canvas, fig_pie)
    update_pie_chart_display() # Initial empty draw

def update_pie_chart_display():
    """Redraws the pie chart based on current state_counts."""
    global ax_pie, fig_agg_pie, state_counts
    if not ax_pie or not fig_agg_pie: return

    ax_pie.clear() # Clear previous pie
    ax_pie.set_title('Predicted State Distribution', fontsize=8, fontweight='bold') # Title more general

    if not state_counts:
        ax_pie.text(0.5, 0.5, 'No data yet', horizontalalignment='center', verticalalignment='center', transform=ax_pie.transAxes)
    else:
        # Use CLASS_LABELS for ordering and ensuring all states can be shown
        labels = CLASS_LABELS
        sizes = [state_counts.get(label, 0) for label in labels] # Get counts, default to 0
        # Filter out states with 0 counts to avoid cluttering the pie
        labels_with_data = [l for l, s in zip(labels, sizes) if s > 0]
        sizes_with_data = [s for s in sizes if s > 0]

        if not sizes_with_data: # Handle case where counts exist but are all zero after filtering
             ax_pie.text(0.5, 0.5, 'No data yet', horizontalalignment='center', verticalalignment='center', transform=ax_pie.transAxes)
        else:
            total = sum(sizes_with_data)
            percentages = [(s / total) * 100 for s in sizes_with_data]

            # Define consistent colors if possible, map to labels
            color_map = {'Low': 'blue', 'Medium': 'green', 'Steady': '#99ff99', 'High': 'red', 'N/A': 'grey', 'Unknown': 'orange', 'Error':'black'}
            colors = [color_map.get(label, 'grey') for label in labels_with_data]

            explode = tuple([0.05] * len(labels_with_data)) # Slight separation for all shown slices

            wedges, texts, autotexts = ax_pie.pie(percentages, explode=explode, labels=labels_with_data, colors=colors,
                                                  autopct='%1.1f%%', startangle=90, shadow=False,
                                                  textprops={'fontsize': 7, 'fontweight': 'bold'},
                                                  pctdistance=0.85) # Adjust pct distance

    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    try:
        fig_agg_pie.draw()
    except Exception as e:
        print(f"Error drawing pie chart: {e}")

# --- Serial Communication and Processing Thread ---
def serial_reader(window: 'sg.Window', port: str, baud: int, predictor_obj, stop_flag: threading.Event, data_q: queue.Queue):
    """
    Reads data from serial port (expects 5 features), performs ML prediction,
    and updates GUI via events/queue.
    """
    print(f"Serial thread started for {port} at {baud} baud.")
    ser = None
    last_state = None
    sample_count = 0
    expected_keys = {'Acc', 'Sp', 'Roll', 'Pitch', 'Yaw'} # Set of expected keys

    try:
        ser = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        print(f"Serial port {port} opened successfully.")
        window.write_event_value('-SERIAL_CONNECTED-', True)

        while not stop_flag.is_set():
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue

                    # --- Parsing Logic for 5 features ---
                    # Example format: "Acc:0.2,Sp:3.1,Roll:1.5,Pitch:-0.8,Yaw:175.2"
                    parsed_data = {}
                    parts = line.split(',')
                    valid_line = True
                    for part in parts:
                        if ':' in part:
                            key, value_str = part.split(':', 1)
                            key = key.strip()
                            try:
                                # Store all potential values as floats
                                parsed_data[key] = float(value_str)
                            except ValueError:
                                # print(f"Warning: Could not parse value for {key} in '{part}'")
                                valid_line = False # Mark line invalid if any part fails
                                break # No need to parse further on this line
                        else:
                            # print(f"Warning: Invalid format part '{part}' in line '{line}'")
                            valid_line = False
                            break

                    # --- Check if all expected keys were found and valid ---
                    if valid_line and expected_keys.issubset(parsed_data.keys()):
                        acc_val = parsed_data['Acc']
                        sp_val = parsed_data['Sp']
                        roll_val = parsed_data['Roll']
                        pitch_val = parsed_data['Pitch']
                        yaw_val = parsed_data['Yaw']

                        # --- ML Prediction ---
                        try:
                            # Pass all 5 values to the predictor
                            predicted_state = predictor_obj(acc_val, sp_val, roll_val, pitch_val, yaw_val) # <--- Pass all 5
                        except Exception as pred_e:
                            print(f"Error during prediction: {pred_e}")
                            predicted_state = "Error"

                        # --- Update State Counts ---
                        valid_states_for_pie = CLASS_LABELS + ["Error", "Unknown"]
                        if predicted_state in valid_states_for_pie:
                            state_counts[predicted_state] += 1
                            if predicted_state != last_state:
                                window.write_event_value('-UPDATE_PIE-', True)
                                last_state = predicted_state

                        # --- Send full data to main thread ---
                        data_payload = {
                            'acc': acc_val, 'sp': sp_val, 'state': predicted_state,
                            'roll': roll_val, 'pitch': pitch_val, 'yaw': yaw_val # Include others
                        }
                        window.write_event_value('-SERIAL_DATA-', data_payload) # <--- Send full payload

                        # --- Put data onto queue for live plot (still just Acc/Sp for plot) ---
                        sample_count += 1
                        data_q.put({'time': sample_count, 'acc': acc_val, 'sp': sp_val})

                    elif valid_line and not expected_keys.issubset(parsed_data.keys()):
                        # print(f"Warning: Missing keys in line: '{line}'. Found: {parsed_data.keys()}")
                        pass # Ignore line with missing keys
                    else:
                        # print(f"Warning: Invalid line format or parse error: '{line}'")
                        pass # Ignore lines with parsing errors

                except serial.SerialException as serial_e:
                    print(f"Serial error: {serial_e}")
                    stop_flag.set()
                except UnicodeDecodeError as decode_e:
                     print(f"Serial decode error: {decode_e}. Ignoring byte.")
                except Exception as read_e:
                    print(f"Error reading/processing serial line: {read_e}")
            else:
                time.sleep(0.01)

    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        window.write_event_value('-SERIAL_ERROR-', str(e))
    except Exception as e:
        print(f"Unexpected error in serial thread: {e}")
    finally:
        if ser and ser.is_open: ser.close()
        print(f"Serial port {port} closed.")
        print("Serial thread finished.")
        window.write_event_value('-SERIAL_DISCONNECTED-', True)

# --- Plot Update Thread (plot_updater) ---
# Remains the same (only uses acc, sp from queue)
def plot_updater(stop_flag: threading.Event, data_q: queue.Queue):
    """
    Continuously reads data from the queue and updates the live plot.
    """
    print("Plot update thread started.")
    while not stop_flag.is_set():
        try:
            new_data_available = False
            while not data_q.empty():
                try:
                    data_point = data_q.get_nowait()
                    time_data.append(data_point['time'])
                    acc_data.append(data_point['acc'])
                    sp_data.append(data_point['sp'])
                    new_data_available = True
                except queue.Empty:
                    break
                except KeyError:
                    print("Warning: Invalid data format in plot queue")

            if new_data_available:
                update_live_plot_data() # Request redraw

            time.sleep(PLOT_UPDATE_INTERVAL)

        except Exception as e:
            print(f"Error in plot update thread: {e}")
            time.sleep(1)

    print("Plot update thread finished.")