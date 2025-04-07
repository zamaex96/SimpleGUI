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
