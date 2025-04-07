# Aqua-Aware: Real-Time Serial Data Monitoring & ML Prediction System

## Purpose of the Code

This application, named "Aqua-Aware," is designed for real-time monitoring and analysis of sensor data with machine learning integration. The code implements a GUI-based system that:

1. Connects to serial devices (like Arduino, ESP32, or custom sensors)
2. Processes multi-parameter input data (acceleration, speed, roll, pitch, yaw)
3. Applies machine learning predictions using a PyTorch model
4. Visualizes data through real-time plots and state distribution charts
5. Provides a user-friendly interface for monitoring and control

The system is targeted for monitoring motion or movement in underwater environments, likely for Underwater Internet of Things (UIoT) applications.

## Step-by-Step Implementation Process

### 1. Environment Setup

```markdown
- Install required Python libraries:
  - PySimpleGUI - For the graphical interface
  - matplotlib - For data visualization
  - numpy - For numerical operations
  - pyserial - For serial port communication
  - torch - For machine learning model support
  - PIL - For image processing
```

### 2. Model Implementation

```markdown
1. Define the PyTorch model architecture:
   - A placeholder model `PlaceholderObjectMonitorNet` is provided
   - The model accepts 5 input features (acc, speed, roll, pitch, yaw)
   - Outputs classification into 4 states: 'Low', 'Medium', 'Steady', 'High'

2. Implement model loading functionality:
   - Function to load a pre-trained PyTorch model
   - Includes handling for DataParallel models
   - Provides fallback predictor if model loading fails
```

### 3. GUI Setup

```markdown
1. Create the main window using PySimpleGUI with multiple tabs:
   - Main tab - Serial connection controls and live readings
   - Logging tab - Tabular data display
   - Output Log tab - System messages and debugging
   - Import tab - File/folder operations
   - Theming tab - Visual customization

2. Design input layout with:
   - Serial port selection dropdown
   - Baud rate input
   - Connect/Disconnect buttons
   - Real-time sensor value displays
   - Prediction state indicator
   - Visualization canvases for plots
```

### 4. Serial Data Processing

```markdown
1. Implement port detection:
   - Function `get_serial_ports()` to identify available serial ports
   - Platform-specific handling for Windows, Linux, and macOS

2. Create serial reading thread:
   - Connect to specified port with user-defined baud rate
   - Parse incoming data format: "Acc:0.2,Sp:3.1,Roll:1.5,Pitch:-0.8,Yaw:175.2"
   - Extract and validate all 5 required values
   - Handle errors and missing data gracefully
```

### 5. ML Prediction Integration

```markdown
1. Process each complete data point:
   - Convert parsed values to PyTorch tensor format
   - Pass data through loaded model
   - Extract prediction result
   - Update state counter and UI elements
   - Handle prediction errors

2. Implement prediction fallback:
   - Simple rule-based classification if model fails to load
   - Ensures system remains functional even without ML component
```

### 6. Data Visualization

```markdown
1. Implement real-time plotting:
   - Create dual-line plot for acceleration and speed
   - Configure matplotlib for embedded canvas display
   - Use deque for efficient rolling data storage

2. Create state distribution pie chart:
   - Track frequency of each predicted state
   - Update dynamically as new predictions arrive
   - Color-code different states for visual clarity
```

### 7. Multi-threading Implementation

```markdown
1. Serial reader thread:
   - Runs independently from main GUI thread
   - Communicates via event system and queue
   - Handles connecting, reading, and disconnecting from serial port

2. Plot updater thread:
   - Monitors data queue for new values
   - Updates visualization elements at specified intervals
   - Prevents UI blocking during intensive rendering
```

### 8. Event Handling

```markdown
1. Implement core event loop:
   - Handle connection/disconnection events
   - Process incoming serial data events
   - Update UI elements based on predictions
   - Manage visualization updates
   - Handle user interface interactions

2. Implement theme switching:
   - Allow user to select from PySimpleGUI themes
   - Reset and reconstruct interface with new theme
   - Maintain application state during theme changes
```

### 9. Deployment Considerations

```markdown
1. Package requirements:
   - Ensure all dependencies are properly installed
   - Replace placeholder model with actual trained model
   - Verify serial data format matches sensor output

2. Resource management:
   - Implement proper thread cleanup on exit
   - Handle serial port exceptions
   - Manage memory for visualization components
```

