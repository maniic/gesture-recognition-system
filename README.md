# Hand Recognition System

This project combines real-time gesture recognition with data fusion techniques using a Kalman filter to create a robust hand-tracking system.

## Features

- **Multi-Hand Tracking**: Detects and tracks both hands simultaneously with distinct visual indicators
- **Advanced Gesture Recognition**: Accurately recognizes up to 15 different hand gestures including:
  - Basic gestures: Fist, Open Hand (all except thumb), Peace Sign, Thumbs Up, Thumbs Down, Pointing
  - Advanced gestures: Three Fingers, Four Fingers (all five fingers extended), OK Sign, Rock On (index and pinky extended), Pinky Promise, Phone Call, Spider-Man, Gun, L Shape
  - Custom gestures with automatic labeling
- **Data Fusion**: Applies a Kalman filter to smooth the noisy wrist position measurements for more stable tracking
- **Orientation Support**: Works with different hand orientations and positions
- **Visual Feedback**: Color-coded visualization with reduced opacity (60%) for better visibility:
  - Left hand: Red (raw) / Green (filtered)
  - Right hand: Blue (raw) / Yellow (filtered)
- **Extended Visualization**: Optional matplotlib-based visualization for detailed tracking analysis
- **Error Handling**: Robust error handling for better stability in various environments

## Requirements

- Python 3.7+ (Tested with Python 3.12.4)
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
   If you encounter issues, try installing packages individually:
   ```
   pip install opencv-python==4.8.1
   pip install mediapipe==0.10.7
   pip install numpy==1.26.0
   pip install matplotlib==3.8.0
   ```
3. Verify installation:
   ```
   python verify_installation.py
   ```

## Usage

### Basic Usage
Run the main script:

```
python main.py
```

### With Visualization
Run with the matplotlib visualization for detailed tracking analysis:

```
python main.py --visualize
```

To also save the tracking plot as an image when you exit:

```
python main.py --visualize --save-plot
```

### Using a Different Camera
If you have multiple cameras, you can specify which one to use:

```
python main.py --camera 1
```

### Test System Setup
To verify that all components of the system are working correctly:

```
python test_system.py
```

This will run a series of tests to check:
- OpenCV installation and camera access
- MediaPipe functionality
- NumPy operations
- Matplotlib visualization
- Kalman filter implementation
- Gesture recognizer 
- Visualization system

### Controls and Interface
- The webcam will start and begin tracking your hand(s)
- Different gestures will be recognized and displayed on screen for each hand
- Color-coded indicators for each hand:
  - Left hand: Red dots/trails (raw) and Green dots/trails (filtered) at 60% opacity
  - Right hand: Blue dots/trails (raw) and Yellow dots/trails (filtered) at 60% opacity
- If visualization is enabled, a separate window shows the tracking paths
- Press 'q' to quit the application

## How It Works

### MediaPipe Hand Tracking
The system uses MediaPipe's hand tracking to detect 21 hand landmarks in real-time for up to two hands simultaneously. It correctly distinguishes between left and right hands from the user's perspective.

### Gesture Recognition
A sophisticated algorithm analyzes the relative positions of fingers to determine specific gestures with confidence scoring. The system can handle different hand orientations and positions (palm facing camera, sideways, etc.) and provides accurate classification using angle-based measurements for finger extension detection.

### Kalman Filter
The Kalman filter fuses raw positional data with predictions based on previous states, resulting in smoother tracking. Each hand has its own Kalman filter instance. The implementation uses a constant velocity model with:
- State vector: [x, y, vx, vy]
- Process and measurement noise parameters that can be tuned for different environments

### Data Visualization
The system offers two types of visualization:
1. **Real-time OpenCV**: Shows the camera feed with hand landmarks, gesture recognition, and position tracking for both hands with distinct color coding at 60% opacity
2. **Matplotlib Analysis**: Plots the raw vs. filtered position data to visualize the Kalman filter's effect for both hands (enabled with `--visualize`)

## Customization

You can adjust several parameters to customize the system:
- `process_noise` and `measurement_noise` in the KalmanFilter constructor to change filtering behavior
- `max_num_hands` in the MediaPipe Hands initialization (set to 2 for tracking both hands)
- `max_positions` in GestureRecognizer to change the length of position trails

## Advanced Analysis

For offline analysis of hand tracking data:
1. Run with the visualization option: `python main.py --visualize --save-plot`
2. Perform hand movements to capture tracking data
3. Exit the application by pressing 'q'
4. The tracking visualization will be saved and also displayed for analysis

## Testing the Kalman Filter

To evaluate the Kalman filter separately:

```
python test_kalman.py
```

For testing multiple parameter combinations:

```
python test_kalman.py --multi
```

For testing with two hands:

```
python test_kalman.py --two-hands
```

This will generate synthetic circular motion with noise and show how the Kalman filter smooths the tracking.

## Recent Updates

### March 14, 2025
- **Fixed**: Left/right hand detection now correctly matches the user's perspective
- **Improved**: Gesture recognition accuracy with confidence-based scoring system
- **Enhanced**: Better detection of thumb and finger positions in various hand orientations
- **Updated**: More robust finger extension detection using angle-based measurements
- **Adjusted**: Reduced color opacity to 60% for more pleasant visualization
- **Corrected**: "Four Fingers" gesture now means all five fingers extended, while "Open Hand" means all fingers except thumb

Happy Tracking!