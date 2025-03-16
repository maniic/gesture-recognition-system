# Gesture Recognition System

A real-time hand gesture recognition system that combines MediaPipe's hand tracking with Kalman filtering for robust hand position tracking.

## Project Overview

This project combines real-time gesture recognition with data fusion techniques to create a robust hand-tracking system. Using MediaPipe's hand tracking module, the system detects and tracks hand landmarks from a webcam feed and applies a heuristic-based gesture recognition algorithm. Based on the relative positions of the landmarks, it classifies common gestures such as Fist, Open Hand, Peace Sign, and Thumbs Up.

In parallel, the project integrates a Kalman filter to smooth the noisy wrist position measurements provided by MediaPipe. The Kalman filter fuses the raw positional data with a prediction based on the system's previous state (including both position and velocity), resulting in a more stable and accurate tracking of the hand's movement.

## Key Components

- **MediaPipe Hand Tracking**: Detects hand landmarks and provides raw positional data.
- **Gesture Recognition**: Analyzes finger positions to determine specific gestures.
- **Data Fusion with Kalman Filter**: Combines noisy measurements with predictive modeling to smooth the hand's tracked position.
- **Visualization**: Displays the raw and fused wrist positions alongside the recognized gesture on the video frame.

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gesture-recognition-system.git
   cd gesture-recognition-system
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```
   python verify_installation.py
   ```

## Usage

Run the main application:
```
python main.py
```

Additional command-line options:
- `--visualize`: Enable matplotlib visualization
- `--save-plot`: Save visualization plot at the end
- `--camera N`: Use camera with index N (default: 0)

Example:
```
python main.py --visualize --camera 1
```

## Testing

Run the Kalman filter tests:
```
python test_kalman.py
```

Run the system tests:
```
python test_system.py
```

## Project Structure

- `main.py`: Main application with gesture recognition system
- `verify_installation.py`: Script to verify dependencies are installed correctly
- `visualize.py`: Visualization utilities for tracking data
- `test_kalman.py`: Tests for the Kalman filter implementation
- `test_system.py`: Comprehensive system tests
- `requirements.txt`: List of required Python packages

## License

This project is licensed under the MIT License - see the LICENSE file for details. 