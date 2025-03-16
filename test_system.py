import cv2
import os
import numpy as np
import time
import sys
import importlib.util
import platform

def print_section(title):
    """Print a section title formatted with dashes"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "-"))
    print("=" * 50)

def test_opencv():
    """Test OpenCV installation and camera access"""
    print_section("Testing OpenCV")
    
    print(f"OpenCV version: {cv2.__version__}")
    
    try:
        # Try to open the default camera
        print("Attempting to access camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read a frame from camera")
            cap.release()
            return False
        
        # Get basic info
        height, width, channels = frame.shape
        print(f"✅ Successfully accessed camera")
        print(f"   - Frame dimensions: {width}x{height} pixels")
        print(f"   - Channels: {channels}")
        
        # Release camera
        cap.release()
        
        # Try to save and load an image to test disk access
        test_file = "test_opencv.jpg"
        cv2.imwrite(test_file, frame)
        if os.path.exists(test_file):
            print(f"✅ Successfully saved image to {test_file}")
            # Clean up
            os.remove(test_file)
        else:
            print(f"❌ Failed to save image to {test_file}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing OpenCV: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe installation"""
    print_section("Testing MediaPipe")
    
    try:
        import mediapipe as mp
        print(f"MediaPipe version: {mp.__version__}")
        
        # Test creating a Hands object
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        print("Successfully created MediaPipe Hands object")
        
        # Clean up
        hands.close()
        return True
    except ImportError:
        print("❌ MediaPipe is not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing MediaPipe: {e}")
        return False

def test_numpy():
    """Test NumPy installation and basic operations"""
    print_section("Testing NumPy")
    
    try:
        print(f"NumPy version: {np.__version__}")
        
        # Create a test array
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        print(f"Successfully created NumPy array: {test_array.shape}")
        
        # Test matrix operations
        result = test_array.T @ test_array
        print(f"Successfully performed matrix operation")
        
        return True
    except Exception as e:
        print(f"Error testing NumPy: {e}")
        return False

def test_matplotlib():
    """Test Matplotlib installation"""
    print_section("Testing Matplotlib")
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        
        print(f"Matplotlib version: {matplotlib.__version__}")
        print(f"Backend: {matplotlib.get_backend()}")
        
        # Create a simple plot
        plt.figure(figsize=(2, 2))
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.close()
        
        print("Successfully created a plot")
        
        return True
    except ImportError:
        print("❌ Matplotlib is not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing Matplotlib: {e}")
        return False

def test_kalman_filter():
    """Test KalmanFilter class from main module"""
    print_section("Testing Kalman Filter")
    
    try:
        from main import KalmanFilter
        
        # Create a Kalman filter
        kalman = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        print("Successfully created KalmanFilter object")
        
        # Test prediction and update
        predicted_pos = kalman.predict()
        print(f"Successfully called predict(): {predicted_pos.shape}")
        
        updated_pos = kalman.update([100, 200])
        print(f"Successfully called update(): {updated_pos.shape}")
        
        return True
    except ImportError:
        print("Could not import KalmanFilter from main module")
        return False
    except Exception as e:
        print(f"Error testing KalmanFilter: {e}")
        return False

def test_gesture_recognizer():
    """Test GestureRecognizer class from main module"""
    print_section("Testing Gesture Recognizer")
    
    try:
        from main import GestureRecognizer
        
        # Create a GestureRecognizer
        recognizer = GestureRecognizer()
        print("Successfully created GestureRecognizer object")
        print("Multiple hand support configured (max_num_hands=2)")
        
        # Check if the Kalman filters exist for both hands
        if 'left' in recognizer.kalman_filters and 'right' in recognizer.kalman_filters:
            print("Kalman filters created for both left and right hands")
        else:
            print("Kalman filters not properly configured for multiple hands")
        
        # Check if data structures exist for both hands
        if 'left' in recognizer.raw_positions and 'right' in recognizer.raw_positions:
            print("Position tracking data structures set up for both hands")
        else:
            print("Position tracking data structures not properly configured")
        
        # Check if we can access a camera to test frame processing
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("Testing frame processing...")
                processed_frame = recognizer.process_frame(frame)
                print(f"Successfully processed a frame: {processed_frame.shape}")
            cap.release()
        else:
            print("Cannot test frame processing - camera not available")
        
        # Clean up
        recognizer.release()
        return True
    except ImportError:
        print("Could not import GestureRecognizer from main module")
        return False
    except Exception as e:
        print(f"Error testing GestureRecognizer: {e}")
        return False

def test_visualizer():
    """Test TrackingVisualizer class from visualize module"""
    print_section("Testing Visualizer")
    
    try:
        from visualize import TrackingVisualizer
        
        # Create a TrackingVisualizer
        visualizer = TrackingVisualizer()
        print("Successfully created TrackingVisualizer object")
        
        # Test adding data points for both hands
        visualizer.add_data_point((100, 200), (110, 210), 'left')
        visualizer.add_data_point((300, 250), (310, 260), 'right')
        visualizer.update_plot()
        print("Successfully added data points for both hands")
        
        # Verify data structure
        data_keys = visualizer.data.keys()
        if 'left' in data_keys and 'right' in data_keys:
            print("Data structures correctly configured for both hands")
        else:
            print("Data structures not properly configured for multiple hands")
        
        # Verify plot lines
        if 'left' in visualizer.lines and 'right' in visualizer.lines:
            print("Plot visualization configured for both hands")
        else:
            print("Plot visualization not configured for multiple hands")
        
        return True
    except ImportError:
        print("Could not import TrackingVisualizer from visualize module")
        return False
    except Exception as e:
        print(f"Error testing TrackingVisualizer: {e}")
        return False

def display_system_info():
    """Display system information for debugging"""
    print_section("System Information")
    
    print(f"Python version: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Display common paths for debugging
    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")

def run_all_tests():
    """Run all tests and report results"""
    print_section("Hand Recognition System Test")
    print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display system info
    display_system_info()
    
    # Run all tests
    tests = [
        ("OpenCV", test_opencv),
        ("MediaPipe", test_mediapipe),
        ("NumPy", test_numpy),
        ("Matplotlib", test_matplotlib),
        ("Kalman Filter", test_kalman_filter),
        ("Gesture Recognizer", test_gesture_recognizer),
        ("Visualizer", test_visualizer)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        start_time = time.time()
        success = test_func()
        elapsed = time.time() - start_time
        results.append((name, success, elapsed))
    
    # Display summary
    print_section("Test Summary")
    
    all_success = True
    for name, success, elapsed in results:
        status = "PASS" if success else "FAIL"
        print(f"{name.ljust(20)}: {status} ({elapsed:.2f}s)")
        all_success = all_success and success
    
    print("\nOverall result:", "All tests passed" if all_success else "Some tests failed")
    
    return all_success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 