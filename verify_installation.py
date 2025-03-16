"""
Simple verification script to ensure that the basic functionality works.
"""

import sys
import os

def main():
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    print("\nChecking required packages:")
    
    # Check NumPy
    try:
        import numpy
        print(f"  ✓ NumPy installed (version: {numpy.__version__})")
    except ImportError:
        print("  ✗ NumPy not installed or not working correctly")
        return False
    
    # Check OpenCV
    try:
        import cv2
        print(f"  ✓ OpenCV installed (version: {cv2.__version__})")
    except ImportError:
        print("  ✗ OpenCV not installed or not working correctly")
        return False
    
    # Check MediaPipe
    try:
        import mediapipe
        print(f"  ✓ MediaPipe installed (version: {mediapipe.__version__})")
    except ImportError:
        print("  ✗ MediaPipe not installed or not working correctly")
        return False
    
    # Check Matplotlib
    try:
        import matplotlib
        print(f"  ✓ Matplotlib installed (version: {matplotlib.__version__})")
    except ImportError:
        print("  ✗ Matplotlib not installed or not working correctly")
        return False
    
    print("\nAll required packages are installed!")
    
    # Try to implement a simple Kalman filter to verify that the basic math works
    print("\nTesting simple matrix operations:")
    try:
        import numpy as np
        
        # Create test matrices
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        # Perform operations
        C = A @ B  # Matrix multiplication
        print(f"  ✓ Matrix multiplication works")
        
        # Try to invert a matrix (needed for Kalman filter)
        A_inv = np.linalg.inv(A)
        print(f"  ✓ Matrix inversion works")
        
        return True
    except Exception as e:
        print(f"  ✗ Error performing matrix operations: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 