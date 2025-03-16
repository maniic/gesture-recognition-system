import numpy as np
import matplotlib.pyplot as plt
from main import KalmanFilter


def generate_noisy_data(num_points=100, noise_level=10.0):
    """Generate a circular path with noise for testing the Kalman filter"""
    t = np.linspace(0, 2 * np.pi, num_points)
    # True circle
    true_x = 300 + 100 * np.cos(t)
    true_y = 200 + 100 * np.sin(t)
    
    # Add noise
    noisy_x = true_x + np.random.normal(0, noise_level, num_points)
    noisy_y = true_y + np.random.normal(0, noise_level, num_points)
    
    return true_x, true_y, noisy_x, noisy_y


def test_kalman_filter(process_noise=0.01, measurement_noise=0.1):
    """Test Kalman filter on a circular path with noise"""
    # Generate test data
    true_x, true_y, noisy_x, noisy_y = generate_noisy_data(noise_level=15.0)
    
    # Initialize Kalman filter
    kalman = KalmanFilter(process_noise=process_noise, 
                          measurement_noise=measurement_noise)
    
    # Process each point through the Kalman filter
    filtered_x = []
    filtered_y = []
    
    for i in range(len(noisy_x)):
        # Prediction
        kalman.predict()
        
        # Update with measurement
        filtered_pos = kalman.update([noisy_x[i], noisy_y[i]])
        
        # Extract values accounting for different possible shapes
        if isinstance(filtered_pos, np.ndarray) and filtered_pos.ndim > 1:
            filtered_x.append(filtered_pos[0][0])
            filtered_y.append(filtered_pos[1][0])
        else:
            filtered_x.append(filtered_pos[0])
            filtered_y.append(filtered_pos[1])
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.plot(true_x, true_y, 'b-', label='True Path')
    plt.plot(noisy_x, noisy_y, 'r.', label='Noisy Measurements')
    plt.plot(filtered_x, filtered_y, 'g-', label='Kalman Filtered')
    plt.title(f'Kalman Filter Test (Process Noise={process_noise}, Measurement Noise={measurement_noise})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Calculate and print error metrics
    true_points = np.column_stack((true_x, true_y))
    noisy_points = np.column_stack((noisy_x, noisy_y))
    filtered_points = np.column_stack((filtered_x, filtered_y))
    
    # Mean Squared Error (MSE)
    noisy_mse = np.mean(np.sum((noisy_points - true_points)**2, axis=1))
    filtered_mse = np.mean(np.sum((filtered_points - true_points)**2, axis=1))
    
    print(f"Noisy Measurements MSE: {noisy_mse:.2f}")
    print(f"Kalman Filtered MSE: {filtered_mse:.2f}")
    print(f"Improvement: {(1 - filtered_mse/noisy_mse) * 100:.2f}%")
    
    plt.show()


def test_multiple_hands():
    """Test with two hands moving in different patterns"""
    # Generate test data for two hands
    t = np.linspace(0, 2 * np.pi, 100)
    
    # Left hand follows a circle
    left_true_x = 200 + 80 * np.cos(t)
    left_true_y = 200 + 80 * np.sin(t)
    left_noisy_x = left_true_x + np.random.normal(0, 15.0, len(t))
    left_noisy_y = left_true_y + np.random.normal(0, 15.0, len(t))
    
    # Right hand follows a different pattern (figure 8)
    right_true_x = 400 + 80 * np.sin(t)
    right_true_y = 200 + 80 * np.sin(2*t)
    right_noisy_x = right_true_x + np.random.normal(0, 15.0, len(t))
    right_noisy_y = right_true_y + np.random.normal(0, 15.0, len(t))
    
    # Initialize Kalman filters
    left_kalman = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
    right_kalman = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
    
    # Process each point through the Kalman filters
    left_filtered_x, left_filtered_y = [], []
    right_filtered_x, right_filtered_y = [], []
    
    for i in range(len(t)):
        # Left hand
        left_kalman.predict()
        left_filtered_pos = left_kalman.update([left_noisy_x[i], left_noisy_y[i]])
        if isinstance(left_filtered_pos, np.ndarray) and left_filtered_pos.ndim > 1:
            left_filtered_x.append(left_filtered_pos[0][0])
            left_filtered_y.append(left_filtered_pos[1][0])
        else:
            left_filtered_x.append(left_filtered_pos[0])
            left_filtered_y.append(left_filtered_pos[1])
        
        # Right hand
        right_kalman.predict()
        right_filtered_pos = right_kalman.update([right_noisy_x[i], right_noisy_y[i]])
        if isinstance(right_filtered_pos, np.ndarray) and right_filtered_pos.ndim > 1:
            right_filtered_x.append(right_filtered_pos[0][0])
            right_filtered_y.append(right_filtered_pos[1][0])
        else:
            right_filtered_x.append(right_filtered_pos[0])
            right_filtered_y.append(right_filtered_pos[1])
    
    # Create plot with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Left hand plot
    ax1.plot(left_true_x, left_true_y, 'b-', label='True Path')
    ax1.plot(left_noisy_x, left_noisy_y, 'r.', label='Noisy Measurements')
    ax1.plot(left_filtered_x, left_filtered_y, 'g-', label='Kalman Filtered')
    ax1.set_title('Left Hand - Circular Motion')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Right hand plot
    ax2.plot(right_true_x, right_true_y, 'b-', label='True Path')
    ax2.plot(right_noisy_x, right_noisy_y, 'r.', label='Noisy Measurements')
    ax2.plot(right_filtered_x, right_filtered_y, 'g-', label='Kalman Filtered')
    ax2.set_title('Right Hand - Figure 8 Motion')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.suptitle('Multiple Hand Tracking with Kalman Filter', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Calculate and print error metrics
    left_true_points = np.column_stack((left_true_x, left_true_y))
    left_noisy_points = np.column_stack((left_noisy_x, left_noisy_y))
    left_filtered_points = np.column_stack((left_filtered_x, left_filtered_y))
    
    right_true_points = np.column_stack((right_true_x, right_true_y))
    right_noisy_points = np.column_stack((right_noisy_x, right_noisy_y))
    right_filtered_points = np.column_stack((right_filtered_x, right_filtered_y))
    
    # Left hand MSE
    left_noisy_mse = np.mean(np.sum((left_noisy_points - left_true_points)**2, axis=1))
    left_filtered_mse = np.mean(np.sum((left_filtered_points - left_true_points)**2, axis=1))
    
    # Right hand MSE
    right_noisy_mse = np.mean(np.sum((right_noisy_points - right_true_points)**2, axis=1))
    right_filtered_mse = np.mean(np.sum((right_filtered_points - right_true_points)**2, axis=1))
    
    print(f"Left Hand - Noisy MSE: {left_noisy_mse:.2f}, Filtered MSE: {left_filtered_mse:.2f}, Improvement: {(1 - left_filtered_mse/left_noisy_mse) * 100:.2f}%")
    print(f"Right Hand - Noisy MSE: {right_noisy_mse:.2f}, Filtered MSE: {right_filtered_mse:.2f}, Improvement: {(1 - right_filtered_mse/right_noisy_mse) * 100:.2f}%")
    
    plt.show()


def test_multiple_parameters():
    """Test Kalman filter with different parameter combinations"""
    process_noise_values = [0.001, 0.01, 0.1]
    measurement_noise_values = [0.01, 0.1, 1.0]
    
    for p_noise in process_noise_values:
        for m_noise in measurement_noise_values:
            print(f"\nTesting with process_noise={p_noise}, measurement_noise={m_noise}")
            test_kalman_filter(process_noise=p_noise, measurement_noise=m_noise)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Kalman Filter')
    parser.add_argument('--multi', action='store_true', 
                        help='Test multiple parameter combinations')
    parser.add_argument('--two-hands', action='store_true',
                        help='Test with two hands moving in different patterns')
    parser.add_argument('--process-noise', type=float, default=0.01,
                        help='Process noise parameter for Kalman filter')
    parser.add_argument('--measurement-noise', type=float, default=0.1,
                        help='Measurement noise parameter for Kalman filter')
    
    args = parser.parse_args()
    
    if args.multi:
        test_multiple_parameters()
    elif args.two_hands:
        test_multiple_hands()
    else:
        test_kalman_filter(process_noise=args.process_noise, 
                          measurement_noise=args.measurement_noise) 