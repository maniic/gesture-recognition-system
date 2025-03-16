import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import threading
import time

# Use Agg backend if running headless or in an environment without display
try:
    # Try to create a figure to test if interactive mode works
    plt.figure()
    plt.close()
except Exception:
    print("Warning: Using non-interactive Agg backend for matplotlib")
    matplotlib.use('Agg')

class TrackingVisualizer:
    def __init__(self, max_points=200):
        self.max_points = max_points
        
        # Initialize data arrays for each hand
        self.data = {
            'left': {
                'raw_x': [], 'raw_y': [],
                'filtered_x': [], 'filtered_y': []
            },
            'right': {
                'raw_x': [], 'raw_y': [],
                'filtered_x': [], 'filtered_y': []
            }
        }
        
        # Lock for thread safety when updating data
        self.data_lock = threading.Lock()
        
        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Create lines and scatter points for each hand
        self.lines = {
            'left': {
                'raw': self.ax.plot([], [], 'r-', lw=1, label='Left Raw')[0],
                'filtered': self.ax.plot([], [], 'g-', lw=2, label='Left Filtered')[0],
                'raw_scatter': self.ax.scatter([], [], c='red', s=30),
                'filtered_scatter': self.ax.scatter([], [], c='green', s=30)
            },
            'right': {
                'raw': self.ax.plot([], [], 'b-', lw=1, label='Right Raw')[0],
                'filtered': self.ax.plot([], [], 'y-', lw=2, label='Right Filtered')[0],
                'raw_scatter': self.ax.scatter([], [], c='blue', s=30),
                'filtered_scatter': self.ax.scatter([], [], c='yellow', s=30)
            }
        }
        
        # Labels and legends
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Hand Tracking: Raw vs. Filtered Positions (Both Hands)')
        self.ax.legend()
        self.ax.grid(True)
        
        # Initial ranges (will be updated dynamically)
        self.ax.set_xlim(0, 640)
        self.ax.set_ylim(0, 480)
        
        # Invert Y axis to match image coordinates
        self.ax.invert_yaxis()
        
        # Flag to track if the plot is closed
        self.is_closed = False
        self.fig.canvas.mpl_connect('close_event', self._handle_close)
        
    def _handle_close(self, event):
        """Handle plot window close event"""
        self.is_closed = True
        
    def add_data_point(self, raw_pos, filtered_pos, hand_label='left'):
        """Add a new data point to the visualization for the specified hand"""
        if self.is_closed:
            return
            
        raw_x, raw_y = raw_pos
        filtered_x, filtered_y = filtered_pos
        
        # Thread-safe data update
        with self.data_lock:
            # Add to arrays
            self.data[hand_label]['raw_x'].append(raw_x)
            self.data[hand_label]['raw_y'].append(raw_y)
            self.data[hand_label]['filtered_x'].append(filtered_x)
            self.data[hand_label]['filtered_y'].append(filtered_y)
            
            # Limit data length
            if len(self.data[hand_label]['raw_x']) > self.max_points:
                self.data[hand_label]['raw_x'] = self.data[hand_label]['raw_x'][-self.max_points:]
                self.data[hand_label]['raw_y'] = self.data[hand_label]['raw_y'][-self.max_points:]
                self.data[hand_label]['filtered_x'] = self.data[hand_label]['filtered_x'][-self.max_points:]
                self.data[hand_label]['filtered_y'] = self.data[hand_label]['filtered_y'][-self.max_points:]
        
        # Update plot limits if needed
        self._update_plot_limits()
    
    def _update_plot_limits(self):
        """Update the plot limits to keep all data visible"""
        if self.is_closed:
            return
            
        all_x = []
        all_y = []
        
        with self.data_lock:
            for hand in ['left', 'right']:
                if self.data[hand]['raw_x']:
                    all_x.extend(self.data[hand]['raw_x'])
                    all_x.extend(self.data[hand]['filtered_x'])
                    all_y.extend(self.data[hand]['raw_y'])
                    all_y.extend(self.data[hand]['filtered_y'])
            
            if not all_x:
                return
            
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            # Add some margin
            margin_x = (max_x - min_x) * 0.1 or 50
            margin_y = (max_y - min_y) * 0.1 or 50
        
        # Set limits outside the lock to avoid potential deadlocks with matplotlib
        try:
            self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
            self.ax.set_ylim(max_y + margin_y, min_y - margin_y)  # Inverted for image coordinates
        except Exception as e:
            print(f"Error updating plot limits: {e}")
    
    def update_plot(self):
        """Update the visualization with current data"""
        if self.is_closed:
            return []
            
        all_artists = []
        
        try:
            with self.data_lock:
                for hand_label in ['left', 'right']:
                    hand_data = self.data[hand_label]
                    
                    if not hand_data['raw_x']:
                        continue
                    
                    # Update line data
                    self.lines[hand_label]['raw'].set_data(hand_data['raw_x'], hand_data['raw_y'])
                    self.lines[hand_label]['filtered'].set_data(hand_data['filtered_x'], hand_data['filtered_y'])
                    
                    # Update scatter with latest points
                    raw_latest = np.column_stack([hand_data['raw_x'][-1:], hand_data['raw_y'][-1:]])
                    filtered_latest = np.column_stack([hand_data['filtered_x'][-1:], hand_data['filtered_y'][-1:]])
                    
                    self.lines[hand_label]['raw_scatter'].set_offsets(raw_latest)
                    self.lines[hand_label]['filtered_scatter'].set_offsets(filtered_latest)
                    
                    # Add to artists list
                    all_artists.extend([
                        self.lines[hand_label]['raw'],
                        self.lines[hand_label]['filtered'],
                        self.lines[hand_label]['raw_scatter'],
                        self.lines[hand_label]['filtered_scatter']
                    ])
            
            return all_artists
        except Exception as e:
            print(f"Error updating plot: {e}")
            return []
    
    def animate(self, interval=50):
        """Start animation for real-time visualization"""
        if self.is_closed:
            return None
            
        ani = FuncAnimation(self.fig, lambda i: self.update_plot(), 
                           interval=interval, blit=True)
        
        try:
            plt.show()
        except Exception as e:
            print(f"Error showing animation: {e}")
        
        return ani
    
    def save_plot(self, filename="hand_tracking_visualization.png"):
        """Save the current plot to a file"""
        if self.is_closed:
            print("Warning: Cannot save plot - window is closed")
            return
            
        try:
            # Update the plot before saving
            self.update_plot()
            self.fig.savefig(filename)
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"Error saving plot to {filename}: {e}")
    
    def show_plot(self):
        """Show the current plot"""
        if self.is_closed:
            print("Warning: Cannot show plot - window is closed")
            return
            
        try:
            # Update the plot before showing
            self.update_plot()
            plt.show()
        except Exception as e:
            print(f"Error showing plot: {e}")


# Example of how to use this class:
if __name__ == "__main__":
    import math
    
    try:
        # Simulate noisy hand movement
        visualizer = TrackingVisualizer()
        
        print("Generating simulated hand movement data...")
        for i in range(100):
            # True position (circle)
            t = i / 10.0
            
            # Left hand - circular motion
            left_true_x = 200 + 80 * math.cos(t)
            left_true_y = 240 + 80 * math.sin(t)
            
            # Right hand - square motion
            right_true_x = 440 + 80 * math.cos(t*0.8)
            right_true_y = 240 + 80 * math.sin(t*1.2)
            
            # Add noise to raw positions
            left_noise_x = np.random.normal(0, 10)
            left_noise_y = np.random.normal(0, 10)
            left_raw_x = left_true_x + left_noise_x
            left_raw_y = left_true_y + left_noise_y
            
            right_noise_x = np.random.normal(0, 10)
            right_noise_y = np.random.normal(0, 10)
            right_raw_x = right_true_x + right_noise_x
            right_raw_y = right_true_y + right_noise_y
            
            # Simulate filtered positions (less noisy)
            left_filtered_x = left_true_x + np.random.normal(0, 3)
            left_filtered_y = left_true_y + np.random.normal(0, 3)
            
            right_filtered_x = right_true_x + np.random.normal(0, 3)
            right_filtered_y = right_true_y + np.random.normal(0, 3)
            
            # Add to visualizer
            visualizer.add_data_point((left_raw_x, left_raw_y), (left_filtered_x, left_filtered_y), 'left')
            visualizer.add_data_point((right_raw_x, right_raw_y), (right_filtered_x, right_filtered_y), 'right')
            
            # Update plot for simulation
            visualizer.update_plot()
            plt.pause(0.05)
        
        # Show the final result
        print("Simulation complete. Showing final plot...")
        visualizer.show_plot()
    except Exception as e:
        print(f"Error in visualization test: {e}") 