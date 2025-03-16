import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from visualize import TrackingVisualizer

class KalmanFilter:
    def __init__(self, process_noise=0.001, measurement_noise=0.1):
        # State: [x, y, vx, vy]
        self.state = np.zeros((4, 1))
        self.covariance = np.eye(4)
        
        # How much we expect the state to change
        self.process_noise = process_noise
        self.Q = np.eye(4) * process_noise
        
        # How noisy our measurements are
        self.measurement_noise = measurement_noise
        self.R = np.eye(2) * measurement_noise
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix (maps state to measurement)
        self.H = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ])
        
        self.initialized = False
        
    def predict(self):
        # Predict the next state
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement):
        if not self.initialized:
            # Initialize state with first measurement
            self.state[0, 0] = measurement[0]
            self.state[1, 0] = measurement[1]
            self.initialized = True
            return self.state[:2]
        
        # Update state with measurement
        measurement = np.array(measurement).reshape(2, 1)
        
        # Calculate Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(self.H @ self.covariance @ self.H.T + self.R)
        
        # Update state
        self.state = self.state + K @ (measurement - self.H @ self.state)
        
        # Update covariance
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        
        return self.state[:2]  # Return updated position


class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track up to 2 hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize Kalman filters for each hand
        self.kalman_filters = {
            'left': KalmanFilter(process_noise=0.01, measurement_noise=0.1),
            'right': KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        }
        
        # Store positions for visualization for each hand
        self.raw_positions = {'left': [], 'right': []}
        self.filtered_positions = {'left': [], 'right': []}
        self.max_positions = 50  # Maximum number of positions to store
        
        # For FPS calculation
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        
        # Hand colors for visualization - 60% opacity
        self.hand_colors = {
            'left': {
                'raw': (0, 0, 153),     # Red
                'filtered': (0, 153, 0)  # Green
            },
            'right': {
                'raw': (153, 0, 0),     # Blue
                'filtered': (153, 153, 0)  # Yellow
            }
        }
        
        # Store detected hand gestures
        self.hand_gestures = {'left': "Unknown", 'right': "Unknown"}
    
    def _is_finger_extended(self, landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx=None):
        """
        Check if a finger is extended based on the position of the landmarks.
        More robust implementation that handles different hand orientations.
        """
        # Calculate the direction of the palm
        wrist = landmarks[0]
        middle_mcp = landmarks[9]  # Middle finger MCP
        palm_direction_vertical = abs(middle_mcp.y - wrist.y) > abs(middle_mcp.x - wrist.x)
        
        # For thumb, we need special handling
        if finger_tip_idx == 4:  # Thumb
            # Determine if hand is facing left or right by checking if pinky is left/right of index
            pinky_mcp = landmarks[17]
            index_mcp = landmarks[5]
            is_hand_facing_right = pinky_mcp.x > index_mcp.x
            
            # Thumb tip, pip, and mcp points
            thumb_tip = landmarks[finger_tip_idx]
            thumb_ip = landmarks[finger_pip_idx]
            thumb_mcp = landmarks[finger_mcp_idx] if finger_mcp_idx else landmarks[2]
            
            # Calculate vectors
            v1 = np.array([thumb_ip.x - thumb_mcp.x, thumb_ip.y - thumb_mcp.y])
            v2 = np.array([thumb_tip.x - thumb_ip.x, thumb_tip.y - thumb_ip.y])
            
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
            v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
            
            # Calculate dot product to find angle
            dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot_product) * 180 / np.pi
            
            # Thumb is extended if the angle is substantial
            angle_threshold = 25  # Degrees
            
            if is_hand_facing_right:
                # Additional check for the position relative to index finger
                index_mcp_x = landmarks[5].x
                is_thumb_out = thumb_tip.x > index_mcp_x and angle > angle_threshold
                return is_thumb_out
            else:
                # Additional check for the position relative to index finger
                index_mcp_x = landmarks[5].x
                is_thumb_out = thumb_tip.x < index_mcp_x and angle > angle_threshold
                return is_thumb_out
            
        # Get the finger tip, pip and mcp points
        finger_tip = landmarks[finger_tip_idx]
        finger_pip = landmarks[finger_pip_idx]
        finger_mcp = landmarks[finger_mcp_idx] if finger_mcp_idx else None
        
        # Main check - if finger tip is higher than PIP
        basic_check = finger_tip.y < finger_pip.y
        
        # For non-vertical palm orientations, use more sophisticated checks
        if not palm_direction_vertical and finger_mcp is not None:
            # Calculate vectors and angles
            v1 = np.array([finger_pip.x - finger_mcp.x, finger_pip.y - finger_mcp.y])
            v2 = np.array([finger_tip.x - finger_pip.x, finger_tip.y - finger_pip.y])
            
            # Check if vectors are reasonably aligned
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
            v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
            
            # Calculate dot product to find angle
            dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot_product) * 180 / np.pi
            
            # Finger is extended if roughly straight and extending beyond PIP
            finger_length_threshold = 0.02  # Minimum length
            length_ratio = np.linalg.norm(v2) / np.linalg.norm(v1)
            
            # Different thresholds for different fingers
            angle_threshold = 45  # General angle threshold (degrees)
            length_ratio_threshold = 0.5  # Minimum ratio of finger segment lengths
            
            is_extended = (angle < angle_threshold and length_ratio > length_ratio_threshold and 
                          np.linalg.norm(v2) > finger_length_threshold)
            
            return is_extended
        
        # For vertical orientation, use the basic check (tip above pip)
        return basic_check
    
    def recognize_gesture(self, landmarks):
        """
        Recognize hand gesture based on finger positions
        Returns: gesture name (string)
        """
        # Check if thumb is extended
        thumb_extended = self._is_finger_extended(landmarks, 4, 3, 2)
        
        # Check if fingers are extended
        index_extended = self._is_finger_extended(landmarks, 8, 6, 5)
        middle_extended = self._is_finger_extended(landmarks, 12, 10, 9)
        ring_extended = self._is_finger_extended(landmarks, 16, 14, 13)
        pinky_extended = self._is_finger_extended(landmarks, 20, 18, 17)
        
        # Count extended fingers
        finger_states = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        extended_fingers = sum(finger_states)
        
        # Calculate finger distance metrics for better accuracy
        wrist = landmarks[0]
        fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        
        # Calculate distances to improve recognition
        # Distance between fingertips for special gestures
        thumb_index_distance = np.sqrt((landmarks[4].x - landmarks[8].x)**2 + 
                                       (landmarks[4].y - landmarks[8].y)**2)
        
        # Determine finger extension confidence
        # If fingers are very clearly extended or closed, use that for more reliable detection
        extension_confidence = {}
        for i, (name, idx_tip, idx_pip, idx_mcp) in enumerate(zip(
            ['thumb', 'index', 'middle', 'ring', 'pinky'],
            [4, 8, 12, 16, 20],  # Fingertip landmarks
            [3, 6, 10, 14, 18],  # PIP landmarks
            [2, 5, 9, 13, 17]    # MCP landmarks
        )):
            # Calculate vectors
            v_mcp_to_pip = np.array([landmarks[idx_pip].x - landmarks[idx_mcp].x, 
                                     landmarks[idx_pip].y - landmarks[idx_mcp].y])
            v_pip_to_tip = np.array([landmarks[idx_tip].x - landmarks[idx_pip].x,
                                     landmarks[idx_tip].y - landmarks[idx_pip].y])
            
            # Normalize vectors
            if np.linalg.norm(v_mcp_to_pip) > 0:
                v_mcp_to_pip = v_mcp_to_pip / np.linalg.norm(v_mcp_to_pip)
            if np.linalg.norm(v_pip_to_tip) > 0:
                v_pip_to_tip = v_pip_to_tip / np.linalg.norm(v_pip_to_tip)
            
            # Calculate dot product to find angle
            dot_product = np.clip(np.dot(v_mcp_to_pip, v_pip_to_tip), -1.0, 1.0)
            angle = np.arccos(dot_product) * 180 / np.pi
            
            # Calculate confidence based on angle
            # If angle is very small: definitely extended
            # If angle is very large: definitely closed
            if angle < 20:  # Nearly straight
                extension_confidence[name] = 0.9  # High confidence it's extended
            elif angle > 70:  # Significantly bent
                extension_confidence[name] = 0.1  # High confidence it's closed
            else:
                extension_confidence[name] = 0.5  # Uncertain
        
        # Define standard thresholds and refined gestures
        # Recognize gestures based on finger states with refined logic
        # Use confidence scores for better accuracy
        
        # Four Fingers - All five fingers extended
        if all(finger_states):
            return "Four Fingers"
            
        # Fist - All fingers must be closed
        elif all(not finger for finger in finger_states):
            return "Fist"
            
        # Open Hand - All except thumb extended
        elif (not thumb_extended and index_extended and middle_extended and 
              ring_extended and pinky_extended and
              extension_confidence['index'] > 0.6 and 
              extension_confidence['middle'] > 0.6):
            return "Open Hand"
            
        # Peace Sign - Only index and middle fingers extended
        elif (index_extended and middle_extended and 
              not ring_extended and not pinky_extended and 
              not thumb_extended and
              extension_confidence['index'] > 0.7 and 
              extension_confidence['middle'] > 0.7):
            return "Peace Sign"
            
        # Thumbs Up - Only thumb extended and pointing upward
        elif (thumb_extended and not any(finger_states[1:]) and 
              landmarks[4].y < landmarks[0].y and  # Thumb tip above wrist
              extension_confidence['thumb'] > 0.7):
            return "Thumbs Up"
            
        # Thumbs Down - Only thumb extended and pointing downward
        elif (thumb_extended and not any(finger_states[1:]) and 
              landmarks[4].y > landmarks[0].y and  # Thumb tip below wrist
              extension_confidence['thumb'] > 0.7):
            return "Thumbs Down"
            
        # Pointing - Index finger extended, possibly with thumb
        elif (index_extended and not middle_extended and 
              not ring_extended and not pinky_extended and
              extension_confidence['index'] > 0.7):
            return "Pointing"
            
        # Three Fingers - Index, middle, and ring extended
        elif (index_extended and middle_extended and ring_extended and 
              not pinky_extended and not thumb_extended and
              extension_confidence['index'] > 0.6 and 
              extension_confidence['middle'] > 0.6 and
              extension_confidence['ring'] > 0.6):
            return "Three Fingers"
            
        # OK Sign - Thumb and index form a circle, others extended
        elif (not thumb_extended and not index_extended and 
              middle_extended and ring_extended and pinky_extended):
            # Check if thumb and index are close
            if thumb_index_distance < 0.1:  # Close together
                return "OK Sign"
            else:
                return "Three Fingers Up"
                
        # Rock On - Index and pinky extended, others closed
        elif (not thumb_extended and index_extended and not middle_extended and 
              not ring_extended and pinky_extended and
              extension_confidence['index'] > 0.7 and 
              extension_confidence['pinky'] > 0.7):
            return "Rock On"
            
        # Pinky Promise - Only pinky extended
        elif (not thumb_extended and not index_extended and not middle_extended and 
              not ring_extended and pinky_extended and
              extension_confidence['pinky'] > 0.8):
            return "Pinky Promise"
            
        # Phone Call - Thumb and pinky extended, like a phone gesture
        elif (thumb_extended and not index_extended and not middle_extended and 
              not ring_extended and pinky_extended and
              extension_confidence['thumb'] > 0.7 and 
              extension_confidence['pinky'] > 0.7):
            return "Phone Call"
            
        # Spider-Man - Thumb, index, and pinky extended (web shooter)
        elif (thumb_extended and index_extended and not middle_extended and 
              not ring_extended and pinky_extended):
            return "Spider-Man"
            
        # Gun gesture - Thumb and index extended, others closed
        elif (thumb_extended and index_extended and not middle_extended and 
              not ring_extended and not pinky_extended and
              extension_confidence['thumb'] > 0.7 and 
              extension_confidence['index'] > 0.7):
            # Check if they form a roughly perpendicular angle
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            # Vectors from wrist to fingertips
            thumb_vector = [thumb_tip.x - wrist.x, thumb_tip.y - wrist.y]
            index_vector = [index_tip.x - wrist.x, index_tip.y - wrist.y]
            
            # Calculate dot product
            dot_product = thumb_vector[0] * index_vector[0] + thumb_vector[1] * index_vector[1]
            # Normalize
            thumb_mag = (thumb_vector[0]**2 + thumb_vector[1]**2)**0.5
            index_mag = (index_vector[0]**2 + index_vector[1]**2)**0.5
            
            if abs(dot_product / (thumb_mag * index_mag)) < 0.3:  # Close to perpendicular
                return "L Shape"
            else:
                return "Gun"
        
        # Return descriptive name for other combinations
        else:
            # For more accurate description of custom poses
            reliable_extended = []
            for i, (name, extended, conf) in enumerate(zip(
                ["Thumb", "Index", "Middle", "Ring", "Pinky"],
                finger_states,
                [extension_confidence.get(x, 0.5) for x in ['thumb', 'index', 'middle', 'ring', 'pinky']]
            )):
                # Only include fingers we're confident about
                if extended and conf > 0.6:
                    reliable_extended.append(name)
                
            if reliable_extended:
                return f"Custom: {', '.join(reliable_extended)}"
            else:
                return "Unknown"
    
    def process_frame(self, frame, visualizer=None):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            # Process for each detected hand
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Determine if it's left or right hand
                # MediaPipe's labels are from the camera's perspective, which is opposite the user's perspective
                # So we need to invert the labels (Left becomes Right and vice versa)
                camera_label = handedness.classification[0].label.lower()  # 'Left' or 'Right' from camera perspective
                hand_label = 'right' if camera_label == 'left' else 'left'  # Invert to match user's perspective
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get landmarks as a list
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark)
                
                # Recognize gesture
                gesture = self.recognize_gesture(landmarks)
                self.hand_gestures[hand_label] = gesture
                
                # Get wrist position (raw measurement)
                wrist_x = int(landmarks[0].x * w)
                wrist_y = int(landmarks[0].y * h)
                raw_position = np.array([wrist_x, wrist_y])
                
                # Predict next position using Kalman filter
                self.kalman_filters[hand_label].predict()
                
                # Update Kalman filter with measurement
                filtered_position = self.kalman_filters[hand_label].update(raw_position)
                # Fix for NumPy scalar conversion warning
                filtered_x = int(filtered_position[0][0]) if isinstance(filtered_position, np.ndarray) and filtered_position.ndim > 1 else int(filtered_position[0])
                filtered_y = int(filtered_position[1][0]) if isinstance(filtered_position, np.ndarray) and filtered_position.ndim > 1 else int(filtered_position[1])
                
                # Store positions for visualization
                self.raw_positions[hand_label].append((wrist_x, wrist_y))
                self.filtered_positions[hand_label].append((filtered_x, filtered_y))
                
                # Keep only last N positions
                if len(self.raw_positions[hand_label]) > self.max_positions:
                    self.raw_positions[hand_label].pop(0)
                    self.filtered_positions[hand_label].pop(0)
                
                # Get colors for this hand
                raw_color = self.hand_colors[hand_label]['raw']
                filtered_color = self.hand_colors[hand_label]['filtered']
                
                # Draw raw position
                cv2.circle(frame, (wrist_x, wrist_y), 5, raw_color, -1)
                
                # Draw filtered position
                cv2.circle(frame, (filtered_x, filtered_y), 5, filtered_color, -1)
                
                # Draw trails
                for i in range(1, len(self.raw_positions[hand_label])):
                    # Raw position trail
                    cv2.line(frame, self.raw_positions[hand_label][i-1], self.raw_positions[hand_label][i], raw_color, 2)
                    
                    # Filtered position trail
                    cv2.line(frame, self.filtered_positions[hand_label][i-1], self.filtered_positions[hand_label][i], filtered_color, 2)
                
                # Text position based on which hand (left hand text on left, right hand text on right)
                text_x = 10 if hand_label == 'left' else w - 300
                text_y_base = 30
                
                # Display gesture for this hand
                cv2.putText(frame, f"{hand_label.capitalize()} hand: {gesture}", 
                           (text_x, text_y_base), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, filtered_color, 2)
                
                # Display position info
                cv2.putText(frame, f"Raw: ({wrist_x}, {wrist_y})", 
                           (text_x, text_y_base + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, raw_color, 2)
                           
                cv2.putText(frame, f"Filtered: ({filtered_x}, {filtered_y})", 
                           (text_x, text_y_base + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, filtered_color, 2)
                
                # Update matplotlib visualization if provided
                if visualizer:
                    visualizer.add_data_point((wrist_x, wrist_y), (filtered_x, filtered_y), hand_label)
            
            # Add legend at the bottom
            legend_y = h - 60
            cv2.putText(frame, "Left hand: Red(raw)/Green(filtered)", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 153, 0), 2)
            cv2.putText(frame, "Right hand: Blue(raw)/Yellow(filtered)", (10, legend_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (153, 153, 0), 2)
        
        # Calculate FPS
        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.curr_frame_time
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def release(self):
        self.hands.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hand Recognition System')
    parser.add_argument('--visualize', action='store_true', help='Enable matplotlib visualization')
    parser.add_argument('--save-plot', action='store_true', help='Save visualization plot at the end')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    args = parser.parse_args()
    
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera with index {args.camera}")
        
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized with resolution: {frame_width}x{frame_height}")
    except Exception as e:
        print(f"Error opening camera: {e}")
        print("Please make sure your webcam is connected and not being used by another application.")
        return
    
    # Initialize GestureRecognizer
    recognizer = GestureRecognizer()
    
    # Initialize visualizer if requested
    visualizer = None
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            plt.ion()  # Enable interactive mode
            visualizer = TrackingVisualizer()
        except ImportError:
            print("Warning: Could not import matplotlib. Visualization will be disabled.")
            args.visualize = False
            args.save_plot = False
    
    print("Starting hand recognition system...")
    print("Press 'q' to quit")
    
    try:
        while cap.isOpened():
            # Read frame from webcam
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break
            
            # Process frame
            processed_frame = recognizer.process_frame(frame, visualizer)
            
            # Display the frame
            cv2.imshow('Hand Recognition System', processed_frame)
            
            # If visualization is active, update the plot
            if visualizer:
                plt.pause(0.001)  # Small pause to update the plot
            
            # Check for key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Save visualization if requested
        if args.save_plot and visualizer:
            try:
                visualizer.save_plot()
            except Exception as e:
                print(f"Error saving plot: {e}")
        
        # Release resources
        print("Cleaning up resources...")
        recognizer.release()
        cap.release()
        cv2.destroyAllWindows()
        
        # Show the final plot if visualization was enabled
        if args.visualize and visualizer:
            try:
                plt.ioff()  # Turn off interactive mode
                visualizer.show_plot()
            except Exception as e:
                print(f"Error displaying plot: {e}")


if __name__ == "__main__":
    main() 