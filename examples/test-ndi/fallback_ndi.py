#!/usr/bin/env python3
"""
Fallback NDI module that simulates NDI functionality for StreamDiffusion examples.
This allows the NDI examples to run even without the NDI SDK installed.
"""

import cv2
import numpy as np
import time
import threading
from PIL import Image

class NDISimulator:
    """
    Simulates NDI functionality for testing when NDI is not available.
    Uses webcam as input source and simply outputs to screen.
    """
    def __init__(self):
        print("NDI SDK not found - using webcam simulator instead")
    
    def initialize(self):
        print("Initializing NDI simulator")
        return True
    
    def terminate(self):
        print("Terminating NDI simulator")
    
    class Finder:
        def wait_for_sources(self, timeout):
            print(f"Waiting for sources with timeout {timeout}ms")
            time.sleep(timeout / 1000)  # Convert ms to seconds
            
        def get_sources(self):
            print("Returning simulated webcam source")
            return [type('NDISource', (), {'name': 'Webcam0'})]
    
    class Receiver:
        def __init__(self):
            self.cap = None
            
        def connect(self, source):
            print(f"Connecting to source: {source.name}")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            return True
            
        def disconnect(self):
            if self.cap and self.cap.isOpened():
                self.cap.release()
            print("Disconnected from source")
            
        def capture_video(self, timeout_ms):
            if not self.cap or not self.cap.isOpened():
                return False, None
                
            ret, frame = self.cap.read()
            if not ret:
                return False, None
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a custom frame-like object
            video_frame = type('VideoFrame', (), {
                'width': frame_rgb.shape[1],
                'height': frame_rgb.shape[0],
                'data': frame_rgb,
                'fourcc': None
            })
            
            return True, video_frame
    
    class Sender:
        def __init__(self, name="NDI Simulator Output"):
            self.name = name
            self.window_name = name
            print(f"Created NDI simulator sender: {name}")
            
        def send_video(self, frame):
            # Display the frame in a window
            img_array = frame.data
            if isinstance(img_array, np.ndarray):
                # Convert RGB to BGR for OpenCV display
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, img_bgr)
                cv2.waitKey(1)
            else:
                print("Warning: Invalid frame data format")
                
        def send_frame(self, frame_data, width, height):
            # Alternative send method
            if isinstance(frame_data, np.ndarray):
                img_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, img_bgr)
                cv2.waitKey(1)
            else:
                print("Warning: Invalid frame data format")
                
        def destroy(self):
            cv2.destroyWindow(self.window_name)
            print(f"Destroyed NDI simulator sender: {self.name}")

# Make the module act like a proper NDI module
initialize = NDISimulator().initialize
terminate = NDISimulator().terminate
Finder = NDISimulator.Finder
Receiver = NDISimulator.Receiver
Sender = NDISimulator.Sender

# Constants that might be used
FOURCC_VIDEO_TYPE_RGBA = 1
FOURCC_VIDEO_TYPE_BGRA = 2
FOURCC_VIDEO_TYPE_UYVY = 3

if __name__ == "__main__":
    print("Testing NDI simulator...")
    
    # Initialize NDI
    if not initialize():
        print("Failed to initialize NDI simulator")
        exit(1)
    
    # Find sources
    finder = Finder()
    finder.wait_for_sources(1000)
    sources = finder.get_sources()
    
    if not sources:
        print("No sources found")
        terminate()
        exit(1)
    
    # Get the first source
    receiver = Receiver()
    if not receiver.connect(sources[0]):
        print("Failed to connect to source")
        terminate()
        exit(1)
    
    # Create a sender
    sender = Sender("Test Output")
    
    # Process 100 frames (or about 5 seconds)
    print("Processing frames for 5 seconds...")
    for i in range(100):
        success, frame = receiver.capture_video(1000)
        if success:
            sender.send_video(frame)
        time.sleep(0.05)
    
    # Clean up
    receiver.disconnect()
    sender.destroy()
    terminate()
    
    print("NDI simulator test completed successfully")