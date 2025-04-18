#!/usr/bin/env python3
"""
NDI driver for StreamDiffusion on macOS.
This script provides a common interface for NDI functionality,
whether using the ndi-python bindings or the fallback simulator.
"""

import sys
import os
import importlib

# First, try to import the NDI Python bindings
try:
    import NDIlib as ndi
    print("Using actual NDI Python bindings")
    HAS_REAL_NDI = True
except ImportError:
    # Fall back to the simulator
    print("NDI Python bindings not found, using simulator")
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    import fallback_ndi as ndi
    HAS_REAL_NDI = False

# Test if NDI works
if not ndi.initialize():
    print("ERROR: Failed to initialize NDI")
    sys.exit(1)

class NDIHandler:
    """
    Wrapper for NDI functionality that works with both 
    real NDI bindings and the simulator.
    """
    
    def __init__(self):
        self.initialized = True
        self.finder = None
        self.sources = []
        self.receivers = {}
        self.senders = {}
    
    def find_sources(self, timeout_ms=1000):
        """Find available NDI sources."""
        if HAS_REAL_NDI:
            # Use the real NDI bindings
            if hasattr(ndi, 'find_create_v2'):
                self.finder = ndi.find_create_v2()
                if self.finder:
                    ndi.find_wait_for_sources(self.finder, timeout_ms)
                    self.sources = ndi.find_get_current_sources(self.finder)
                    return self.sources
            return []
        else:
            # Use the simulator
            finder = ndi.Finder()
            finder.wait_for_sources(timeout_ms)
            self.sources = finder.get_sources()
            return self.sources
    
    def create_receiver(self, source):
        """Create a receiver for an NDI source."""
        if HAS_REAL_NDI:
            # Use the real NDI bindings
            if hasattr(ndi, 'recv_create_v3'):
                receiver_id = f"receiver_{len(self.receivers)}"
                settings = ndi.RecvCreateV3()
                settings.color_format = ndi.RECV_COLOR_FORMAT_RGBX_RGBA
                settings.source_to_connect_to = source
                receiver = ndi.recv_create_v3(settings)
                if receiver:
                    self.receivers[receiver_id] = receiver
                    ndi.recv_connect(receiver, source)
                    return receiver_id
            return None
        else:
            # Use the simulator
            receiver_id = f"receiver_{len(self.receivers)}"
            receiver = ndi.Receiver()
            receiver.connect(source)
            self.receivers[receiver_id] = receiver
            return receiver_id
    
    def create_sender(self, name="StreamDiffusion Output"):
        """Create an NDI sender."""
        if HAS_REAL_NDI:
            # Use the real NDI bindings
            if hasattr(ndi, 'send_create'):
                sender_id = f"sender_{len(self.senders)}"
                settings = ndi.SendCreate()
                settings.p_ndi_name = name
                sender = ndi.send_create(settings)
                if sender:
                    self.senders[sender_id] = sender
                    return sender_id
            return None
        else:
            # Use the simulator
            sender_id = f"sender_{len(self.senders)}"
            sender = ndi.Sender(name)
            self.senders[sender_id] = sender
            return sender_id
    
    def receive_frame(self, receiver_id, timeout_ms=1000):
        """Receive a frame from an NDI source."""
        if receiver_id not in self.receivers:
            return None
        
        receiver = self.receivers[receiver_id]
        
        if HAS_REAL_NDI:
            # Use the real NDI bindings
            if hasattr(ndi, 'recv_capture_v2'):
                frame_type, video_frame, audio_frame, metadata_frame = ndi.recv_capture_v2(
                    receiver, timeout_ms
                )
                if frame_type == ndi.FRAME_TYPE_VIDEO:
                    # Process the video frame
                    return video_frame
                return None
            return None
        else:
            # Use the simulator
            success, frame = receiver.capture_video(timeout_ms)
            if success:
                return frame
            return None
    
    def send_frame(self, sender_id, frame):
        """Send a frame to NDI."""
        if sender_id not in self.senders:
            return False
        
        sender = self.senders[sender_id]
        
        if HAS_REAL_NDI:
            # Use the real NDI bindings
            if hasattr(ndi, 'send_send_video_v2'):
                return ndi.send_send_video_v2(sender, frame)
            return False
        else:
            # Use the simulator
            sender.send_video(frame)
            return True
    
    def cleanup(self):
        """Clean up NDI resources."""
        # Clean up receivers
        for receiver_id, receiver in self.receivers.items():
            if HAS_REAL_NDI:
                if hasattr(ndi, 'recv_destroy'):
                    ndi.recv_destroy(receiver)
            else:
                receiver.disconnect()
        
        # Clean up senders
        for sender_id, sender in self.senders.items():
            if HAS_REAL_NDI:
                if hasattr(ndi, 'send_destroy'):
                    ndi.send_destroy(sender)
            else:
                sender.destroy()
        
        # Clean up finder
        if HAS_REAL_NDI and self.finder:
            if hasattr(ndi, 'find_destroy'):
                ndi.find_destroy(self.finder)
        
        # Clean up NDI
        if HAS_REAL_NDI:
            if hasattr(ndi, 'destroy'):
                ndi.destroy()
        else:
            ndi.terminate()
        
        self.initialized = False

# Example usage
if __name__ == "__main__":
    handler = NDIHandler()
    
    print("Finding NDI sources...")
    sources = handler.find_sources(1000)
    
    if sources:
        print(f"Found {len(sources)} NDI sources")
        
        # Create a receiver for the first source
        source = sources[0]
        print(f"Creating receiver for source: {source}")
        receiver_id = handler.create_receiver(source)
        
        if receiver_id:
            print("Receiving 10 frames...")
            # Create a sender
            sender_id = handler.create_sender("NDI Driver Test")
            
            # Receive and send 10 frames
            for i in range(10):
                frame = handler.receive_frame(receiver_id)
                if frame:
                    print(f"Received frame {i+1}")
                    if sender_id:
                        handler.send_frame(sender_id, frame)
                        print(f"Sent frame {i+1}")
    else:
        print("No NDI sources found")
    
    # Clean up
    handler.cleanup()
    print("NDI driver test completed")