#!/usr/bin/env python3
"""
NDI-OSC integration for StreamDiffusion on macOS.
Receives video from NDI sources, applies AI image generation based on prompts received via OSC,
and outputs the processed video as an NDI source.
"""

import os
import sys
import time
import threading
import argparse
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
import numpy as np
import fire
import tkinter as tk
import cv2
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer, ThreadingOSCUDPServer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from examples.ndi.fixed_wrapper import FixedStreamDiffusionWrapper

# Define NDISimulator class globally
class NDISimulator:
    def __init__(self):
        print("NDI SDK not found - using webcam simulator instead")
    
    def initialize(self):
        return True
    
    def terminate(self):
        pass
    
    def destroy(self):
        pass
    
    class Finder:
        def wait_for_sources(self, timeout):
            time.sleep(timeout / 1000)
            
        def get_sources(self):
            return [type('NDISource', (), {'name': 'Webcam0'})]

# Try to import NDI module from Python 3.10 environment
try:
    # First try importing the NDI Python binding from Python 3.10 environment
    import NDIlib as ndi
    print("Using NDIlib for NDI functionality")
    using_simulator = False
except ImportError:
    # Create a simple fallback
    ndi = NDISimulator()
    using_simulator = True
    print("NDI SDK not found - using webcam simulator instead")

# Global variables
inputs = []
current_prompt = "cyberpunk neon city, vaporwave aesthetic, vibrant colors"
prompt_lock = threading.Lock()

def webcam_receiver(
    event: threading.Event,
    height: int = 512,
    width: int = 512,
):
    """Webcam fallback receiver"""
    global inputs
    
    try:
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            event.set()
            return
            
        while not event.is_set():
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert from BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and resize
            img = PIL.Image.fromarray(frame_rgb)
            img = img.resize((width, height))
            
            # Add to inputs queue
            inputs.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1))
            
            # Sleep briefly to control frame rate
            time.sleep(0.03)  # ~30fps
    except Exception as e:
        print(f"Error in webcam receiver: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print('Exited webcam receiver')

def ndi_receiver(
    event: threading.Event,
    height: int = 512,
    width: int = 512,
    ndi_source_name: Optional[str] = None,
):
    """NDI receiver thread"""
    global inputs
    
    # Initialize NDI
    try:
        if not ndi.initialize():
            print("Failed to initialize NDI")
            event.set()
            return webcam_receiver(event, height, width)
        
        # Find NDI sources
        sources = []
        if using_simulator:
            finder = ndi.Finder()
            finder.wait_for_sources(5000) 
            sources = finder.get_sources()
        else:
            finder = ndi.find_create_v2()
            if finder:
                ndi.find_wait_for_sources(finder, 5000)
                sources = ndi.find_get_current_sources(finder)
                
        if not sources:
            print("No NDI sources found, falling back to webcam")
            if finder and not using_simulator:
                ndi.find_destroy(finder)
            return webcam_receiver(event, height, width)
        
        # Select NDI source
        selected_source = None
        if ndi_source_name:
            for source in sources:
                source_name = None
                if hasattr(source, 'name'):
                    source_name = source.name
                elif hasattr(source, 'ndi_name'):
                    source_name = source.ndi_name
                else:
                    source_name = str(source)
                
                if ndi_source_name in source_name:
                    selected_source = source
                    break
        
        if not selected_source:
            selected_source = sources[0]
            source_name = None
            if hasattr(selected_source, 'name'):
                source_name = selected_source.name
            elif hasattr(selected_source, 'ndi_name'):
                source_name = selected_source.ndi_name
            else:
                source_name = str(selected_source)
            print(f"Using NDI source: {source_name}")
        
        # Create receiver
        receiver = None
        if using_simulator:
            receiver = ndi.Receiver()
            receiver.connect(selected_source)
        else:
            if hasattr(ndi, 'recv_create_v3'):
                settings = ndi.RecvCreateV3()
                if hasattr(settings, 'color_format'):
                    settings.color_format = ndi.RECV_COLOR_FORMAT_RGBX_RGBA
                if hasattr(settings, 'source_to_connect_to'):
                    settings.source_to_connect_to = selected_source
                receiver = ndi.recv_create_v3(settings)
                if receiver:
                    ndi.recv_connect(receiver, selected_source)
            
        if not receiver:
            print("Failed to create NDI receiver, falling back to webcam")
            if finder and not using_simulator:
                ndi.find_destroy(finder)
            return webcam_receiver(event, height, width)
        
        # Main loop to receive frames
        while not event.is_set():
            # Receive video frame
            frame = None
            if using_simulator:
                success, frame = receiver.capture_video(1000)
                if not success:
                    continue
            else:
                frame_type, video_frame, _, _ = ndi.recv_capture_v2(receiver, 1000)
                if frame_type == ndi.FRAME_TYPE_VIDEO:
                    frame = video_frame
                if frame is None:
                    continue
            
            # Convert NDI frame to PIL Image
            try:
                # Extract frame data
                img_array = None
                if hasattr(frame, 'data'):
                    img_array = np.array(frame.data)
                else:
                    img_array = np.array(frame)
                
                # Convert to PIL Image and resize
                img = PIL.Image.fromarray(img_array).convert('RGB')
                img = img.resize((width, height))
                
                # Add to inputs queue
                inputs.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1))
            except Exception as e:
                print(f"Error processing NDI frame: {e}")
                continue
        
        # Clean up
        if using_simulator:
            receiver.disconnect()
        else:
            if finder:
                ndi.find_destroy(finder)
            if receiver:
                ndi.recv_destroy(receiver)
        
        print('Exited NDI receiver')
    except Exception as e:
        print(f"Error in NDI receiver: {e}")
        return webcam_receiver(event, height, width)

def ndi_sender(
    event: threading.Event,
    output_queue: Queue,
    height: int = 512,
    width: int = 512,
    ndi_output_name: str = "StreamDiffusion Output",
):
    """NDI sender thread"""
    try:
        # Create a CV2 window as fallback display
        cv2.namedWindow(ndi_output_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ndi_output_name, width, height)
        
        # Create NDI sender if possible
        sender = None
        if not using_simulator:
            if hasattr(ndi, 'send_create'):
                sender_config = ndi.SendCreate()
                if hasattr(sender_config, 'p_ndi_name'):
                    sender_config.p_ndi_name = ndi_output_name
                sender = ndi.send_create(sender_config)
                if sender:
                    print(f"Created NDI output: {ndi_output_name}")
        
        while not event.is_set():
            if not output_queue.empty():
                # Get the next frame from queue
                output_tensor = output_queue.get(block=False)
                
                # Convert tensor to numpy array
                output_array = (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                # Always show in OpenCV window
                cv2.imshow(ndi_output_name, cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                
                # Send via NDI if sender exists
                if sender:
                    try:
                        # Create video frame
                        if hasattr(ndi, 'VideoFrameV2'):
                            video_frame = ndi.VideoFrameV2()
                            video_frame.data = output_array
                            video_frame.width = width
                            video_frame.height = height
                            ndi.send_send_video_v2(sender, video_frame)
                    except Exception as e:
                        print(f"Error sending NDI frame: {e}")
            
            time.sleep(0.001)
    except Exception as e:
        print(f"Error in NDI sender: {e}")
    finally:
        cv2.destroyAllWindows()
        if sender and not using_simulator:
            ndi.send_destroy(sender)
        print('Exited NDI sender')

def handle_prompt(address, *args):
    """OSC handler for /prompt messages"""
    global current_prompt
    if len(args) > 0:
        new_prompt = str(args[0])
        with prompt_lock:
            current_prompt = new_prompt
        print(f"Received new prompt via OSC: {new_prompt}")

def handle_negative_prompt(address, *args):
    """OSC handler for /negative_prompt messages"""
    global current_negative_prompt
    if len(args) > 0:
        new_negative_prompt = str(args[0])
        with prompt_lock:
            current_negative_prompt = new_negative_prompt
        print(f"Received new negative prompt via OSC: {new_negative_prompt}")

def handle_guidance_scale(address, *args):
    """OSC handler for /guidance_scale messages"""
    global current_guidance_scale
    if len(args) > 0:
        try:
            new_guidance_scale = float(args[0])
            with prompt_lock:
                current_guidance_scale = new_guidance_scale
            print(f"Received new guidance scale via OSC: {new_guidance_scale}")
        except (ValueError, TypeError):
            print(f"Invalid guidance scale value: {args[0]}")

def start_osc_server(ip="127.0.0.1", port=9001):
    """Start OSC server to receive prompt updates"""
    dispatcher = Dispatcher()
    dispatcher.map("/prompt", handle_prompt)
    dispatcher.map("/negative_prompt", handle_negative_prompt)
    dispatcher.map("/guidance_scale", handle_guidance_scale)
    
    # Non-blocking server
    server = ThreadingOSCUDPServer((ip, port), dispatcher)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    print(f"OSC server listening on {ip}:{port}")
    print(f"Available OSC commands:")
    print(f"  /prompt <text>            - Set new prompt")
    print(f"  /negative_prompt <text>   - Set new negative prompt")
    print(f"  /guidance_scale <float>   - Set new guidance scale (typically 1.0-7.0)")
    
    return server, server_thread

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    output_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    ndi_source_name: Optional[str] = None,
    osc_ip: str = "127.0.0.1",
    osc_port: int = 9001,
) -> None:
    """Image generation process using fixed StreamDiffusion wrapper"""
    global inputs, current_prompt, current_negative_prompt, current_guidance_scale
    
    # Initialize global variables for OSC control
    current_prompt = prompt
    current_negative_prompt = negative_prompt
    current_guidance_scale = guidance_scale
    
    # Start OSC server
    osc_server, osc_thread = start_osc_server(osc_ip, osc_port)
    
    # Use the fixed wrapper that doesn't have t_index_list issues
    stream = FixedStreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],  # Fixed t_index_list
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    # Prepare the model with our fixed wrapper - no t_index_list parameter
    stream.prepare(
        prompt=current_prompt,
        negative_prompt=current_negative_prompt,
        num_inference_steps=50,
        guidance_scale=current_guidance_scale,
        delta=delta
    )

    # Create input and output threads
    event = threading.Event()
    input_thread = threading.Thread(target=ndi_receiver, args=(event, height, width, ndi_source_name))
    output_thread = threading.Thread(target=ndi_sender, args=(event, output_queue, height, width))
    
    input_thread.start()
    output_thread.start()
    
    print("Starting StreamDiffusion processing...")
    print(f"Initial prompt: {current_prompt}")
    
    fps = 0
    last_prompt_update = ""
    last_negative_prompt_update = ""
    last_guidance_scale_update = 0.0
    
    try:
        while True:
            if not close_queue.empty():  # closing check
                break
                
            # Check if prompt needs updating
            with prompt_lock:
                # Only update if there's a change
                if current_prompt != last_prompt_update or \
                   current_negative_prompt != last_negative_prompt_update or \
                   current_guidance_scale != last_guidance_scale_update:
                    print(f"Updating prompt to: {current_prompt}")
                    try:
                        # Save current values to track changes
                        last_prompt_update = current_prompt
                        last_negative_prompt_update = current_negative_prompt
                        last_guidance_scale_update = current_guidance_scale
                        
                        # Update the model with the new prompt
                        stream.prepare(
                            prompt=current_prompt,
                            negative_prompt=current_negative_prompt,
                            num_inference_steps=50,
                            guidance_scale=current_guidance_scale,
                            delta=delta
                        )
                    except Exception as e:
                        print(f"Error updating prompt: {e}")
            
            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
                
            start_time = time.time()
            sampled_inputs = []
            
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
                
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu()
            
            if frame_buffer_size == 1:
                output_images = [output_images]
                
            for output_image in output_images:
                queue.put(output_image, block=False)  # For display
                output_queue.put(output_image, block=False)  # For NDI output

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing image generation process...")
        event.set()  # stop capture threads
        input_thread.join()
        output_thread.join()
        
        # Stop OSC server
        if hasattr(osc_server, 'shutdown') and callable(osc_server.shutdown):
            osc_server.shutdown()
        if osc_thread.is_alive():
            osc_thread.join(timeout=1.0)
        
        print(f"Final fps: {fps}")

def main(
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "cyberpunk neon city, vaporwave aesthetic, vibrant colors",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
    ndi_source_name: Optional[str] = None,
    ndi_output_name: str = "StreamDiffusion Output",
    osc_ip: str = "127.0.0.1",
    osc_port: int = 9001,
) -> None:
    """
    Main function to start the stream processing with StreamDiffusion and OSC control.
    
    In addition to the regular NDI parameters, this version adds:
    
    Parameters
    ----------
    osc_ip : str
        IP address for the OSC server to listen on, default is 127.0.0.1 (localhost)
    osc_port : int
        Port for the OSC server to listen on, default is 9001
    
    OSC Commands:
    - /prompt <text>          : Set a new prompt for image generation
    - /negative_prompt <text> : Set a new negative prompt 
    - /guidance_scale <float> : Set a new guidance scale value
    """
    # Show NDI sources if available
    if not using_simulator:
        if ndi.initialize():
            finder = ndi.find_create_v2()
            if finder:
                ndi.find_wait_for_sources(finder, 2000)
                sources = ndi.find_get_current_sources(finder)
                
                print("Available NDI sources:")
                for i, source in enumerate(sources):
                    source_name = None
                    if hasattr(source, 'name'):
                        source_name = source.name
                    elif hasattr(source, 'ndi_name'):
                        source_name = source.ndi_name
                    else:
                        source_name = str(source)
                    print(f"{i}: {source_name}")
                
                ndi.find_destroy(finder)
                ndi.destroy()
    
    # Set up multiprocessing
    ctx = get_context('spawn')
    queue = ctx.Queue()  # For display
    fps_queue = ctx.Queue()
    close_queue = Queue()
    output_queue = ctx.Queue()  # For NDI output

    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            output_queue,
            model_id_or_path,
            lora_dict,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            ndi_source_name,
            osc_ip,
            osc_port,
        ),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # Wait for viewer to close
    process2.join()
    print("Viewer closed. Terminating processing...")
    close_queue.put(True)
    process1.join(5)  # with timeout
    if process1.is_alive():
        print("Process still alive. Force killing...")
        process1.terminate()
    process1.join()
    print("All processes terminated.")

if __name__ == "__main__":
    fire.Fire(main)