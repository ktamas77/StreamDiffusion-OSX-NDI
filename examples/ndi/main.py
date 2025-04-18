import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
import numpy as np
from streamdiffusion.image_utils import pil2tensor
import fire
import tkinter as tk
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

# Global variables for input/output frames
inputs = []
ndi_output_frame = None

class NDISimulator:
    """
    Simulates NDI functionality for testing when NDI is not available.
    Uses webcam as input source and simply outputs to screen.
    """
    def __init__(self):
        print("NDI SDK not found - using webcam simulator instead")
    
    def initialize(self):
        pass
    
    def terminate(self):
        pass
    
    class Finder:
        def wait_for_sources(self, timeout):
            pass
        
        def get_sources(self):
            return [type('NDISource', (), {'name': 'Webcam0'})]

# Try to import NDI module
try:
    # First try importing the most common NDI Python binding
    import NDIlib as ndi
    print("Using NDIlib for NDI functionality")
except ImportError:
    try:
        # Try another common binding
        import PyNDI as ndi
        print("Using PyNDI for NDI functionality")
    except ImportError:
        # Fall back to webcam simulation
        ndi = NDISimulator()
        print("NDI SDK not found - using webcam simulator instead")

def webcam_receiver(
    event: threading.Event,
    height: int = 512,
    width: int = 512,
):
    """
    Thread function to receive webcam video as a fallback when NDI is not available.
    
    Parameters
    ----------
    event : threading.Event
        Event to signal thread termination.
    height : int
        Height to resize input images to.
    width : int
        Width to resize input images to.
    """
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
            inputs.append(pil2tensor(img))
            
            # Sleep briefly to control frame rate
            time.sleep(0.03)  # ~30fps
    except Exception as e:
        print(f"Error in webcam receiver: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print('Exited: webcam receiver')

def mock_ndi_receiver(
    event: threading.Event,
    height: int = 512,
    width: int = 512,
    ndi_source_name: Optional[str] = None,
):
    """
    Thread function that simulates receiving NDI video using webcam when NDI is not available.
    
    Parameters
    ----------
    event : threading.Event
        Event to signal thread termination.
    height : int
        Height to resize input images to.
    width : int
        Width to resize input images to.
    ndi_source_name : Optional[str]
        Ignored in simulation mode.
    """
    webcam_receiver(event, height, width)

def ndi_receiver(
    event: threading.Event,
    height: int = 512,
    width: int = 512,
    ndi_source_name: Optional[str] = None,
):
    """
    Thread function to receive NDI video stream and prepare input images for processing.
    Falls back to webcam if NDI is not available.
    
    Parameters
    ----------
    event : threading.Event
        Event to signal thread termination.
    height : int
        Height to resize input images to.
    width : int
        Width to resize input images to.
    ndi_source_name : Optional[str]
        Name of the NDI source to connect to. If None, will use the first available source.
    """
    # Check if we're using the simulator
    if isinstance(ndi, NDISimulator):
        return mock_ndi_receiver(event, height, width, ndi_source_name)
    
    global inputs
    
    # Initialize NDI - implementation varies by library
    try:
        # NDIlib style initialization
        if hasattr(ndi, 'initialize') and callable(ndi.initialize):
            if not ndi.initialize():
                print("Failed to initialize NDI")
                event.set()
                return
        
        # Find NDI sources - implementation varies by library
        sources = []
        if hasattr(ndi, 'find_sources'):
            # Some libraries have a direct find_sources function
            sources = ndi.find_sources(5000)  # 5 second timeout
        elif hasattr(ndi, 'Finder'):
            # NDIlib style finder
            finder = ndi.Finder()
            if hasattr(finder, 'wait_for_sources'):
                finder.wait_for_sources(5000)
            sources = finder.get_sources()
        
        if not sources:
            print("No NDI sources found, falling back to webcam")
            return mock_ndi_receiver(event, height, width)
        
        # Select NDI source
        selected_source = None
        if ndi_source_name:
            for source in sources:
                source_name = source.name if hasattr(source, 'name') else str(source)
                if ndi_source_name in source_name:
                    selected_source = source
                    break
        
        if not selected_source:
            selected_source = sources[0]
            source_name = selected_source.name if hasattr(selected_source, 'name') else str(selected_source)
            print(f"Using NDI source: {source_name}")
        
        # Create receiver - implementation varies by library
        receiver = None
        if hasattr(ndi, 'Receiver'):
            receiver = ndi.Receiver()
            if hasattr(receiver, 'connect'):
                receiver.connect(selected_source)
        elif hasattr(ndi, 'create_receiver'):
            receiver = ndi.create_receiver(selected_source)
        
        if not receiver:
            print("Failed to create NDI receiver, falling back to webcam")
            return mock_ndi_receiver(event, height, width)
        
        # Main loop to receive frames
        while not event.is_set():
            # Receive video frame - implementation varies by library
            frame = None
            if hasattr(receiver, 'capture_video'):
                _, frame = receiver.capture_video(5000)
            elif hasattr(receiver, 'receive_video'):
                frame = receiver.receive_video(5000)
            
            if frame is None:
                continue
            
            # Convert NDI frame to PIL Image - implementation varies by library
            try:
                # Extract frame data - varies by library
                if hasattr(frame, 'data'):
                    img_array = np.copy(frame.data)
                elif hasattr(frame, 'get_data'):
                    img_array = np.copy(frame.get_data())
                else:
                    img_array = np.copy(frame)
                
                # Process based on format - varies by library
                img = None
                if hasattr(frame, 'fourcc') and frame.fourcc == getattr(ndi, 'FOURCC_VIDEO_TYPE_UYVY', None):
                    # Convert UYVY to RGB (simplified)
                    if len(img_array.shape) == 3:
                        img = PIL.Image.fromarray(img_array).convert('RGB')
                    else:
                        img_array = img_array.reshape(frame.height, frame.width, -1)
                        img = PIL.Image.fromarray(img_array).convert('RGB')
                else:
                    # Assume RGB/BGR format
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img = PIL.Image.fromarray(img_array)
                    else:
                        img = PIL.Image.fromarray(img_array).convert('RGB')
                
                # Resize to target dimensions
                img = img.resize((width, height))
                
                # Add to inputs
                inputs.append(pil2tensor(img))
            except Exception as e:
                print(f"Error processing NDI frame: {e}")
                continue
            
    except Exception as e:
        print(f"Error in NDI receiver: {e}")
        # Fall back to webcam
        return mock_ndi_receiver(event, height, width)
    finally:
        # Cleanup
        if 'receiver' in locals() and receiver:
            if hasattr(receiver, 'disconnect'):
                receiver.disconnect()
            elif hasattr(receiver, 'destroy'):
                receiver.destroy()
        
        if hasattr(ndi, 'terminate') and callable(ndi.terminate):
            ndi.terminate()
        
        print('Exited: NDI receiver')

def fake_ndi_sender(
    event: threading.Event,
    output_queue: Queue,
    height: int = 512,
    width: int = 512,
    ndi_output_name: str = "StreamDiffusion Output",
):
    """
    Thread function that simulates sending NDI stream when NDI is not available.
    Simply does nothing but consume frames from the queue.
    
    Parameters
    ----------
    event : threading.Event
        Event to signal thread termination.
    output_queue : Queue
        Queue containing processed images to send.
    height : int
        Height of output images.
    width : int
        Width of output images.
    ndi_output_name : str
        Name for the NDI output stream.
    """
    print(f"NDI output would be named: {ndi_output_name}")
    
    try:
        while not event.is_set():
            if not output_queue.empty():
                # Just consume the frame
                _ = output_queue.get(block=False)
            time.sleep(0.001)
    except Exception as e:
        print(f"Error in fake NDI sender: {e}")
    finally:
        print('Exited: fake NDI sender')

def ndi_sender(
    event: threading.Event,
    output_queue: Queue,
    height: int = 512,
    width: int = 512,
    ndi_output_name: str = "StreamDiffusion Output",
):
    """
    Thread function to send processed images as an NDI stream.
    Falls back to a dummy consumer if NDI is not available.
    
    Parameters
    ----------
    event : threading.Event
        Event to signal thread termination.
    output_queue : Queue
        Queue containing processed images to send.
    height : int
        Height of output images.
    width : int
        Width of output images.
    ndi_output_name : str
        Name for the NDI output stream.
    """
    # Check if we're using the simulator
    if isinstance(ndi, NDISimulator):
        return fake_ndi_sender(event, output_queue, height, width, ndi_output_name)
    
    try:
        # Initialize NDI
        if hasattr(ndi, 'initialize') and callable(ndi.initialize):
            if not ndi.initialize():
                print("Failed to initialize NDI for sending")
                return fake_ndi_sender(event, output_queue, height, width, ndi_output_name)
        
        # Create a sender - implementation varies by library
        sender = None
        if hasattr(ndi, 'Sender'):
            sender = ndi.Sender(name=ndi_output_name)
        elif hasattr(ndi, 'create_sender'):
            sender = ndi.create_sender(ndi_output_name)
        
        if not sender:
            print("Failed to create NDI sender")
            return fake_ndi_sender(event, output_queue, height, width, ndi_output_name)
        
        # Create video frame structure - implementation varies by library
        print(f"Started NDI output: {ndi_output_name}")
        
        while not event.is_set():
            if not output_queue.empty():
                # Get the next frame from queue
                output_tensor = output_queue.get(block=False)
                
                # Convert tensor to numpy array
                pil_image = PIL.Image.fromarray(
                    (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                )
                
                # Convert to format needed by NDI - implementation varies by library
                numpy_frame = np.array(pil_image.convert("RGBA" if hasattr(ndi, 'FOURCC_VIDEO_TYPE_BGRA') else "RGB"))
                
                # Send the frame - implementation varies by library
                if hasattr(sender, 'send_video'):
                    # Create video frame structure if needed
                    if 'video_frame' not in locals():
                        video_frame = ndi.VideoFrameV2() if hasattr(ndi, 'VideoFrameV2') else type('VideoFrame', (), {})
                        if hasattr(video_frame, 'fourcc') and hasattr(ndi, 'FOURCC_VIDEO_TYPE_BGRA'):
                            video_frame.fourcc = ndi.FOURCC_VIDEO_TYPE_BGRA
                        video_frame.width = width
                        video_frame.height = height
                    
                    # Update frame data
                    video_frame.data = numpy_frame
                    
                    # Send frame
                    sender.send_video(video_frame)
                elif hasattr(sender, 'send_frame'):
                    # Alternative send method
                    sender.send_frame(numpy_frame, width, height)
                
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
    except Exception as e:
        print(f"Error in NDI sender: {e}")
        return fake_ndi_sender(event, output_queue, height, width, ndi_output_name)
    finally:
        # Cleanup
        if 'sender' in locals() and sender:
            if hasattr(sender, 'destroy'):
                sender.destroy()
        
        if hasattr(ndi, 'terminate') and callable(ndi.terminate):
            ndi.terminate()
        
        print('Exited: NDI sender')

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
) -> None:
    """
    Process for generating images based on NDI input stream using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in for display.
    fps_queue : Queue
        The queue to put the calculated fps.
    close_queue : Queue
        Queue to monitor for termination signals.
    output_queue : Queue
        Queue to put generated images for NDI output.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
    prompt : str
        The prompt to generate images from.
    negative_prompt : str
        The negative prompt to use.
    frame_buffer_size : int
        The frame buffer size for denoising batch.
    width : int
        The width of the image.
    height : int
        The height of the image.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The acceleration method.
    use_denoising_batch : bool
        Whether to use denoising batch or not.
    seed : int
        The seed. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"]
        The cfg_type for img2img mode.
    guidance_scale : float
        The CFG scale.
    delta : float
        The delta multiplier of virtual residual noise.
    do_add_noise : bool
        Whether to add noise for following denoising steps or not.
    enable_similar_image_filter : bool
        Whether to enable similar image filter or not.
    similar_image_filter_threshold : float
        The threshold for similar image filter.
    similar_image_filter_max_skip_frame : int
        The max skip frame for similar image filter.
    ndi_source_name : Optional[str]
        Name of the NDI source to connect to. If None, will use the first available source.
    """
    
    global inputs
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],
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

    # Don't pass t_index_list parameter that's already set in StreamDiffusionWrapper
    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta
    )

    event = threading.Event()
    input_thread = threading.Thread(target=ndi_receiver, args=(event, height, width, ndi_source_name))
    input_thread.start()
    
    # Create NDI output sender thread
    output_thread = threading.Thread(target=ndi_sender, args=(event, output_queue, height, width))
    output_thread.start()
    
    print("Starting StreamDiffusion processing...")
    time.sleep(2)  # Give time for connections to establish

    while True:
        try:
            if not close_queue.empty():  # closing check
                break
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
            break

    print("Closing image_generation_process...")
    event.set()  # stop capture threads
    input_thread.join()
    output_thread.join()
    print(f"Final fps: {fps}")

def main(
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
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
) -> None:
    """
    Main function to start the stream processing with StreamDiffusion.
    Falls back to webcam if NDI is not available.
    
    Parameters
    ----------
    model_id_or_path : str
        The model to use for diffusion.
    lora_dict : Optional[Dict[str, float]]
        Dictionary of LoRA models to apply with their weights.
    prompt : str
        The prompt to guide the diffusion.
    negative_prompt : str
        Negative prompt to avoid certain characteristics.
    frame_buffer_size : int
        Number of frames to buffer for processing.
    width, height : int
        Dimensions for processing.
    acceleration : str
        Acceleration method to use.
    use_denoising_batch : bool
        Whether to use denoising batches.
    seed : int
        Seed for random generation.
    cfg_type : str
        Configuration for guidance.
    guidance_scale : float
        Scale for classifier-free guidance.
    delta : float
        Delta for noise scheduling.
    do_add_noise : bool
        Whether to add noise during processing.
    enable_similar_image_filter : bool
        Whether to filter similar consecutive frames.
    similar_image_filter_threshold : float
        Threshold for similarity detection.
    similar_image_filter_max_skip_frame : float
        Maximum frames to skip when similar.
    ndi_source_name : Optional[str]
        Name of NDI source to connect to (uses first available if None).
    ndi_output_name : str
        Name for the NDI output stream.
    """
    # Try to check for NDI sources
    try:
        if not isinstance(ndi, NDISimulator):
            if hasattr(ndi, 'initialize') and callable(ndi.initialize):
                ndi.initialize()
            
            sources = []
            if hasattr(ndi, 'Finder'):
                finder = ndi.Finder()
                if hasattr(finder, 'wait_for_sources'):
                    finder.wait_for_sources(2000)
                sources = finder.get_sources()
            elif hasattr(ndi, 'find_sources'):
                sources = ndi.find_sources(2000)
            
            print("Available NDI sources:")
            for i, source in enumerate(sources):
                source_name = source.name if hasattr(source, 'name') else str(source)
                print(f"{i}: {source_name}")
            
            if hasattr(ndi, 'terminate') and callable(ndi.terminate):
                ndi.terminate()
        else:
            print("Using webcam as input source (NDI SDK not available)")
    except Exception as e:
        print(f"Error listing NDI sources: {e}")
        print("Will use webcam as fallback")

    # Try to install OpenCV if needed for the webcam fallback
    try:
        import cv2
    except ImportError:
        print("Installing OpenCV for webcam support...")
        os.system("pip install opencv-python")
        try:
            import cv2
            print("OpenCV installed successfully")
        except ImportError:
            print("Failed to install OpenCV. Continuing anyway...")
    
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
            ),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # terminate
    process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5)  # with timeout
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate()  # force kill...
    process1.join()
    print("process1 terminated.")

if __name__ == "__main__":
    fire.Fire(main)