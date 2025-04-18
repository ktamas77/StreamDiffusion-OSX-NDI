import os
import sys
from typing import List, Optional, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.wrapper import StreamDiffusionWrapper

# Fix the prepare method that causes issues with t_index_list
class FixedStreamDiffusionWrapper(StreamDiffusionWrapper):
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference - fixed version that doesn't pass t_index_list
        which is already set in the StreamDiffusion constructor.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        negative_prompt : str
            The negative prompt to use.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        # Don't pass t_index_list parameter that's already set in StreamDiffusionWrapper
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta
        )