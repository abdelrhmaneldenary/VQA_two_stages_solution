import os
import gc
import torch
import numpy as np
from PIL import Image

# 1. The Correct Official SAM 3 Imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class Stage2Segmenter:
    def __init__(self, model_id):
        local_path = os.path.abspath(model_id) if os.path.exists(model_id) else model_id
        print(f"🚀 Loading Stage 2 (Native SAM 3): {local_path}")
        
        # Find the checkpoint file (.pt or .pth) in the directory
        weight_file = None
        if os.path.isdir(local_path):
            for file in os.listdir(local_path):
                if file.endswith(('.pt', '.pth')):
                    weight_file = os.path.join(local_path, file)
                    break
        else:
            weight_file = local_path
            
        if not weight_file:
             raise FileNotFoundError(f"Could not find a .pt/.pth file in {local_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # TF32 Optimizations for modern GPUs
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 2. Load the native model using the official SAM 3 Builder
        self.model = build_sam3_image_model(
            checkpoint_path=weight_file, 
            load_from_HF=False, 
            device=device
        )
        
        # Use bfloat16 to save memory
        if device == "cuda":
            self.model.bfloat16()
            
        # 3. Use the official SAM 3 Image Processor
        self.processor = Sam3Processor(self.model)

    def _to_image(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert("RGB")
        return Image.open(image_or_path).convert("RGB")

    def generate_masks(self, image_or_path, labels):
        image = self._to_image(image_or_path)
        
        masks_out = []
        scores_out = []

        try:
            # Process the image ONCE (Huge speedup over a loop)
            inference_state = self.processor.set_image(image)

            # Enable mixed precision for inference
            with torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad():
                for label in labels:
                    try:
                        # NATIVE INFERENCE: Pass the text prompt
                        output = self.processor.set_text_prompt(
                            state=inference_state,
                            prompt=label
                        )
                        
                        masks = output["masks"]
                        scores = output["scores"]
                        
                        if len(masks) > 0:
                            # Convert to numpy
                            if torch.is_tensor(scores):
                                scores = scores.cpu().numpy()
                            if torch.is_tensor(masks):
                                masks = masks.cpu().numpy()
                                
                            best_idx = np.argmax(scores)
                            # SAM 3 usually returns logits; threshold at 0.0 to get binary mask
                            mask = (masks[best_idx] > 0.0)
                            score = float(scores[best_idx])
                            
                            masks_out.append(mask)
                            scores_out.append(score)
                        else:
                            masks_out.append(np.zeros((image.height, image.width), dtype=bool))
                            scores_out.append(0.0)

                    except Exception as e:
                        print(f"⚠️ Stage 2 Prediction Error on label '{label}': {e}")
                        masks_out.append(np.zeros((image.height, image.width), dtype=bool))
                        scores_out.append(0.0)

        except Exception as e:
            print(f"⚠️ Stage 2 Setup Error for image: {e}")
            for _ in labels:
                 masks_out.append(np.zeros((image.height, image.width), dtype=bool))
                 scores_out.append(0.0)

        finally:
            torch.cuda.empty_cache()
            gc.collect()

        return masks_out, scores_out