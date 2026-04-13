import os
import gc
import torch
import numpy as np
from PIL import Image

# Import the native SAM 3 builder (assuming 'sam3' is installed in your Kaggle environment)
from sam3.build_sam3 import build_sam3
from sam3.sam3_image_predictor import SAM3ImagePredictor

class Stage2Segmenter:
    def __init__(self, model_id):
        local_path = os.path.abspath(model_id) if os.path.exists(model_id) else model_id
        print(f"🚀 Loading Stage 2 (Native SAM 3): {local_path}")
        
        # 1. Determine the config based on the folder name or contents
        # (Usually sam3_h.yaml, sam3_l.yaml, etc. Defaulting to huge 'h' if unknown)
        # Note: You may need to change this string if your config is different!
        config_file = "sam3_configs/sam3.yaml" 
        
        # Look for the .pt or .pth file in the directory
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

        # 2. Load the native model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # We load in bfloat16 to save memory
        self.model = build_sam3(config_file, weight_file, device=device)
        self.model.bfloat16()
        
        # 3. Use the native Predictor wrapper
        self.predictor = SAM3ImagePredictor(self.model)

    def _to_image(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert("RGB")
        return Image.open(image_or_path).convert("RGB")

    def generate_masks(self, image_or_path, labels):
        image = self._to_image(image_or_path)
        # SAM 3 predictor expects a numpy array, not a PIL image
        image_np = np.array(image)
        
        masks_out = []
        scores_out = []

        try:
            # Tell the predictor what image we are working with
            self.predictor.set_image(image_np)

            for label in labels:
                try:
                    # NATIVE INFERENCE: Pass the text prompt directly to the predictor
                    masks, scores, _ = self.predictor.predict(
                        text_prompt=label,
                        multimask_output=False # We just want the best mask
                    )
                    
                    # SAM3 returns arrays. Grab the highest scoring mask.
                    best_idx = np.argmax(scores)
                    mask = masks[best_idx]
                    score = scores[best_idx]
                    
                    masks_out.append(mask)
                    scores_out.append(score)

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
            self.predictor.reset_image()
            torch.cuda.empty_cache()
            gc.collect()

        return masks_out, scores_out