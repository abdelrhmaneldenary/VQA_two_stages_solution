import os
import gc
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor

class Stage2Segmenter:
    def __init__(self, model_id):
        # 1. Force the Kaggle absolute path
        local_path = os.path.abspath(model_id) if os.path.exists(model_id) else model_id
        print(f"🚀 Loading Stage 2 (HF SAM 3): {local_path}")
        
        # 2. Load Processor safely
        self.processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )
        
        # 3. Load Model with EAGER ATTENTION to bypass the SDPA crash
        self.model = AutoModelForMaskGeneration.from_pretrained(
                    local_path,
                    device_map="cuda:0", # <--- CHANGE THIS FROM "auto"
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    attn_implementation="eager",
                    trust_remote_code=True,
                    local_files_only=True if os.path.exists(model_id) else False
                )
        self.model.eval()

    def _to_image(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert("RGB")
        return Image.open(image_or_path).convert("RGB")

    def generate_masks(self, image_or_path, labels):
        image = self._to_image(image_or_path)
        masks = []
        scores = []

        for label in labels:
            try:
                # 1. THE PROCESSOR BYPASS (Fixes the 'text' kwarg error)
                image_inputs = self.processor.image_processor(images=image, return_tensors="pt")
                # No padding needed for single word labels
                text_inputs = self.processor.tokenizer(text=label, return_tensors="pt") 
                
                # Merge the dictionaries explicitly
                inputs = {**image_inputs, **text_inputs}
                
                # Strip out the text attention mask if it exists to prevent further SDPA collisions
                if "attention_mask" in inputs:
                    del inputs["attention_mask"]

                inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

                # 2. THE FORWARD PASS
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 3. NATIVE MASK EXTRACTION
                if hasattr(outputs, "pred_masks"):
                    mask = (outputs.pred_masks > 0.0).squeeze().cpu().numpy()
                else:
                    mask = (outputs[0] > 0.0).squeeze().cpu().numpy()

                # Ensure mask is exactly 2D (Height x Width)
                if mask.ndim > 2:
                    mask = mask[0]

                # SAM usually outputs IoU scores; default to 1.0 if not found
                if hasattr(outputs, "iou_scores"):
                    score = outputs.iou_scores.max().item() 
                else:
                    score = 1.0

                masks.append(mask)
                scores.append(score)

            except Exception as e:
                print(f"⚠️ Stage 2 OOM/Error on label '{label}': {e}")
                # Emit a blank boolean mask to keep the pipeline moving
                blank_mask = np.zeros((image.height, image.width), dtype=bool)
                masks.append(blank_mask)
                scores.append(0.0)

            finally:
                # Per-label memory flush to prevent mid-image OOM spikes
                torch.cuda.empty_cache()
                gc.collect()

        return masks, scores