import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration

class Stage2Segmenter:
    def __init__(self, model_id):
        local_path = os.path.abspath(model_id) if os.path.exists(model_id) else model_id
        print(f"🚀 Loading Stage 2 (HF SAM 3 - Text Prompt Armored): {local_path}")
        
        self.processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )
        
        # Start on CPU for Model Ping-Pong
        self.model = AutoModelForMaskGeneration.from_pretrained(
            local_path,
            device_map="cpu", # Will be moved to cuda:0 during the loop
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager", # Bypasses the Hugging Face text-attention bug
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )
        self.model.eval()

    def generate_masks(self, image_path, labels):
        raw_image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = raw_image.size
        
        # --- THE VRAM SHIELD: DYNAMIC RESOLUTION CAPPING ---
        safe_h, safe_w = orig_h, orig_w
        if safe_h > 1024 or safe_w > 1024:
            scale_factor = 1024.0 / max(safe_h, safe_w)
            safe_h = int(safe_h * scale_factor)
            safe_w = int(safe_w * scale_factor)
            
        safe_image = raw_image.resize((safe_w, safe_h), Image.Resampling.LANCZOS)
        
        final_masks = []
        mask_scores = []
        
        for label in labels:
            try:
                # 1. PROCESSOR BYPASS (Explicit text & image merging)
                image_inputs = self.processor.image_processor(images=safe_image, return_tensors="pt")
                text_inputs = self.processor.tokenizer(text=label, return_tensors="pt")
                
                inputs = {**image_inputs, **text_inputs}
                
                # Fix SDPA crash for text/vision mask collision
                if "attention_mask" in inputs:
                    del inputs["attention_mask"]
                    
                inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

                # 2. FORWARD PASS
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 3. EXTRACTION
                if hasattr(outputs, "pred_masks"):
                    mask = (outputs.pred_masks > 0.0).squeeze().cpu().numpy()
                else:
                    mask = (outputs[0] > 0.0).squeeze().cpu().numpy()

                if mask.ndim > 2:
                    mask = mask[0]
                    
                mask_uint8 = mask.astype(np.uint8)

                # 4. CPU UPSCALER (Nearest Neighbor to strictly preserve 0s and 1s)
                if (safe_w, safe_h) != (orig_w, orig_h):
                    mask_uint8 = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                score = outputs.iou_scores.max().item() if hasattr(outputs, "iou_scores") else 1.0

                final_masks.append(mask_uint8)
                mask_scores.append(score)

            except Exception as e:
                print(f"⚠️ Stage 2 Error on label '{label}': {e}")
                final_masks.append(np.zeros((orig_h, orig_w), dtype=np.uint8))
                mask_scores.append(0.0)
                
            finally:
                del inputs
                torch.cuda.empty_cache()
                gc.collect()

        return final_masks, mask_scores