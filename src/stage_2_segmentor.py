import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model

class Stage2Segmentor:
    def __init__(self, model_id=None):
        print(f"🚀 Initializing Stage 2 (SAM 3) from LOCAL PATH: {model_id}")

        try:
            self.processor = Sam3Processor.from_pretrained(
                model_id, 
                local_files_only=True,   
                trust_remote_code=True   
            )
            
            self.model = Sam3Model.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=True,   
                trust_remote_code=True
            ).to("cuda:0") 
            
        except Exception as e:
            print(f"❌ Local Load Failed: {e}")
            raise e

        self.model.eval()

    def generate_masks(self, image_or_path, bimodal_tuples):
        if isinstance(image_or_path, Image.Image):
            raw_image = image_or_path if image_or_path.mode == "RGB" else image_or_path.convert("RGB")
        else:
            raw_image = Image.open(image_or_path).convert("RGB")
        orig_w, orig_h = raw_image.size
        
        # --- THE VRAM SHIELD: DYNAMIC RESOLUTION CAPPING ---
        # Caps the GPU upscaler to 1 Megapixel to prevent 7GB+ OOM spikes.
        safe_h, safe_w = orig_h, orig_w
        if safe_h > 1024 or safe_w > 1024:
            scale_factor = 1024.0 / max(safe_h, safe_w)
            safe_h = int(safe_h * scale_factor)
            safe_w = int(safe_w * scale_factor)
            
        final_masks = []
        mask_scores = []
        
        for candidate_text, (point_x, point_y) in bimodal_tuples:
            
            # 1. THE PROCESSOR HANDSHAKE
            inputs = self.processor(
                images=raw_image,
                text=candidate_text,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            # 2. THE MANUAL AFFINE TRANSLATION
            scale = 1024.0 / max(orig_w, orig_h)
            scaled_x = point_x * scale
            scaled_y = point_y * scale
            
            # 3. THE SURGICAL TENSOR INJECTION
            point_tensor = torch.tensor([[[[scaled_x, scaled_y]]]], device=self.model.device, dtype=torch.float32)
            label_tensor = torch.tensor([[[1]]], device=self.model.device, dtype=torch.long)
            
            inputs["input_points"] = point_tensor
            inputs["input_labels"] = label_tensor

            # 4. THE FORWARD PASS
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 5. POST PROCESSING (Protected by VRAM Shield)
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0,  # <--- Threshold dropped to 0.0 to fix blurry rejection
                target_sizes=[(safe_h, safe_w)] # <--- Bounded safe dimensions
            )[0]

            if len(results['masks']) > 0:
                best_idx = torch.argmax(results['scores'])
                
                # Immediately offload to CPU as a lightweight 8-bit array
                mask_array = results['masks'][best_idx].cpu().numpy().astype(np.uint8)
                
                # --- THE CPU UPSCALER ---
                # Use Nearest Neighbor interpolation to strictly preserve 0s and 1s
                if (safe_w, safe_h) != (orig_w, orig_h):
                    mask_array = cv2.resize(mask_array, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    
                final_masks.append(mask_array)
                mask_scores.append(results['scores'][best_idx].item())
            else:
                final_masks.append(np.zeros((orig_h, orig_w), dtype=np.uint8))
                mask_scores.append(0.0)
            
            # Aggressive Garbage Collection
            del inputs, outputs, point_tensor, label_tensor

        torch.cuda.empty_cache()
        return final_masks, mask_scores