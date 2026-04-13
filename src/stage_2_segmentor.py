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

    def generate_masks(self, image_path, bimodal_tuples):
        raw_image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = raw_image.size
        
        final_masks = []
        mask_scores = []
        
        for candidate_text, (point_x, point_y) in bimodal_tuples:
            
            # 1. THE PROCESSOR HANDSHAKE (Image & Text Only)
            # We deliberately omit the point here to bypass the strict API validation crash.
            inputs = self.processor(
                images=raw_image,
                text=candidate_text,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            # 2. THE MANUAL AFFINE TRANSLATION
            # SAM processes images by scaling the longest edge to 1024.
            # We must apply this exact mathematical scale to our raw Qwen point.
            scale = 1024.0 / max(orig_w, orig_h)
            scaled_x = point_x * scale
            scaled_y = point_y * scale
            
            # 3. THE SURGICAL TENSOR INJECTION
            # Format: [Batch, Num_Instances, Num_Points, 2_Coords] -> [1, 1, 1, 2]
            point_tensor = torch.tensor([[[[scaled_x, scaled_y]]]], device=self.model.device, dtype=torch.float32)
            
            # 1 = Foreground target, 0 = Background avoidance
            label_tensor = torch.tensor([[[1]]], device=self.model.device, dtype=torch.long)
            
            # Forcibly write them into the kwargs dictionary right before the forward pass
            inputs["input_points"] = point_tensor
            inputs["input_labels"] = label_tensor

            # 4. THE FORWARD PASS
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 5. POST PROCESSING
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.0, 
                target_sizes=[(orig_h, orig_w)]
            )[0]

            if len(results['masks']) > 0:
                best_idx = torch.argmax(results['scores'])
                final_masks.append(results['masks'][best_idx].cpu().numpy())
                mask_scores.append(results['scores'][best_idx].item())
            else:
                final_masks.append(np.zeros((orig_h, orig_w)))
                mask_scores.append(0.0)
            
            del inputs, outputs, point_tensor, label_tensor
            torch.cuda.empty_cache()

        return final_masks, mask_scores