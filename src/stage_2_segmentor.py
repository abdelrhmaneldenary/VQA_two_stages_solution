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
        final_masks = []
        mask_scores = []
        
        for candidate_text, point_coords in bimodal_tuples:
            
            # --- THE NATIVE API HANDSHAKE ---
            # We pass the bimodal prompt (Text + Point) directly through the front door.
            # The processor natively handles all 1024-padding and latent offsets automatically!
            inputs = self.processor(
                images=raw_image,
                text=candidate_text,
                input_points=point_coords, 
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.1, 
                target_sizes=[(raw_image.size[1], raw_image.size[0])]
            )[0]

            if len(results['masks']) > 0:
                best_idx = torch.argmax(results['scores'])
                final_masks.append(results['masks'][best_idx].cpu().numpy())
                mask_scores.append(results['scores'][best_idx].item())
            else:
                final_masks.append(np.zeros((raw_image.size[1], raw_image.size[0])))
                mask_scores.append(0.0)
            
            del inputs, outputs
            torch.cuda.empty_cache()

        return final_masks, mask_scores