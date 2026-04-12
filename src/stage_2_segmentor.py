import torch
import numpy as np
import os
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import HfFolder

class Stage2Segmentor:
    def __init__(self, model_id="facebook/sam3"):
        print(f"🚀 Initializing Stage 2 (SAM 3)...")
        
        # CRITICAL FIX: Get the token from the session memory
        token = HfFolder.get_token()
        if not token:
            print("⚠️ Warning: No HF token found. Ensure you ran login(token=...)")

        # FIX FOR OSERROR: Force 'local_files_only=False' and pass token explicitly
        # We also use Sam3Processor directly instead of AutoProcessor
        try:
            self.processor = Sam3Processor.from_pretrained(
                model_id, 
                token=token,
                local_files_only=False 
            )
            
            self.model = Sam3Model.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token,
                local_files_only=False
            )
        except Exception as e:
            print(f"❌ Error loading SAM 3: {e}")
            print("💡 Tip: Ensure there is NO folder named 'sam3' or 'facebook' in your current directory.")
            raise e

        self.model.eval()

    def generate_masks(self, image_path, bimodal_tuples):
        raw_image = Image.open(image_path).convert("RGB")
        final_masks = []
        mask_scores = []
        
        for candidate_text, dense_logit_prior in bimodal_tuples:
            # SAM 3 expects (Batch, Channels, H, W) -> (1, 1, 256, 256)
            dense_prompt_tensor = dense_logit_prior.unsqueeze(0).unsqueeze(0)
            
            inputs = self.processor(
                images=raw_image,
                text=candidate_text,
                input_masks=dense_prompt_tensor,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use SAM 3's specific post-processing
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
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