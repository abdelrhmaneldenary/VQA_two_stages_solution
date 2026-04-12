import torch
import numpy as np
import os
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import huggingface_hub

class Stage2Segmentor:
    def __init__(self, model_id="facebook/sam3"):
        print(f"🚀 Initializing Stage 2 (SAM 3)...")
        
        # 1. Force retrieval of token
        token = huggingface_hub.get_token()
        if not token:
            # If the session token is missing, check the environment variable
            token = os.getenv("HF_TOKEN")
        
        if not token:
            raise ValueError("❌ No Hugging Face token found! Run login(token='...') in a cell first.")

        # 2. Defensive Loading
        try:
            # We explicitly pass the token and set local_files_only=False
            print(f"📡 Fetching configuration from Hugging Face Hub...")
            self.processor = Sam3Processor.from_pretrained(
                model_id, 
                token=token,
                local_files_only=False,
                trust_remote_code=True
            )
            
            print(f"🧠 Downloading/Loading SAM 3 Weights...")
            self.model = Sam3Model.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token,
                local_files_only=False,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Error Detail: {str(e)}")
            raise e

        self.model.eval()

    def generate_masks(self, image_path, bimodal_tuples):
        raw_image = Image.open(image_path).convert("RGB")
        final_masks = []
        mask_scores = []
        
        for candidate_text, dense_logit_prior in bimodal_tuples:
            # Shape: (Batch=1, Channels=1, H=256, W=256)
            dense_prompt_tensor = dense_logit_prior.unsqueeze(0).unsqueeze(0)
            
            inputs = self.processor(
                images=raw_image,
                text=candidate_text,
                input_masks=dense_prompt_tensor,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process back to original image dimensions
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