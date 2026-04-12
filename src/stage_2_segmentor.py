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
            
            # --- THE MULTI-GPU FIX ---
            # We remove device_map="auto" so the model doesn't get split across 2 GPUs.
            # Instead, we load it into RAM and manually move the whole thing to cuda:0.
            self.model = Sam3Model.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=True,   
                trust_remote_code=True
            ).to("cuda:0") # Force entire model onto GPU 0
            # -------------------------
            
        except Exception as e:
            print(f"❌ Local Load Failed: {e}")
            raise e

        self.model.eval()
    def generate_masks(self, image_path, bimodal_tuples):
            raw_image = Image.open(image_path).convert("RGB")
            final_masks = []
            mask_scores = []
            
            for candidate_text, dense_logit_prior in bimodal_tuples:
                # 1. Boost the Signal
                # A logit of 3.85 is a bit soft for SAM (it expects -20 to +20).
                # We multiply by 5 to turn the 98% probability into a booming command.
                amplified_prior = dense_logit_prior * 5.0
                dense_prompt_tensor = amplified_prior.to(torch.float32) 
                
                inputs = self.processor(
                    images=raw_image,
                    text=candidate_text,
                    segmentation_maps=dense_prompt_tensor, 
                    return_tensors="pt"
                ).to(self.model.device, dtype=torch.bfloat16)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 2. Disable the Self-Doubt Filter
                # We drop the threshold from 0.5 to 0.1 so SAM hands over the geometry 
                # even if it isn't 100% confident in the object's boundaries.
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.1,  # <--- THE FIX
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