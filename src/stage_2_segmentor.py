import torch
import numpy as np
from PIL import Image
# Use the explicit SAM 3 classes for better feature support
from transformers import Sam3Processor, Sam3Model 

class Stage2Segmentor:
    def __init__(self, model_id="facebook/sam3"):
        print(f"Loading Stage 2 Geometric Engine (SAM 3): {model_id}")
        
        # SAM 3 uses a complex tokenizer for text prompts; use_fast=False is safer here
        self.processor = Sam3Processor.from_pretrained(model_id, use_fast=False)
        
        self.model = Sam3Model.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def generate_masks(self, image_path, bimodal_tuples):
        raw_image = Image.open(image_path).convert("RGB")
        final_masks = []
        mask_scores = []
        
        for candidate_text, dense_logit_prior in bimodal_tuples:
            # 1. Format the Bridge Output
            # SAM 3 expects (Batch, 1, 256, 256) for mask inputs
            dense_prompt_tensor = dense_logit_prior.unsqueeze(0).unsqueeze(0)
            
            # 2. Bimodal Prompting (Text + Latent Bridge Mask)
            inputs = self.processor(
                images=raw_image,
                text=candidate_text, # Passing the Qwen-generated answer
                input_masks=dense_prompt_tensor,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 3. Post-Process (SAM 3 specialized method)
            # This handles resizing the mask back to the original VizWiz image size
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                target_sizes=[(raw_image.size[1], raw_image.size[0])]
            )[0]

            if len(results['masks']) > 0:
                # We take the highest confidence mask for this concept
                best_idx = torch.argmax(results['scores'])
                final_masks.append(results['masks'][best_idx].cpu().numpy())
                mask_scores.append(results['scores'][best_idx].item())
            else:
                # Fallback if SAM 3 finds nothing for that specific text
                final_masks.append(np.zeros((raw_image.size[1], raw_image.size[0])))
                mask_scores.append(0.0)
            
            del inputs, outputs
            torch.cuda.empty_cache()

        return final_masks, mask_scores