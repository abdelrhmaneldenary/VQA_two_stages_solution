import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration

class Stage2Segmentor:
    def __init__(self, model_id="facebook/sam3-base"):
        """
        Initializes SAM 3. We load this in bfloat16 to optimize VRAM usage.
        Because SAM 3 is primarily a vision encoder/decoder, 16-bit precision 
        maintains perfect geometric accuracy without the memory footprint of FP32.
        """
        print(f"Loading Stage 2 Geometric Engine: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.model = AutoModelForMaskGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def generate_masks(self, image_path, bimodal_tuples):
        """
        Takes the original image and the bimodal tuples (text, logit_prior) 
        and returns the final binary masks and their SAM-predicted IoU scores.
        """
        raw_image = Image.open(image_path).convert("RGB")
        
        final_masks = []
        mask_scores = []
        
        for candidate_text, dense_logit_prior in bimodal_tuples:
            # 1. Formatting the Dense Mask Prompt
            # SAM 3 expects dense prompts to have a channel dimension: (1, 256, 256)
            dense_prompt_tensor = dense_logit_prior.unsqueeze(0)
            
            # 2. Synergistic Bimodal Prompting
            # We pass BOTH the text and the Qwen-derived logit matrix
            inputs = self.processor(
                images=raw_image,
                text=[candidate_text],
                dense_prompts=[dense_prompt_tensor],
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            # 3. Single Forward Pass (Zero-Shot Grounding)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 4. Extracting the Logits and Scores
            # SAM 3 typically outputs 3 levels of mask granularity (whole, part, subpart)
            # Shape is usually (batch_size, num_masks, H, W)
            pred_masks_logits = outputs.pred_masks.squeeze(0) 
            iou_scores = outputs.iou_scores.squeeze(0)

            # 5. Granularity Selection
            # We trust SAM 3's internal confidence metric to select the best geometric fit
            best_mask_idx = torch.argmax(iou_scores)
            best_mask_logits = pred_masks_logits[best_mask_idx]
            best_score = iou_scores[best_mask_idx].item()

            # 6. The Native Zero-Level Threshold
            # Converts the continuous logit output into a strict binary polygon (1s and 0s)
            # We do NOT use a 0.4 threshold here. Logit math naturally pivots at 0.0.
            binary_mask = (best_mask_logits > 0.0).cpu().numpy()

            final_masks.append(binary_mask)
            mask_scores.append(best_score)
            
            # Explicitly free memory inside the loop
            del inputs, outputs, pred_masks_logits, iou_scores
            torch.cuda.empty_cache()

        return final_masks, mask_scores