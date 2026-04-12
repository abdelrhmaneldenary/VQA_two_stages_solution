import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam2Model, Sam2Processor

class Stage2Segmentor:
    def __init__(self, model_id="facebook/sam2.1-hiera-large"):
        """
        Initializes SAM 2.1. This is the 'Hiera' architecture.
        It is faster and more memory-efficient than SAM-ViT-Huge.
        """
        print(f"Loading Stage 2 Geometric Engine (SAM 2.1): {model_id}")
        
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def generate_masks(self, image_path, bimodal_tuples):
        """
        Uses Qwen's logit heatmaps as 'mask prompts' for SAM 2.
        """
        raw_image = Image.open(image_path).convert("RGB")
        width, height = raw_image.size
        
        final_masks = []
        mask_scores = []
        
        for candidate_text, dense_logit_prior in bimodal_tuples:
            # 1. Prepare the mask prompt (SAM 2 expects 256x256 logits)
            # We unsqueeze to add batch and object dimensions: [1, 1, 256, 256]
            input_masks = dense_logit_prior.unsqueeze(0).unsqueeze(0)
            
            # 2. Process inputs for SAM 2
            # Note: We provide the original image and the mask prompt.
            inputs = self.processor(
                images=raw_image, 
                mask_input=input_masks, 
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            # 3. Forward Pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 4. Post-processing for SAM 2
            # SAM 2 outputs masks at a lower resolution; we must upscale to the original image size.
            masks = outputs.pred_masks.squeeze(0).squeeze(0) # [3, 256, 256]
            iou_scores = outputs.iou_scores.squeeze()        # [3]

            # 5. Best Mask Selection (Granularity handling)
            best_idx = torch.argmax(iou_scores)
            best_mask_logits = masks[best_idx]
            
            # 6. Binary Threshold & Upscale
            # We move to CPU and convert to uint8 for high-speed OpenCV resizing
            mask_np = (best_mask_logits > 0.0).cpu().numpy().astype(np.uint8)
            
            # Upscale to match the exact original image resolution for Stage 3 math
            final_binary_mask = cv2.resize(
                mask_np, 
                (width, height), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            final_masks.append(final_binary_mask)
            mask_scores.append(iou_scores[best_idx].item())
            
            # Memory Hygiene
            del inputs, outputs
            torch.cuda.empty_cache()

        return final_masks, mask_scores