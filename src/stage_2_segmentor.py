import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model

class Stage2Segmentor:
    def __init__(self, model_id=None):
        print(f"🚀 Initializing Stage 2 (SAM 3) from LOCAL PATH: {model_id}")

        # 1. Loading from Local Disk (No Tokens Required!)
        try:
            self.processor = Sam3Processor.from_pretrained(
                model_id, 
                local_files_only=True,   # 🔒 Forces STRICT OFFLINE mode
                trust_remote_code=True   # Still needed for SAM 3 architecture
            )
            
            self.model = Sam3Model.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,   # 🔒 Forces STRICT OFFLINE mode
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Local Load Failed: {e}")
            print(f"💡 PRO-TIP: Run '!ls {model_id}' to ensure the path is exactly correct.")
            raise e

        self.model.eval()

    def generate_masks(self, image_path, bimodal_tuples):
            raw_image = Image.open(image_path).convert("RGB")
            final_masks = []
            mask_scores = []
            
            for candidate_text, dense_logit_prior in bimodal_tuples:
                # Prepare SAM 3 inputs: (Batch, Channels, H, W)
                # Ensure shape is [1, 1, 256, 256] and dtype is float32 for the processor
                dense_prompt_tensor = dense_logit_prior.unsqueeze(0).unsqueeze(0).to(torch.float32)
                
                # --- THE KEYWORD FIX ---
                # Changed 'input_masks' to 'mask_input' to align with SAM 3 API
                inputs = self.processor(
                    images=raw_image,
                    text=candidate_text,
                    mask_input=dense_prompt_tensor, 
                    return_tensors="pt"
                ).to(self.model.device, dtype=torch.bfloat16)
                # -----------------------

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # SAM 3 native post-processing
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
                
                # Memory safety: clear the prompt tensors immediately
                del inputs, outputs
                torch.cuda.empty_cache()

            return final_masks, mask_scores