import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import inspect

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
            self.model = Sam3Model.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                local_files_only=True,   
                trust_remote_code=True
            ).to("cuda:0") 
            # -------------------------
            
        except Exception as e:
            print(f"❌ Local Load Failed: {e}")
            raise e

        self.model.eval()

        # ==========================================
        # 🔍 THE KWARG DISCOVERY PROTOCOL
        # ==========================================
        # Hugging Face frequently renames kwargs (mask_input vs input_masks vs mask_inputs).
        # We dynamically inspect the loaded model's forward pass to find the exact key it requires.
        forward_params = inspect.signature(self.model.forward).parameters
        self.valid_mask_key = None
        
        for possible_key in ["mask_inputs", "input_masks", "mask_input", "dense_prompt"]:
            if possible_key in forward_params:
                self.valid_mask_key = possible_key
                break
                
        if self.valid_mask_key:
            print(f"✅ Discovered dense mask kwarg: '{self.valid_mask_key}'")
        elif "kwargs" in forward_params:
            print("⚠️ Explicit mask kwarg hidden. Relying on **kwargs fallback using 'mask_inputs'")
            self.valid_mask_key = "mask_inputs" # Default Meta fallback
        else:
            raise ValueError("❌ Model signature does not accept dense masks or kwargs! Architecture mismatch.")
        # ==========================================


    def generate_masks(self, image_path, bimodal_tuples):
        raw_image = Image.open(image_path).convert("RGB")
        final_masks = []
        mask_scores = []
        
        for candidate_text, dense_logit_prior in bimodal_tuples:
            
            # 1. Standard Processor Encoding (Image + Text)
            inputs = self.processor(
                images=raw_image,
                text=candidate_text,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            # 2. THE DIRECT INJECTION
            # The Latent Bridge tensor is already perfectly padded and clamped.
            # Format: [Batch, Channels, H, W] -> [1, 1, 256, 256]
            dense_prompt = dense_logit_prior.unsqueeze(0).unsqueeze(0).to(
                device=self.model.device, 
                dtype=torch.bfloat16
            )
            
            # Inject using the dynamically discovered key
            inputs[self.valid_mask_key] = dense_prompt

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 3. Post Processing
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
                # If SAM still rejects it, output a blank mask safely
                final_masks.append(np.zeros((raw_image.size[1], raw_image.size[0])))
                mask_scores.append(0.0)
            
            del inputs, outputs, dense_prompt
            torch.cuda.empty_cache()

        return final_masks, mask_scores