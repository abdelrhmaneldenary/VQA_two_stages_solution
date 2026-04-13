import os
import torch
from transformers import AutoModel, AutoProcessor # Using AutoModel here
import gc
class Stage2Segmenter:
    def __init__(self, model_id):
        # 1. Force the Kaggle absolute path
        local_path = os.path.abspath(model_id) if os.path.exists(model_id) else model_id
        print(f"🚀 Loading Stage 2 (SAM 3): {local_path}")
        
        # 2. Load Processor safely
        self.processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )
        
        # 3. Load Model using AutoModel instead of AutoModelForVision2Seq
        self.model = AutoModel.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )
        self.model.eval()       

    def _to_image(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert("RGB")
        return Image.open(image_or_path).convert("RGB")

    def _blank_mask(self, image):
        w, h = image.size
        return np.zeros((h, w), dtype=np.uint8)

    def _mask_from_generated(self, generated, inputs, image):
        w, h = image.size

        if hasattr(self.processor, "post_process_instance_segmentation"):
            try:
                results = self.processor.post_process_instance_segmentation(
                    generated,
                    threshold=0.0,
                    target_sizes=[(h, w)],
                )[0]
                masks = results.get("masks", [])
                if len(masks) > 0:
                    idx = int(torch.argmax(results["scores"]))
                    return masks[idx].detach().cpu().numpy().astype(np.uint8), float(results["scores"][idx])
            except Exception:
                pass

        if hasattr(generated, "pred_masks"):
            try:
                pred_masks = generated.pred_masks
                if pred_masks is not None and pred_masks.numel() > 0:
                    mask = pred_masks[0, 0]
                    mask = (mask > 0).detach().cpu().numpy().astype(np.uint8)
                    return mask, 1.0
            except Exception:
                pass

        if hasattr(self.processor, "post_process_masks"):
            try:
                processed = self.processor.post_process_masks(generated, target_sizes=[(h, w)])
                if processed and len(processed[0]) > 0:
                    mask = processed[0][0]
                    if torch.is_tensor(mask):
                        mask = mask.detach().cpu().numpy()
                    mask = (mask > 0).astype(np.uint8)
                    return mask, 1.0
            except Exception:
                pass

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            if hasattr(self.processor, "post_process_instance_segmentation"):
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.0,
                    target_sizes=[(h, w)],
                )[0]
                masks = results.get("masks", [])
                if len(masks) > 0:
                    idx = int(torch.argmax(results["scores"]))
                    return masks[idx].detach().cpu().numpy().astype(np.uint8), float(results["scores"][idx])
        except Exception:
            pass

        return self._blank_mask(image), 0.0



    def generate_masks(self, image_or_path, labels):
        image = self._to_image(image_or_path)
        masks = []
        scores = []

        for label in labels:
            try:
                # 1. THE PROCESSOR BYPASS (Fixes the 'text' kwarg error)
                image_inputs = self.processor.image_processor(images=image, return_tensors="pt")
                text_inputs = self.processor.tokenizer(text=label, padding=True, return_tensors="pt")
                
                # Merge the dictionaries explicitly
                inputs = {**image_inputs, **text_inputs}
                inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

                # 2. THE FORWARD PASS (Fixes the 'generate' missing attribute error)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 3. NATIVE MASK EXTRACTION (SAM outputs spatial masks directly)
                if hasattr(outputs, "pred_masks"):
                    # Standard SAM format
                    mask = (outputs.pred_masks > 0.0).squeeze().cpu().numpy()
                else:
                    # Fallback if custom weights use a tuple output
                    mask = (outputs[0] > 0.0).squeeze().cpu().numpy()

                # Ensure mask is 2D (Height x Width)
                if mask.ndim > 2:
                    mask = mask[0]

                # SAM usually outputs IoU scores; default to 1.0 if not found
                score = outputs.iou_scores.squeeze().cpu().numpy() if hasattr(outputs, "iou_scores") else 1.0

                masks.append(mask)
                scores.append(score)

            except Exception as e:
                print(f"⚠️ Stage 2 OOM/Error on label '{label}': {e}")
                # Emit a blank mask to keep the pipeline moving
                blank_mask = np.zeros((image.height, image.width), dtype=bool)
                masks.append(blank_mask)
                scores.append(0.0)

            finally:
                # Per-label memory flush to prevent mid-image OOM spikes
                torch.cuda.empty_cache()
                gc.collect()

        return masks, scores
