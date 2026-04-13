import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class Stage2Segmenter:
    def __init__(self, model_id="facebook/sam3-h"):
        print(f"🚀 Loading Stage 2 (SAM 3): {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
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
        h, w = image.size[1], image.size[0]

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
            inputs = self.processor(text=label, images=image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=16)

            mask, score = self._mask_from_generated(generated, inputs, image)
            masks.append(mask)
            scores.append(score)

        return masks, scores
