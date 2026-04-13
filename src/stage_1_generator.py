import ast
import re
import os # <-- Added for path resolution
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class Stage1Generator:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        # 1. Force the model_id into an absolute local path
        local_path = os.path.abspath(model_id) if os.path.exists(model_id) else model_id
        print(f"🚀 Loading Stage 1 (Qwen2.5-VL): {local_path}")
        
        # 2. Add local_files_only=True to prevent HF Hub validation crashes
        self.processor = AutoProcessor.from_pretrained(
            local_path, 
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )
        self.model.eval()


    def _run_prompt(self, image_path, prompt, max_new_tokens=64):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _extract_count(self, text, max_count):
        match = re.search(r"\d+", text or "")
        if not match:
            return 1
        value = int(match.group())
        return max(1, min(value, max_count))

    def _extract_labels(self, text):
        text = (text or "").strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        text = text.replace("\n", ",")
        parts = [p.strip(" -•\t\"'") for p in text.split(",")]
        return [p for p in parts if p]

    def generate_grounding_plan(self, image_path, question, max_count=5):
        count_prompt = (
            "Based on the image and question, how many distinct visual regions are needed "
            "to answer correctly? Answer with a single number.\n"
            f"Question: {question}"
        )
        count_text = self._run_prompt(image_path, count_prompt, max_new_tokens=8)
        candidate_count = self._extract_count(count_text, max_count=max_count)

        labels_prompt = (
            f"List the {candidate_count} names of the distinct visual objects or text labels "
            "that provide the answer. Return only a Python list of strings.\n"
            f"Question: {question}"
        )
        labels_text = self._run_prompt(image_path, labels_prompt, max_new_tokens=64)
        labels = self._extract_labels(labels_text)

        if not labels:
            labels = ["object"]

        if len(labels) < candidate_count and labels:
            labels.extend([labels[-1]] * (candidate_count - len(labels)))

        return candidate_count, labels[:candidate_count]
