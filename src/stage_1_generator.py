import ast
import re
import os
import gc
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

class Stage1Generator:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        # 1. Force absolute local path
        local_path = os.path.abspath(model_id) if os.path.exists(model_id) else model_id
        print(f"🚀 Loading Stage 1 (Qwen2.5-VL Simple Armored): {local_path}")
        
        # 2. VRAM Armor: 4-Bit Quantization (Crushes 8GB down to ~2.5GB)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.processor = AutoProcessor.from_pretrained(
            local_path, 
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )

        # 3. Load with Quantization and Eager Attention (Bypasses HF SDPA crash)
        # Note: 4-bit models must stay on the GPU, but since it's only 2.5GB, 
        # it leaves plenty of room for SAM 3 to operate alongside it.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path,
            quantization_config=bnb_config,
            device_map="cuda:0", 
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True if os.path.exists(model_id) else False
        )
        self.model.eval()
        
        # Strip deprecated generation warnings
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.do_sample = False
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(self.model.generation_config, attr):
                    setattr(self.model.generation_config, attr, None)

    def _run_prompt(self, image_or_path, prompt, max_new_tokens=64):
        # Dynamically handle both PIL Images (from your pipeline) and raw paths
        if isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGB")
        else:
            image = Image.open(image_or_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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
        
        # Aggressive memory clearing immediately after text generation
        del inputs
        torch.cuda.empty_cache()
        
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

    def generate_grounding_plan(self, image_or_path, question, max_count=5):
        count_prompt = (
            "Based on the image and question, how many distinct visual regions are needed "
            "to answer correctly? Answer with a single number.\n"
            f"Question: {question}"
        )
        count_text = self._run_prompt(image_or_path, count_prompt, max_new_tokens=8)
        candidate_count = self._extract_count(count_text, max_count=max_count)

        labels_prompt = (
            f"List the {candidate_count} names of the distinct visual objects or text labels "
            "that provide the answer. Return only a Python list of strings.\n"
            f"Question: {question}"
        )
        labels_text = self._run_prompt(image_or_path, labels_prompt, max_new_tokens=64)
        labels = self._extract_labels(labels_text)

        # Fallback if the LLM completely fails to generate a list
        if not labels:
            labels = ["object"]

        # Pad the list if the LLM generated fewer items than the count
        if len(labels) < candidate_count and labels:
            labels.extend([labels[-1]] * (candidate_count - len(labels)))

        # Final garbage collection before handing the baton back to run_pipeline
        gc.collect()

        return candidate_count, labels[:candidate_count]