import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import ast
import re
from enum import Enum


class QuestionSkill(str, Enum):
    TEXT = "TEXT"
    OBJECT = "OBJECT"
    COLOR = "COLOR"
    COUNT = "COUNT"

class Stage1Generator:
    GENERIC_BANLIST = {"object", "target", "item", "picture"}
    TEXT_PATTERNS = [
        re.compile(p) for p in [
            r"\btext\b", r"\blabel\b", r"\bwriting\b", r"\bword\b", r"\bwords\b",
            r"\bbrand\b", r"\bname\b", r"\bread\b", r"\bsay\b", r"\bwritten\b",
            r"\bletter\b", r"\bletters\b",
        ]
    ]
    COLOR_PATTERNS = [re.compile(p) for p in [r"\bcolor\b", r"\bcolour\b", r"\bshade\b", r"\bhue\b"]]
    COUNT_PATTERNS = [re.compile(p) for p in [r"\bhow many\b", r"\bnumber of\b", r"\bcount\b"]]

    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        print(f"🚀 Loading Stage 1 Semantic Engine: {model_id}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda:1", 
            attn_implementation="eager",
            low_cpu_mem_usage=True, # Added to optimize RAM during weight loading
            trust_remote_code=True  
        )
        self.model.eval()

        self.model.config.output_attentions = True

        # ==========================================
        # 🧠 THE CLEAN ATTENTION TAP (FIXED)
        # ==========================================
        attn_modules = []
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn"):
                attn_modules.append((name, module))
                
        if not attn_modules:
            raise AttributeError("❌ Could not dynamically find any 'self_attn' layers!")
            
        # Target an upper-middle layer (75% depth) instead of the final text-generation layer.
        target_idx = int(len(attn_modules) * 0.5)
        target_layer_name, target_layer_module = attn_modules[target_idx]
            
        print(f"✅ Tapping crisp spatial Eager layer: {target_layer_name} (Layer {target_idx}/{len(attn_modules)})")

        self.captured_attentions = []
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                # Detach and move to CPU immediately to prevent VRAM accumulation
                self.captured_attentions.append(output[1].detach().cpu())

        self.hook = target_layer_module.register_forward_hook(hook_fn)
        # ==========================================

    def _parse_candidate_strings(self, decoded_text):
        """
        Robustly extract candidate noun phrases from model output without naive
        comma splitting (which breaks malformed JSON-like strings).
        """
        if not decoded_text:
            return []

        candidates = []

        # 1) Strict Python-list parse path
        try:
            parsed_list = ast.literal_eval(decoded_text)
            if isinstance(parsed_list, list):
                for item in parsed_list:
                    if isinstance(item, str):
                        clean = item.strip()
                        if clean and clean.lower() not in self.GENERIC_BANLIST:
                            candidates.append(clean)
                if candidates:
                    return candidates
        except Exception:
            pass

        # 2) Key:value JSON-like text extraction (handles malformed braces safely).
        # 'object' can appear as a structural key in malformed outputs, while
        # generic values like "object" are still filtered by GENERIC_BANLIST.
        kv_text_hits = re.findall(
            r"""(?i)["']?(?:text|label|object|name|noun)["']?\s*:\s*["']([^"']+)["']""",
            decoded_text,
        )
        for hit in kv_text_hits:
            clean = hit.strip()
            if clean and clean.lower() not in self.GENERIC_BANLIST:
                candidates.append(clean)
        if candidates:
            return candidates

        # 3) Quoted phrase extraction fallback
        quoted_hits = re.findall(r"""["']([^"']{1,80})["']""", decoded_text)
        for hit in quoted_hits:
            clean = hit.strip()
            if clean and clean.lower() not in self.GENERIC_BANLIST and not clean.isdigit():
                candidates.append(clean)
        if candidates:
            return candidates

        # 4) Conversational noun-phrase fallback (no comma-based list assumptions)
        raw = decoded_text.strip().strip("[]{}()")
        raw = re.sub(r"(?i)\b(?:answer|answers|output|objects?)\s*:\s*", "", raw)
        chunks = re.split(r"[\n;|/]|\b(?:and|or|then)\b", raw, flags=re.IGNORECASE)
        for chunk in chunks:
            clean = chunk.strip(" -:,.\"'")
            clean = re.sub(r"(?i)^(?:a|an|the)\s+", "", clean).strip()
            if not clean:
                continue
            if not re.search(r"[a-zA-Z]", clean):
                continue
            if clean.lower() in self.GENERIC_BANLIST:
                continue
            candidates.append(clean)

        return candidates

    def _classify_question_skill(self, question):
        q = (question or "").strip().lower()
        if not q:
            return QuestionSkill.OBJECT.value

        if any(p.search(q) for p in self.COUNT_PATTERNS):
            return QuestionSkill.COUNT.value
        if any(p.search(q) for p in self.TEXT_PATTERNS):
            return QuestionSkill.TEXT.value
        if any(p.search(q) for p in self.COLOR_PATTERNS):
            return QuestionSkill.COLOR.value
        return QuestionSkill.OBJECT.value

    def _build_skill_bias_instruction(self, skill):
        if skill == QuestionSkill.TEXT.value:
            return "Skill-Bias(TEXT): Focus strictly on distinct textual labels. Do not group or merge different strings."
        if skill == QuestionSkill.OBJECT.value:
            return "Skill-Bias(OBJECT): Focus on macro-objects. Ignore sub-parts like buttons, screens, handles, and components. List only the parent entity."
        if skill == QuestionSkill.COLOR.value:
            return "Skill-Bias(COLOR): Focus on the object whose color is being asked about; avoid part-level mentions unless the part itself is the queried object."
        return "Skill-Bias(COUNT): Focus on distinct countable entities at object level; avoid splitting one physical object into parts."

    def generate_candidates(self, image_path, question, context_string, num_beams=4, diversity_penalty=0.5):
        """
        Executes Diverse Beam Search and extracts semantic candidates.
        Includes an aggressive parser to prevent '0/1' geometric failures.
        """
        self.captured_attentions = []
        predicted_skill = self._classify_question_skill(question)
        skill_instruction = self._build_skill_bias_instruction(predicted_skill)


        prompt_text = (
            f"{context_string}\n"
            f"System Skill Token: {predicted_skill}\n"
            f"{skill_instruction}\n"
            f"Task: You are simulating a diverse crowd of human annotators. Answer the question by listing the distinct, concrete, physical objects visible in the image.\n"
            f"Rules:\n"
            f"1. NO DENSE CAPTIONING: Name the macro-object. Never list the component parts of a single item (e.g., output 'laptop', never 'screen' and 'keyboard'). EXCEPTION: If the question asks about TEXT, LABELS, WRITING, BRANDS, or WORDS on an object, output each distinct, spatially-separate text string or label as its own list entry.\n"
            f"2. PHYSICAL NOUNS ONLY: Only output tangible, physical items or exact text strings. Never use generic words ('object', 'item', 'picture').\n"
            f"3. PRESERVE DISTINCT SEPARATION: If there are multiple spatially separate, distinct objects or text strings that answer the question (e.g., a 'laptop' and a 'coffee cup', two separate 'cars', or two separate text labels), you MUST list each one individually.\n"
            f"4. DIVERSITY: If humans might disagree on which specific object or text answers the question, list all plausible distinct objects.\n"
            f"5. Output ONLY a Python bracketed list of string noun phrases.\n\n"
            f"Target Image Analysis:\n"
            f"Question: '{question}'\n"
            f"Plausible Visual Answers:"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "max_pixels": 200704}, # Capped pixels for memory safety
                {"type": "text", "text": prompt_text}
            ]
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)

        # --- THE ASPECT RATIO & PATCH MERGE FIX ---
        image_grid_thw = inputs["image_grid_thw"][0] # Shape: (Time, Height, Width)
        grid_h = image_grid_thw[1].item() // 2       # Divide by 2
        grid_w = image_grid_thw[2].item() // 2       # Divide by 2
        # -----------------------------

        input_ids = inputs["input_ids"][0].cpu()
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        
        start_idx = image_indices[0].item()
        end_idx = image_indices[-1].item() + 1

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                custom_generate="transformers-community/group-beam-search", 
                trust_remote_code=True,                                     
                num_beams=num_beams,
                num_beam_groups=num_beams,         
                diversity_penalty=diversity_penalty, 
                num_return_sequences=num_beams,
                return_dict_in_generate=True,      
                output_attentions=True, 
                do_sample=False,
                use_cache=False 
            )

        if not self.captured_attentions:
            raise ValueError("❌ Hook failed to capture attention. Ensure eager mode is active.")
        
        final_bridge_attentions = [(self.captured_attentions[-1],)]

        # --- AGGRESSIVE VRAM/RAM CLEANUP ---
        # Beam search creates dozens of attention tensors. We only need the last one. 
        # Delete the rest instantly to avoid memory fragmentation.
        self.captured_attentions.clear() 
        # -----------------------------------

        candidates_data = [] 
        
        for seq_idx in range(num_beams):
            generated_ids = outputs.sequences[seq_idx][len(inputs["input_ids"][0]):]
            decoded_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            extracted_items = self._parse_candidate_strings(decoded_text)
            for item in extracted_items:
                candidates_data.append((item, seq_idx))

        del inputs
        torch.cuda.empty_cache()
        
        return candidates_data, final_bridge_attentions, start_idx, end_idx, grid_h, grid_w, predicted_skill
