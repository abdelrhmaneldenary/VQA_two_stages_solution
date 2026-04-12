import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import ast

# Required for the Hybrid Attention Surgery
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention 

class Stage1Generator:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        print(f"🚀 Loading Stage 1 Semantic Engine: {model_id}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # 1. LOAD WITH SDPA (Massive Speed & VRAM Savings)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa" 
        )
        self.model.eval()

        # ==========================================
        # 🧠 THE HYBRID ATTENTION SURGERY 
        # ==========================================
        # 1. Find the very last attention layer
        last_layer_attn = self.model.model.layers[-1].self_attn

        # 2. Morph ONLY this specific layer back to "Eager" mode
        last_layer_attn.__class__ = Qwen2VLAttention

        # 3. Pre-Hook: Sneak the 'output_attentions=True' flag into this layer ONLY.
        def pre_hook_fn(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs
            
        last_layer_attn.register_forward_pre_hook(pre_hook_fn, with_kwargs=True)

        # 4. Post-Hook: Catch the materialized matrix safely on the CPU
        self.captured_attentions = []
        def post_hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.captured_attentions.append(output[1].detach().cpu())

        self.hook = last_layer_attn.register_forward_hook(post_hook_fn)
        print("✅ Hybrid Attention Active: 99% SDPA Speed, 1% Eager Extraction")
        # ==========================================

    def generate_candidates(self, image_path, question, context_string, num_beams=4, diversity_penalty=0.5):
        # Clear buffer
        self.captured_attentions = []

        # 1. Prompt Construction
        # 1. Prompt Construction (AGGRESSIVE ALIGNMENT)
        prompt_text = (
            f"{context_string}"
            f"You are a precise bounding-box and segmentation detector. "
            f"Look at the image and answer the following question using ONLY specific, concrete, physical nouns. "
            f"STRICT RULE: NEVER use generic words like 'object', 'target', 'item', or 'picture'.\n"
            f"Question: '{question}'\n"
            f"Concrete Physical Answer:"
        )        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path, "max_pixels": 313600},
            {"type": "text", "text": prompt_text}
        ]}]

        # 2. Processing
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)

        # 3. Dynamic Token Locator for the Bridge
        input_ids = inputs["input_ids"][0].cpu()
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        start_idx = image_indices[0].item()
        end_idx = image_indices[-1].item() + 1

        # 4. DBS Execution (CRITICAL: output_attentions=True is REMOVED from here)
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
                do_sample=False,
                use_cache=False 
            )

        # 5. Extract Hooked Attention
        if not self.captured_attentions:
            raise ValueError("❌ Hook failed to capture attention. Check Hybrid Surgery setup.")
        final_bridge_attentions = [ (self.captured_attentions[-1],) ]

        # 6. Parse Sequences
        candidates_data = [] 
        for seq_idx in range(num_beams):
            generated_ids = outputs.sequences[seq_idx][len(inputs["input_ids"][0]):]
            decoded_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            try:
                parsed_list = ast.literal_eval(decoded_text)
                if isinstance(parsed_list, list):
                    for item in parsed_list: candidates_data.append((item, seq_idx))
                else: candidates_data.append((decoded_text, seq_idx))
            except: candidates_data.append((decoded_text, seq_idx))

        # Memory Cleanup
        del inputs
        torch.cuda.empty_cache()
        
        return candidates_data, final_bridge_attentions, start_idx, end_idx