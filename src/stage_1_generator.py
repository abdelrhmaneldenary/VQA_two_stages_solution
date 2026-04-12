import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import ast

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
        # 🧠 THE HYBRID ATTENTION SURGERY (MONKEY-PATCH)
        # ==========================================
        target_layer_module = None
        target_layer_name = ""
        
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn"):
                target_layer_module = module
                target_layer_name = name
                
        if target_layer_module is None:
            raise AttributeError("❌ Could not dynamically find a 'self_attn' layer!")
            
        print(f"✅ Found exact target layer for surgery: {target_layer_name}")

        # 1. The Monkey-Patch
        # We save the original mathematical forward pass
        original_forward = target_layer_module.forward

        def patched_forward(*args, **kwargs):
            new_args = list(args)
            # The signature is: forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, ...)
            # If it's passed as the 5th positional argument (index 4), overwrite it.
            if len(new_args) > 4:
                new_args[4] = True
                # Remove from kwargs to prevent Python "multiple values" crash
                kwargs.pop('output_attentions', None) 
            else:
                # If it wasn't passed positionally, force it via kwarg
                kwargs['output_attentions'] = True
                
            return original_forward(*new_args, **kwargs)

        # We violently overwrite the layer's forward pass with our patched version
        target_layer_module.forward = patched_forward

        # 2. Keep the Post-Hook to catch the matrix on its way out
        self.captured_attentions = []
        def post_hook_fn(module, input, output):
            # output is (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.captured_attentions.append(output[1].detach().cpu())

        self.hook = target_layer_module.register_forward_hook(post_hook_fn)
        print("✅ Hybrid Attention Active: Monkey-Patch deployed successfully.")
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