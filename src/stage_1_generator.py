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
        
        # 1. LOAD WITH NATIVE EAGER MODE ON ISOLATED GPU
        # Since SAM 3 is on cuda:0, Qwen gets cuda:1 all to itself.
        # We have plenty of VRAM, so we don't need brittle SDPA hacks!
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda:1", 
            attn_implementation="eager" 
        )
        self.model.eval()

        # Force native attention output at the config level
        self.model.config.output_attentions = True

        # ==========================================
        # 🧠 THE CLEAN ATTENTION TAP
        # ==========================================
        target_layer_module = None
        target_layer_name = ""
        
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn"):
                target_layer_module = module
                target_layer_name = name
                
        if target_layer_module is None:
            raise AttributeError("❌ Could not dynamically find a 'self_attn' layer!")
            
        print(f"✅ Tapping clean Eager layer: {target_layer_name}")

        # A standard hook (No monkey-patching required)
        self.captured_attentions = []
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.captured_attentions.append(output[1].detach().cpu())

        self.hook = target_layer_module.register_forward_hook(hook_fn)
        # ==========================================



    def generate_candidates(self, image_path, question, context_string, num_beams=4, diversity_penalty=0.5):
        """
        Executes Diverse Beam Search and extracts semantic candidates.
        Includes an aggressive parser to prevent '0/1' geometric failures.
        """
        # Reset attention hook storage for this specific image run
        self.captured_attentions = []

        # 1. Rigorous System Prompting
        # We force the model into 'List Mode' to simplify parsing.
        prompt_text = (
            f"{context_string}\n"
            f"Task: Answer the question by listing all distinct, concrete, physical nouns visible in the image.\n"
            f"Rules: NEVER use generic words ('object', 'item'). Output ONLY a bracketed list of noun phrases.\n\n"
            f"Target Image Analysis:\n"
            f"Question: '{question}'\n"
            f"Plausible Visual Answers:"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "max_pixels": 313600},
                {"type": "text", "text": prompt_text}
            ]
        }]

        # 2. Tokenization and Index Tracking
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)

        # Calculate exactly where the image tokens live in the sequence
        input_ids = inputs["input_ids"][0].cpu()
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        
        # These indices tell the Latent Bridge which part of the attention matrix to 'crop'
        start_idx = image_indices[0].item()
        end_idx = image_indices[-1].item() + 1

        # 3. Diverse Beam Search Generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
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
        
        # Wrap the last captured attention tensor for the Latent Bridge
        final_bridge_attentions = [(self.captured_attentions[-1],)]

        # 4. The Aggressive Parser (Fixes the 0/1 Valid Masks Bug)
        candidates_data = [] 
        
        for seq_idx in range(num_beams):
            # Slice off the prompt to get only the new tokens
            generated_ids = outputs.sequences[seq_idx][len(inputs["input_ids"][0]):]
            decoded_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Step A: Attempt clean Python list evaluation
            try:
                parsed_list = ast.literal_eval(decoded_text)
                if isinstance(parsed_list, list):
                    for item in parsed_list:
                        # Filter out generic noise that Qwen might still output
                        if item.lower() not in ["object", "target", "item", "picture"]:
                            candidates_data.append((item.strip(), seq_idx))
                    continue # Successfully parsed this beam
            except:
                pass # Fallback to manual cleaning if ast fails
                
            # Step B: Regex-style manual cleaning fallback
            # This strips [ ] ' " and splits by commas to recover the nouns
            clean_text = decoded_text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            manual_items = [c.strip() for c in clean_text.split(",") if c.strip()]
            
            for item in manual_items:
                if item.lower() not in ["object", "target", "item", "picture"]:
                    candidates_data.append((item, seq_idx))

        # Clean up memory
        del inputs
        torch.cuda.empty_cache()
        
        return candidates_data, final_bridge_attentions, start_idx, end_idx