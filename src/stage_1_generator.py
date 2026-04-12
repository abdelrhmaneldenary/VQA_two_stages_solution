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
        
        # --- THE FIX: SET ATTN_IMPLEMENTATION TO 'EAGER' ---
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda:1",  # <--- Moved entirely to GPU 1
            attn_implementation="eager" # Required to allow output_attentions
        )
        # ----------------------------------------------------
        
        self.model.eval()
        
        # Now this line will work without crashing
        self.model.config.output_attentions = True

        self.captured_attentions = []

        def hook_fn(module, input, output):
            # In 'eager' mode, output[1] will now contain the attention weights
            if isinstance(output, tuple) and len(output) > 1:
                if output[1] is not None:
                    self.captured_attentions.append(output[1].detach().cpu())

        target_layer = None
        for name, module in self.model.named_modules():
            if "self_attn" in name:
                target_layer = module
        
        if target_layer:
            print(f"✅ Tapping: {target_layer.__class__.__name__}")
            self.hook = target_layer.register_forward_hook(hook_fn)


    def generate_candidates(self, image_path, question, context_string, num_beams=4, diversity_penalty=0.5):
        self.captured_attentions = []

        # 1. Standard Prompting
        prompt_text = (f"{context_string}Target Image Analysis:\nQuestion: '{question}'\nPlausible Visual Answers:")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path, "max_pixels": 156800},
            {"type": "text", "text": prompt_text}
        ]}]

        # 2. Processor
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)

        # 3. Dynamic Token Locator
        input_ids = inputs["input_ids"][0].cpu()
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        start_idx = image_indices[0].item()
        end_idx = image_indices[-1].item() + 1

        # 4. DBS Execution with Explicit Attention Flag
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
                output_attentions=True, # We keep this here too
                do_sample=False,
                use_cache=False 
            )

        # 5. The Hook Check
        if not self.captured_attentions:
            # Last ditch effort: Try to find any attention-like tensor in the output object
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # If the script actually returned them in the standard way, use those
                final_bridge_attentions = outputs.attentions
            else:
                raise ValueError("❌ Hook AND Native Attention both failed. Custom script is blocking attention flow.")
        else:
            final_bridge_attentions = [ (self.captured_attentions[-1],) ]

        # 6. Parse sequences
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

        del inputs
        torch.cuda.empty_cache()
        return candidates_data, final_bridge_attentions, start_idx, end_idx