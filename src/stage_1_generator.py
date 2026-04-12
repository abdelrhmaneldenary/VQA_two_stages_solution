import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import ast

class Stage1Generator:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initializes the Semantic Brain (Qwen2.5-VL).
        4-bit Quantization (NF4) to fit MRoPE-based weights in Kaggle VRAM.
        """
        print(f"🚀 Loading Stage 1 Semantic Engine: {model_id}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()

    def generate_candidates(self, image_path, question, context_string, num_beams=4, diversity_penalty=0.5):
        """
        DIVERSE BEAM SEARCH (DBS): Mathematically forces semantic exploration.
        """
        prompt_text = (
            f"{context_string}"
            f"Target Image Analysis:\n"
            f"Question: '{question}'\n"
            f"Plausible Visual Answers:"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path, "max_pixels": 313600},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # Process via MRoPE-aware template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        # Dynamic Token Locator for Latent Bridge
        input_ids = inputs["input_ids"][0].cpu()
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        start_idx = image_indices[0].item()
        end_idx = image_indices[-1].item() + 1

        # DBS Execution
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                custom_generate="transformers-community/group-beam-search",
                trust_remote_code=True,
                num_beams=num_beams,
                num_beam_groups=num_beams, 
                num_return_sequences=num_beams,        
                diversity_penalty=diversity_penalty, 
                return_dict_in_generate=True,      
                output_attentions=True,            
                do_sample=False,
                use_cache=False # Bypass KV-Cache crash in community script
            )

        candidates_data = [] 
        for seq_idx in range(num_beams):
            generated_ids = outputs.sequences[seq_idx][len(inputs["input_ids"][0]):]
            decoded_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            try:
                parsed_list = ast.literal_eval(decoded_text)
                if isinstance(parsed_list, list):
                    for item in parsed_list:
                        candidates_data.append((item, seq_idx))
                else:
                    candidates_data.append((decoded_text, seq_idx))
            except:
                candidates_data.append((decoded_text, seq_idx))

        del inputs
        torch.cuda.empty_cache()

        return candidates_data, outputs, start_idx, end_idx