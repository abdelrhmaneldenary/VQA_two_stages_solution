import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import ast

class Stage1Generator:
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initializes the VLM. We use 4-bit quantization by default to ensure 
        this can run on Kaggle/Colab GPUs without OOM (Out Of Memory) errors.
        """
        print(f"🚀 Loading Stage 1 Semantic Engine: {model_id}")
        
        # 4-bit Quantization is mandatory for local/Kaggle compute
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # local_files_only=True if you are fully offline, otherwise keep default
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval() # Ensure we are in inference mode (no gradients)

    def generate_candidates(self, image_path, question, context_string, num_beams=4, diversity_penalty=0.5):
        """
        Executes Diverse Beam Search to find ambiguous interpretations.
        Returns the parsed candidate list, raw attention tensors, and the exact token boundaries of the image.
        """
        # 1. Construct the final prompt using the Few-Shot context
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
                    # We limit max_pixels to prevent VRAM explosion on 4K images
                    {"type": "image", "image": image_path, "max_pixels": 313600},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        # 2. Process inputs for Qwen2.5-VL
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        # ---------------------------------------------------------
        # NEW: FIND THE IMAGE TOKENS DYNAMICALLY
        # Qwen2-VL uses <|image_pad|> tokens for the visual patches.
        # We need to find exactly where they sit in the 1D sequence.
        # ---------------------------------------------------------
        input_ids = inputs["input_ids"][0].cpu()
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
        # Get all indices where the token is an image patch
        image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        
        if len(image_indices) == 0:
            raise ValueError("❌ Could not locate <|image_pad|> tokens in the sequence!")
            
        start_idx = image_indices[0].item()
        end_idx = image_indices[-1].item() + 1 # +1 for Python slicing
        # ---------------------------------------------------------

        # 3. Execute Diverse Beam Search (DBS) with Tensor Extraction Hooks
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                custom_generate="transformers-community/group-beam-search", # Required for v5.x
                trust_remote_code=True,
                num_beams=num_beams,
                num_beam_groups=num_beams,         # Required to activate DBS in HuggingFace
                diversity_penalty=diversity_penalty, # The Lambda parameter from our CSV
                return_dict_in_generate=True,      # Forces output as a dictionary, not just a tensor
                output_attentions=True ,            # CRITICAL: Exposes the hidden cross-attention maps!
                do_sample=False
            )

        # 4. Parse the output text into a Python list
        # We only take the top-ranked sequence (outputs.sequences[0]) because DBS 
        # forces the diversity into a single sequence containing our bracketed list.
        generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]
        decoded_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        try:
            # Safely evaluate the string "['cup', 'shadow']" into an actual Python list
            candidates_list = ast.literal_eval(decoded_text)
            if not isinstance(candidates_list, list):
                candidates_list = [decoded_text] # Fallback if model forgets brackets
        except (ValueError, SyntaxError):
            # Fallback if the model hallucinates non-list formatting
            candidates_list = [decoded_text]

        # Clean up memory immediately to save space for SAM 3
        del inputs
        torch.cuda.empty_cache()

        # RETURN THE FULL PAYLOAD for the Latent Bridge
        return candidates_list, outputs, start_idx, end_idx