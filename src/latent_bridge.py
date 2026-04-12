import torch
import torch.nn.functional as F
import spacy
import numpy as np
import math

class LatentBridge:
    def __init__(self, logit_scale_factor=1.0, sam3_prompt_size=(256, 256)):
        print("🚀 Initializing Latent Bridge & SpaCy POS Tagger...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            
        self.logit_scale_factor = logit_scale_factor
        self.sam3_prompt_size = sam3_prompt_size
        self.valid_pos = {"NOUN", "PROPN", "ADJ", "VERB"}

    def _probability_to_logits(self, prob_matrix, epsilon=1e-5):
        # Move to float32 for stable log math if it isn't already
        prob_matrix = prob_matrix.to(torch.float32)
        prob_matrix = torch.clamp(prob_matrix, min=epsilon, max=1.0 - epsilon)
        logits = torch.log(prob_matrix / (1.0 - prob_matrix))
        return logits * self.logit_scale_factor

    def _extract_and_reshape_attention(self, all_steps_attentions, image_token_start, image_token_end):
        """
        Aggregates attention across ALL generation steps, safely reshapes it into a 2D grid.
        """
        # 1. Stack all generation steps: Tuple of length (Num_Steps) -> Tensor (Num_Steps, Layers, Heads, 1, Seq_Len)
        # We only care about the LAST layer [-1] for semantic routing.
        # Shape becomes: (Num_Steps, Batch=1, Num_Heads, 1, Seq_Len)
        last_layer_all_steps = torch.stack([step_attn[-1] for step_attn in all_steps_attentions])
        
        # 2. Average across steps (Time) AND heads (Features)
        # Shape becomes: (1, Seq_Len) -> squeeze to (Seq_Len)
        avg_attn = last_layer_all_steps.mean(dim=0).mean(dim=1).squeeze()
        
        # 3. Isolate visual tokens
        image_attn_1d = avg_attn[image_token_start:image_token_end]
        
        # 4. Normalize
        if image_attn_1d.max() > 0:
            image_attn_1d = image_attn_1d / image_attn_1d.max()
            
        # 5. DYNAMIC GRID CALCULATION (Prevents .view() crashes)
        # Assuming Qwen creates a square-ish grid of tokens
        num_tokens = image_attn_1d.shape[0]
        grid_h = int(math.sqrt(num_tokens))
        grid_w = num_tokens // grid_h 
        
        # Safety fallback if it's not perfectly rectangular
        if grid_h * grid_w != num_tokens:
            print(f"⚠️ Warning: Visual tokens ({num_tokens}) do not form a perfect grid. Pad/Truncate needed.")
            # Simple truncation to make it fit a square for the bridge
            target_size = grid_h * grid_w
            image_attn_1d = image_attn_1d[:target_size]

        image_attn_2d = image_attn_1d.view(1, 1, grid_h, grid_w)
        return image_attn_2d

    def process_bimodal_tuples(self, candidates_list, outputs, image_token_start, image_token_end):
        bimodal_tuples = []
        
        for candidate in candidates_list:
            doc = self.nlp(candidate)
            semantic_tokens = [token.text for token in doc if token.pos_ in self.valid_pos]
            if not semantic_tokens:
                semantic_tokens = [candidate]
                
            # Use ALL attention steps, not just [-1]
            attn_grid_2d = self._extract_and_reshape_attention(
                outputs.attentions, 
                image_token_start, 
                image_token_end
            )
            
            # Interpolate to 256x256 (Ensure it's float32 for interpolate)
            attn_grid_256 = F.interpolate(
                attn_grid_2d.to(torch.float32), 
                size=self.sam3_prompt_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze() 
            
            # Convert to Logits
            dense_logit_prior = self._probability_to_logits(attn_grid_256)
            
            bimodal_tuples.append((candidate, dense_logit_prior))
            
        return bimodal_tuples