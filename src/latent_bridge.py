import torch
import torch.nn.functional as F
import spacy
import numpy as np

class LatentBridge:
    def __init__(self, logit_scale_factor=1.0, sam3_prompt_size=(256, 256)):
        """
        Initializes the Latent Bridge. Loads a lightweight NLP model purely 
        for Part-of-Speech (POS) tagging to filter out stop-words.
        """
        print("Initializing Latent Bridge & SpaCy POS Tagger...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            
        self.logit_scale_factor = logit_scale_factor
        self.sam3_prompt_size = sam3_prompt_size
        
        # Valid semantic targets. We ignore determiners ("the", "a") and prepositions ("on")
        self.valid_pos = {"NOUN", "PROPN", "ADJ", "VERB"}

    def _probability_to_logits(self, prob_matrix, epsilon=1e-5):
        """
        Translates raw attention probabilities into SAM 3 Logit Space.
        Formula: Logit = ln(p / (1 - p))
        """
        # Clip probabilities to prevent log(0) or division by zero (infinity)
        prob_matrix = torch.clamp(prob_matrix, min=epsilon, max=1.0 - epsilon)
        
        # Log-odds conversion
        logits = torch.log(prob_matrix / (1.0 - prob_matrix))
        
        return logits * self.logit_scale_factor

    def _extract_and_reshape_attention(self, raw_attentions, image_token_start, image_token_end, grid_size):
        """
        Extracts the attention weights for the image tokens, averages them across 
        transformer heads, and reshapes the 1D array back into a 2D spatial grid.
        """
        # Attentions format during generation: (batch_size, num_heads, 1, sequence_length)
        # We average across all attention heads in the final transformer layer
        final_layer_attn = raw_attentions[-1] 
        avg_head_attn = torch.mean(final_layer_attn, dim=1).squeeze(0).squeeze(0)
        
        # Isolate ONLY the tokens corresponding to the visual image patches
        image_attn_1d = avg_head_attn[image_token_start:image_token_end]
        
        # Normalize so the maximum attention spike equals 1.0 (Probability peak)
        if image_attn_1d.max() > 0:
            image_attn_1d = image_attn_1d / image_attn_1d.max()
            
        # Reshape from a flat 1D sequence back into the 2D visual grid (e.g., 16x16)
        image_attn_2d = image_attn_1d.view(1, 1, grid_size[0], grid_size[1])
        return image_attn_2d

    def process_bimodal_tuples(self, candidates_list, outputs, image_token_start, image_token_end, visual_grid_size):
        """
        The main bridge function. Takes text candidates and neural outputs, 
        and returns a list of (text, SAM3_Dense_Logit_Mask) tuples.
        """
        bimodal_tuples = []
        
        for candidate in candidates_list:
            # 1. POS Masking: Identify which words actually matter
            doc = self.nlp(candidate)
            semantic_tokens = [token.text for token in doc if token.pos_ in self.valid_pos]
            
            # If the NLP filter wiped out everything (e.g., candidate was just "it"), fallback to the whole string
            if not semantic_tokens:
                semantic_tokens = [candidate]
                
            # 2. Extract raw attention for this specific candidate
            # (Note: In a full implementation, you map the semantic_tokens to their exact generation step index. 
            # For brevity in this core logic, we assume 'candidate_attention_tuple' holds those specific steps).
            candidate_attention_tuple = outputs.attentions[-1] # Simplification: tracking the final token's focus
            
            # 3. Reshape the 1D visual tokens back into a 2D abstract grid
            attn_grid_2d = self._extract_and_reshape_attention(
                candidate_attention_tuple, 
                image_token_start, 
                image_token_end, 
                visual_grid_size
            )
            
            # 4. Interpolate from Qwen's low-res grid (e.g. 16x16) to SAM 3's high-res prompt space (256x256)
            attn_grid_256 = F.interpolate(
                attn_grid_2d, 
                size=self.sam3_prompt_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze() # Remove batch/channel dims -> Shape: (256, 256)
            
            # 5. The Hallucination Filter: Convert Probability (0 to 1) to Logits (-X to +X)
            dense_logit_prior = self._probability_to_logits(attn_grid_256)
            
            # Append the completed bimodal tuple ready for Stage 2
            bimodal_tuples.append((candidate, dense_logit_prior))
            
        return bimodal_tuples