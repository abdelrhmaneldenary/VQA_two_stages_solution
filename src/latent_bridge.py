import torch
import torch.nn.functional as F
import spacy
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
        prob_matrix = prob_matrix.to(torch.float32)
        prob_matrix = torch.clamp(prob_matrix, min=epsilon, max=1.0 - epsilon)
        logits = torch.log(prob_matrix / (1.0 - prob_matrix))
        return logits * self.logit_scale_factor

    def _extract_and_reshape_attention(self, all_steps_attentions, image_token_start, image_token_end, seq_idx=0):
        # 1. FOOLPROOF TUPLE FILTERING
        # group-beam-search inserts empty tuples during beam transitions. We filter them out.
        valid_steps = [s for s in all_steps_attentions if isinstance(s, tuple) and len(s) > 0]
        if not valid_steps:
            raise ValueError("❌ Attention tensors are entirely empty!")

        # 2. Get the last valid step, and the last transformer layer
        last_step = valid_steps[-1]
        last_layer_attn = last_step[-1] 

        # 3. Handle Beam Search Batching (Crucial for DBS!)
        # Shape is usually (batch * num_beams, num_heads, seq_len_q, seq_len_k)
        # We select the specific sequence index to get its unique attention!
        if last_layer_attn.shape[0] > seq_idx:
            beam_attn = last_layer_attn[seq_idx]
        else:
            beam_attn = last_layer_attn[0]

        # 4. Average across heads
        avg_attn = beam_attn.mean(dim=0)

        # 5. Extract what the final generated token is looking at
        if avg_attn.dim() == 2:
            focus_attn = avg_attn[-1, :]
        else:
            focus_attn = avg_attn

        # 6. Isolate visual tokens
        image_attn_1d = focus_attn[image_token_start:image_token_end]

        # 7. Normalize
        if image_attn_1d.max() > 0:
            image_attn_1d = image_attn_1d / image_attn_1d.max()
            
        # 8. DYNAMIC GRID (Crash-Proof Reshaping)
        num_tokens = image_attn_1d.shape[0]
        grid_h = int(math.sqrt(num_tokens))
        grid_w = num_tokens // grid_h 
        
        if grid_h * grid_w != num_tokens:
            target_size = grid_h * grid_w
            image_attn_1d = image_attn_1d[:target_size]

        image_attn_2d = image_attn_1d.view(1, 1, grid_h, grid_w)
        return image_attn_2d

    def process_bimodal_tuples(self, candidates_data, outputs, image_token_start, image_token_end):
        bimodal_tuples = []
        
        for item in candidates_data:
            # Safely unpack the Text and the Sequence Index
            if isinstance(item, tuple) and len(item) == 2:
                candidate_text, seq_idx = item
            else:
                candidate_text = item
                seq_idx = 0 # Fallback if indexing fails
                
            doc = self.nlp(str(candidate_text))
            semantic_tokens = [token.text for token in doc if token.pos_ in self.valid_pos]
            if not semantic_tokens:
                semantic_tokens = [str(candidate_text)]
                
            # Cross the bridge with the specific sequence index!
            attn_grid_2d = self._extract_and_reshape_attention(
                outputs.attentions, 
                image_token_start, 
                image_token_end,
                seq_idx=seq_idx
            )
            
            attn_grid_256 = F.interpolate(
                attn_grid_2d.to(torch.float32), 
                size=self.sam3_prompt_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze() 
            
            dense_logit_prior = self._probability_to_logits(attn_grid_256)
            bimodal_tuples.append((str(candidate_text), dense_logit_prior))
            
        return bimodal_tuples