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
        # 1. Access the captured tensor
        # Our hook returned: [ (attention_tensor,) ]
        if not all_steps_attentions or len(all_steps_attentions) == 0:
            raise ValueError("❌ Latent Bridge received NULL attention from the Semantic Brain!")
            
        # Grab the last layer from the final step
        last_step_tensor = all_steps_attentions[-1][0] 
        # Shape: (Batch * Beams, Heads, Q_Seq, K_Seq)

        # 2. Slice the correct Beam
        if last_step_tensor.shape[0] > seq_idx:
            beam_attn = last_step_tensor[seq_idx]
        else:
            beam_attn = last_step_tensor[0]

        # 3. Average across heads
        avg_attn = beam_attn.mean(dim=0) # (Q_Seq, K_Seq)

        # 4. Focus: What is the VERY LAST generated token looking at?
        # In a full-sequence attention matrix, the last row [-1, :] represents the focus of the final answer token.
        focus_attn = avg_attn[-1, :]

        # 5. Geometric Extraction
        image_attn_1d = focus_attn[image_token_start:image_token_end]

        if image_attn_1d.max() > 0:
            image_attn_1d = image_attn_1d / image_attn_1d.max()
            
        num_tokens = image_attn_1d.shape[0]
        grid_h = int(math.sqrt(num_tokens))
        grid_w = num_tokens // grid_h 
        
        image_attn_2d = image_attn_1d[:grid_h*grid_w].view(1, 1, grid_h, grid_w)
        return image_attn_2d    

    def process_bimodal_tuples(self, candidates_data, outputs, image_token_start, image_token_end):
            bimodal_tuples = []
            
            # --- ROBUST HANDSHAKE ---
            if isinstance(outputs, (list, tuple)):
                attentions_to_process = outputs
            elif hasattr(outputs, 'attentions'):
                attentions_to_process = outputs.attentions
            else:
                raise ValueError(f"❌ Latent Bridge received unrecognized output type: {type(outputs)}")
            # ------------------------
            
            for item in candidates_data:
                if isinstance(item, tuple) and len(item) == 2:
                    candidate_text, seq_idx = item
                else:
                    candidate_text = item
                    seq_idx = 0 
                    
                doc = self.nlp(str(candidate_text))
                semantic_tokens = [token.text for token in doc if token.pos_ in self.valid_pos]
                if not semantic_tokens:
                    semantic_tokens = [str(candidate_text)]
                    
                # Use the verified attentions_to_process
                attn_grid_2d = self._extract_and_reshape_attention(
                    attentions_to_process, 
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
                
            
                # ==========================================
                # 🌉 Z-SCORE SPATIAL CALIBRATION
                # ==========================================
                # 1. Calculate the statistical mean and spread of the attention
                mean_val = attn_grid_256.mean()
                std_val = attn_grid_256.std()
                
                # 2. Z-Score Standardization
                # This shifts the average pixel to 0.0. 
                # Background becomes negative. The object body becomes positive.
                if std_val > 1e-6:
                    z_scored_attn = (attn_grid_256 - mean_val) / std_val
                else:
                    z_scored_attn = attn_grid_256 - mean_val
                    
                # 3. Scale to SAM 3's Native Logit Range
                # We multiply by 3.0 to stretch the Z-scores into a booming [-10 to +10] range.
                # No clamped Log-Odds math required! This IS the logit tensor.
                dense_logit_prior = z_scored_attn * 3.0
                # ==========================================

                bimodal_tuples.append((str(candidate_text), dense_logit_prior))
                
            return bimodal_tuples