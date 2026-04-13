import torch
import torch.nn.functional as F
import spacy

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

    def _extract_and_reshape_attention(self, all_steps_attentions, image_token_start, image_token_end, grid_h, grid_w, seq_idx=0):
        # 1. Access the captured tensor
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
        focus_attn = avg_attn[-1, :]

        # 5. Geometric Extraction
        image_attn_1d = focus_attn[image_token_start:image_token_end]

        if image_attn_1d.max() > 0:
            image_attn_1d = image_attn_1d / image_attn_1d.max()
            
        # --- THE ASPECT RATIO FIX ---
        # No more math.isqrt() shear! We strictly use the true physical constraints.
        # We ensure the token length perfectly matches the grid, safely truncating any 
        # sequence padding Qwen might have appended to the 1D array.
        expected_tokens = grid_h * grid_w
        image_attn_2d = image_attn_1d[:expected_tokens].view(1, 1, grid_h, grid_w)
        return image_attn_2d    

    def process_bimodal_tuples(self, candidates_data, outputs, image_token_start, image_token_end, grid_h, grid_w, raw_image_size):
        bimodal_tuples = []
        
        # --- ROBUST HANDSHAKE ---
        if isinstance(outputs, (list, tuple)):
            attentions_to_process = outputs
        elif hasattr(outputs, 'attentions'):
            attentions_to_process = outputs.attentions
        else:
            raise ValueError(f"❌ Latent Bridge received unrecognized output type: {type(outputs)}")
        # ------------------------
        
        # --- SAM'S AFFINE SCALING & PADDING MATH ---
        # SAM processes images by resizing the longest edge to 1024 and padding the short edge.
        # Its latent prompt encoder operates exactly at 1/4 resolution of that canvas (256x256 max).
        orig_w, orig_h = raw_image_size
        scale = 1024.0 / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        latent_w, latent_h = new_w // 4, new_h // 4
        # -------------------------------------------
        
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
                
            attn_grid_2d = self._extract_and_reshape_attention(
                attentions_to_process, 
                image_token_start, 
                image_token_end,
                grid_h,
                grid_w,
                seq_idx=seq_idx
            )
            
            # --- MULTI-SPACE COORDINATE FIX ---
            # Interpolate ONLY to the active image dimensions, not the full padded canvas.
            active_attn = F.interpolate(
                attn_grid_2d.to(torch.float32), 
                size=(latent_h, latent_w), 
                mode='bilinear', 
                align_corners=False
            ) 
            
            # --- Z-SCORE MATHEMATICAL DRIFT FIX ---
            # Normalize ONLY against the non-zero semantic activations (the active image),
            # ignoring the void space that we are about to pad.
            mean_val = active_attn.mean()
            std_val = active_attn.std()
            
            if std_val > 1e-6:
                z_scored_active = (active_attn - mean_val) / std_val
            else:
                z_scored_active = active_attn - mean_val
                
            active_logits = z_scored_active * 3.0
            
            # Prevent the "37.08 Bomb" on the active pixels
            active_logits = torch.clamp(active_logits, min=-5.0, max=5.0)
            
            # --- THE PADDED VOID ---
            # Pad the right and bottom edges with -5.0. This acts as a mathematical 
            # command to SAM: "There is nothing in the black padding, do not draw here."
            pad_right = 256 - latent_w
            pad_bottom = 256 - latent_h
            
            dense_logit_prior = F.pad(active_logits, (0, pad_right, 0, pad_bottom), value=-5.0).squeeze()
            # ==========================================

            # --- THE DIVERSITY COLLAPSE FIX ---
            # Indented correctly to append all K candidate paths
            bimodal_tuples.append((str(candidate_text), dense_logit_prior))
            
        return bimodal_tuples