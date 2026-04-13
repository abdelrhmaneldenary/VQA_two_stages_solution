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
        if not all_steps_attentions or len(all_steps_attentions) == 0:
            raise ValueError("❌ Latent Bridge received NULL attention from the Semantic Brain!")
            
        last_step_tensor = all_steps_attentions[-1][0] 
        
        if last_step_tensor.shape[0] > seq_idx:
            beam_attn = last_step_tensor[seq_idx]
        else:
            beam_attn = last_step_tensor[0]

        avg_attn = beam_attn.mean(dim=0) 
        focus_attn = avg_attn[-1, :]

        image_attn_1d = focus_attn[image_token_start:image_token_end]
        expected_tokens = grid_h * grid_w

        
        if image_attn_1d.shape[0] > expected_tokens:
            # Drop the first token (often a global/prefix token) to find the real pixels
            image_attn_1d = image_attn_1d[1:expected_tokens+1]
            
        image_attn_2d = image_attn_1d[:expected_tokens].view(1, 1, grid_h, grid_w)
        return image_attn_2d    

    def process_bimodal_tuples(self, candidates_data, outputs, image_token_start, image_token_end, grid_h, grid_w, raw_image_size):
        bimodal_tuples = []
        # Tracks which attention-grid cells have already been claimed by a
        # prior candidate from the same beam sequence.  Keyed by seq_idx.
        used_peaks = {}

        if isinstance(outputs, (list, tuple)):
            attentions_to_process = outputs
        elif hasattr(outputs, 'attentions'):
            attentions_to_process = outputs.attentions
        else:
            raise ValueError(f"❌ Latent Bridge received unrecognized output type: {type(outputs)}")
            
        orig_w, orig_h = raw_image_size
        
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
                
            # --- FIXED INDENTATION STARTS HERE ---
            attn_grid_2d = self._extract_and_reshape_attention(
                attentions_to_process, 
                image_token_start, 
                image_token_end,
                grid_h,
                grid_w,
                seq_idx=seq_idx
            )
            
            # --- THE ATTENTION SINK DESTROYER (3x3 SMOOTHING) ---
            # ViTs dump excess attention on 1-pixel corner artifacts. 
            # We apply average pooling to dilute isolated spikes and 
            # force the argmax to find the dense "center of mass".
            smoothed_attn = F.avg_pool2d(
                attn_grid_2d.to(torch.float32), 
                kernel_size=3, 
                stride=1, 
                padding=1
            ).squeeze()
            
            smoothed_attn[0, :] = 0.0   # Top edge
            smoothed_attn[-1, :] = 0.0  # Bottom edge
            smoothed_attn[:, 0] = 0.0   # Left edge
            smoothed_attn[:, -1] = 0.0

            # --- ATTENTION PEAK SUPPRESSION ---
            # When multiple candidates originate from the same beam (seq_idx),
            # they share an identical attention map.  Without suppression every
            # candidate would produce the same argmax anchor, causing SAM to
            # generate identical masks and the NMS to incorrectly collapse all
            # text labels into a single entry.
            # We zero-out a window around each previously claimed peak so that
            # the k-th candidate finds the k-th distinct hotspot.
            # Suppression radius = 25% of the larger grid dimension.
            # This is wide enough to prevent two candidates from locking
            # onto the same broad attention peak while still allowing
            # distinct hotspots within adjacent quadrants to remain
            # independently discoverable.
            suppression_radius = max(1, max(grid_h, grid_w) // 4)
            suppressed_attn = smoothed_attn.clone()
            for prev_r, prev_c in used_peaks.get(seq_idx, []):
                r_lo = max(0, prev_r - suppression_radius)
                r_hi = min(grid_h, prev_r + suppression_radius + 1)
                c_lo = max(0, prev_c - suppression_radius)
                c_hi = min(grid_w, prev_c + suppression_radius + 1)
                suppressed_attn[r_lo:r_hi, c_lo:c_hi] = 0.0
            # -----------------------------------

            flat_idx = torch.argmax(suppressed_attn)
            # ---------------------------------------------------
            
            r = (flat_idx // grid_w).item()
            c = (flat_idx % grid_w).item()

            # Register this peak so subsequent same-beam candidates are
            # directed to a different spatial location
            if seq_idx not in used_peaks:
                used_peaks[seq_idx] = []
            used_peaks[seq_idx].append((r, c))
            
            point_x = int(((c + 0.5) / grid_w) * orig_w)
            point_y = int(((r + 0.5) / grid_h) * orig_h)
            
            bimodal_tuples.append((str(candidate_text), (point_x, point_y)))
            
        return bimodal_tuples