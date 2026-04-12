import os
import json
import random

class VQADatasetLoader:
    def __init__(self, base_dir):
        """
        Initializes the data loader with the root directory of the Kaggle dataset.
        """
        self.base_dir = base_dir
        self.train_img_dir = os.path.join(base_dir, "train")
        self.val_img_dir = os.path.join(base_dir, "val")
        
        # We store the training data in memory after the first load 
        # so we can quickly pull Few-Shot examples later without re-reading the disk.
        self._train_cache = None 

    def _format_coco_filename(self, image_id):
        """
        Helper function to fix the VQA/COCO image naming convention.
        """
        return f"COCO_train2014_{str(image_id).zfill(12)}.jpg"

    def load_and_balance(self, split="val", force_balance=True):
        """
        Loads the JSON files, verifies image existence, and handles the 85/15 class skew.
        """
        raw_dataset = []
        img_dir = self.train_img_dir if split == "train" else self.val_img_dir
        
        # 1. Load VizWiz Data
        vizwiz_path = os.path.join(self.base_dir, f"VizWiz_{split}.json")
        if os.path.exists(vizwiz_path):
            with open(vizwiz_path, "r") as f:
                vizwiz_data = json.load(f)
                for item in vizwiz_data:
                    item["resolved_image_path"] = os.path.join(img_dir, str(item["image_id"]))
                    item["source"] = "VizWiz"
                raw_dataset.extend(vizwiz_data)

        # 2. Load VQA Data (COCO Format)
        vqa_path = os.path.join(self.base_dir, f"VQA_{split}.json")
        if os.path.exists(vqa_path):
            with open(vqa_path, "r") as f:
                vqa_data = json.load(f)
                for item in vqa_data:
                    item["resolved_image_path"] = os.path.join(img_dir, self._format_coco_filename(item["image_id"]))
                    item["source"] = "VQA"
                raw_dataset.extend(vqa_data)

        # 3. Filter Valid Images and Split by Class
        single_class = []
        multiple_class = []
        
        for item in raw_dataset:
            if not os.path.exists(item["resolved_image_path"]):
                continue # Skip if the image file is missing
                
            if item.get("binary_label") == "single":
                single_class.append(item)
            elif item.get("binary_label") == "multiple":
                multiple_class.append(item)

        # 4. Handle Class Imbalance
        if force_balance:
            random.seed(42) # Ensures our evaluation is identical every time we run it
            random.shuffle(single_class)
            random.shuffle(multiple_class)
            
            # Cut the majority class down to match the minority class
            min_size = min(len(single_class), len(multiple_class))
            balanced_data = single_class[:min_size] + multiple_class[:min_size]
            random.shuffle(balanced_data)
            
            # Save to cache if we are loading the training set
            if split == "train":
                self._train_cache = balanced_data
                
            return balanced_data
        
        return single_class + multiple_class

    def get_few_shot_context(self, n_shots=3):
            """
            Retrieves random examples to teach Qwen to output physical nouns, NOT classifications.
            """
            if self._train_cache is None:
                self.load_and_balance(split="train", force_balance=True)
                
            examples = random.sample(self._train_cache, n_shots)
            
            # The System Instruction is absolutely rigid. No ambiguity.
            context_string = (
                "You are a precise visual extraction engine. Given an image and a question, "
                "you must list all plausible physical objects in the image that could answer the question. "
                "Do not explain. Do not classify. Output ONLY a bracketed list of short noun phrases.\\n\\n"
            )
            
            for i, ex in enumerate(examples):
                # Assuming the JSON contains the raw text answers from the crowdworkers
                # If the dataset has a list of unique answers, we join them.
                # Example fallback if the JSON structure is slightly different: 
                raw_answers = ex.get("unique_answers", ["the target object"]) 
                
                # We format it to look exactly like a Python list of strings
                formatted_answers = "[" + ", ".join([f"'{ans}'" for ans in raw_answers]) + "]"
                
                context_string += f"Example {i+1}:\\n"
                context_string += f"Question: '{ex['question']}'\\n"
                context_string += f"Plausible Visual Answers: {formatted_answers}\\n\\n"
                
            return context_string