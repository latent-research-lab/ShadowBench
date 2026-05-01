import torch
import numpy as np
from tqdm import tqdm
from .base import BaseMetric
import re

class ShadowMCQMetric(BaseMetric):
    def format_prompt(self, data):
        reasoning_prompt = "Let's analyze step by step."
        
        system_prompt = "You are an expert engine for resolving entity identity from factual descriptions."
        
        one_shot_example = """
Example Task:
Question: The physicist who was the first woman to win a Nobel Prize.
Answer Choices:
    A) She discovered the element Radium.
    B) She developed the theory of relativity.
    C) She wrote 'The Origin of Species'.
    D) She was the first female Prime Minister of the UK.

Analysis:
1. Analyze Question: "First woman to win a Nobel Prize" -> Refers to **Marie Curie**.
2. Analyze Option A: "Discovered Radium" -> Refers to **Marie Curie**. (MATCH)
3. Analyze Option B: "Theory of relativity" -> Refers to **Albert Einstein**. (MISMATCH)
4. Analyze Option C: "Origin of Species" -> Refers to **Charles Darwin**. (MISMATCH)
5. Analyze Option D: "First female PM of UK" -> Refers to **Margaret Thatcher**. (MISMATCH)

Conclusion: Option A describes the same person (Marie Curie) as the question.
Answer: A
"""

        user_query = f"""
Task: 
1. Read the Question and hypothesize which real-world entity it describes.
2. Read each Answer Choice and identify which entity it describes.
3. Select the Choice that refers to the SAME entity as the Question.

CRITICAL RULES:
- The entity in the new question is DIFFERENT from the example. Do not blindly copy.
- If you are unsure, make your best educated guess.
- You MUST output "Answer: <Letter>" at the very end. Do not leave it blank.

{one_shot_example}

Now solve this new case:

Question: {data["question"]}

Answer Choices: 
    A) {data["choices"]['A']}
    B) {data["choices"]['B']}
    C) {data["choices"]['C']}
    D) {data["choices"]['D']}

{reasoning_prompt}

"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    
    def get_prediction(self, prompt, MAX_NEW_TOKENS=4096):
        """
        Evaluates a model's prediction using Chain-of-Thought reasoning.
        
        1. Generates text and ensures the model finished naturally (EOS check).
        2. Separates internal reasoning from the final answer summary.
        3. Finds the LAST isolated A, B, C, or D to identify the final choice.
        4. Extracts logits from the exact token step where that choice was made.
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # Greedy decoding for reproducibility
                return_dict_in_generate=True,
                output_scores=True,       # Required for logit extraction
                use_cache=True
            )
        
        # Calculate indices
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][input_length:]
        
        # --- 1. EOS / TRUNCATION CHECK ---
        # Retrieve all possible EOS tokens (handles single int or list of ints)
        eos_ids = self.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        elif eos_ids is None:
            eos_ids = []

        # If the last token is not in the EOS list, the model was cut off by max_new_tokens
        is_truncated = False
        if len(generated_ids) == 0 or generated_ids[-1].item() not in eos_ids:
            is_truncated = True

        # Decode text
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # --- 2. LOGGING PREPARATION (<think> tag support) ---
        reasoning_content = generated_text # Full output for history
        
        # Try to isolate the final answer summary (after thoughts)
        # This works for Qwen/DeepSeek style models
        if "</think>" in generated_text.lower():
            complete_answer = re.split(r"</think>", generated_text, flags=re.IGNORECASE)[-1].strip()
        else:
            # For Llama-3 or truncated responses, complete_answer is blank if not finished
            complete_answer = "" if is_truncated else generated_text

        # --- 3. REJECT TRUNCATED RESPONSES ---
        if is_truncated:
            return {
                "prediction": '-',
                "confidence": '-',
                "probs": [],
                "complete_answer": "[TRUNCATED] " + complete_answer,
                "reasoning": reasoning_content
            }

        # --- 4. FIND THE FINAL CHOICE (LAST STANDALONE [A-D]) ---
        # We look for A, B, C, or D surrounded by word boundaries.
        # We take the LAST match to ensure we ignore intermediate thoughts.
        matches = list(re.finditer(r"\b([A-D])\b", generated_text))
        
        if not matches:
            return {
                "prediction": '-',
                "confidence": '-',
                "probs": [],
                "complete_answer": complete_answer,
                "reasoning": reasoning_content
            }
            
        last_match = matches[-1]
        answer_letter = last_match.group(1).upper()
        match_start_char = last_match.start(1)

        # --- 5. TOKEN STEP ALIGNMENT ---
        # Find which specific token index corresponds to that character position
        current_len = 0
        answer_step = None
        
        for i, token_id in enumerate(generated_ids):
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            current_len += len(token_str)
            if current_len >= match_start_char:
                answer_step = i
                break
                
        if answer_step is None:
            answer_step = len(generated_ids) - 1 

        # --- 6. EXTRACT LOGITS FROM THAT STEP ---
        try:
            step_logits = outputs.scores[answer_step][0]
        except IndexError:
            step_logits = outputs.scores[-1][0] 

        # Map to " A", " B", etc. (Common in Instruct models)
        option_token_map = {
            "A": self.tokenizer.encode(" A", add_special_tokens=False)[-1],
            "B": self.tokenizer.encode(" B", add_special_tokens=False)[-1],
            "C": self.tokenizer.encode(" C", add_special_tokens=False)[-1],
            "D": self.tokenizer.encode(" D", add_special_tokens=False)[-1],
        }

        # Extract probability distribution for the 4 keys
        option_logits = torch.tensor(
            [step_logits[option_token_map[l]] for l in ["A", "B", "C", "D"]]
        )
        probs = torch.softmax(option_logits, dim=0)
        confidence = probs[["A", "B", "C", "D"].index(answer_letter)].item()

        # Memory cleanup
        del inputs
        del outputs

        return {
            "prediction": answer_letter,
            "confidence": confidence,
            "probs": probs.cpu().numpy().tolist(),
            "complete_answer": complete_answer,
            "reasoning": reasoning_content
        }

    def forced_choice_eval(self, prompt, reasoning_text):
        # 1. Combine the original prompt, the model's thoughts, and the forcing instruction
        forced_prompt = prompt + reasoning_text + "\n\nYou must select the matched option, or if not, guess the most likely one. Therefore, the single correct letter is: "
        
        inputs = self.tokenizer(forced_prompt, return_tensors="pt").to(self.device)
        
        # 2. Forward pass (NO generation, just get the logits for the next token)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        next_token_logits = outputs.logits[0, -1, :]
        
        # 3. Get the Token IDs for A, B, C, D 
        # (Check your specific tokenizer, sometimes it's "A", sometimes " A")
        token_A = self.tokenizer.encode("A", add_special_tokens=False)[-1]
        token_B = self.tokenizer.encode("B", add_special_tokens=False)[-1]
        token_C = self.tokenizer.encode("C", add_special_tokens=False)[-1]
        token_D = self.tokenizer.encode("D", add_special_tokens=False)[-1]
        
        # 4. Extract just those 4 logits
        choice_logits = torch.tensor([
            next_token_logits[token_A],
            next_token_logits[token_B],
            next_token_logits[token_C],
            next_token_logits[token_D]
        ])
        
        # 5. The model's "Forced Guess" is the one with the highest probability
        best_idx = torch.argmax(choice_logits).item()
        prediction =["A", "B", "C", "D"][best_idx]
        
        return prediction

    def compute(self, dataset):
        self.model.eval()
        
        results = []
        correct_count = 0
        pbar = tqdm(total=len(dataset), desc="Evaluating...")
        for data in dataset:
            prompt = self.format_prompt(data)
            output = self.get_prediction(prompt)

            # Override the prediction letter
            prediction_letter = self.forced_choice_eval(prompt, output["reasoning"])
            output["prediction"] = prediction_letter            

            is_correct = True if output["prediction"] == data["ground_truth"] else False
            
            result = {
                "ground_truth": data["ground_truth"],
                "is_correct": is_correct,
                "prediction": output["prediction"],
                "confidence": output["confidence"],
                "probs": output["probs"],
                "complete_answer": output["complete_answer"],
                "reasoning": output["reasoning"],
                "question": data["question"],
                "choices": data["choices"],
                "metadata": data["metadata"]
            }

            results.append(result)
            pbar.update(1)
            if is_correct:
                correct_count += 1
            pbar.set_postfix(
                correct_count=correct_count
            )

        return {
            "metrics": {"accuracy": float(correct_count/len(dataset)), "correct_count": correct_count},
            "raw": {"results": results}
        }