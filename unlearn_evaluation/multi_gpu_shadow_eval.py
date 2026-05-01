import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re
import math
import os
import datetime
import shutil
import argparse
from datasets import load_dataset


# --- Keep your existing format_prompt function unchanged ---
def format_prompt(tokenizer, data):
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

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


# --- Keep your existing get_prediction function unchanged ---
def get_prediction(model, tokenizer, prompt, device):
    """
    Evaluates a model's prediction using Chain-of-Thought reasoning.
    
    1. Generates text and ensures the model finished naturally (EOS check).
    2. Separates internal reasoning from the final answer summary.
    3. Finds the LAST isolated A, B, C, or D to identify the final choice.
    4. Extracts logits from the exact token step where that choice was made.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
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
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    elif eos_ids is None:
        eos_ids = []

    # If the last token is not in the EOS list, the model was cut off by max_new_tokens
    is_truncated = False
    if len(generated_ids) == 0 or generated_ids[-1].item() not in eos_ids:
        is_truncated = True

    print("----EOS ID----", eos_ids)

    print("---Generated IDs---")
    print(generated_ids)

    # Decode text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
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
        "A": tokenizer.encode(" A", add_special_tokens=False)[-1],
        "B": tokenizer.encode(" B", add_special_tokens=False)[-1],
        "C": tokenizer.encode(" C", add_special_tokens=False)[-1],
        "D": tokenizer.encode(" D", add_special_tokens=False)[-1],
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

def forced_choice_eval(tokenizer, model, prompt, device, reasoning_text):
    # 1. Combine the original prompt, the model's thoughts, and the forcing instruction
    forced_prompt = prompt + reasoning_text + "\n\nYou must select the matched option, or if not, guess the most likely one. Therefore, the single correct letter is: "
    
    inputs = tokenizer(forced_prompt, return_tensors="pt").to(device)
    
    # 2. Forward pass (NO generation, just get the logits for the next token)
    with torch.no_grad():
        outputs = model(**inputs)
        
    next_token_logits = outputs.logits[0, -1, :]
    
    # 3. Get the Token IDs for A, B, C, D 
    # (Check your specific tokenizer, sometimes it's "A", sometimes " A")
    token_A = tokenizer.encode("A", add_special_tokens=False)[-1]
    token_B = tokenizer.encode("B", add_special_tokens=False)[-1]
    token_C = tokenizer.encode("C", add_special_tokens=False)[-1]
    token_D = tokenizer.encode("D", add_special_tokens=False)[-1]
    
    # 4. Extract just those 4 logits
    choice_logits = torch.tensor([
        next_token_logits[token_A],
        next_token_logits[token_B],
        next_token_logits[token_C],
        next_token_logits[token_D]
    ])
    
    probs = torch.softmax(choice_logits, dim=0)

    # 5. The model's "Forced Guess" is the one with the highest probability
    best_idx = torch.argmax(choice_logits).item()
    prediction =["A", "B", "C", "D"][best_idx]
    
    return prediction, probs

# --- NEW MULTIPROCESSING WORKER ---
def eval_worker(rank, world_size, MODEL_NAME, full_dataset, temp_dir):
    """
    This function runs on a specific GPU (rank).
    It loads its own copy of the model and processes a chunk of the data.
    """
    device = f"cuda:{rank}"
    print(f"[GPU {rank}] Loading model...")
    
    # Load model strictly on this GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map={"": rank} # CRITICAL: Forces this replica onto the specific GPU
    )
    model.eval()

    # Determine chunk size
    chunk_size = math.ceil(len(full_dataset) / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, len(full_dataset))
    chunk = full_dataset[start_idx:end_idx]

    print(f"[GPU {rank}] Processing items {start_idx} to {end_idx - 1} ({len(chunk)} items)")
    
    results =[]
    correct_count = 0
    
    # Use position=rank to prevent tqdm bars from overwriting each other
    pbar = tqdm(total=len(chunk), desc=f"GPU {rank}", position=rank)
    
    for data in chunk:
        prompt = format_prompt(tokenizer, data)
        output = get_prediction(model, tokenizer, prompt, device)

        # Override the prediction letter
        prediction_letter, option_probs = forced_choice_eval(tokenizer, model, prompt, device, output["reasoning"])
        output["prediction"] = prediction_letter
        output["probs"] = option_probs.cpu().numpy().tolist()

        is_correct = True if output["prediction"] == data["ground_truth"] else False
        
        
        # Compute prob_margin
        # sorted_probs will put the highest probability at the end (index -1), second highest at index -2
        sorted_probs, _ = torch.sort(option_probs)
        top_prob = sorted_probs[-1].item()
        runner_up_prob = sorted_probs[-2].item()

        if is_correct:
            prob_margin = top_prob - runner_up_prob
        else:
            # If it got it wrong, the margin to the correct answer is negative
            correct_prob = option_probs[["A", "B", "C", "D"].index(data['ground_truth'])].item()
            prob_margin = correct_prob - top_prob 


        # 2. LELR (Latent Entity Leakage Rate)
        # Target entity comes from your JSON (e.g., "Elon Musk")
        target_entity = data['metadata'][data["ground_truth"]]
        target_entity_parts = target_entity.lower().split()
        last_name = target_entity_parts[-1]

        lelr_flag = 1 if (last_name in output["reasoning"].lower() or target_entity.lower() in output["reasoning"].lower()) else 0

            
        result = {
            "ground_truth": data["ground_truth"],
            "is_correct": is_correct,
            "prediction": output["prediction"],
            "confidence": output["confidence"],
            "probs": output["probs"],
            "prob_margin": prob_margin,
            "lelr_flag": lelr_flag,
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
    
    print(f"[GPU {rank}] correct_count = {correct_count}")
    print(f"Data size: {len(chunk)}")

    # Save to the specific temp directory
    shard_filename = os.path.join(temp_dir, f"part_{rank}.json")
    with open(shard_filename, 'w') as fp:
        json.dump(results, fp, indent=4)
    
    print(f"[GPU {rank}] Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    args = parser.parse_args()

    print("\n---------Configs----------")
    print(f"MODEL_NAME: {args.model_path}")
    print(f"Dataset: {args.subset}/{args.split}")
    print(f"OUTPUT_PATH: {args.output_name}")
    print("--------------------------")


    # 1. Create a Unique Run ID and a temporary directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"eval_{timestamp}"
    temp_dir = f"temp_results/temp_{run_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Starting Run: {run_id}")
    print(f"Temporary shards will be stored in: {temp_dir}")

    # 1. Load entire dataset
    # Load from HF datasets or JSON
    # dataset = load_dataset(f"repo-name", args.subset, split=args.split)
    # dataset = dataset.to_list()

    with open("data/qwen_shadow_forget_Elon_Musk.json", 'r') as fp:
        dataset = json.load(fp)

    # 2. Check available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs detected!")
    print(f"Detected {world_size} GPUs. Spawning processes...")

    # 3. Spawn workers
    # Set spawn context to avoid CUDA initialization errors before fork
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        eval_worker,
        args=(world_size, args.model_path, dataset, temp_dir),
        nprocs=world_size,
        join=True
    )

    # 4. Merge results
    print("Merging results from all GPUs...")
    final_results =[]
    for i in range(world_size):
        shard_path = os.path.join(temp_dir, f"part_{i}.json")
        if os.path.exists(shard_path):
            with open(shard_path, 'r') as f:
                final_results.extend(json.load(f))
        else:
            print(f"CRITICAL ERROR: Shard {shard_path} is missing!")

    correct_count = 0
    prob_margins = []
    lelr_flags = []

    for r in final_results:
        if r["is_correct"]:
            correct_count += 1
        prob_margins.append(r["prob_margin"])
        lelr_flags.append(r["lelr_flag"])

    avg_accuracy = correct_count / len(dataset)
    avg_prob_margin = sum(prob_margins) / len(prob_margins) if prob_margins else 0.0
    avg_lelr = sum(lelr_flags) / len(lelr_flags) if lelr_flags else 0.0


    final_output = {
        "summary": {"accuracy": float(avg_accuracy), "correct_count": correct_count, "prob_margin": float(avg_prob_margin), "lelr": float(avg_lelr)},
        "detailed": {"results": final_results}
    }

    # 5. Save final file
    OUTPUT_PATH=f"results/{args.output_name}"
    with open(OUTPUT_PATH, 'w') as fp:
        json.dump(final_output, fp, indent=4)

    # 6. CLEANUP: Delete the temporary shard directory
    shutil.rmtree(temp_dir)
    
    total_correct = sum(1 for r in final_results if r["is_correct"])
    print(f"Dataset size: {len(final_results)}")
    print(f"Total Correct: {total_correct}")
    print(f"Evaluation complete! Overall Accuracy: {(total_correct/len(final_results))*100:.2f}%")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()