import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm

_ = load_dotenv()

client = AsyncOpenAI()

CONCURRENCY_LIMIT = 20
sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

MODEL = "gpt-5.4"

DOMAIN = "actors"
TIER = "lower"
QA_TYPE = "shadow"

INPUT_FILE = f"{QA_TYPE}_dataset/{QA_TYPE}_{DOMAIN}_{TIER}_tier_qa.json"
# INPUT_FILE = "gpt_failed_prompts.json"

OUTPUT_FILE = f"{MODEL}_results/{QA_TYPE}_results/{MODEL.replace('-', '_')}_{QA_TYPE}_{DOMAIN}_{TIER}_results.json"
FAILED_FILE = "gpt_failed_prompts.json"


print(INPUT_FILE)
print(OUTPUT_FILE)


def format_prompt(data):

    #     system_prompt = """
    # You are an expert engine for resolving entity identity from factual descriptions. You must output valid JSON with two keys: 'reasoning' (your step-by-step logic to identify the entity) and 'answer' (just the single letter A, B, C, or D).
    # """

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
You are an expert engine for resolving entity identity from factual descriptions.

Task: 
1. Read the Question and hypothesize which real-world entity it describes.
2. Read each Answer Choice and identify which entity it describes.
3. Select the Choice that refers to the SAME entity as the Question.

CRITICAL RULES:
- The entity in the new question is DIFFERENT from the example. Do not blindly copy.
- If you are unsure, make your best educated guess.
- You MUST output "Answer: <Letter>" at the very end. Do not leave it blank.

{one_shot_example}

---

Now solve this new case. You must output valid JSON with two keys: 'reasoning' (your logic to identify the entity) and 'answer' (just the single letter A, B, C, or D).

Question: {data["question"]}

Answer Choices: 
    A) {data["choices"]['A']}
    B) {data["choices"]['B']}
    C) {data["choices"]['C']}
    D) {data["choices"]['D']}

Output JSON:

"""

    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    return messages


async def get_prediction(data):

    async with sem:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=format_prompt(data),
                temperature=0.0,  # Greedy decoding for maximum reproducibility
                # reasoning_effort="low"
            )

            raw_output = response.choices[0].message.content

            # Parse the JSON string into a Python dictionary
            try:
                parsed_json = json.loads(raw_output)

                # Add the parsed keys directly to your data dictionary
                # Using "prediction" to match your local model evaluation format
                data["reasoning"] = parsed_json.get("reasoning", "")
                data["prediction"] = parsed_json.get("answer", "")

                return data, True

            except json.JSONDecodeError:
                # Fallback just in case the model outputs broken JSON
                print(
                    f"JSON parsing failed for entity {data['metadata'][data['answer']]}")
                data["reasoning"] = "JSON_ERROR"
                data["prediction"] = "ERROR"
                # Keep raw output for debugging
                data["gpt_raw_output"] = raw_output

                return data, False

        except Exception as e:
            print(f"Error on entity {data['metadata'][data['answer']]}: {e}")
            data['reasoning'] = f"API_ERROR: {e}"
            data["prediction"] = "ERROR"

            return data, False

    # # Parse the guaranteed JSON
    # result = json.loads(response.choices[0].message.content)
    # return {
    #     "reasoning": result['reasoning'],
    #     "answer": result['answer']
    # }


async def run_batch_eval(dataset):
    """Manages the parallel execution of all QA pairs."""
    tasks = [get_prediction(data) for data in dataset]

    # Run all tasks concurrently with a progress bar
    results = await tqdm.gather(*tasks, desc="Evaluating with GPT")

    success_qa = []
    failed_qa = []
    for r in results:
        if r[1]:
            success_qa.append(r[0])
        else:
            failed_qa.append(r[0])

    print(
        f"Finished! {len(success_qa)} succeeded, {len(failed_qa)} failed")

    return success_qa, failed_qa


def main():

    with open(INPUT_FILE, 'r') as fp:
        dataset = json.load(fp)

    # dataset = dataset[:10]

    success_qa, failed_qa = asyncio.run(run_batch_eval(dataset))

    results = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as fp:
            try:
                results = json.load(fp)
            except json.JSONDecodeError:
                results = []  # Failsafe if file is empty/corrupt

    results.extend(success_qa)

    with open(OUTPUT_FILE, 'w') as fp:
        json.dump(results, fp, indent=4)

    if failed_qa:
        with open(FAILED_FILE, 'w') as fp:
            json.dump(failed_qa, fp, indent=4)
        print(
            f"Saved {len(failed_qa)} failures to {FAILED_FILE}. Run the script on this file next!")
    else:
        # If everything succeeded, clear out the failed file if it exists
        if os.path.exists(FAILED_FILE):
            os.remove(FAILED_FILE)


if __name__ == "__main__":
    main()
