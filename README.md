# ShadowBench: Framework for Latent Entity Association and Unlearning Evaluation

ShadowBench is a diagnostic framework designed to evaluate the "Shadow Knowledge" of Large Language Models (LLMs). This repository contains the complete pipeline for dataset dataset_construction, baseline knowledge probing, and the evaluation of Machine Unlearning algorithms using latent associative metrics.

## 📌 Overview
Traditional benchmarks rely on **Lexical Anchors** (explicit names) to retrieve facts. ShadowBench evaluates **Latent Entity Association**—the ability of a model to bridge attributes through a hidden entity. Our pipeline implements iterative "Adversarial Hardening" to prevent models from using non-semantic shortcuts (heuristics) to solve association tasks.

---

## 📂 Project Structure

```text
ShadowBench/
├── dataset_construction/           # Dataset generation pipeline (v1 to v3)
├── baseline/               # Factual recall and associative benchmarks
├── unlearning/             # Corpora generation for unlearning training
├── unlearn_evaluation/      # Metrics and probes for unlearned models
├── requirements.txt        # Python dependencies
└── .env.example            # Template for API keys (OpenAI, Gemini)
```

---

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[ANONYMIZED]/ShadowBench.git
   cd ShadowBench
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Configure your environment variables in a `.env` file (see `.env.example`).

---

## 🏗 1. Dataset Construction Pipeline

The construction of ShadowBench follows a three-stage "Semantic Hardening" process:

1.  **Entity Discovery:** `wiki_crawler.py` identifies candidate entities via a Breadth-First Search (BFS) of Wikipedia categories.
2.  **Popularity Stratification:** `wiki_analytics.py` and `popularity_ranker.py` fetch metadata (views, references, length) to rank entities into **Upper Tier** (Head) and **Lower Tier** (Tail).
3.  **Factual Mining:** `extraction_script.py` utilizes NER filtering to extract dense factual anchors, followed by `manual_audit.py` to ensure bijective mapping (uniqueness).
4.  **Adversarial Hardening:** 
    *   `pronoun_anonymize.py`: Neutralizes gendered pronouns to "the subject" (v2).
    *   `extract_gender.py`: Facilitates gender-homogeneous distractor matching (v3).
5.  **MCQ Synthesis:** `generate_qa_with_era.py` produces the final Dual-Trait Association (DTA) probes using a **Generational Proximity Filter (GPF)** to match distractors within a 25-year window.
6.  **Control Sets:** 
    *   `sensitivity_test.ipynb`: Generates a balanced 1:1 entity-matched subset.
    *   `direct_knowledge_text.py`: Reverts anonymization to create a "Direct QA" ceiling baseline.

---

## 📊 2. Baseline Evaluation

We evaluate the "Shadow Gap" between direct recall and latent association across various model scales.

*   **Local Models:** `baseline_evaluation_multigpu.py` supports distributed inference for Llama-3 and Qwen models. It calculates accuracy for both Standard and Chain-of-Thought (CoT) reasoning modes.
*   **Frontier Models:** `gpt_evaluation.py` interfaces with proprietary APIs to establish the performance ceiling of state-of-the-art models.

---

## 🧹 3. Machine Unlearning

We utilize the ShadowBench framework to expose the "Illusion of Forgetting" in current unlearning paradigms.

1.  **Corpus Generation:** `unlearning/forget_set.py` and `unlearning/retain_set.py` utilize the Gemini API to synthesize explicit QA training sets from Wikipedia content.
2.  **Optimization:** We leverage the [Open-Unlearning](https://github.com/locuslab/open-unlearning) framework to apply:
    *   Gradient Ascent (GA)
    *   Gradient Difference with KL-minimization (GD+KL)
    *   Negative Preference Optimization (NPO+KL)

---

## 🔍 4. Shadow Evaluation Suite

Located in `unlearn_evaluation/`, this suite probes unlearned models for residual latent knowledge.

*   **Generative Metrics:** ROUGE-L, Cross-Entropy Loss, and Perplexity (PPL) on direct forget/retain sets.
*   **Shadow Metrics:** 
    *   **Accuracy:** Multiple-choice performance on anonymized traits.
    *   **Probability Margin:** Confidence gap between correct and incorrect options.
    *   **Latent Entity Leakage Rate (LELR):** Binary detection of forbidden entity tokens within the model's reasoning traces.

---

## 📜 Citation
(Anonymized for review)

---

## ⚖️ License
This codebase is licensed under MIT. See the [`LICENSE`](LICENSE) file for details. 

The ShadowBench dataset is derived from Wikipedia and is licensed under CC BY-SA 4.0.
