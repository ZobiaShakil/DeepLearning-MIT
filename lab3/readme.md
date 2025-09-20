
## MIT 6.S191 - Lab 3: LLM Fine-Tuning for Style Transfer

### Executive Summary

This lab successfully fine-tuned Google's **Gemma-2B** language model to generate text in two distinct styles: **Leprechaun** and **Yoda**. Using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA**, the model was adapted with minimal computational cost. The model's performance was quantitatively evaluated using an innovative **"LLM-as-a-Judge"** technique, where a larger model (`Gemma-2-9B-it`) scored the generated outputs for style adherence. The fine-tuned model showed a significant improvement in generating style-specific text, moving from a baseline score of `0.01` to a generated text score of `0.24`, approaching the gold-standard training data score of `0.44`.

### 1. Methodology & Implementation

#### 1.1 Templating and Tokenization
*   **Templating:** A structured chat template was used to format conversations, clearly distinguishing between user prompts and model responses. This provided the model with the necessary context to generate coherent replies.
    ```python
    <start_of_turn>user
    What is your name?<end_of_turn>
    <start_of_turn>model
    My name is Gemma!<end_of_turn>
    ```
*   **Tokenization:** The Gemma tokenizer, using **Byte-Pair Encoding (BPE)**, converted text into a sequence of numerical tokens (vocabulary size: 256,000). This allowed the model to process and generate language.

#### 1.2 Parameter-Efficient Fine-Tuning (PEFT) with LoRA
*   Instead of updating all ~2.6 billion parameters of Gemma-2B, **LoRA (Low-Rank Adaptation)** was applied to efficiently fine-tune only **0.4%** (10.3 million) of the parameters.
*   The model was trained on custom datasets containing question-answer pairs, where answers were written in the target style (Leprechaun or Yoda).
*   The training loop involved:
    1.  Formatting data with the chat template.
    2.  Tokenizing the text.
    3.  Computing loss **only on the answer tokens** using a mask, forcing the model to focus on learning the style.
    4.  Updating the LoRA parameters using the **Lion optimizer**.

#### 1.3 LLM-as-a-Judge Evaluation
*   A larger model, **Gemma-2-9B-it**, accessed via the **OpenRouter API**, was used as an impartial "judge."
*   The judge was given a **system prompt** instructing it to evaluate how well a given text adhered to Yoda's style and to output a score between 0 and 10.
*   This setup was integrated with **Comet ML's Opik** framework to create a custom evaluation metric (`LLMJudgeEvaluator`).
*   Three categories of text were scored:
    *   **Base:** Original, non-stylized answers (Negative Control).
    *   **Generated:** Answers produced by the fine-tuned model.
    *   **Style (Train):** Gold-standard style examples from the training set (Positive Control).

### 2. Results & Analysis

#### 2.1 Qualitative Results
The model's ability to adopt the target style was evident from its generated responses.

**Leprechaun Style Example:**
*   **Prompt:** "What is a good story about tennis"
*   **Output:** `"Top o' the mornin' to ye, me hearty! Let me tell ye a tale fit for a grand ol' dame o' the court! Now, there was this wee lad named Finnigan, a right scrapper on the court, known as "The Tiger" for his fiery spirit..."`

**Yoda Style Example:**
*   **Prompt:** "What is the capital of France?"
*   **Output:** `"Paris, the capital of France is."`

#### 2.2 Quantitative Evaluation
The judge LLM provided quantitative scores, confirming the qualitative observations. The results clearly show the fine-tuned model learned the style effectively.

| Text Type | Average Score | Standard Deviation |
| :--- | :--- | :--- |
| **Base** (Negative Control) | 0.01 | ± 0.02 |
| **Generated** by our Model | 0.24 | ± 0.25 |
| **Style/Train** (Positive Control) | 0.44 | ± 0.26 |

**Interpretation:**
*   The **Base** text received near-zero scores, confirming the judge correctly identified non-stylized English.
*   The **Generated** text scores show a massive improvement over the base, proving the fine-tuning was successful.
*   The **Style** text from the training set received the highest scores, serving as a valid positive control that the judge recognizes true Yoda-speak.

#### 2.3 Final Model Performance
The model's final performance was measured by computing the **cross-entropy loss** on a held-out test sample of true Yoda-speak. A lower loss indicates the model assigns a higher probability to the correct style.
*   **Yoda test loglikelihood: `2.71`**

### 3. Conclusion

This lab demonstrated a complete pipeline for style-specific fine-tuning of a large language model. Key successes include:
1.  **Effective Style Transfer:** The model successfully learned to generate text in the distinctive styles of a Leprechaun and Yoda.
2.  **Efficient Training:** Using LoRA, this was achieved by updating less than 1% of the model's parameters, making the process computationally feasible.
3.  **Robust Evaluation:** The "LLM-as-a-Judge" method provided a scalable, quantitative way to evaluate subjective qualities like writing style, moving beyond simple qualitative assessment.

The project highlights the power of fine-tuning to customize LLMs for specific applications and the importance of developing robust evaluation metrics for generative AI tasks.
