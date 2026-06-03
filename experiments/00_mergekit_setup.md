# MergeKit Setup — Week 5

## Environment
- Platform: Kaggle (CPU mode for installation)
- MergeKit installed via pip
- Verification: `import mergekit` succeeds, `mergekit-yaml --help` prints command-line usage

## Installation Process

### Step 1: Account setup
- Created Kaggle account
- Verified phone number to enable Internet and GPU access

### Step 2: Notebook setup
- Created notebook named `mergekit-setup-week1`
- Accelerator set to None (CPU) for installation phase — GPU not needed yet
- Internet toggled ON in notebook settings

### Step 3: Installation
- Command used: `!pip install mergekit -q`
- Took ~3 minutes to complete

### Issues encountered and resolved

**Issue 1: DNS error on first install attempt**
- Error message: `Temporary failure in name resolution`
- Cause: Internet access was OFF in notebook settings by default
- Resolution: Toggled Internet ON in the right-side settings panel
- Lesson: Kaggle notebooks default to Internet off even after phone verification

**Issue 2: Dependency conflict warnings during install**
- Output included red text mentioning conflicts with packages like google-adk, kaggle-environments, ydata-profiling, etc.
- Cause: Pip's dependency resolver flagging conflicts with Kaggle's pre-installed packages
- Resolution: No action needed. These conflicts don't affect MergeKit's functionality
- Lesson: Red text ≠ error. Look for actual Python tracebacks vs informational warnings

**Issue 3: Version check command failed**
- Tried: `print(mergekit.__version__)`
- Error: `AttributeError: module 'mergekit' has no attribute '__version__'`
- Cause: MergeKit doesn't expose version via the standard `__version__` attribute
- Resolution: Use `!pip show mergekit` instead
- Lesson: Different packages handle version reporting differently. Don't assume standard attributes exist.

## Concepts learned this week

### Configuration structure
- MergeKit uses YAML config files to specify merges
- Run with: `mergekit-yaml config.yaml output_directory`

### Key parameters
- `models` — list of fine-tuned models to merge (Hugging Face paths)
- `weight` — relative contribution of each model in the merge (normalized across all models)
- `density` — TIES's trim ratio (fraction of delta parameters to keep)
- `base_model` — the pretrained checkpoint (θ_pre); required for methods operating on delta parameters
- `merge_method` — which algorithm to use (linear, task_arithmetic, ties, dare_ties, etc.)
- `dtype` — floating point precision (float16, bfloat16, float32)

### Weight normalization
- `weight` parameters are coefficients, not percentages
- They get normalized across all models: each model's actual contribution = its_weight / sum_of_all_weights
- Example: weights of 2.0 and 3.0 mean 40% and 60% contribution

### Method requirements
- Linear merge: no base model needed (operates on fine-tuned weights directly)
- TIES, DARE, Task Arithmetic: require base_model (operate on delta parameters)
- Linear is the simplest baseline; other methods add interference handling

### Advanced features (for later experimentation)
- Density and weight can be specified as gradients across layers (e.g., `density: [1, 0.7, 0.1]`)
- Filters can restrict parameters to specific layer types (e.g., `filter: mlp`)
- `normalize: true` rescales merged weights for stability

## First successful merge: 2026-06-01

### What worked
- Linear merge between two copies of `google/t5-v1_1-base` (weights 0.5 + 0.5)
- Output saved to ./merged_model/ — 593MB safetensors file
- Model loads successfully via `T5ForConditionalGeneration.from_pretrained()`
- Parameter count: 296,926,464 (matches T5-v1.1-base)

### Initial error and resolution
- First attempt with `google-t5/t5-base` failed with tensor mismatch error
- T5-v1 has single `DenseReluDense.wi` weight
- T5-v1.1 has gated `wi_0` and `wi_1` weights — MergeKit expects this variant
- Lesson: model version matters; v1 and v1.1 are not interchangeable

### Warnings (informational, not blocking)
- Triton not detected — CPU mode, not relevant
- Tied weights warning — embeddings stored separately, model still loads correctly

### Status
- MergeKit infrastructure: fully functional
- Ready for next step: merging two *different* fine-tuned models

## Week 6: First meaningful merge

Date: 2026-06-02

### Setup
- Models merged:
  - google/flan-t5-base (generalist instruction-tuned)
  - mrm8488/flan-t5-base-finetuned-openai-summarize_from_feedback (summarization specialist)
- Method: linear merge, 0.5 + 0.5 weights
- dtype: float16
- Compute: CPU

### Result
- Merge ran successfully on first attempt
- Output: ./merged_flan_model (~594MB)
- Model loads correctly with 296,926,464 parameters

### Qualitative comparison on summarization task
Tested all three models on a paragraph about James Webb Space Telescope.

- FLAN-T5-base produced a single terse sentence
- Summarization fine-tune produced 3 detailed sentences
- Merged model produced 4 sentences with the most factual content of the three

The merged model inherited summarization behavior (multi-sentence output)
from the fine-tune while producing outputs distinct from both inputs.

### What I learned
- A linear merge between meaningfully different fine-tunes produces a model
  that behaves like a blend of both
- Output content isn't a simple average of input model outputs
- One example is a demonstration, not evaluation - real eval needs many inputs
  and quantitative metrics
## Week 6 Task 1: Cross-input testing

### Test text
EU-Japan trade agreement news paragraph (different style from previous science test)

### Three outputs

FLAN-T5-base:
- 1 terse sentence
- Captured main event + tariff fact

Summarization fine-tune:
- 3 sentences  
- Captured economic benefit, critics, supporters
- Did not include tariff percentage

Merged model (0.5 + 0.5):
- 3 sentences
- Captured tariff fact (FLAN-style) AND critics/supporters debate (fine-tune-style)
- Made a small factual error (said "eliminates 90 percent of goods" instead of "eliminates tariffs on 90 percent of goods")

### Observation
The merged model showed consistent behavior across both science and news inputs:
- Always produced multi-sentence summaries (like the fine-tune)
- Pulled fact-level details (like FLAN)
- Combined elements from both parents

### Caveat
Small factual error in merged output suggests merging can disturb specific 
details. Not catastrophic on this example but worth tracking in future evaluations.
## Week 6 Task 2: Weight ratio comparison

### Setup
- Three merge ratios tested: 0.7/0.3, 0.5/0.5, 0.3/0.7 (FLAN/fine-tune)
- Same test input as Task 1 (EU-Japan trade news paragraph)
- Compared against pure FLAN and pure fine-tune baselines

### Prediction
Higher FLAN weight → more terse, FLAN-like outputs
Higher fine-tune weight → more detailed, fine-tune-like outputs
Expected smooth gradient between the two extremes

### Result
The prediction was partially wrong. All three merged ratios produced 
similar multi-sentence summaries with critics/supporters structure - 
none showed clearly "more FLAN-like" or "more fine-tune-like" behavior.

The fine-tune's behavioral signature appears even at 0.3 weight.

### Failure modes observed
- 0.7/0.3 and 0.5/0.5: dropped "tariffs on" - factual error
- 0.3/0.7: produced a stutter ("90 percent tariffs on more than 90 percent")
- Pure models had no such errors

### Interpretation
- Linear merging does not produce smooth behavioral gradients
- Fine-tune influence is disproportionately strong relative to its weight
- Merging can introduce new failure modes not present in either input
- This empirically demonstrates the interference problem that TIES, DARE, 
  and other methods are designed to address
## Week 7: TIES vs Linear comparison

### Setup
- TIES merge with density 0.2, base_model: google/t5-v1_1-base
- Tested on EU-Japan trade news paragraph
- Compared against linear merges and pure models

### Reproducible findings (verified by re-running)
- Linear merges all produced clean, factually mostly-correct summaries
- TIES density 0.2 produced incoherent output ("Europe's farmers say they can't 
  protect small farmers...") — missing main event, garbled meaning
- TIES density 0.5 produced cleaner output but still slightly degraded vs linear
- Result is reproducible: re-running the test produced the same outputs

### Hypothesis
TIES's density 0.2 default may not transfer well to merging similar models 
(both FLAN-derived). When task vectors heavily overlap, aggressive trimming 
may discard shared useful information rather than resolving conflicts.

### What I learned
- Paper defaults don't always transfer to new setups
- Reproducibility matters - re-running confirmed the finding wasn't noise  
- One input still isn't enough for strong claims, but consistent reproduction 
  on one input is more reliable than I initially thought
- Genuine negative results are valuable - they point to questions worth asking

### Open question for future work
Why does TIES fail on similar models? Possible follow-ups:
- Try FLAN as base_model instead of T5-v1.1-base
- Try higher density values (0.7, 0.9)
- Test with less similar models (different tasks)
## Week 8: DARE-TIES density sweep

### Setup
Tested DARE-TIES on the same FLAN model pair at three density values:
- 0.9 (keep 90%, drop 10%)
- 0.5 (keep 50%, drop 50%)
- 0.1 (keep 10%, drop 90%)

Same input news paragraph as Week 7. Same base model. Only density varies.

### Results

Density 0.9: Clean coherent summary with main event, critics, supporters.
Slight phrasing redundancy but otherwise good output.

Density 0.5: Complete collapse. Output: "microspor as as as in as as..."
Repetitive English-like gibberish.

Density 0.1: Complete collapse. Output: "ingrijire bebelus bebelus..."
Romanian-language baby-care vocabulary. Non-English hallucination.

### Finding
DARE-TIES on this model pair works only at very low drop rates (around 10%).
At 50% or higher drop rates, the model collapses entirely.

This contradicts the DARE paper's claim that dropping 90-99% of deltas
works well. The paper's experiments used:
- Larger models (Llama-2-13B+)
- Task-different fine-tunes (math, code, instruction)

My setup is different:
- Smaller model (T5-base, ~250M params)
- Highly similar fine-tunes (both FLAN-derived)

### Hypothesis
DARE's drop-and-rescale strategy may only work when task vectors are roughly
orthogonal (different task domains). When source models are highly similar
(overlapping task vectors), random dropping disrupts consistent patterns and
2x rescaling amplifies noise into model collapse.

### Open questions
- Would DARE-TIES work on less similar T5 fine-tunes (different tasks)?
- Does the failure threshold (around density 0.5-0.7) generalize across model pairs?
- Could task vector cosine similarity predict the maximum tolerable drop rate?

## Week 9 — Day 1: Evaluation infrastructure

### Setup
- Created Kaggle Dataset (flan-merged-models-week6-8) containing all 8 
  merged models from Weeks 6-8
- Started new notebook (mergekit-week9-evaluation) with the Dataset attached
- Path to merged models: /kaggle/input/datasets/sriyaguttikonda/flan-merged-models-week6-8/

### Day 1 work
- Installed datasets library
- Loaded CNN/DailyMail test split (11,490 examples)
- Each example has 'article' (full text), 'highlights' (reference summary), 'id'
- Confirmed dataset loads correctly with a sample example

## Week 9 — Day 2: ROUGE setup

### Decisions
- Evaluation subset: first 50 examples of CNN/DailyMail test split
- Deterministic (range(50)) for fair model comparison and reproducibility
- All 8 merged models will be evaluated on these same 50 articles

### Data characteristics (50-example subset)
- Article length: 630 to 9,257 chars (mean 3,608)
- Summary length: 104 to 346 chars (mean 199)
- Note: most articles exceed T5-base's 512-token context window
- Articles will be truncated during tokenization; evaluation effectively 
  measures "summarize the first ~400 words"

### ROUGE setup
- Installed rouge-score library
- Verified ROUGE-1, ROUGE-2, ROUGE-L compute correctly
- Identical text gives 1.0 across all variants
- Unrelated text gives ~0.13 ROUGE-1 (common word noise floor), 
  0.00 ROUGE-2 (bigrams don't coincidentally match)

## Week 9 — Day 3: Building evaluation pipeline

### Built
- `evaluate_model()` function that runs a model on the 50-example subset 
  and returns averaged ROUGE-1/2/L scores
- Uses beam search (num_beams=4), max_length=128 for generation

### Critical finding during pipeline testing
First evaluation of FLAN-T5-base produced low scores (ROUGE-1: 0.18) with 
generated summaries that were broken or near-empty.

Sanity check showed example output "Palestinian authority's authority." 
when the reference was a real two-sentence summary.

Tested four prompt formats. Found that prompt format dramatically affects 
output quality:
- "summarize: {article}" → broken outputs
- "Article: ... Summary:" → quotes, not summaries  
- "Please write a short summary of this article: {article}" → coherent summaries

### Updated evaluation
Re-ran FLAN-T5-base with "Please write a short summary..." prompt:
- ROUGE-1: 0.1819 → 0.2600 (+43%)
- ROUGE-2: 0.0773 → 0.1233 (+60%)
- ROUGE-L: 0.1549 → 0.2105 (+36%)

### Methodological note
Prompt format affects ROUGE scores by 36-60% with no other changes.
This means: any merge evaluation must hold prompt format constant across 
all models being compared. Using different prompts per model would mix 
prompt effects with merge effects.

All 8 models will be evaluated with the same "Please write a short summary" prompt.

### Tomorrow (Day 4)
Run remaining 7 models through the same pipeline, build comparison table.
