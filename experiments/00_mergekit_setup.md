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

## Next steps
- Continue reading MergeKit README and example configs
- Identify 3-5 specific questions about merge methods
- Pick two small fine-tuned models for first merge (likely T5-Base or Llama-3.2-1B variants)
- Plan first merge: start with simple linear merge before attempting TIES
