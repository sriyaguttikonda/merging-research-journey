# Method Design Notes — Activation-Informed Merging

Working notes from method-design session (Week 13 pivot). We shifted from workshop paper on empirical observations (Option A from Day 7) to first-author conference paper proposing a novel merging algorithm (Option C). Sriya first author, Pruthvi co-author (compute + review).

Motivation carries over from Week 11-12 findings: standard merge methods (TIES, DARE) fail catastrophically on continued-pretraining-style deltas (Qwen Math+Coder) and degrade even on clean SFT deltas (Mistral Instruct+OpenHermes). See experiments/week11_task_vector_findings.md and experiments/week12_baselines.md for full data.

## Literature review summary

### DARE (Yu et al., 2023 — arXiv 2311.03099)

DARE basically says that when a model is fine-tuned, a lot of the changes from the original model are actually redundant, so instead of keeping the entire task vector, we can randomly drop a large percentage of the delta parameters and then rescale the remaining ones. The task vector is the difference between the fine-tuned model and the original pretrained model, and DARE does this by dropping each delta parameter with probability p and multiplying the surviving parameters by 1/(1−p), so that the expected value of the delta stays approximately the same. The main reason this works is that the fine-tuning deltas are generally very small compared to the original model weights and there is a lot of redundancy, so randomly removing some of them does not completely change the model behaviour. The paper shows that extremely high sparsity can still work surprisingly well and that the resulting sparse task vectors can then be merged with other task vectors. The main thing DARE fixes is the redundancy problem in model merging, while also making very sparse merging possible. But the weakness is that it is still random pruning, so it does not actually know which parameters are important for the model. Also, when the deltas become large or the pruning rate becomes extreme, the 1/(1−p) rescaling factor becomes very large and can amplify the remaining updates, which is where the assumptions behind DARE start breaking. So DARE is basically the starting point for the question of whether we can do something smarter than random pruning.

### DARE-the-Extreme

DARE-the-Extreme looks specifically at the cases where normal DARE starts failing instead of just assuming that DARE will continue to work at any sparsity level. The main claim is that DARE's behaviour depends strongly on the magnitude and distribution of the delta parameters as well as the pruning rate. When the pruning rate becomes very high, the 1/(1−p) rescaling factor becomes huge, and when the delta parameters themselves are large, this rescaling can cause much larger changes to the model representations and eventually hurt performance. The paper proposes DAREx-q, where instead of forcing the standard 1/(1−p) rescaling, the rescaling can be controlled using a different factor, which reduces the amplification problem at extreme sparsity. They also propose DAREx-L2, where the delta parameters are regularized during fine-tuning so that the resulting deltas are more suitable for aggressive pruning. They also investigate importance-based pruning instead of purely random pruning, especially in the large-delta regime. Their experiments show that the original DARE can degrade at extreme pruning rates, while the proposed modifications can make extreme sparsification work much better. What this paper does not solve completely is the general problem of finding the actual importance of every delta parameter during merging. It shows that magnitude/importance starts becoming important when the delta values are large, but it does not give us a complete solution for how parameter importance should be defined when multiple task vectors interact and conflict with each other.

### DELLA (DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling)

DELLA is basically where the idea of magnitude-aware sparsification becomes much more explicit. Instead of DARE giving every parameter the same probability of being dropped, DELLA makes the dropout probability depend on the magnitude/rank of the delta parameter. So large-magnitude delta parameters are less likely to be dropped, while smaller-magnitude parameters are more likely to be dropped. The important difference is that DELLA is still probabilistic like DARE, but the probability is now different for different parameters instead of using one global dropout probability. After the magnitude-based pruning, DELLA still uses the Elect and Fuse stages, so the overall idea is basically magnitude-aware Drop followed by sign election and merging. This means that the intuition of "don't randomly drop everything, use the magnitude of the delta to decide what is more likely to survive" has already been explored. DELLA shows that this magnitude-based sampling can improve merging compared with random pruning and other baselines. However, the main limitation is that magnitude is still being used as a proxy for importance. A large delta is assumed to be more important, but magnitude alone does not necessarily tell us how much that parameter actually affects the model's output or how it interacts with the other task vectors. So DELLA basically closes the obvious gap between DARE and magnitude-aware pruning, but it leaves the bigger question of whether we can define a better importance measure than magnitude alone, especially one that considers activation importance or conflicts between different task vectors.

### Progression across the three papers

Random pruning (DARE) → Extreme-regime analysis (DARE-the-Extreme) → Magnitude-aware pruning (DELLA)

The important part for our research: we should NOT say "let's invent magnitude-aware merging" because DELLA has already done that.

The question left open: Is magnitude actually the best measure of delta-parameter importance, or can we use information about activations, task conflicts, or the structure of the model to decide which deltas should survive?

This is the gap carried forward.

## Wanda paper analysis (Sun et al. — arXiv 2306.11695)

### Core mechanism

Basically, Wanda is an improved version of magnitude pruning for LLMs. Instead of saying "small weight = less important", it also checks how strongly the corresponding input feature is activated. So the importance is basically weight magnitude × activation magnitude. They get these activation values by passing a small calibration dataset through the model. Then they prune per output, meaning they compare the weights connected to the same output and remove the least important ones. This works better for LLMs because some input features can have very large activations, so even a small weight connected to them can be important. The main advantage is that Wanda can prune the model without retraining or updating the remaining weights, while still giving results close to more expensive methods like SparseGPT.

### Why Wanda beats magnitude pruning

Basically, Wanda works better because LLMs don't treat every input feature equally. Some hidden-state features can have much larger activations than others, so just looking at the weight magnitude can give the wrong idea about its importance. For example, a small weight connected to a very large activation can have a bigger effect on the output than a large weight connected to a small activation. Magnitude pruning would remove the small weight, while Wanda can recognize that it is actually important because its activation is large. This is why adding activation information makes the pruning decision more meaningful for LLMs. The experiments support this: for LLaMA-7B at 50% sparsity, magnitude pruning gives a WikiText perplexity of 17.29, while Wanda gives 7.26, which is almost the same as SparseGPT's 7.22. At the same time, Wanda does not perform the expensive weight-update process used by SparseGPT. So basically, Wanda gets much better pruning than magnitude pruning by making the importance activation-aware, while still keeping the method simple and cheap.

### Wanda's limitations

Basically, Wanda is not perfect — pruning still causes some performance loss, especially when we push the sparsity higher. The performance also depends on the type of sparsity being used. Wanda performs very well with unstructured sparsity and is competitive with SparseGPT, but for some structured settings, especially 2:4 sparsity on smaller models, SparseGPT can perform better. Also, even though Wanda can create a sparse model, simply making many weights zero does not automatically guarantee a proportional inference speedup. The actual speedup depends on whether the hardware and matrix-multiplication kernels can efficiently exploit that particular sparse pattern. The paper also shows that fine-tuning can recover much of the performance lost during pruning, but then we lose some of Wanda's main advantage of being a no-retraining pruning method. So basically, Wanda gives a very good trade-off between simplicity, pruning efficiency and model performance, but there is still a trade-off between sparsity, accuracy and actual hardware speedup.

## Transfer analysis — does Wanda's approach apply to merging?

### Transfer question 1: whose activations do we use?

Basically, for merging Mistral-Instruct + OpenHermes, I think the most reasonable starting point is to use activations from the base model, rather than taking only Mistral-Instruct or only OpenHermes activations. The reason is that the task vectors represent the changes made to the same base model, so the base model gives a common reference for measuring how important each parameter is before applying either delta. If I use only Mistral-Instruct activations, I am assuming that the importance of the delta should be judged mainly according to the Instruct model's behavior; similarly, using OpenHermes activations makes the metric biased toward OpenHermes. Averaging the two could be another option, but it mixes the two task-specific behaviors and may hide differences between them. So my first intuition is base-model activations as the common reference, and then compare this against using the individual fine-tuned model activations as an ablation. The important thing is that this is an assumption that needs to be experimentally tested, not something Wanda directly tells us. Wanda itself only has the one-model setting, where the model's own input activations are used to score its weights.

### Transfer question 2: what does "per-output" mean for task vectors?

Basically, this is where transferring Wanda to merging becomes less straightforward. In Wanda, per-output pruning works because we have one weight matrix and a clear output neuron, so we can compare all incoming weights to that particular output. But when merging two task vectors, we are not simply deciding whether a single weight is important or not. We have two delta values at the same parameter location, and we need to decide whether to keep the Mistral delta, the OpenHermes delta, both, or neither. So I don't think directly doing "per-output across both models" is enough, because it assumes that the corresponding neurons and their functional roles are aligned. Since both task vectors come from the same base architecture, the parameter positions technically correspond, but that does not necessarily mean the learned features are semantically aligned. My first approach would therefore be to preserve Wanda's structural idea but redefine the comparison group around the same parameter/tensor structure, rather than assuming neuron-level semantic correspondence. For example, within each output row of a corresponding weight tensor, we could compare the activation-weighted magnitudes of the two deltas and decide which contribution to retain. But this itself becomes a research assumption that needs validation.

### Transfer question 3: does the small-weight-large-activation phenomenon show up in task vector deltas?

Basically, we don't know yet — and I think this is actually the most important point. In Week 11, I have the per-tensor magnitudes of the Qwen Math+Coder and Mistral fine-tune deltas, so I know how large the deltas are, but I have not measured the corresponding activations on a calibration dataset. Therefore, I cannot currently claim that magnitude is a poor measure of delta importance. It is completely possible that the delta magnitude already captures most of the useful information for merging, in which case adding Wanda-style activation information may not provide much benefit. The key experiment would therefore be to take the same calibration data, measure the activations, calculate something like |ΔW| × ||X||₂ for the task-vector parameters, and compare its ranking with the ranking obtained from |ΔW| alone. If the activation-weighted delta identifies substantially different parameters, especially small deltas connected to highly activated features, then Wanda's idea has evidence for being useful in merging. If the rankings are almost identical, then activation information may not add much beyond magnitude. So right now, the data does not answer this question — it motivates the first experiment.

## Where this leaves us

The gap: activation-informed importance for merging task vectors.

Current assumptions to test:
- Use base model activations as the common reference (Q1)
- Preserve Wanda's per-output structural idea but redefine for two-task-vector setting (Q2)
- The premise itself — that magnitude ranking differs meaningfully from activation-weighted ranking on task-vector deltas — has not yet been tested (Q3)

## Next session

1. Search for prior work on activation-informed merging. Key queries: "activation-aware model merging", "Wanda merging model", "task vector importance activation", "activation-informed pruning task vector". If someone has published this in 2025-2026, adjust scope.

2. Assuming no clean prior work, sketch method as pseudocode. One page. Send to Pruthvi for review before implementing.

3. First experiment: on Qwen Math+Coder and Mistral Instruct+OpenHermes, compute per-parameter |ΔW| × ||X||₂ using base model activations on a small calibration set. Compare ranking against |ΔW| alone. If rankings differ meaningfully → evidence to build on. If identical → premise fails, rethink.

## Not yet decided

- Calibration set: what data, how large, from where
- Whether activation is computed per-token then averaged, or per-sequence, or something else
- What "output" means concretely for the per-output constraint in the merging setting
- Whether we're proposing a full new method or a preprocessing step for existing methods (TIES, DARE)
- Target venue (was ICLR 2027 workshop; now conference-quality — likely ICLR 2027 main track or NeurIPS 2027)
