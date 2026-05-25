# TIES-Merging Paper Summary

## 1. Introduction

*TIES-Merging* (Yadav et al., NeurIPS 2023) is a method for combining multiple fine-tuned models into a single multi-task model without retraining. The method improves upon earlier approaches such as *Task Arithmetic* by addressing two major causes of interference during model merging: redundant parameter updates and sign conflicts between task vectors. TIES introduces a three-step pipeline — **Trim, Elect Sign, and Disjoint Merge** — to reduce these problems and produce a stronger merged model.

The method requires all fine-tuned models to share the same pretrained base checkpoint and architecture because task vectors only make sense in the same parameter coordinate system. Unlike ensembles or routing networks, TIES outputs a single merged model with the same architecture and size as the original models.

## 2. Problem Statement

Existing model merging approaches often fail because they directly average task vectors or model weights. This creates two major issues:

1. **Redundancy Interference:** Most parameters change only slightly during fine-tuning, yet these small noisy updates still contribute to the average and dilute the truly important parameter changes.

2. **Sign Conflict Interference:** Different task vectors may push the same parameter in opposite directions. Averaging then moves the parameter toward zero, partially cancelling information from both models.

The paper shows that sign conflicts increase as more models are merged, making naive averaging increasingly ineffective.

## 3. Methodology

TIES addresses these problems through a three-step procedure.

### 3.1 Step 1: Trim

The first step removes redundant updates by keeping only the top-$k$% largest parameter changes (by absolute magnitude) in each task vector and setting the remaining values to zero.

$$\hat{\tau}_t = \text{keep-top-k}(\tau_t, k)$$

where:
- $\tau_t$ is the task vector
- $k$ is the sparsity ratio

The paper shows that keeping only the top 20% of values achieves performance comparable to using 100% of the parameters, indicating that most updates are redundant.

### 3.2 Step 2: Elect Sign

The second step resolves directional conflicts. For each parameter, the trimmed task vector values are summed across all tasks, and the sign of this sum becomes the elected sign.

$$\gamma_m[p] = \mathrm{sign}\left(\sum_t \hat{\tau}_t[p]\right)$$

This acts as a magnitude-weighted voting mechanism because larger parameter updates influence the decision more strongly than smaller ones.

### 3.3 Step 3: Disjoint Merge

The third step averages only the task vector values whose signs agree with the elected sign. Parameters with opposite signs are excluded from the merge for that specific parameter location.

The aligned task set is defined as:

$$A_p = \{ t : \mathrm{sign}(\hat{\tau}_t[p]) = \gamma_m[p] \}$$

The final merged task vector is then computed as:

$$\tau_m[p] = \frac{1}{|A_p|} \sum_{t \in A_p} \hat{\tau}_t[p]$$

Finally, the merged task vector is added back to the pretrained model weights:

$$\theta_m = \theta_{\mathrm{init}} + \lambda \tau_m$$

where:
- $\theta_{\mathrm{init}}$ is the pretrained base model
- $\lambda$ is the scaling coefficient

## 4. Novelty

The novelty of TIES is not simply merging models, but explicitly separating and addressing the two interference types:

- **Trim** handles redundancy interference
- **Elect-Sign + Disjoint Merge** handle sign conflicts

Prior approaches such as *Task Arithmetic* simply summed task vectors without interference handling.

## 5. Baselines

The paper compares TIES against several existing merging methods:

1. **Simple Averaging:** Direct averaging of model weights.
2. **Task Arithmetic:** Summation of task vectors scaled by a coefficient.
3. **Fisher Merging:** Weighted averaging using Fisher information importance scores.
4. **RegMean:** Closed-form least-squares merging based on activation matching.

The most important comparison is with *Task Arithmetic*, which is the direct predecessor method that TIES improves upon.

## 6. Experimental Results

TIES achieved strong results across both NLP and vision models, including T5 and ViT architectures. The paper reports that TIES outperformed baselines in 9 out of 10 experimental settings, with an average improvement of approximately +2.0 points over Task Arithmetic.

One particularly important finding was strong out-of-domain generalization, especially on T5-Large where TIES achieved a +4.4 improvement.

However, improvements were not uniform across all models. For example, T5-Base showed only a modest gain (+0.7). Interestingly, the pattern does not correlate cleanly with model size, since ViT-L/14 (+1.5) shows a smaller improvement than ViT-B/32 (+1.8), suggesting that task redundancy and sign-conflict structure may matter more than parameter count alone.

This suggests a possible direction for future research: understanding when interference-aware merging is most beneficial.

## 7. Limitations and Future Work

TIES relies on a heuristic sign election strategy based on the sign of summed task vectors. An oracle experiment showed a large performance gap of approximately +5.6 points between heuristic sign selection and oracle sign selection, indicating that better sign estimation could significantly improve merging quality.

Possible future research directions include:

- Learned sign prediction
- Adaptive trimming ratios
- Layer-wise merging strategies
- Confidence-weighted voting mechanisms

Even though newer methods such as DARE and DELLA extend some of TIES's ideas, TIES remains one of the foundational and most influential interference-aware model merging techniques.
