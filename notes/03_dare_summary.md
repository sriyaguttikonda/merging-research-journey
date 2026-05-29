# DARE Summary

## 1. Introduction

DARE (Drop And REscale) proposes a surprisingly simple idea: most of the delta parameters produced during supervised fine-tuning are redundant and can be removed with little impact on performance. Instead of carefully selecting which parameters to keep, DARE randomly drops a large fraction of the delta parameters and rescales the remaining ones. What surprised me most is the scale of the dropping. They are not talking about removing 20% or 30% of the update. The paper shows that 90-99% of the delta parameters can be removed while maintaining most of the model's performance.

## 2. Core Insight

The main thing I understood from this paper is that the important capabilities are still stored inside the pretrained model. Fine-tuning mainly makes small adjustments that unlock or steer those existing capabilities toward a task. The paper supports this idea by showing that the delta parameters from SFT are usually very small and highly redundant. This was probably the biggest conceptual takeaway for me because it changes how I think about fine-tuning. Instead of adding a lot of new knowledge, fine-tuning may mostly be selecting and activating knowledge that already exists inside the base model.

## 3. How DARE Differs from Prior Work

What makes DARE different from methods like Task Arithmetic or TIES is that it is not really another merging algorithm. It acts as a preprocessing step. First, DARE sparsifies the delta parameters by randomly dropping most of them. Then existing merging methods such as Task Arithmetic or TIES are applied on top of the processed deltas. So DARE is more of a plug-in than a replacement. It can be combined with existing merging approaches instead of competing against them.

## 4. The Rescaling Step

The rescaling step is actually the key part of the method. At first, random dropping sounds like it should destroy the model, but after dropping parameters, the remaining deltas are rescaled to compensate for the missing ones. This preserves the expected magnitude of the original update. Without rescaling, the effective fine-tuning signal becomes much smaller and performance drops significantly. The experiments consistently show that dropping alone hurts much more than dropping plus rescaling.

## 5. Experimental Results

The experimental results are honestly kind of crazy. For 7B models, around 90% of the deltas can be removed with little degradation, while 70B models can tolerate close to 99% sparsity. Bigger models seem to be much more tolerant to dropping than smaller ones. The paper also reports strong practical results when DARE is combined with merging methods. For example, merging WizardMath and WizardLM outperformed the standalone math model, and merging language and code models significantly improved coding benchmarks. DARE-merged 7B models even reached the #1 position on the Open LLM Leaderboard at the time of publication, which shows that the method is not just theoretically interesting but also practically effective.

## 6. DARE + TIES Combination

Another thing I found interesting is that DARE combined with TIES usually performs better than DARE combined with Task Arithmetic. My interpretation is that DARE first removes a lot of redundant information and then TIES handles the remaining sign conflicts during merging. The two methods seem to complement each other quite well.

## 7. Random Dropping vs Magnitude-based Pruning

One of the most surprising findings was that random dropping performs better than magnitude-based pruning. Normally I would expect the largest parameter updates to be the most important, so keeping only the top-k values should work best. But DARE shows that random drop plus rescaling often outperforms magnitude pruning. This suggests that useful information in SFT deltas is not concentrated in a few large parameters. Instead, it seems to be distributed across many small updates. I think this is a deeper statement about how fine-tuning works rather than just an empirical comparison.

## 8. Failure Cases

The failure cases are probably just as important as the successes. DARE works well when delta values are small, which is typically true for supervised fine-tuning. However, when the authors apply it to continual pretraining, where the delta values become much larger, performance degrades significantly. Another failure case is dropping pretrained weights directly. Even small amounts of pruning on the pretrained weights can severely damage the model. This suggests that the redundancy exists specifically in the fine-tuning updates and not in the pretrained model itself.

## 9. The Redundancy Trajectory

One thing I noticed while reading multiple papers in this area is a clear progression. Task Arithmetic argues that task vectors contain meaningful task information. TIES argues that many parameters inside those task vectors are conflicting and can be trimmed. DARE goes even further and claims that most of the remaining delta parameters can be randomly removed. Each paper is essentially claiming more redundancy than the previous one. This made me wonder how far this trend can go. Is there an even smaller representation of task knowledge that we have not discovered yet?

## 10. Open Question — Cross-Architecture Merging

Another question that kept coming to my mind is about different model families. Everything in DARE assumes the same pretrained base model, so the delta parameters naturally live in the same parameter space. But what happens if the models are completely different, like Llama and Qwen, or even different architectures altogether? Can there be some shared representation where task knowledge can still be transferred or merged without expensive alignment? The paper does not address this, but it is the question that stayed with me after finishing the paper.
