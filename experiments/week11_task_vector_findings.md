# Week 11 Day X — Task Vector Similarity Analysis

## What I did

- new kaggle notebook
- cpu only because no point burning gpu for similarity calculation
- downloaded model index json for all 3 qwen models
- downloaded all 12 shards (~60GB)
- computed task vector similarity tensor by tensor using lazy loading
- runtime ~8 mins for all 339 tensors

## Models

- Base -> Qwen2.5-7B
- Math -> Qwen2.5-Math-7B-Instruct
- Coder -> Qwen2.5-Coder-7B-Instruct

all 339 tensors, same names, same architecture

## Overall numbers

    mean = 0.4005
    median = 0.2813
    min = -0.0472
    max = 0.9816
    std = 0.2676

first thing i noticed...0.40 is much lower than i expected. i was kinda expecting the task vectors to be much more aligned.

## Distribution

0.2-0.4 bucket contains 213 tensors lol.
thats like majority of the model.
very high similarity tensors (>0.8) = 57 only.
so the picture is not "everything is different" but also definitely not "everything is same".

## Finding 1 — tensor type matters a lot

all the really high similarity tensors are basically layernorms.

example:

    layer 0 input_layernorm = 0.903

whereas things like mlp projections are sitting around 0.3

example:

    layer 0 mlp down_proj = 0.336

i think both finetunes barely touched layernorms. probably thats why they look almost identical.
the actual differences seem to be inside attention and mlp blocks.
need to verify this properly though.

## Finding 2 — weird U shape

excluding layernorms:

    layer 0  -> 0.39
    layer 1  -> 0.43
    layer 11 -> 0.22
    layer 21 -> 0.23
    layer 27 -> 0.41

this was unexpected.
middle layers are much lower.
then similarity increases again near the end.
not sure if this means middle layers contain more task-specific stuff or if im overinterpreting.
need plots tomorrow.

## Negative tensors

only 5 negative tensors.
all tiny attention k/v biases.
largest negative = -0.047
basically nothing.
so there is almost no strong disagreement between task vectors.
that part surprised me.

## Initial thoughts

the thing i keep looking at is the 0.40 mean similarity.
if math and coder task vectors are only around 0.40 similar on average then maybe the assumptions behind TIES are already getting weaker.
week 10 TIES became garbage at density 0.5.
at first i thought maybe implementation issue.
now im wondering if the vectors themselves are the issue.

also interesting that most tensors are in the low positive range instead of being near zero or near one.
its like they are related but not aligned enough.

## Day 2 — Magnitude Analysis

After yesterday's similarity analysis, I wanted to check something else. Similarity tells me whether two task vectors point in similar directions, but it doesn't tell me how large those task vectors actually are. So today I looked at the magnitude of the updates themselves.

### Relative magnitude of Math and Coder vs base

I started by computing the relative magnitude of Math and Coder compared to the base Qwen2.5-7B model. The first thing that stood out was that both models changed by almost the same amount overall. The cumulative ratio between them was around 0.925, which is surprisingly close. So whatever training procedure was used, both models seem to have undergone a similar amount of modification from the base model.

| Model | Mean | Median | Min | Max | Stdev |
|---|---|---|---|---|---|
| Math vs Base | 0.7692 | 0.9744 | 0.0311 | 1.4618 | 0.5144 |
| Coder vs Base | 0.8312 | 0.9859 | 0.0936 | 2.2112 | 0.4276 |

Looking at the per-tensor results, most of the largest changes came from MLP up projection weights, especially in the early layers. For Math, the top changed tensors were almost entirely mlp.up_proj weights. Coder showed a similar pattern, although a few attention k_proj bias terms appeared at the top because their original values are extremely small, which makes the relative change look huge. The least changed tensors for both models were mostly attention bias terms.

**Top 5 changed by Math (all early-layer mlp.up_proj weights):**

| Tensor | Relative magnitude |
|---|---|
| layer 5 mlp.up_proj | 1.4618 |
| layer 4 mlp.up_proj | 1.4480 |
| layer 3 mlp.up_proj | 1.4376 |
| layer 9 mlp.up_proj | 1.4239 |
| layer 0 mlp.up_proj | 1.4143 |

**Top 5 changed by Coder (note: top k-bias values inflated by small base norms):**

| Tensor | Relative magnitude |
|---|---|
| layer 11 self_attn.k_proj.bias | 2.2112 |
| layer 20 self_attn.k_proj.bias | 1.8331 |
| layer 18 self_attn.k_proj.bias | 1.7278 |
| layer 24 self_attn.k_proj.bias | 1.4379 |
| layer 5 mlp.up_proj.weight | 1.3568 |

### Per-layer U-shape

Something else I noticed is that the per-layer magnitude pattern also forms a U-shape, although weaker than yesterday's similarity results. The middle layers show smaller magnitudes while the beginning and end layers show larger magnitudes. Yesterday I saw a U-shape in similarity. Today I see a U-shape in magnitude. Seeing the same pattern twice makes me think this probably isn't random noise anymore. I'm still not completely sure what causes it, but it seems like the middle layers are being modified differently from the edge layers.

| Layer | Math mean mag | Coder mean mag |
|---|---|---|
| 0 | 0.8967 | 0.9807 |
| 1 | 0.9255 | 1.0451 |
| 11 | 0.8819 | 1.0997 |
| 17 | 0.8062 | 0.8399 (min) |
| 19 | 0.7975 (min) | 0.8028 |
| 27 | 0.9029 | 0.8880 |

(Full per-layer table available in saved JSON; abbreviated here.)

### The biggest finding — per-parameter absolute delta

The biggest finding today came from looking at the average absolute delta values.

To make sense of the numbers, I compared my results against Table 8 from the DARE-the-Extreme paper (arxiv 2410.09344). That paper reports average parameter updates for several LLaMA-2 fine-tunes. Typical supervised fine-tuning models such as WizardMath, MetaMath and Abel fall somewhere around 0.0004 to 0.0007. Models that underwent continued pretraining show much larger values around 0.017.

When I computed the same metric for Qwen2.5-Math and Qwen2.5-Coder, I got 0.0161 and 0.0148 respectively.

Honestly this was not what I expected.

These values are not slightly larger than the DARE SFT examples. They are roughly 20-70 times larger. More importantly, they are almost identical to the continued-pretraining example reported in the paper.

At first I wondered if this was just a Qwen thing. Maybe every Qwen fine-tune has large deltas. To test that, I added Qwen2.5-7B-Instruct as a control model.

The control completely changed my interpretation.

Qwen2.5-Instruct came out at 0.000228. Not only is that small, it is actually smaller than WizardMath from the DARE paper.

| Model | Mean abs delta | Type / source |
|---|---|---|
| Qwen-2.5-7B-Instruct (ours, control) | 0.000228 | Standard instruction fine-tune |
| WizardMath-7B (LLaMA-2) | 0.000400 | Standard SFT; DARE-the-Extreme 2024 Table 8 |
| MetaMath-7B (LLaMA-2) | 0.000720 | Standard SFT; DARE-the-Extreme 2024 Table 8 |
| Abel-7B (LLaMA-2) | 0.000730 | Standard SFT; DARE-the-Extreme 2024 Table 8 |
| Qwen-2.5-Coder-7B (ours) | 0.014813 | Released as fine-tune |
| Qwen-2.5-Math-7B (ours) | 0.016086 | Released as fine-tune |
| MetaMath-LLeMA-7B | 0.017000 | Continued pretrained; DARE-the-Extreme 2024 Table 8 |

Total parameters analyzed per model: 7,615,616,512.

This rules out my original explanation that all Qwen fine-tunes naturally produce large deltas. If that were true, Instruct should also have large values. Instead, Instruct behaves exactly like a normal supervised fine-tune while Math and Coder behave much more like the continued-pretraining example.

### Interpretation

Because of that, my current hypothesis is that Qwen2.5-Math and Qwen2.5-Coder were trained using something much closer to continued pretraining than standard instruction tuning. I can't prove that from these experiments alone because Alibaba's training pipeline is not fully public. But the similarity between my numbers and the DARE continued-pretraining example is difficult to ignore.

This also gives me a possible explanation for what happened during Week 10. Most model merging methods such as Task Arithmetic and TIES are designed around the assumption that fine-tuning updates are relatively small modifications to a pretrained model. But if the updates themselves become extremely large, then operations like addition, trimming and sign election may behave very differently. The merge failures I observed might not be caused by implementation mistakes at all. They may be a consequence of applying SFT-oriented merging methods to models whose updates look much more like continued pretraining.

### Caveats

That being said, I need to be careful here because I only have correlation right now. I have shown that large deltas and merge failures appear together. I have not shown that the large deltas caused the failures.

A few honest limitations:
- Only tested one model family (Qwen) so far
- Cannot prove continued pretraining without Qwen's full training documentation, which isn't public
- Correlation between large deltas and merge failure has been observed but not causally tested
- Only one "specialist" pair and one "clean fine-tune" data point measured directly
- Some top-ranked tensors by relative magnitude (k_proj biases) are inflated by small base norms; absolute change in those tensors is still small

### Next step

The next step is probably to test another model family. If a more traditional SFT pair shows small deltas and merges successfully, that would strengthen this hypothesis. If it also fails, then I need a different explanation.

Right now the strongest conclusion I feel comfortable making is that Qwen2.5-Math and Qwen2.5-Coder look fundamentally different from the fine-tuning examples used in papers like DARE, and that difference is large enough that it may explain some of the strange behavior I observed during merging experiments.
