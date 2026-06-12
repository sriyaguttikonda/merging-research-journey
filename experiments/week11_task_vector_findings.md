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
## Day 3 — Mistral Family Analysis

After yesterday's results, I wanted to check whether the large delta magnitudes I observed in Qwen Math and Qwen Coder were actually unusual or if I was just looking at a weird metric. So today I switched to the Mistral family and repeated the same analysis on three popular fine-tunes: Mistral Instruct, Zephyr and OpenHermes.

### Setup

The first thing I checked was whether the models were compatible for direct comparison. All four models had exactly the same tensor structure with 291 tensors each. The only issue was that all three fine-tunes added two extra vocabulary tokens compared to the base model. This caused shape mismatches for the embedding and lm_head tensors, so I excluded those two tensors from the analysis. Since this affects less than 1% of the parameters and MergeKit already uses special handling for embeddings, I think this is a reasonable approximation, but it is still something worth mentioning as a limitation.

Models used:

| Label | Hugging Face ID |
|---|---|
| base | mistralai/Mistral-7B-v0.1 |
| instruct | mistralai/Mistral-7B-Instruct-v0.2 |
| zephyr | HuggingFaceH4/zephyr-7b-beta |
| openhermes | teknium/OpenHermes-2.5-Mistral-7B |

### Magnitude analysis

The main result is honestly pretty striking.

Mistral Instruct came out at 0.000421. Zephyr was even smaller at 0.000098. OpenHermes was smaller still at 0.000061.

When I put these numbers next to yesterday's Qwen results, the difference becomes hard to ignore. Qwen Math and Qwen Coder were around 0.015-0.016. These Mistral fine-tunes are sitting around 0.00006-0.00042.

That is not a small difference. That is roughly two orders of magnitude.

Full comparison across all data points collected this week + literature reference:

| Model | Mean per-param |Δ| | Type | Source |
|---|---|---|---|
| OpenHermes-2.5-Mistral-7B | 0.000061 | Clean SFT | Day 3 |
| Zephyr-7B-beta | 0.000098 | Clean SFT | Day 3 |
| Qwen-2.5-7B-Instruct | 0.000228 | Clean SFT | Day 2 |
| WizardMath-7B (LLaMA-2) | 0.000400 | Clean SFT | DARE-the-Extreme Table 8 |
| Mistral-7B-Instruct-v0.2 | 0.000421 | Clean SFT | Day 3 |
| MetaMath-7B (LLaMA-2) | 0.000720 | Clean SFT | DARE-the-Extreme Table 8 |
| Abel-7B (LLaMA-2) | 0.000730 | Clean SFT | DARE-the-Extreme Table 8 |
| Qwen-2.5-Coder-7B | 0.014813 | Released as fine-tune | Day 2 |
| Qwen-2.5-Math-7B | 0.016086 | Released as fine-tune | Day 2 |
| MetaMath-LLeMA-7B | 0.017000 | Continued pretraining | DARE-the-Extreme Table 8 |

What's interesting is that the Mistral numbers look exactly like what I would expect from the DARE paper. WizardMath, MetaMath and Abel all fall into the same general range. In other words, the Mistral family behaves like the literature says normal supervised fine-tuning should behave.

Qwen Math and Qwen Coder do not.

This makes yesterday's result much more difficult to dismiss as a Qwen-specific quirk or a bug in my implementation. If the metric itself was broken, I would expect all models to show strange values. Instead, the Mistral models line up very nicely with both the DARE results and the Qwen Instruct control model from Day 2.

At this point my interpretation is becoming a bit stronger. Yesterday I suspected that Qwen Math and Qwen Coder might have undergone something closer to continued pretraining than standard supervised fine-tuning. Today's results don't prove that, but they do remove one alternative explanation. The alternative explanation was that all fine-tuned models naturally produce large deltas and I was overreacting to the Qwen numbers. The Mistral results make that explanation much harder to defend.

Another thing that stood out is how small Zephyr and OpenHermes are. These are highly capable instruction-following models, yet their average parameter changes are tiny. That fits surprisingly well with the DARE interpretation that supervised fine-tuning is mostly unlocking or steering capabilities that already exist in the base model rather than creating entirely new capabilities from scratch.

### Pairwise similarity between Mistral fine-tunes

After looking at delta magnitudes, I wanted to see whether the Mistral fine-tunes were also pointing in similar directions. Since all three models are instruction-tuned variants of the same base model, my initial expectation was that they would probably be more aligned than the Qwen Math/Coder pair.

The first results were confusing.

One of the model pairs showed cosine values ranging almost from -1 to 1. After digging into the tensors, I realized this was mostly coming from layernorm parameters. The actual deltas for those tensors are extremely small, so even tiny numerical differences can produce unstable cosine values. Since the denominator becomes very small, the cosine similarity becomes unreliable.

Raw similarity statistics (including layernorms):

| Pair | n | mean | median | min | max | stdev |
|---|---|---|---|---|---|---|
| instruct vs zephyr | 240 | 0.0179 | 0.0177 | -0.0005 | 0.1031 | 0.0113 |
| instruct vs openhermes | 241 | 0.0489 | 0.0455 | -0.0267 | 0.0847 | 0.0236 |
| zephyr vs openhermes | 239 | 0.0409 | 0.0367 | -1.0000 | 1.0000 | 0.1773 |

Investigation of the suspicious zephyr_vs_openhermes range showed that 8 tensors with |cosine_similarity| > 0.5 were all layernorms. Distribution of |cos_sim| values for that pair:

| Range | Count | Percentage |
|---|---|---|
| < 0.05 | 208 | 87.0% |
| 0.05 - 0.10 | 20 | 8.4% |
| 0.10 - 0.50 | 3 | 1.3% |
| 0.50 - 0.95 | 1 | 0.4% |
| 0.95 - 1.00 (layernorm artifact) | 7 | 2.9% |

After excluding layernorm tensors and recomputing the statistics, the picture became much cleaner:

| Pair | n | mean | median | min | max | stdev |
|---|---|---|---|---|---|---|
| instruct vs zephyr | 224 | 0.0184 | 0.0179 | 0.0037 | 0.1031 | 0.0094 |
| instruct vs openhermes | 224 | 0.0528 | 0.0492 | 0.0165 | 0.0847 | 0.0195 |
| zephyr vs openhermes | 224 | 0.0329 | 0.0367 | -0.0020 | 0.0614 | 0.0149 |

The mean similarities for the three model pairs were all very low, ranging from roughly 0.018 to 0.053.

Honestly this was the opposite of what I expected.

Yesterday the Qwen Math/Coder pair had an average similarity around 0.40. I came into today's experiment expecting clean supervised fine-tunes to be more aligned than specialist models. Instead they were much less aligned.

My current interpretation is that clean supervised fine-tuning does not necessarily push models in the same direction. Even though all three models are instruction-tuned and start from the same base model, they are still learning from different datasets and optimization trajectories. The end result seems to be that each model develops its own task vector direction.

So at least from these experiments, "same type of fine-tune" does not automatically mean "high task-vector similarity."

### Per-layer similarity pattern

I also looked at similarity layer-by-layer after excluding layernorm tensors.

The pattern was surprisingly consistent across all three model pairs in shape, although the amplitude varied (instruct vs openhermes peaks much higher than instruct vs zephyr).

Sample per-layer values:

| Layer | instruct_vs_zephyr | instruct_vs_openhermes | zephyr_vs_openhermes |
|---|---|---|---|
| 0 | 0.0325 | 0.0272 | 0.0136 |
| 5 | 0.0129 | 0.0354 | 0.0152 |
| 10 | 0.0168 | 0.0464 | 0.0275 |
| 15 | 0.0220 | 0.0585 | 0.0400 |
| 18 | 0.0243 | 0.0665 | 0.0470 |
| 19 | 0.0224 | 0.0674 | 0.0438 |
| 20 | 0.0206 | 0.0653 | 0.0417 |
| 25 | 0.0164 | 0.0606 | 0.0398 |
| 30 | 0.0209 | 0.0675 | 0.0500 |
| 31 | 0.0234 | 0.0711 | 0.0550 |

Similarity starts very low in the early layers, usually around 0.01-0.03. It then gradually increases through the middle layers, reaches its highest values around layers 18-19, drops slightly around layers 24-26, and then increases again near the final layers.

Overall the shape looks more like a hump or inverted-U.

What makes this interesting is that it looks completely different from the Qwen Math/Coder results from Day 1.

For Qwen, similarity was highest near the beginning and end of the network and lowest in the middle layers. That produced a U-shaped curve.

For Mistral, similarity is lowest near the edges and highest in the middle layers, producing almost the opposite pattern.

I don't want to over-interpret this yet, but it suggests that different training procedures may leave different signatures across network depth. The Qwen specialist models already looked unusual because of their large delta magnitudes. Now they also look unusual in terms of where those changes occur across the network.

### Updated interpretation

At the start of this week I was expecting similarity and magnitude to tell the same story.

Something like:

small deltas -> high similarity

large deltas -> low similarity

That is not what happened.

The magnitude result remains very strong. Mistral fine-tunes behave like the supervised fine-tuning examples reported in the literature, while Qwen Math and Qwen Coder behave much more like continued-pretraining examples.

The directional result is much less straightforward.

In fact, the Qwen Math/Coder pair is substantially more aligned than the clean Mistral fine-tune pairs.

This means similarity and magnitude are measuring different things. A model can undergo very large changes while still moving in a relatively similar direction to another model. Likewise, two models can have very small deltas while moving in almost completely different directions.

Right now the clearest signal for distinguishing the two groups is magnitude, not similarity.

The layer-wise results make the story even more interesting because the shape of the similarity curve changes completely between the Qwen and Mistral families. That suggests the training procedure may influence not only how much a model changes, but also where in the network those changes become aligned or divergent.

At this point my strongest conclusion is that delta magnitude appears to be the most reliable indicator separating the Qwen specialist models from conventional supervised fine-tunes. Similarity seems to vary much more independently than I originally expected.

### Caveats

- Excluded `model.embed_tokens.weight` and `lm_head.weight` (~0.9% of parameters) due to vocabulary mismatch (32000 → 32002, fine-tunes added chat template tokens)
- Only one continued-pretrained example measured directly (Qwen Math, Qwen Coder); the LLaMA-2 reference (MetaMath-LLeMA at 0.017000) comes from literature
- Tested directional similarity for clean SFT pairs but not for any other continued-pretrained pair besides Qwen Math/Coder
- Have not directly shown causation between large magnitudes and merge failure; only correlation
- Cosine similarity becomes numerically unstable when delta norms are very small (as seen in layernorm tensors); needed to exclude these for reliable statistics

### Saved data

- Magnitude analysis: Kaggle dataset `sriyaguttikonda/week11-day3-mistral-magnitudes`
- Similarity analysis: Kaggle dataset `sriyaguttikonda/week11-day3-mistral-similarities`

# Week 11 Day 4 - Magnitude vs Similarity Analysis

After looking at magnitude separately and similarity separately over the last few days, I wanted to see whether they are actually related to each other.

The question today was pretty simple.

If a tensor has a larger delta, does that also mean it is more aligned with the corresponding tensor in another fine-tuned model?

Or are magnitude and similarity completely independent things?

To check this, I rebuilt the Qwen results from scratch, attached the Mistral datasets from yesterday and created a side-by-side scatter plot.

For each tensor:

* x-axis = average per-parameter |Δ|
* y-axis = cosine similarity between the two task vectors

I only kept attention and MLP tensors because those are the main components and it removes a lot of noise from layernorms, biases and embedding-related stuff.

The first thing that jumped out at me was how different the scales are.

For Qwen, most tensors sit around 0.01-0.025 magnitude and 0.2-0.4 similarity.

For Mistral, most tensors sit around 0.0002-0.0004 magnitude and 0.01-0.03 similarity.

So before even looking at patterns, the two plots are already operating in completely different regimes.

Honestly that was the first thing my eyes went to.

The Qwen plot and Mistral plot don't even look like they came from the same type of training process.

The Qwen plot shows a pretty clear positive trend.

Its not a perfect line or anything, but as magnitude increases, similarity also tends to increase.

The MLP tensors form a fairly tight cluster in the upper-middle part of the plot, while the attention tensors are spread out over a much larger range.

What I found interesting is that the MLP cluster sits above most of the attention points. So the MLP tensors are not only changing a lot, they are also changing in more similar directions between Math and Coder.

That feels important.

The Mistral plot looks completely different.

I went into this expecting clean SFT models to probably be more aligned than Qwen.

That is not what happened.

The similarities are much lower overall and I don't really see the same positive trend.

If anything, the MLP points show a weak downward trend where larger magnitudes correspond to slightly lower similarity.

The attention tensors are also split into two separate groups instead of forming one cloud.

One cluster sits near the MLP points while another appears at much larger magnitudes but still very low similarity.

I honestly don't know why yet.

That is probably the most interesting unresolved thing in today's plot.

The main takeaway for me is that magnitude and similarity are not telling the same story.

A few days ago I was almost expecting something like:

small deltas -> high similarity

large deltas -> low similarity

But that is clearly not what the data says.

Qwen has very large deltas and relatively high similarity.

Mistral has tiny deltas and very low similarity.

So magnitude and direction seem to be measuring different properties of the training process.

The Qwen result is actually making more sense to me now.

If Qwen Math and Qwen Coder both went through some common continued-pretraining style stage before specializing, then it would explain why the largest deltas are often the most aligned. The biggest changes may be coming from something both models experienced together, while the later task-specific tuning introduces the differences.

Not saying I've proven that.

But the positive correlation is at least consistent with that story.

For Mistral, the picture feels different.

The deltas are tiny and the directions seem much more independent.

Almost like each instruction-tuning dataset nudged the model in its own direction regardless of how much a particular tensor changed.

So the headline from today's experiment is not just that Qwen and Mistral have different magnitudes.

Its that the relationship between magnitude and direction itself looks different.

Qwen gives me one big cloud with an upward trend.

Mistral gives me separated structures with almost no positive relationship at all.

That feels like a stronger result than looking at magnitude or similarity alone.

Because now it is not just "how much did the models change?" or "how aligned are they?"

It is "how does alignment change as the updates get larger?"

And for Qwen and Mistral, the answers seem completely different.

## Caveats

A few things I need to be careful about.

First, this is only one Mistral pair (Instruct vs Zephyr). Another Mistral pair could look different.

Second, I only have one specialist pair on the Qwen side.

Third, I used average per-parameter |Δ| as the magnitude metric. A different metric such as relative magnitude might produce a different picture.

Also, I removed biases, layernorms and embeddings from the main analysis. That makes the comparison cleaner, but it also means I am not looking at the whole model.

## What I want to check next

The attention split in Mistral is bothering me.

I want to know whether those two clusters correspond to specific layers or specific attention projections.

I also want to see if the same pattern appears in Instruct vs OpenHermes and Zephyr vs OpenHermes.

Right now the strongest thing I feel comfortable saying is that magnitude and similarity are not coupled in a universal way.

The relationship itself seems to depend on the training procedure.
