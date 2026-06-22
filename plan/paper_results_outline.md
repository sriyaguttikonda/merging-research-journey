# Paper Results section outline

Draft outline for the empirical Results section of the ICLR 2027 workshop submission. Each subsection lists what it shows, data sources, figures/tables, and the headline claim. This is a planning document, not a draft.

## 3.0 Experimental design

What it shows: Experimental setup and rationale. We compare TIES merging outcomes across two model pairs that differ primarily in input delta magnitude, holding the merge configuration constant. Baseline runs of each parent model individually allow attribution of merge errors.

Data sources: Week 10 Day 4 (Qwen merge config), Week 11 Day 5 (Mistral merge config), Week 12 Day 1 (baseline protocol).

Figures/tables: None — narrative section.

Headline: We compare TIES merging outcomes on continued-pretraining-style and clean-SFT-style input pairs, holding merge configuration constant (density 0.5, weight 0.5/0.5, normalize=true, int8_mask=true, bfloat16).

## 3.1 Magnitude characterization across model families

What it shows: Per-parameter mean absolute delta differs by roughly two orders of magnitude between clean SFT fine-tunes and continued-pretrained-style models. We characterize this across three model families (LLaMA-2 from literature, Qwen, Mistral) and find consistent ranges.

Data sources: Week 11 Day 2 (Qwen magnitudes: Math 0.016086, Coder 0.014813, Instruct 0.000228), Week 11 Day 3 (Mistral magnitudes: Instruct 0.000421, Zephyr 0.000098, OpenHermes 0.000061), DARE-the-Extreme Table 8 (LLaMA-2 references: WizardMath 0.000400, MetaMath 0.000720, Abel 0.000730, MetaMath-LLeMA 0.017000).

Figures/tables: Table 1: 10-row table grouped by category (clean SFT, continued-pretrained), columns for model name, family, per-param mean |Δ|, source. Have all data; needs formatting.

Sketch of Table 1:

| Model | Family | Per-param mean Δ | Category | Source |
|---|---|---|---|---|
| OpenHermes-2.5-Mistral-7B | Mistral | 0.000061 | Clean SFT | Day 3 |
| Zephyr-7B-beta | Mistral | 0.000098 | Clean SFT | Day 3 |
| Qwen2.5-7B-Instruct | Qwen | 0.000228 | Clean SFT | Day 2 |
| WizardMath-7B | LLaMA-2 | 0.000400 | Clean SFT | DARE-the-Extreme Table 8 |
| Mistral-7B-Instruct-v0.2 | Mistral | 0.000421 | Clean SFT | Day 3 |
| MetaMath-7B | LLaMA-2 | 0.000720 | Clean SFT | DARE-the-Extreme Table 8 |
| Abel-7B | LLaMA-2 | 0.000730 | Clean SFT | DARE-the-Extreme Table 8 |
| Qwen2.5-Coder-7B-Instruct | Qwen | 0.014813 | Continued-PT-style | Day 2 |
| Qwen2.5-Math-7B-Instruct | Qwen | 0.016086 | Continued-PT-style | Day 2 |
| MetaMath-LLeMA-7B | LLaMA-2 | 0.017000 | Continued-PT | DARE-the-Extreme Table 8 |

Headline: Clean SFT fine-tunes consistently produce per-parameter deltas in the range 10⁻⁴ to 10⁻³ across three model families and seven independent fine-tunes, while continued-pretrained models cluster around 10⁻² — a roughly two-order-of-magnitude separation that holds across architectures.

## 3.2 Independence of magnitude and direction

What it shows: Delta magnitude and task-vector direction are not the same property. Models with very small deltas can have extremely low directional alignment, while models with very large deltas can remain relatively aligned. This suggests that magnitude and similarity should be treated as independent dimensions when analyzing mergeability.

Data sources: Week 11 Day 1 (Qwen cosine similarity: mean 0.4005, layer-wise U-shape), Week 11 Day 3 (Mistral pairwise cosine similarities: means 0.018-0.053 after layernorm removal), Week 11 Day 4 (scatter plot analysis combining magnitude and similarity).

Figures/tables: Figure 1: Qwen vs Mistral magnitude-similarity scatter plot (already exists: week11_day4_scatter.png). Table 2: Pairwise similarity statistics for Qwen and Mistral model pairs (have data; needs formatting).

Sketch of Table 2:

| Pair | n (no LN) | Mean cosine | Median | Notes |
|---|---|---|---|---|
| Qwen Math vs Coder | 196 | 0.4005 | 0.2813 | U-shape across layers |
| Mistral Instruct vs Zephyr | 224 | 0.0184 | 0.0179 | Inverted-U across layers |
| Mistral Instruct vs OpenHermes | 224 | 0.0528 | 0.0492 | Inverted-U across layers |
| Mistral Zephyr vs OpenHermes | 224 | 0.0329 | 0.0367 | Inverted-U across layers |

Headline: Qwen specialists exhibit higher directional alignment (mean cosine 0.40) than clean Mistral SFT pairs (mean cosine 0.018-0.053), despite having 100× larger deltas — magnitude and direction vary independently across model families.

## 3.3 Controlled merge experiments

### 3.3.1 Qwen Math + Coder (continued-pretraining-style deltas)

What it shows: Applying TIES with a standard configuration to Qwen Math and Qwen Coder produces catastrophic failure despite both parent models being individually competent. The merged model generates incoherent outputs on both math and coding tasks.

Data sources: Week 10 Day 4 (merge configuration), Week 10 Day 6 (Qwen TIES outputs), Week 12 Day 1 (Qwen-Math and Qwen-Coder baselines).

Figures/tables: Table 3: Qwen parent model outputs vs merged model outputs on math and code prompts (have data; needs formatting). Figure 2: Example generations showing catastrophic failure (need formatted excerpts).

Sketch of Table 3:

| Model | Math prompt result | Code prompt result |
|---|---|---|
| Qwen2.5-Math-7B-Instruct alone | Correct: 36/7 ≈ 5.14 hrs (correct LCM 36) | Correct working memoized Fibonacci |
| Qwen2.5-Coder-7B-Instruct alone | Correct: 36/7 ≈ 5.14 hrs (full answer) | Correct, idiomatic closure-based implementation |
| Qwen-Math + Qwen-Coder TIES merged | Complete gibberish from token 1 | Complete gibberish from token 1 |

Headline: A standard TIES configuration can completely destroy capabilities of individually competent specialist models when applied to continued-pretraining-style deltas.

### 3.3.2 Mistral Instruct + OpenHermes (clean SFT deltas)

What it shows: The same TIES configuration applied to a clean SFT pair produces a very different outcome. The merged model remains coherent and task-aware but exhibits degraded reasoning and coding performance relative to its parents.

Data sources: Week 11 Day 5 (Mistral TIES merge experiment), Week 12 Day 1 (Mistral-Instruct and OpenHermes baselines).

Figures/tables: Table 4: Mistral parent model outputs vs merged model outputs on math and code prompts (have data; needs formatting). Figure 3: Example generations illustrating graceful degradation (need formatted excerpts).

Sketch of Table 4:

| Model | Math prompt result | Code prompt result |
|---|---|---|
| Mistral-7B-Instruct-v0.2 alone | Forgot pipe C, wrong denominator, gave up | Clean structure, broken memoization (exponential time) |
| OpenHermes-2.5-Mistral-7B alone | Right setup, wrong denominator (18), got 3.6 hrs (incorrect) | Correct working code (anti-pattern mutable default) |
| Mistral-Instruct + OpenHermes TIES merged | Right setup, same wrong denominator as OpenHermes, cut off | Clean structure, KeyError on n ≥ 3 |

Headline: Under identical merge settings, clean SFT models experience graceful degradation rather than catastrophic collapse.

## 3.4 Baseline-confirmed attribution

What it shows: Baseline testing of all parent models allows merge errors to be attributed more precisely. Some merged-model behaviors are inherited directly from a parent model, while others are introduced by the merge itself.

Data sources: Week 12 Day 1 baseline experiments, Week 11 Day 5 merged model outputs, Week 10 Day 6 Qwen merged outputs.

Figures/tables: Table 5: Attribution matrix showing whether observed behaviors originate from parent models or emerge after merging (needs creation from existing results).

Sketch of Table 5:

| Observed behavior in merged model | Origin | Evidence |
|---|---|---|
| Mistral merge math error (wrong denominator 18) | Inherited from OpenHermes | OpenHermes alone made identical error |
| Mistral merge code break (KeyError on n ≥ 3) | Introduced by merge | OpenHermes alone wrote working code |
| Qwen merge math gibberish | Introduced by merge | Qwen-Math alone solved correctly |
| Qwen merge code gibberish | Introduced by merge | Qwen-Coder alone wrote correct code |

Headline: Baselines reveal that merging is neither purely preservative nor purely destructive; it selectively preserves some parent behaviors while introducing new failures.

## 3.5 Severity-magnitude correlation (the main empirical claim)

What it shows: Across controlled experiments, merge degradation severity appears to correlate with input delta magnitude. Models with clean SFT-scale deltas (~10⁻⁴) remain coherent after merging, while models with continued-pretraining-scale deltas (~10⁻²) collapse into gibberish under the same merge configuration.

Data sources: Week 11 Day 2 (Qwen magnitudes), Week 11 Day 3 (Mistral magnitudes), Week 11 Day 5 (Mistral merge results), Week 12 Day 1 (baseline-confirmed attribution), Week 10 Day 6 (Qwen merge results).

Figures/tables: Figure 4: Conceptual comparison of merge outcomes across delta magnitude regimes (needs creation). Table 6: Summary table linking model pair, delta magnitude range, and observed merge outcome category (catastrophic failure vs graceful degradation).

Sketch of Table 6:

| Model pair | Mean per-param Δ | Merge outcome category |
|---|---|---|
| Qwen-Math + Qwen-Coder | ~0.015-0.016 | Catastrophic failure (gibberish) |
| Mistral-Instruct + OpenHermes | ~0.0001-0.0004 | Graceful degradation (coherent but incorrect) |

Headline: In our controlled comparison, the dimension most associated with merge outcome was the scale of the underlying fine-tuning deltas: SFT-scale deltas led to graceful degradation, whereas continued-pretraining-scale deltas led to catastrophic collapse under the same TIES configuration.

## 3.6 Practical implications (optional — pending)

What it shows: A concrete diagnostic check practitioners can apply before merging — compute per-parameter mean |Δ| on input fine-tunes; values approaching 10⁻² predict merge instability under standard TIES configurations and warrant alternative approaches.

Data sources: All of Sections 3.1 through 3.5.

Figures/tables: None — narrative section. Optionally a small recommendation box / pseudocode for the diagnostic check.

Headline: Practitioners can apply a simple delta-magnitude check before merging to anticipate whether standard merge methods are likely to succeed; values above approximately 10⁻³ in either input fine-tune predict significant degradation risk in our experiments.

Note: Whether this section is included depends on whether the diagnostic recommendation is genuinely defensible from the current evidence (N=2 controlled pairs). May be safer to move this content into the Discussion section rather than presenting it as a Results-level finding.

## Out of scope for this paper

- Algorithmic proposals (e.g., magnitude-aware merging methods). This paper is empirical observation only.
- Theoretical justification for why magnitude affects merge stability. Belongs in Discussion if at all; speculation should be clearly flagged.
- Multiple density values. Catastrophic failure was observed at density 0.5; other densities not yet tested.
- Multiple continued-pretraining-style pairs. Only one pair (Qwen Math+Coder) is currently in our evidence base — a real limitation, to be acknowledged in Discussion.
- Real benchmark scores (GSM8K, HumanEval). Planned for Week 13-14; will replace or supplement N=2 anecdotal prompts.
