## Week 12 Day 1 — Baseline experiments

This is probably the strongest evidence I have so far for the main hypothesis.

Earlier in Week 11 I showed that Qwen Math and Qwen Coder have unusually large deltas (~0.015-0.016) compared to normal SFT models (~0.0001-0.0007), but that alone was just correlation. Now I have baseline results showing that both Qwen specialists can solve the math and code prompts correctly on their own, yet the TIES merge completely destroys those capabilities and produces gibberish.

The Mistral results make the contrast even stronger. The individual models are not perfect, but they are coherent, and the merged model remains coherent even though it degrades some capabilities. So the failure mode is completely different. Right now it looks like small SFT-style deltas lead to graceful degradation, while large continued-pretraining-style deltas lead to catastrophic collapse. I am not claiming this proves causation yet, but after the magnitude analysis, similarity analysis, merge experiments and now the baseline tests, the same story keeps showing up from multiple directions.

### Setup

The goal was to test each unmerged input model individually on the same two prompts used in Day 5 (Mistral TIES merge) and Week 10 Day 6 (Qwen TIES merge). Without these baselines I couldn't attribute the merged model errors to the merge itself — maybe the input models were already broken.

Setup matched Day 5 exactly: GPU T4 x2, transformers 4.49.0, sequential model loading to manage memory.

Models tested:

| Model | Chat format | Per-param Δ |
|---|---|---|
| Mistral-7B-Instruct-v0.2 | [INST]...[/INST] manual | 0.000421 |
| OpenHermes-2.5-Mistral-7B | ChatML manual | 0.000061 |
| Qwen2.5-Math-7B-Instruct | tokenizer.apply_chat_template | ~0.016 |
| Qwen2.5-Coder-7B-Instruct | tokenizer.apply_chat_template | ~0.015 |

Both Mistral models needed manual chat formatting because the tokenizers we got didn't have chat templates set. Both Qwen models have official templates baked in.

Same two prompts as Day 5:

- Math: water tank with three pipes (A fills in 6 hours, B fills in 9 hours, C empties in 12 hours). Correct answer: 36/7 ≈ 5.14 hours.
- Code: Python fibonacci(n) with memoization using a dictionary, with docstring and edge case handling.

### Math results

**Mistral-Instruct alone:** Forgot pipe C entirely. Only computed (1/6 + 1/9). Used 6 as the common denominator (wrong — should be 18 for those two). Got combined rate of 1/2 tank per hour, then gave up at the end claiming we don't know the tank capacity. Multiple errors.

**OpenHermes alone:** Identified all three pipes correctly. Set up the right equation (1/6 + 1/9 - 1/12). But used 18 as the common denominator — which doesn't work because 12 doesn't divide 18 cleanly. Wrote (2/18 + 2/18 - 1/18) — the first two values are wrong (should be 3/18 and 2/18). Got 5/18 net rate, computed reciprocal as 18/5 = 3.6 hours. Confidently incorrect.

**Qwen-Math alone:** Identified all three pipes. Set up correct equation. Explicitly said "The least common multiple of 6, 9, and 12 is 36." Correctly converted to 6/36, 4/36, 3/36. Computed 7/36. Got cut off by max_tokens just before the final reciprocal step. The reasoning is correct throughout — the model would clearly produce 36/7 ≈ 5.14 with a larger token budget.

**Qwen-Coder alone:** Identified all three pipes. Set up correct equation. Used LCM 36 explicitly. Correctly converted fractions, computed 7/36, took reciprocal. Final answer: 36/7 ≈ 5.14 hours. **Correct.**

### Code results

**Mistral-Instruct alone:** Wrote a function with clean structure, docstring, isinstance type check, proper base cases. But the memoization dictionary `memo = {0: 0, 1: 1}` is created inside the function — it gets reinitialized on every recursive call. So the memoization never persists. The function returns correct values but with exponential time complexity. Defeats the entire purpose of memoization.

**OpenHermes alone:** Concise correct implementation using `def fibonacci(n, memo={}):`. Uses the mutable-default-argument pattern, which is a known Python anti-pattern but actually works as intended here (the dict persists across calls because mutable defaults persist). Correctly memoizes. Handles base cases. Includes a test producing 55 for fibonacci(10). **Correct but uses anti-pattern.**

**Qwen-Math alone:** Uses `def fibonacci(n, memo={0: 0, 1: 1}):`. Adds ValueError for negative input. Passes memo explicitly in recursive calls. Memoization works correctly. Output ends with `\boxed{55}` formatting which is leftover from Qwen-Math's math-output training but doesn't affect the code. **Correct.**

**Qwen-Coder alone:** Uses a closure pattern — outer function does validation, inner `helper(x)` does computation with `fib_cache` in the enclosing scope. Avoids the mutable-default anti-pattern. Clean docstring with Args/Returns/Raises sections. ValueError for negative input. **Correct and idiomatic — the best implementation among all six models tested.**

### Comparison with merged models

| Model | Math result | Code result |
|---|---|---|
| Mistral-Instruct alone | Forgot pipe C, gave up | Clean code, broken memoization (exponential time) |
| OpenHermes alone | Used 18 wrongly, got 3.6 hrs (incorrect) | Correct code (anti-pattern but works) |
| Mistral TIES merge (Day 5) | Used 18 wrongly (same as OpenHermes), cut off mid-error | Clean structure, KeyError on n ≥ 3 |
| **Qwen-Math alone** | **Correct (reasoning to 36/7, cut off at last step)** | **Correct working code** |
| **Qwen-Coder alone** | **Correct (full answer 5.14 hrs)** | **Correct, best implementation** |
| Qwen TIES merge (Week 10) | Complete gibberish from token 1 | Complete gibberish from token 1 |

### Key findings

Both Qwen specialists individually solve both prompts correctly. This was not guaranteed before today. They give the correct LCM (36), correct arithmetic, correct fibonacci implementations. They are demonstrably competent models on these tasks.

The Qwen TIES merge produced complete gibberish from two correctly-functioning specialists. The merge is not averaging broken models. It is destroying competent ones.

The Mistral merge inherited OpenHermes' specific math error. Both OpenHermes alone and the merged Mistral model used 18 as the wrong common denominator. This is too specific to be coincidence — the merge is preserving OpenHermes' reasoning pattern but with worse completion.

The Mistral merge degraded OpenHermes' code competence. OpenHermes alone wrote working memoized fibonacci. The merged model produces clean-looking code that KeyErrors on any n ≥ 3. This is real capability loss.

The severity of merge degradation differs dramatically based on input model properties. Qwen merge produced complete catastrophic failure; Mistral merge produced graceful degradation (coherent but worse than either parent). The severity appears to correlate with the magnitude of the input deltas — Qwen's deltas at ~0.016 are roughly 40x larger than Mistral's at ~0.0001-0.0004.

### What this changes about the Week 11 Day 5 writeup

In Day 5 I wrote a caveat that I couldn't fully attribute the merged model's errors to the merge because I hadn't tested the input models alone. That gap is now filled.

Updated attributions:

- Day 5 merged model's math error (wrong common denominator 18): **inherited from OpenHermes alone**, not introduced by the merge.
- Day 5 merged model's broken fibonacci code: **caused by the merge** — OpenHermes alone wrote working code; the merge broke it.

So the merge does not behave as a clean "preserve" or "destroy" operation. It does both selectively, depending on the tensor type and the task. It can preserve one parent's specific reasoning patterns while destroying another parent's specific capabilities.

### Caveats

- Still N=2 prompts per model. Real benchmarks (GSM8K, HumanEval) needed before stronger claims.
- The "merge degraded OpenHermes' code" claim depends on the specific fibonacci prompt. Broader code evaluation needed.
- Did not test the Qwen merge at lower densities. The catastrophic failure was specifically at density 0.5 in Week 10.
- Did not test whether the Qwen merged model fails on simpler prompts that Qwen-Math and Qwen-Coder both trivially handle.
- Qwen-Math got cut off by max_tokens; the trajectory was clearly correct, but this is a minor measurement caveat.

### What this means for the paper

The central claim now has controlled experimental support:

Standard merge methods like TIES at conventional densities can catastrophically destroy capabilities of individually competent input models, especially when input models have continued-pretraining-style delta magnitudes (~10⁻²) rather than clean SFT magnitudes (~10⁻⁴).

The evidence base now includes:

- 8 baseline measurements (4 models × 2 prompts) from today
- 2 merge experiments (Qwen Week 10, Mistral Week 11 Day 5) with documented outputs
- 7 clean SFT references from Week 11 Days 2-3 confirming the magnitude range across families
- Magnitude vs similarity scatter from Day 4 showing the relationship between deltas and direction differs across training procedures
- Continued-pretraining literature reference (MetaMath-LLeMA at 0.017) matching the Qwen specialist magnitude range

The same story keeps showing up from multiple directions. Magnitude analysis says Qwen specialists are unusual. Cross-family comparison confirms the unusual magnitude is not a measurement artifact. Merge experiments show the unusual magnitudes correlate with merge failure. Baselines show the merge failure cannot be attributed to broken inputs.

### Next steps

Week 13-14: Set up GSM8K and HumanEval evaluation. Run all merged and unmerged models. Convert anecdotal N=2 results into quantitative scores.

This is the biggest credibility upgrade still on the table.


## Comparative analysis across all data points

Now that I have both the merge results and the baseline results, I think the most important thing is not any individual model output but the comparison across all of them together.

The first pattern that stands out is the difference in severity of the merge failures. The Qwen TIES merge and the Mistral TIES merge were created using the exact same configuration. Same merge method, same density (0.5), same weights, same normalize setting and same int8_mask setting. The only real difference was the input models. Yet the outcomes were completely different. The Qwen merge produced complete gibberish on both prompts from the first few tokens, while the Mistral merge remained coherent and attempted to solve the tasks even though it made mistakes. I am not claiming that delta magnitude is the cause, but the pattern is hard to ignore. The Qwen specialists have per-parameter deltas around 0.015-0.016, whereas the Mistral fine-tunes are around 0.0001-0.0004. In this controlled comparison, larger deltas appear to be associated with much more severe merge degradation.

The baseline experiments also revealed something I did not expect about the Mistral merge. Before testing the parent models individually, it was easy to think of the merge as either preserving capabilities or destroying them. The actual behavior is more complicated. In the math prompt, the merged model made the exact same denominator mistake that OpenHermes made when tested alone. Both used 18 as the common denominator and both followed a very similar reasoning path. That suggests the merge preserved at least part of OpenHermes' reasoning pattern. However, the code prompt tells a different story. OpenHermes alone produced a working memoized Fibonacci implementation, while the merged model produced code that looked correct but failed logically. In this case the merge introduced a new failure that was not present in the parent model. So the merge is not acting as a simple preserve-or-destroy operation. It seems capable of preserving some behaviors while simultaneously degrading others.

The baseline data also helps clarify what is actually established versus what is still speculation. One thing that is now much clearer is that the Qwen TIES failure is not happening because the input models are weak. Both Qwen-Math and Qwen-Coder solved the prompts correctly on their own. The merge is taking two competent models and producing complete nonsense. Similarly, the Mistral merge is not simply averaging broken parents. OpenHermes wrote working code on its own, while the merged model did not. The degradation is therefore occurring during the merge process rather than originating entirely from the parent models. Taken together, these observations are consistent with the continued-pretraining-style delta hypothesis developed in Week 11 and do not contradict it.

At the same time, there are still several important things I do not know. The entire comparison is currently based on two prompts, which is enough to identify interesting behavior but not enough to make strong performance claims. I also do not know whether the catastrophic Qwen failure is specific to density 0.5 or whether it would appear at higher densities as well. The continued-pretraining side of the comparison is still represented by a single pair, which is the most important methodological limitation in the current evidence base. I have not yet tested whether the same pattern appears on simpler tasks where both parent models should succeed easily. So while the evidence is becoming more consistent, there is still a lot of validation work left before I can claim this is a general property of delta-based merging methods.
