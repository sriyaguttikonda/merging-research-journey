
# Experiment 1: Math-delta kappa sweep (Task Arithmetic)
# merged = base + kappa*delta_Math + 1.0*delta_Coder
# Qwen2.5-7B base, Math-Instruct + Coder-Instruct. Greedy, 400 tokens. CCR V100.

| kappa | math prompt | code prompt |
|-------|-------------|-------------|
| 1.0   | gibberish (2222)                | gibberish |
| 0.8   | gibberish (12.12)               | gibberish (1 line then collapse) |
| 0.6   | coherent start, collapses to 111 | coherent start, collapses to etc |
| 0.4   | COHERENT, correct (LCM36, 7/36)  | COHERENT, correct working code |
| 0.2   | COHERENT, correct (36/7 ~5.14h)  | COHERENT, correct |
| 0.0   | COHERENT, correct (base+Coder)   | COHERENT, correct |

## Finding
- Shrinking the Math delta RECOVERS the catastrophic merge.
- Sharp transition between kappa=0.6 (broken) and kappa=0.4 (fixed).
- kappa=0.0 sanity endpoint (base+Coder) coherent as expected.
- Confirms DisTaC's "shrink helps" prescription TRANSFERS to the naturally-occurring
  large-delta / continued-pretraining regime (not just artificial LR-induced disparity).
MDEOF
cat /scratch/pmolugur/merging_project/RESULTS_math_sweep.md
# Experiment 1: Math-delta kappa sweep (Task Arithmetic)
# merged = base + kappa*delta_Math + 1.0*delta_Coder
# Qwen2.5-7B base, Math-Instruct + Coder-Instruct. Greedy, 400 tokens. CCR V100.

| kappa | math prompt | code prompt |
|-------|-------------|-------------|
| 1.0   | gibberish (2222)                | gibberish |
| 0.8   | gibberish (12.12)               | gibberish (1 line then collapse) |
| 0.6   | coherent start, collapses to 111 | coherent start, collapses to etc |
| 0.4   | COHERENT, correct (LCM36, 7/36)  | COHERENT, correct working code |
| 0.2   | COHERENT, correct (36/7 ~5.14h)  | COHERENT, correct |
| 0.0   | COHERENT, correct (base+Coder)   | COHERENT, correct |

## Finding
- Shrinking the Math delta RECOVERS the catastrophic merge.
- Sharp transition between kappa=0.6 (broken) and kappa=0.4 (fixed).
- kappa=0.0 sanity endpoint (base+Coder) coherent as expected.
- Confirms DisTaC's "shrink helps" prescription TRANSFERS to the naturally-occurring
  large-delta / continued-pretraining regime (not just artificial LR-induced disparity).
cat > /scratch/pmolugur/merging_project/RESULTS_coder_sweep.md << 'MDEOF'
# Experiment 1b: Coder-delta kappa sweep (Task Arithmetic)
# merged = base + 1.0*delta_Math + kappa*delta_Coder
# Compare against Math sweep (RESULTS_math_sweep.md)

| kappa | math prompt | code prompt |
|-------|-------------|-------------|
| 1.0   | gibberish (2222) [matches math k=1.0, consistency OK] | gibberish |
| 0.8   | gibberish (121212)               | gibberish |
| 0.6   | gibberish (10000)                | gibberish (1 line then collapse) |
| 0.4   | coherent method but WRONG answer (36 not 36/7), loops | broken code, repetition |
| 0.2   | coherent, correct method, trails off | coherent-ish, syntax errors, repetition |
| 0.0   | coherent (base+Math)             | code degenerates into ``` repetition |

## Finding — ASYMMETRY between the two sweeps
- Shrinking MATH delta -> CLEAN full recovery by k=0.4 (correct answer + working code).
- Shrinking CODER delta -> only PARTIAL improvement; never fully clean, code stays broken.
- The two deltas are ~equal magnitude (Math 0.0161, Coder 0.0148, ~8% apart) yet NOT
  equally responsible for the failure. Math delta appears more destabilizing.
- Endpoint clue: base+Math-alone (coder k=0.0) shows code degeneration/repetition even
  without a competing delta -> Math delta may carry instability on its own.
- => Failure is asymmetric in the task vectors, NOT explained by norm magnitude alone.
  Motivates looking at directional/structural differences, not just scale.

## STATUS: qualitative (N=2 prompts, eyeballed). HYPOTHESIS, not confirmed.
## NEXT: GSM8K subset for quantitative accuracy per kappa per sweep to confirm asymmetry.
# Experiment 1b: Coder-delta kappa sweep (Task Arithmetic)
# merged = base + 1.0*delta_Math + kappa*delta_Coder
# Compare against Math sweep (RESULTS_math_sweep.md)

| kappa | math prompt | code prompt |
|-------|-------------|-------------|
| 1.0   | gibberish (2222) [matches math k=1.0, consistency OK] | gibberish |
| 0.8   | gibberish (121212)               | gibberish |
| 0.6   | gibberish (10000)                | gibberish (1 line then collapse) |
| 0.4   | coherent method but WRONG answer (36 not 36/7), loops | broken code, repetition |
| 0.2   | coherent, correct method, trails off | coherent-ish, syntax errors, repetition |
| 0.0   | coherent (base+Math)             | code degenerates into ``` repetition |

## Finding — ASYMMETRY between the two sweeps
- Shrinking MATH delta -> CLEAN full recovery by k=0.4 (correct answer + working code).
- Shrinking CODER delta -> only PARTIAL improvement; never fully clean, code stays broken.
- The two deltas are ~equal magnitude (Math 0.0161, Coder 0.0148, ~8% apart) yet NOT
  equally responsible for the failure. Math delta appears more destabilizing.
- Endpoint clue: base+Math-alone (coder k=0.0) shows code degeneration/repetition even
  without a competing delta -> Math delta may carry instability on its own.
- => Failure is asymmetric in the task vectors, NOT explained by norm magnitude alone.
  Motivates looking at directional/structural differences, not just scale.

## STATUS: qualitative (N=2 prompts, eyeballed). HYPOTHESIS, not confirmed.
## NEXT: GSM8K subset for quantitative accuracy per kappa per sweep to confirm asymmetry.
MDEOF
cat /scratch/pmolugur/merging_project/RESULTS_coder_sweep.md
# Experiment 1b: Coder-delta kappa sweep (Task Arithmetic)
# merged = base + 1.0*delta_Math + kappa*delta_Coder
# Compare against Math sweep (RESULTS_math_sweep.md)

| kappa | math prompt | code prompt |
|-------|-------------|-------------|
| 1.0   | gibberish (2222) [matches math k=1.0, consistency OK] | gibberish |
| 0.8   | gibberish (121212)               | gibberish |
| 0.6   | gibberish (10000)                | gibberish (1 line then collapse) |
| 0.4   | coherent method but WRONG answer (36 not 36/7), loops | broken code, repetition |
| 0.2   | coherent, correct method, trails off | coherent-ish, syntax errors, repetition |
| 0.0   | coherent (base+Math)             | code degenerates into ``` repetition |

## Finding — ASYMMETRY between the two sweeps
- Shrinking MATH delta -> CLEAN full recovery by k=0.4 (correct answer + working code).
- Shrinking CODER delta -> only PARTIAL improvement; never fully clean, code stays broken.
- The two deltas are ~equal magnitude (Math 0.0161, Coder 0.0148, ~8% apart) yet NOT
  equally responsible for the failure. Math delta appears more destabilizing.
- Endpoint clue: base+Math-alone (coder k=0.0) shows code degeneration/repetition even
  without a competing delta -> Math delta may carry instability on its own.
- => Failure is asymmetric in the task vectors, NOT explained by norm magnitude alone.
  Motivates looking at directional/structural differences, not just scale.

## STATUS: qualitative (N=2 prompts, eyeballed). HYPOTHESIS, not confirmed.
## NEXT: GSM8K subset for quantitative accuracy per kappa per sweep to confirm asymmetry.
