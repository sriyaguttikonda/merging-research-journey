(merge) pmolugur@cpn-b02-24:/scratch/pmolugur/merging_project$ cat > /scratch/pmolugur/merging_project/RESULTS_math_sweep.md << 'MDEOF'
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
