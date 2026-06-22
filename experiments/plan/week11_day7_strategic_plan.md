# Week 11 Day 7 - Strategic Plan

## Success condition (in my own words)

For MBZUAI the goal should not be "somehow get an acceptance before Dec 2026" because realistically that is not fully under my control. What is under my control is having a solid research story, a complete paper draft and enough experiments that I can defend every claim if someone asks me about it in an interview.

For Option A (lean into what I have, target ICLR 2027 workshop), the focus should be on strengthening the work rather than rushing to submit. That means testing more model pairs, running proper benchmarks instead of only anecdotal prompts, checking whether the delta magnitude hypothesis still holds across families and being honest about where the evidence is strong and where it is only a correlation.

The GitHub repo becomes important here. If someone opens the repo, they should be able to see the whole research process and not just the final paper. The weekly logs, failed experiments, plots, reasoning and limitations are all part of the story.

The success condition is not "published by Dec 2026". The success condition is having a defensible paper, a clear research narrative and enough evidence that a reviewer, professor or interviewer can see that I actually did research and understand my own work.

## Primary target

ICLR 2027 workshop submission. Deadline February 2027.

For MBZUAI Dec 2026 application, status will be "paper in preparation" or "submitted, under review." That is acceptable. Acceptance is not the determining factor.

## Current research findings (Week 11 end)

1. Qwen2.5-Math and Qwen2.5-Coder have per-parameter delta magnitudes around 0.015-0.016, roughly 100x larger than clean SFT models and matching continued-pretrained references from literature.
2. Clean SFT models across three families (Qwen-Instruct, three Mistral fine-tunes, four LLaMA-2 references from DARE-the-Extreme) consistently fall in 0.00006 to 0.00073 range.
3. Magnitude and similarity are independent dimensions of fine-tuning behavior.
4. Per-layer similarity patterns differ qualitatively between continued-pretrained (Qwen U-shape) and clean SFT (Mistral inverted-U) pairs.
5. Controlled merge experiment: same TIES configuration produces complete gibberish on Qwen Math+Coder but coherent (degraded) output on Mistral Instruct+OpenHermes.

## Week 12 to 19 plan

| Week | Goal | Time | Notes |
|---|---|---|---|
| 12 | Baselines: unmerged models on test prompts | 2-3 days | Run Mistral-Instruct, OpenHermes, Qwen-Math, Qwen-Coder individually on Day 5 prompts. |
| 13-14 | Benchmark evaluation (GSM8K and HumanEval) | 4-5 days | New infrastructure: never set up these benchmarks before. Budget realistically. |
| 15 | One more pair OR strengthen existing | 3-4 days | Decision based on Week 13-14 results. If Mistral merge benchmarks poorly, may need second pair. |
| 16 | Paper outline + draft Intro, Related Work, Conclusion | 4-5 days | LaTeX setup. |
| 17 | Draft Methods, Results, Analysis | 5-6 days | Heaviest writing week. |
| 18 | Discussion, limitations, revision pass; share with professor | 4-5 days | Outside reader sees draft here. |
| 19+ | Final polish, submission prep | Open | Buffer before Feb 2027 ICLR deadline. |

Target: complete draft by late August 2026. Polished version September-October 2026. MBZUAI application Dec 2026 with "paper in preparation for ICLR 2027 workshop submission."

## Action items (next 1-2 weeks)

1. Identify a specific professor I could realistically ask to read my draft in Week 18. Do not ask them yet. Just have a name and decide by Week 12 start.
2. Future strategic re-planning happens only if something major changes (e.g., Week 13 benchmarks show something unexpected). Otherwise execute the plan above.

## Honest limitations and risks

- Never set up GSM8K or HumanEval evaluation harness before. Week 13 may take longer than budgeted.
- No identified outside reader yet for Week 18.
- Mistral merge benchmark could score very low and require a second pair (Week 15 contingency).
- N=1 continued-pretrained pair (Qwen Math+Coder) is the weakest part of the current evidence. Reviewers may push back on generalization.
- No proposed algorithmic contribution. Paper is empirical observation, not method proposal.

## What is NOT in this plan

- Algorithm development (Option C). Deferred. May become future work after this paper.
- Cross-family specialist pair search (Option B). Deferred unless Week 15 contingency requires it.
- Publishing somewhere other than ICLR workshop as fallback. Not pre-planned for. Decided then if needed.
