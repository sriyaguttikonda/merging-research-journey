## Research Question

**Primary question:** Why do existing model merging methods (Linear, TIES, DARE-TIES) fail on certain model pairs even when used with parameter settings the original papers claim should work?

**Specific instance from preliminary work:** Qwen2.5-Math-7B-Instruct + Qwen2.5-Coder-7B-Instruct produced catastrophic failures across linear merging (repetition loops) and TIES at density 0.5 (complete gibberish). Phase 2 work on similar T5 models also showed unexpected failure modes (DARE-TIES collapse at low densities).

**What I want to understand:**
1. What property of the input model pair determines whether merging succeeds or fails?
2. Is there a measurable characteristic of the model pair (e.g., task vector similarity) that predicts which merging method will work?
3. Under what specific conditions does each method (Linear, TIES, DARE-TIES) work vs. fail?

**Why this matters:**
- Existing papers' claims (e.g., "top 20% works") assume conditions that may not generalize
- If we can predict when methods fail, we can either avoid those settings or design methods specifically for them
- This is a foundation for the bigger ambition of designing a better merging algorithm

## Defining Outcomes

When evaluating a merged model's output, I categorize it into one of these states:

1. **Broken** — Output has no grammatical structure or coherent text. Example from my work: TIES merge of Math+Coder produced "0000000000:0a,('s0 youc in0..." from the first generated token.

2. **Degraded** — Output produces grammatical text but cannot complete its reasoning, gets stuck in loops or produces incoherent flow. Example from my work: Linear merge of Math+Coder started "To determine how long it will take to fill the water tank..." then collapsed into "Pipe A's rate + Pipe B's rate + Pipe C's rate:" repeating forever.

3. **Fluent but incorrect** — Output is grammatically clean and complete, but the answer is wrong. (Hypothetical: model says the tank fills in 7.5 hours instead of 5.14.)

4. **Working** — Output is grammatically clean, complete, and arrives at the correct answer with valid reasoning steps.

