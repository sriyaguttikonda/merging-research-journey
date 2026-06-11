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

## Questions for tomorrow

- math vs base separately
- coder vs base separately
- plot layerwise similarity
- check if U shape is actually real
- maybe compare against literature numbers
- decide if path B is needed

current feeling: this result is more interesting than i expected. i thought i would just confirm the models are similar and move on. now im not so sure.
