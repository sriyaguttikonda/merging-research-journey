# Phase 2 Replication Findings: Comparative Evaluation of Model Merging Methods

## Goal

The goal of this evaluation was to compare all the merged models i created during Weeks 6-8 and see which merging method actually works best on a real task. Up until this point i had created multiple models using Linear, TIES and DARE-TIES, but i didnt have any quantitative results showing which one was actually better. The main question i wanted to answer was whether the more advanced methods like TIES and DARE-TIES really outperform simple linear merging when the source models are already very similar.

## Setup

### Models Compared

I evaluated a total of 10 models. This includes 8 merged models created during Weeks 6-8 and 2 baseline models. The merged models were created using Linear, TIES and DARE-TIES with different weight ratios and densities. The baselines were FLAN-T5-base and the summarization fine-tuned model.

### Evaluation Dataset

For evaluation i used the CNN/DailyMail test split. The full test set contains 11,490 examples, but i evaluated only the first 50 examples. I selected the first 50 using a deterministic range(50) approach so that every model would be tested on exactly the same articles. This makes the comparison fair and reproducible.

### Metrics Used

I used ROUGE-1, ROUGE-2 and ROUGE-L F1 scores. ROUGE-1 measures word overlap between the generated summary and reference summary. ROUGE-2 measures bigram overlap and is generally harder to score well on. ROUGE-L measures the longest common subsequence and gives an idea of overall summary similarity.

### Compute Setup

The evaluation was run on Kaggle using T4 GPUs. For generation i used beam search with num_beams=4 and max_length=128. All models used the same generation settings so that differences in performance would come only from the merge method.

## Methodological Notes

One thing i discovered while building the evaluation pipeline is that prompt format matters way more than i expected. My first evaluation of FLAN-T5-base gave a ROUGE-1 score of only 0.18 and the generated summaries looked broken. One example output was just "Palestinian authority's authority."

After testing different prompts i found that changing the prompt to "Please write a short summary of this article" improved ROUGE-1 from 0.1819 to 0.2600, ROUGE-2 from 0.0773 to 0.1233 and ROUGE-L from 0.1549 to 0.2105. Thats a huge improvement without changing the model itself. Because of this, i kept the prompt fixed for every model in the evaluation.

Another thing to note is that most CNN/DailyMail articles are longer than T5-base's 512-token context window. This means the articles get truncated before generation. So the evaluation is really measuring how well the models summarize the first part of the article rather than the full article.

Its also important to remember that ROUGE only measures overlap with the reference summary. It does not directly measure factual accuracy or whether the generated summary is actually better written.

## Results

| Model                   | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ----------------------- | ------- | ------- | ------- |
| Linear 0.3/0.7          | 0.2904  | 0.1210  | 0.2247  |
| Linear 0.5/0.5          | 0.2820  | 0.1179  | 0.2121  |
| Linear 0.7/0.3          | 0.2804  | 0.1209  | 0.2170  |
| Summarization fine-tune | 0.2765  | 0.1023  | 0.2083  |
| DARE-TIES density 0.9   | 0.2738  | 0.1234  | 0.2220  |
| FLAN-T5-base            | 0.2600  | 0.1233  | 0.2105  |
| TIES density 0.2        | 0.2022  | 0.0701  | 0.1578  |
| TIES density 0.5        | 0.1999  | 0.0775  | 0.1638  |
| DARE-TIES density 0.5   | 0.0343  | 0.0000  | 0.0294  |
| DARE-TIES density 0.1   | 0.0009  | 0.0000  | 0.0009  |

## Key Findings

* Linear 0.3/0.7 achieved the best performance overall with ROUGE-1 of 0.29.
* All three linear merges performed very similarly and clustered around 0.28 ROUGE-1.
* TIES underperformed linear merging at both tested densities, scoring around 0.20 ROUGE-1.
* DARE-TIES is extremely sensitive to density. At 0.9 it scored 0.27, but it dropped to 0.03 at 0.5 and to 0.001 at 0.1.
* DARE-TIES at densities 0.5 and 0.1 essentially collapsed and produced non-functional outputs.
* More complicated merging methods did not automatically lead to better performance.

## Mechanistic Interpretation

Initially i thought DARE-TIES should perform better because DARE is supposed to reduce interference before merging. But after looking at the results i dont think interference is even the main problem here.

The first thing i noticed is DARE-TIES completely dies as density goes down. At density 0.9 it is still somewhat ok but at 0.5 and especially 0.1 it is basically unusable. I think this is happening because the models are already very similar. So when DARE randomly drops delta parameters, it is also dropping a lot of overlapping useful information. Then whatever survives gets rescaled. The higher the drop rate, the larger the rescaling. So maybe instead of preserving signal, it is amplifying the mistakes also. Thats why the drop is not gradual. It just falls off a cliff.

TIES looks different to me. It doesnt collapse like DARE-TIES. But it is still worse than linear. Initially i was confused because TIES is supposed to be smarter than linear merging. But now i think the issue is again the similarity between models. If both models already contain similar information, then trimming might be removing useful overlapping deltas. Then sign election comes and removes even more information. So TIES is trying to resolve conflicts when maybe there are not many important conflicts to resolve in the first place.

Linear merging is actually doing the dumbest thing here and somehow winning. But maybe thats exactly why it works. It is not trimming anything. It is not sparsifying anything. It is just averaging. Since the models are already close to each other, the shared information stays intact instead of getting removed.

So right now my understanding is that both TIES and DARE-TIES assume some level of independence between task vectors. But in my case the models are too similar. Because of that, sparsification becomes harmful instead of helpful. DARE-TIES fails because random dropping plus rescaling becomes unstable. TIES fails because trimming and sign election throw away useful overlapping information. Linear doesnt have these problems because it preserves everything.

I think the biggest thing i learned from this evaluation is that more complicated merging methods are not automatically better. They seem to work only when the assumptions behind them are true. If those assumptions break, then the simplest method can actually perform the best.

## Limitations

There are a few limitations in this evaluation. First, the evaluation uses only 50 examples, which is a relatively small subset of the dataset. Second, only one dataset was tested, so the results may not generalize to other tasks. Third, article truncation means the models never see the full article for many examples. Fourth, only one prompt format was used for the final evaluation even though prompt choice clearly affects performance. Finally, T5-base is much smaller than the models commonly used in many recent model merging papers.

## Next Steps

The next thing i would like to explore is whether the DARE-TIES collapse holds when merging less similar models. For example, merging a math fine-tune and a code fine-tune of the same base would test whether the collapse only occurs on highly similar source models. If the collapse goes away in that setting, that would support my hypothesis that the failure depends on task vector overlap. I would also like to test whether these observations hold on larger evaluation sets, on tasks other than summarization, and at larger model scales.
