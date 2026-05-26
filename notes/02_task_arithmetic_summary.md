# Task Arithmetic Paper Summary

## 1. Introduction

Task Arithmetic (Ilharco et al., ICLR 2023) proposes that the difference between a pretrained model and its finetuned version can be treated as a "task vector" which supports arithmetic operations like addition, negation and analogies.

## 2. Core Idea

The main thing i understood from Task Arithmetic is that the task vector is not just some random weight difference after finetuning...they are saying it actually contains the direction for that specific task. So instead of thinking like "ok finetuned model done", they are treating those vectors almost like objects that can be manipulated. Like if a model learned toxicity or sentiment or some vision task, then the change in weights itself stores that behaviour in some meaningful way i guess.

## 3. Comparison with TIES

And unlike TIES which mainly focuses on summing the trimmed vectors carefully so interference is less, this paper is saying u can do more operations also. Like adding vectors to combine capabilities, negating them to remove some behaviour and even analogies where one task relation can help another task. So overall its less about just merging and more about understanding weight space structure itself i guess. TIES feels more like "how to merge properly" while this paper is more like "what even are these task vectors and what all can we do with them".

## 4. Negation Experiments

The negation experiments were actually interesting because they showed that u can remove one behaviour without completely destroying the model. Like for GPT-2 they reduced toxic generations a lot but the normal language modelling capability was mostly preserved. Same for vision models also...the target task accuracy drops heavily but control tasks dont collapse. Thats the important thing i think because random subtraction or gradient ascent baselines were messing up everything. So this kinda supports their claim that the task vector contains localized task-specific information and not just random noise spread everywhere.

## 5. Addition Experiments

The addition experiments are basically where multiple task vectors are added together to create multitask models. What i found interesting is that they got around 92% specialist performance without separately training a huge multitask model from scratch. Also as more task vectors were added the average performance improved which was kinda surprising because i would have expected more interference. I guess one reason addition works is because many task vectors are almost orthogonal unless tasks are semantically related. So the directions dont strongly clash with each other. Another surprising thing was that sometimes adding a task vector from one task improves performance on a different target task also, which kinda suggests some task vectors contain more general transferable improvements and not just task-specific capability.

## 6. Analogy Experiments

The analogy part was honestly the most unexpected thing for me. They basically show that relationships between tasks can transfer. Like if u know relation between task A and B and also know task C, then u can estimate task D even without training data for D. They tested this for sentiment transfer and image subpopulation generalization. This is the point where it stopped feeling like just merging and started feeling more like representation geometry or embedding arithmetic but happening directly in model weight space. And for analogies to even work, the relation between two task vectors has to actually mean something consistent and transferable, not just random parameter difference. Thats a much stronger claim about weight space structure than just saying task vectors are localized.

## 7. Limitations and Open Questions

One limitation i noticed is that everything assumes same base pretrained model. Like the subtraction/addition only makes sense because parameter spaces are aligned already. They also mention it doesnt directly handle adapters or new classification heads because dimensions and structures change. But the bigger thing i kept thinking about was...what happens for completely different architectures or model families. Like Llama and Qwen or maybe even vision-language models trained differently. Can some common representation space exist where task directions can still interact meaningfully without huge retraining or expensive alignment. I didnt really think about this kind of question before reading the paper honestly.
