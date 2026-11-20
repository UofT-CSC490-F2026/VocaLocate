# Part 5: On-Policy Distillation

## 5.1 Method and hypothesis

**Choice of method.** We continue from the same model as Parts 3 and 4: the d20 baseline-long SFT
checkpoint (step 932, 2.73% on GSM8K). Our team did not run midtraining, so this is SFT-only rather than
SFT plus midtraining. We use on-policy distillation in the form described by Thinking Machines Lab (2025).
The student samples its own completions, a teacher scores every sampled token, and the student is updated
to lower the per-token reverse KL, KL(student ‖ teacher), on those samples. The teacher has to share our
32,768-token tokenizer. Karpathy's released nanochat models use a different vocabulary, so the only
stronger teacher we have is our own Part 3 RL model (step 466, 20.09%). For the ablations we also use
the SDPO idea (Hübotter et al., 2026), where the teacher is the student itself with privileged information
in its prompt.

**Implementation** (`a4/part-5-distill/chat_distill.py`). We copied nanochat's `chat_rl.py` and changed
only the advantage. For token y_t of a sampled completion,

A_t = λ_r (r − mean_group r) + λ_d [log p_T(y_t | c_T, y_<t) − log p_S(y_t | x, y_<t)],

with the teacher term under stop-gradient and discount 0, as in the Thinking Machines post. The loss is
`chat_rl.py`'s token-normalized policy gradient, so with λ_r = 0 and λ_d = 1 its gradient is the
per-token reverse KL to the teacher. Sampling, data order, batch (16 questions × 16 samples), learning
rates and the in-training pass@k evaluation are unchanged. Distillation therefore sees exactly the same
first batch as RL; step 0 of every run reproduces Part 3's pass@1 = 0.020, pass@8 = 0.128 and training
reward of 0.039. We tested the code on CPU with tiny models. A teacher identical to the student gives a
reverse KL of exactly 0, and every teacher mode trains and saves checkpoints. A 10-step smoke run on the
real models raised training accuracy from 0.04 to 0.30 and lowered the KL from 1.16 to 0.85 nats per token.

**Hypothesis.** We committed it to GitHub in `529fe48`, before any Part 5 GPU run
(`a4/part-5-distill/HYPOTHESIS.md`). In short: RL gives one bit per completion, and half of Part 3's
groups gave no gradient. Distillation gives a signal on every token, so we expected the student to
recover most of the RL teacher's accuracy with fewer than half of RL's samples (200 steps vs 467). We
expected at least 17% on GSM8K, no collapse, and a mistake profile like the teacher's. Because reverse KL
is mode-seeking and bounded by the teacher, we did not expect the student to exceed it.

## 5.2 Runs

All four runs: 200 steps, 8×H100, one seed, start from SFT step 932, learning rate decaying linearly to
zero over 200 steps. Results are logged to W&B project `a4-distill`.

| Run | Teacher | Advantage |
|---|---|---|
| D00 | RL step 466 (frozen) | teacher log-ratio |
| D01 | the student itself (EMA 0.05), with the GSM8K reference solution in its prompt | teacher log-ratio |
| D02 | RL step 120 (frozen) | teacher log-ratio |
| D03 | RL step 466 (frozen) | teacher log-ratio + (r − mean r) |
| D01b (exploratory) | the student itself, frozen, with the reference solution | 0.9 (r − mean r) + 0.1 × teacher log-ratio |

The runs used volume `vocalocate-a4-p5` in a separate Modal workspace, holding copies of the SFT and RL
checkpoints. Each final evaluation ran on 2 H100s, so it could share the workspace's GPU limit with an
8-GPU training job. Re-evaluating the Part 3 RL model this way reproduced its scores exactly (GSM8K 20.09%,
ARC-E 57.79, ARC-C 46.42, MMLU 35.44, HumanEval 11.59, ChatCORE 0.2357). The RL step-120 checkpoint,
D02's teacher, scores 14.94% on GSM8K.

## 5.3 Results

*(to be filled from `analysis/compare_runs.py`)*

## 5.4 Ablations

**D02, a weaker teacher (RL step 120).** The student tracks its teacher closely. Its KL to the teacher
fell from 0.98 to 0.015 nats per token. In-training pass@1 and pass@8 at step 120 were 0.135 and 0.303;
RL had 0.130 and 0.303 at its own step 120, the checkpoint used as the teacher. On the full test set D02
scores **14.71%**, against **14.94%** for the teacher (3 questions apart, out of 1,319). The other tasks
also move with the teacher: ARC-E 55.1 vs 56.3, MMLU 35.1 vs 35.1, HumanEval 7.3 vs 6.7. This supports
the prediction that the student is bounded by its teacher (prediction met). Two side effects stand out.
First, D02 had a smaller, earlier version of RL's collapse. Around steps 40–60, up to 35% of its rollouts
contained junk text (RL peaked near 60%), and the KL briefly rose again before falling. The step-120
teacher was trained through RL's own collapse, so it likely still gives some probability to those
outputs. Second, D02 almost never gives up: it writes `####` on 98% of test answers (RL: 72%), and only 2%
of its wrong answers end without one (RL: 34%). The step-120 teacher predates the habit of stopping early
that RL picked up later.

**D01, SDPO-style self-teacher.** This run failed completely: **0.0%** on GSM8K and **0.0%** on HumanEval.
Multiple-choice scores were unchanged (ARC-E 57.7, ARC-C 46.9, MMLU 35.6, vs SFT 58.8, 46.5, 36.3),
because those tasks only compare answer probabilities. The failure has a clear cause. On 1,248 of the
1,319 test questions, the greedy answer is exactly the last line of the teacher's prompt, "Correctly solve
the original question." Most of the rest read "The correct solution to the original question is
(indeed) correct." The self-teacher sees phrases like "Correct solution:" in its prompt. A 0.5B model
readily copies text from its context, so it gives those phrases high probability as continuations.
Reverse KL then pulls the student toward producing them, even though the student never sees that prompt:
the privileged context leaks into the student. The EMA teacher tracks the student and reinforces the
habit, and with no reward term nothing pushes back. The signal was also weak from the start: the step-0
KL to the self-teacher was 0.18 nats per token, against 1.16 for the RL teacher. We predicted D01 would
be clearly worse than D00 and RL (≤ 10%), because a 3% model cannot use the reference solution in context.
The direction held, but the failure was a collapse rather than slow learning. This matches the SDPO
paper's caveat that the method relies on in-context learning and falls behind GRPO below about 1.5B
parameters. The paper also never puts ground-truth solutions in the teacher's prompt; it uses the model's
own successful answers, which our model rarely produces.

**D03, distillation plus the RL reward.** Adding (r − mean r) to the distillation term changed almost
nothing. D03's training curves follow D00's within a few hundredths at every logged step, including the
collapse (Section 5.3). The per-token log-ratios are much larger than the reward advantage, which is at
most ±1 per sequence and averaged over tokens, so the distillation term dominates the update. Final
full-test accuracy was **18.20%**, 1.9 points below RL (64 questions gained, 89 lost; McNemar p = 0.052),
and its general-task scores were slightly lower (HumanEval 10.4 vs 11.6). We predicted at least
D00 + 1 point; the difference from D00 is reported in Table 5.2.

**D01b (exploratory, not pre-registered).** After D01 collapsed we ran the SDPO paper's own fix for small
models: a frozen self-teacher, with the advantage 0.9 (r − mean r) + 0.1 × self-distillation.
*(results to be filled)*

## 5.5 Distillation vs RL

*(to be filled)*
