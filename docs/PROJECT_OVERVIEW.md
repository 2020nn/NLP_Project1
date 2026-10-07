# Project Overview: Restoring Obfuscated Korean Reviews

*Study-group project for a DACON competition. Status: in progress.*

## 1. The problem

Korean reviews were deliberately obfuscated by swapping parts of each syllable. The goal is to restore the original text.

```
input : 절테 간면 않 된는 굣 멥몫
output: 절대 가면 안 되는 곳 메모
```

- **Data**: downloaded from DACON. 11,263 training sentences and 1,689 test sentences.
- **Goal of the project**: raise restoration accuracy step by step.

## 2. How the team works

1. One teammate built the baseline model and runs the experiments on GPU.
2. The other members analyse the results and propose ideas to improve accuracy.
3. The runner tests each idea. If accuracy improves, the idea is adopted as the next stage.

This is model training, not prompt engineering: an idea is a change to the model, the inputs or the decoding.
For example, "the model invents non-existent words, so add a model that knows words".

## 3. Where the pipeline is now

| Stage | What it does | Why it was added |
|---|---|---|
| Stages 1–2 | Character-level model (KoCharELECTRA) predicts the original character at each position | Input and output have the same length, so this is per-character classification |
| Stage 3 | Adds a word-level model (klue/roberta-large) on top of the stage 2 output | Remaining errors were mostly non-existent words, e.g. "아저시들"; a character model lacks word knowledge |

Code: [`../team`](../team).

## 4. My role

I joined after the pipeline was already running. My task is to **propose ideas that raise accuracy**.

My plan:
1. **Understand the data from scratch**: rebuild simple baselines myself ([`../practice`](../practice)).
2. **Error analysis**: compare the team's stage 2 output with the answers and find which errors remain.
3. **Propose ideas** based on those errors, and test them where I can.

## 5. Progress log

| Date | What I did | Result |
|---|---|---|
| 2026-10-07 | Data exploration | Inputs and answers have the same length; 60.3% of characters are changed |
| 2026-10-07 | Frequency-mapping baseline (5-fold CV) | 80.6% char accuracy, 1.2% sentence accuracy: context is needed |
| 2026-10-07 | Stage 3 smoke test on Colab T4 | Out of GPU memory at batch size 32; retry with `--bs 8` pending |

## 6. Next steps

- Error analysis on the team's stage 2 output (`oof_v2.csv`)
- Turn the most common error types into concrete ideas for the team
- Final report with results: `docs/REPORT.md` (at the end of the project)

---

## 한국어 요약

- **무엇**: DACON 대회 데이터로 난독화된 한글 리뷰를 원래 문장으로 복원하고, 정확도를 단계별로 높이는 스터디 프로젝트입니다.
- **팀 진행 방식**: 팀원 한 명이 베이스라인을 만들고 GPU로 계속 돌립니다. 나머지는 결과를 분석해 정확도를 올릴 아이디어를 내고, 효과가 있으면 다음 단계로 채택합니다. 프롬프트를 만드는 방식이 아니라 모델을 직접 학습시키는 방식입니다.
- **현재 단계**: 글자 단위 모델(1~2단계) 위에 단어 단위 모델을 붙인 3단계까지 진행됐습니다.
- **내 역할**: 중간에 합류했고, 정확도를 높일 아이디어를 내는 역할입니다. 원본 데이터로 직접 베이스라인을 만들어 보고(`practice/`), 팀 모델의 오류를 분석해 아이디어를 제안합니다.
- **최종 보고서**는 결과가 나오면 `docs/REPORT.md`로 추가합니다.
