# Restoring Obfuscated Korean Reviews (NLP)

Restore Korean product/place reviews whose characters were deliberately scrambled, back to the original text.
Built for a DACON competition, using character-level classification and a multi-stage correction cascade with pretrained Korean language models.

```
input : 절테 간면 않 된는 굣 멥몫
output: 절대 가면 안 되는 곳 메모
```

> 한국어 요약은 [아래](#한국어-요약)에 있습니다.

## Problem framing

- **Data**: 11,263 training sentences (average 93 characters) and 1,689 test sentences (average 168 characters).
- **Key observation**: after stripping trailing spaces from the targets, every input has *exactly* the same length as its target.
  So the task becomes **per-character classification**: for each position, predict the original character.
- About **60% of characters are changed**. Copying the input gives 39.7% character accuracy.
- Obfuscation works at the **jamo level** (initial consonant / vowel / final consonant), so jamo are used as extra inputs and as decoding constraints.

## Approach

| Step | Method | Why |
|---|---|---|
| 1. EDA | Length checks, change rate | Decide how to frame the problem |
| 2. Statistical baseline | Map each obfuscated character to its most frequent original | A floor that any model must beat |
| 3. Deep learning baseline | KoCharELECTRA (character-level transformer) token classification | Uses sentence context, which the statistical baseline cannot |
| 3b. + Jamo input | Add initial/medial/final jamo embeddings | Unseen obfuscated characters can still be decoded from their jamo |
| Stages 1–2 | Character-level KoCharELECTRA cascade (script not included yet) | Remaining errors were mostly *non-existent words*, e.g. "아저시들" — a character model lacks word knowledge |
| Stage 3 | Character model **+ word-level model (klue/roberta-large)**, fused per character | Adds word knowledge to fix non-word errors |
| Stage 4 | Stacking: run stage 3 again on stage 3's out-of-fold output | Optional, if stage 3 still improves |

### Stage 3 model (`train_stage3.py`)

- **Two encoders**: the character model reads the previous stage's output plus the obfuscated jamo. The word model reads the same text in word pieces.
- **Alignment**: word-piece outputs are mapped back to character positions using tokenizer offsets, plus a "position inside the word piece" embedding.
- **Fusion**: concatenate → linear layer → one Transformer layer, so characters sharing a word piece can still differ.
- **Multi-task heads**: syllable + initial + medial + final jamo. Log-probabilities are summed at inference.
- **Constrained decoding**: syllables whose jamo were never produced by the observed obfuscated jamo in training are masked out.
- **Validation**: 5-fold CV with out-of-fold (OOF) predictions. OOF outputs become the next stage's training input, so each stage never sees its own training labels in its inputs.
- **Long text**: 384-character windows with 192-character stride; overlapping predictions are averaged, weighted toward window centres.
- **Engineering**: fp16 mixed precision, separate learning rates for the large word model (2e-5) and the rest (1e-4), and fold caching to resume after Colab disconnects.

## Results

| Method | Char accuracy | Sentence accuracy | Validation |
|---|---|---|---|
| Copy input | 0.397 | – | all training data |
| Frequency mapping baseline | **0.806** | 0.012 | 5-fold CV |
| Char model baseline (step 3) | TBD | TBD | 10% holdout |
| Stages 1–2 cascade | TBD | TBD | 5-fold OOF |
| Stage 3 (+ word model) | TBD | TBD | 5-fold OOF |
| DACON leaderboard | TBD | | public LB |

*TBD values will be filled in after the full runs finish.*

## How to run (Google Colab, GPU runtime)

1. Download `train.csv` and `test.csv` from the competition page (not included in this repo).
2. Upload them with the scripts to Colab and run:

```python
!python 01_eda.py
!python 02_freq_baseline.py                       # no GPU needed
!python 03_char_baseline.py --limit 300 --epochs 1 --out smoke.csv   # quick smoke test
!python 03_char_baseline.py --jamo                 # char model baseline with jamo input
```

Stage 3 needs the stage 2 outputs (`oof_v2.csv`, `submission_cascade_v2_stage2.csv`) in the same folder:

```python
!python train_stage3.py --folds_run 1 --bs 8       # one fold first, to check it beats stage 2
!python train_stage3.py --bs 8                     # all 5 folds (resumes from cache)
!python train_stage4.py --bs 8                     # optional stacking
```

On a free Colab T4 (15 GB), the default batch size of 32 runs out of memory with two transformer encoders. `--bs 8` fits.

## Repository structure

```
01_eda.py             Step 1  data exploration
02_freq_baseline.py   Step 2  frequency-mapping baseline (5-fold CV)
03_char_baseline.py   Step 3  KoCharELECTRA baseline (--jamo for jamo input)
train_stage3.py       Stage 3 char + word model corrector (5-fold, ensemble)
train_stage4.py       Stage 4 stacking wrapper around stage 3
```

The stage 2 training script (`train_cascade_v2.py`) is not included yet.

## Lessons learned

- **Look at the data first**: the equal-length observation turned a sequence-to-sequence problem into a much simpler tagging problem.
- **Baselines set the bar**: a 10-line frequency model already reached 80.6%, so a deep model has to clearly beat that.
- **Let errors pick the next model**: the move to a word-level model came from error analysis, not guesswork.
- **Keep validation leak-free**: each stage is trained only on out-of-fold outputs of the previous stage.
- **Fit the hardware**: two large encoders need batch-size tuning on a 15 GB GPU.

---

## 한국어 요약

**난독화된 한글 리뷰를 원래 문장으로 복원하는 DACON 대회 프로젝트**입니다.

- **문제 정의**: 정답 끝의 공백을 지우면 입력과 정답의 길이가 100% 같아서, **글자마다 원래 글자를 맞히는 분류 문제**로 풀었습니다. 글자의 약 60%가 바뀌어 있습니다.
- **베이스라인**: 난독화 글자를 가장 자주 대응된 정답 글자로 바꾸는 통계 방법이 5-fold 기준 **글자 정확도 80.6%**입니다. 하지만 문장 정확도는 1.2%로, 문맥을 보는 모델이 필요합니다.
- **딥러닝**: 글자 단위 모델(KoCharELECTRA)에 초성·중성·종성 입력을 더했습니다.
- **오류 분석 → 개선**: 남은 오류가 대부분 "아저시들"처럼 없는 단어였습니다. 그래서 3단계에서 **단어 단위 모델(klue/roberta-large)을 글자 모델과 합친 교정기**를 만들었습니다.
- **검증**: 5-fold 교차검증과 OOF 예측을 썼습니다. 각 단계는 이전 단계의 OOF 결과만 입력으로 받아 정보 누수를 막았습니다.
- **자원 제약 해결**: 코랩 T4(15GB)에서 메모리가 부족해 배치를 32에서 8로 줄였습니다.

대회 데이터는 규칙상 포함하지 않았습니다. 대회 페이지에서 받아 같은 폴더에 두고 실행하면 됩니다.
