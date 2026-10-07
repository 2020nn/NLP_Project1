# Team pipeline

Code from the team's pipeline.

| File | Author | What it does |
|---|---|---|
| `train_stage3.py` | Team (latest version) | Stage 3 corrector: character model (KoCharELECTRA) + word model (klue/roberta-large), 5-fold ensemble |
| `train_stage4.py` | Me | Thin wrapper that reruns stage 3 on stage 3's own out-of-fold output (stacking) |

Stages 1–2 (`train_cascade_v2.py`) are run by the team and are not included here. Stage 3 needs their outputs:
`oof_v2.csv` (out-of-fold restorations of the training set) and `submission_cascade_v2_stage2.csv` (restorations of the test set).

## Run on Colab (GPU)

```python
!python train_stage3.py --limit 200 --epochs 1 --folds_run 1 --bs 8 --out smoke.csv   # smoke test
!python train_stage3.py --folds_run 1 --bs 8                                          # one fold: does it beat stage 2?
!python train_stage3.py --bs 8                                                        # all folds
```

On a free Colab T4 (15 GB), the default `--bs 32` runs out of GPU memory, so use `--bs 8`.

## 한국어 요약
팀 파이프라인 코드입니다. `train_stage3.py`는 팀의 최신 버전이고, `train_stage4.py`는 제가 덧붙인 스태킹용 래퍼입니다. 1~2단계 코드는 팀에서 돌리며 여기에는 없습니다.
