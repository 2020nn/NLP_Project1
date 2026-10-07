# Practice: rebuilding from raw data

My own from-scratch practice, written to understand the problem before proposing ideas to the team.
These scripts are **not** the team's pipeline (that is in [`../team`](../team)).

| File | What it does | GPU | Result |
|---|---|---|---|
| `01_eda.py` | Data sizes, length check, how many characters are changed | No | 60.3% of characters changed; copying the input = 39.7% char accuracy |
| `02_freq_baseline.py` | Replace each obfuscated character with its most frequent original (5-fold CV) | No | **80.6%** char accuracy, 1.2% sentence accuracy |
| `03_char_baseline.py` | KoCharELECTRA character tagger, optional `--jamo` input | Yes | Not run yet |

## How to run

Put DACON's `train.csv` and `test.csv` in the same folder as the scripts, then:

```bash
python 01_eda.py
python 02_freq_baseline.py
python 03_char_baseline.py --limit 300 --epochs 1 --out smoke.csv   # quick check on a GPU
python 03_char_baseline.py --jamo
```

## 한국어 요약
팀 아이디어를 내기 전에 문제를 이해하려고 원본 데이터로 처음부터 다시 만들어 보는 연습용 폴더입니다. 팀의 실제 코드는 `team/` 폴더에 있습니다.
