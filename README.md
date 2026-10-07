# Restoring Obfuscated Korean Reviews (NLP)

Study-group project for a DACON competition: restore deliberately scrambled Korean reviews to their original text, and raise accuracy step by step.

```
input : 절테 간면 않 된는 굣 멥몫
output: 절대 가면 안 되는 곳 메모
```

## Start here

| Where | What |
|---|---|
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | The problem, how the team works, my role, progress log |
| [`practice/`](practice) | My from-scratch practice: EDA and baselines built from the raw data |
| [`team/`](team) | The team's stage 3 model, plus my stage 4 stacking wrapper |
| `docs/REPORT.md` | Final report (added at the end of the project) |

## Results so far

| Method | Char accuracy | Sentence accuracy |
|---|---|---|
| Copy the input | 0.397 | – |
| Frequency-mapping baseline (5-fold CV) | 0.806 | 0.012 |
| Team pipeline (stages 2–3) | in progress | in progress |

## Data

Competition data is not included. Download `train.csv` and `test.csv` from DACON and put them next to the scripts.

## 한국어 요약

DACON 대회 데이터로 난독화된 한글 리뷰를 복원하고 정확도를 높이는 스터디 프로젝트입니다. 프로젝트 설명은 `docs/PROJECT_OVERVIEW.md`, 직접 만든 연습 코드는 `practice/`, 팀 코드는 `team/`에 있습니다. 최종 보고서는 프로젝트가 끝나면 `docs/REPORT.md`로 추가합니다.
