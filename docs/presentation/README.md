# Final presentation

Deck used for the study group's final presentation. It compares how two teams solved the same DACON task.
**We are Team B**: each member ran the Team B pipeline (see [`../../team`](../../team)).

| File | Language |
|---|---|
| `review_restoration_team_comparison_ko.pptx` | Korean (original) |
| `review_restoration_team_comparison_en.pptx` | English translation, including speaker notes |

## Key points

- **Shared start**: inputs and answers have the same length, so both teams framed restoration as per-character classification.
- **Team A**: restore first, then a second model (KcELECTRA) corrects the restored text, applied iteratively. Leaderboard F1 0.989.
- **Team B (us)**: a character model (KoCharELECTRA) reads the hint plus the original jamo, a word model (klue/roberta-large) is added, and finally a language model (polyglot-ko-5.8b) re-selects disputed words among candidates from several runs. Public F1 0.98908.
- **Lesson**: data observation and error analysis decided each next design. The two scores use different precision and conditions, so they do not rank the teams.

Team B's scores come from practice submissions made after the competition closed.

## 한국어 요약
스터디 최종 발표 자료입니다. 같은 DACON 과제를 두 팀이 어떻게 풀었는지 비교합니다. 우리는 **B팀**이고, 팀원들이 B팀 파이프라인을 각자 한 번씩 돌려 봤습니다. 한국어 원본과 영어 번역본(발표자 노트 포함)이 있습니다.
