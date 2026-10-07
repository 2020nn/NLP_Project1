"""난독화된 한글 리뷰 복원: 4단계 (3단계 결과를 다시 힌트로 쓰는 스태킹)

train_stage3.py는 힌트 파일/열/출력 이름을 옵션으로 받으므로, 4단계는 그 기본값만 바꿔서 그대로 돌린다.
  학습 힌트: submission_stage3_oof.csv 의 stage3 열 (모델이 본 적 없는 문장의 복원 결과)
  테스트 힌트: submission_stage3.csv 의 output 열
  결과: submission_stage4.csv (+ submission_stage4_oof.csv, 캐시 submission_stage4_cache.pt)

  python train_stage4.py --folds_run 1     # fold 하나로 3단계보다 나아지는지 먼저 확인
  python train_stage4.py                   # 전체 (끝난 fold는 캐시에서 이어받는다)

train_stage3.py와 같은 폴더에 두고, 3단계의 전체 fold 결과가 있어야 한다.
그 밖의 옵션(--epochs, --bs, --lr 등)은 train_stage3.py와 같고, 직접 주면 아래 기본값보다 우선한다.
"""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
defaults = {
    "--hints": "submission_stage3_oof.csv",
    "--hint_col": "stage3",
    "--test_hints": "submission_stage3.csv",
    "--out": "submission_stage4.csv",
}
given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
extra = [x for k, v in defaults.items() if k not in given for x in (k, v)]

if "--hints" not in given:
    import pandas as pd

    f = defaults["--hints"]
    assert os.path.exists(f), f"{f} 가 없다. 먼저 train_stage3.py를 전체 fold로 끝까지 돌려야 한다"
    blank = pd.read_csv(f)[defaults["--hint_col"]].isna().sum()
    assert blank == 0, f"{f} 에 비어 있는 문장이 {blank}개다. 3단계를 --folds_run 없이 전체 fold로 끝내야 한다"
if "--test_hints" not in given:
    assert os.path.exists(defaults["--test_hints"]), f"{defaults['--test_hints']} 가 없다. 먼저 train_stage3.py를 끝까지 돌려야 한다"

sys.argv = [os.path.join(here, "train_stage3.py")] + sys.argv[1:] + extra
runpy.run_path(sys.argv[0], run_name="__main__")
