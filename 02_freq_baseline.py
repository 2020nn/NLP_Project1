"""Step 2: 가장 단순한 베이스라인. GPU 필요 없음.

난독화 글자마다 학습 데이터에서 가장 자주 대응된 정답 글자로 바꾼다. 5-fold 교차검증으로 점수를 잰다.
  python 02_freq_baseline.py
"""
import pandas as pd, numpy as np
from collections import Counter, defaultdict
train = pd.read_csv("train.csv")
train["output"] = train["output"].str.rstrip()
print("rstrip 후 길이 같은 비율", (train.input.str.len() == train.output.str.len()).mean())
X, Y = list(train.input), list(train.output)

def acc(P, G):
    n = sum(len(g) for g in G); ok = sum(a == b for p, g in zip(P, G) for a, b in zip(p, g))
    return round(ok / n, 4), round(sum(p == g for p, g in zip(P, G)) / len(G), 4)

def fit(xs, ys):
    cnt = defaultdict(Counter)
    for x, y in zip(xs, ys):
        for a, b in zip(x, y): cnt[a][b] += 1
    return {a: c.most_common(1)[0][0] for a, c in cnt.items()}

def predict(m, xs):
    return ["".join(m.get(a, a) for a in x) for x in xs]

fold = np.random.RandomState(42).permutation(len(X)) % 5
scores = []
for f in range(5):
    tr, ho = np.where(fold != f)[0], np.where(fold == f)[0]
    m = fit([X[i] for i in tr], [Y[i] for i in tr])
    s = acc(predict(m, [X[i] for i in ho]), [Y[i] for i in ho]); scores.append(s); print("fold", f + 1, s)
print("평균 글자 정확도", round(np.mean([s[0] for s in scores]), 4), "문장", round(np.mean([s[1] for s in scores]), 4))
