"""Step 1: 데이터 살펴보기 (EDA). GPU 필요 없음.

  python 01_eda.py
"""
import pandas as pd
train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
print(train.shape, test.shape)
print(train.head(3).to_string())
same_len = (train.input.str.len() == train.output.str.len()).mean()
print("입력=정답 길이 비율", same_len)
diff = sum(a != b for x, y in zip(train.input, train.output) for a, b in zip(x, y))
tot = train.output.str.len().sum()
print("바뀐 글자 비율", round(diff / tot, 4))
print("입력 그대로 낼 때 글자 정확도", round(1 - diff / tot, 4))
