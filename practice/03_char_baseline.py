"""Step 3: 딥러닝 베이스라인 (글자 단위 모델 하나)

입력과 정답의 길이가 같으므로, 글자마다 원래 글자를 맞히는 분류 문제로 푼다.
--jamo 를 주면 난독화 글자의 초성/중성/종성을 따로 넣어 준다 (처음 보는 글자도 자모로 짐작할 수 있게).

  python 03_char_baseline.py --limit 300 --epochs 1      # 동작 확인 (몇 분)
  python 03_char_baseline.py                             # 베이스라인
  python 03_char_baseline.py --jamo --out sub_jamo.csv   # 개선 1: 자모 입력 추가
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="monologg/kocharelectra-base-discriminator")
p.add_argument("--jamo", action="store_true", help="자모 입력 추가")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--chunk", type=int, default=256, help="한 번에 넣는 최대 글자 수")
p.add_argument("--limit", type=int, default=0, help="동작 확인용: 학습 문장 수 제한")
p.add_argument("--out", default="submission_baseline.csv")
args = p.parse_args()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)


def jamo(c):
    o = ord(c) - 0xAC00
    if 0 <= o < 11172:
        return o // 588, (o % 588) // 28, o % 28
    return 19, 21, 28  # 한글이 아니면 패딩용 번호


# 1) 데이터: 정답 끝의 공백을 지우면 입력과 길이가 같아진다
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["output"] = train["output"].str.rstrip()
if args.limit:
    train = train.iloc[: args.limit]

# 2) 검증: 10%를 떼어 내서 점수를 잰다
idx = np.random.permutation(len(train))
n_val = max(1, len(train) // 10)
val, trn = train.iloc[idx[:n_val]], train.iloc[idx[n_val:]]
print("학습", len(trn), "검증", len(val))

# 3) 어휘: 입력은 모델 어휘, 출력은 정답에 나온 모든 글자
tok = AutoTokenizer.from_pretrained(args.model)
in_vocab = tok.get_vocab()
UNK, CLS, SEP, PAD = tok.unk_token_id, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
out_chars = sorted({c for s in train.output for c in s})
out_id = {c: i for i, c in enumerate(out_chars)}


def pieces(s):
    """긴 문장을 chunk 글자씩 자른다."""
    return [s[i : i + args.chunk] for i in range(0, max(len(s), 1), args.chunk)]


def collate(xs, ys=None):
    n = max(len(x) for x in xs) + 2
    ids, j, tgt = [], [], []
    for k, x in enumerate(xs):
        pad = n - len(x) - 2
        ids.append([CLS] + [in_vocab.get(c, UNK) for c in x] + [SEP] + [PAD] * pad)
        j.append([(19, 21, 28)] + [jamo(c) for c in x] + [(19, 21, 28)] * (pad + 1))
        if ys is not None:
            tgt.append([-100] + [out_id.get(c, -100) for c in ys[k]] + [-100] * (pad + 1))
    ids = torch.tensor(ids, device=device)
    j = torch.tensor(j, device=device)
    tgt = torch.tensor(tgt, device=device) if ys is not None else None
    return ids, j, tgt


class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = AutoModel.from_pretrained(args.model)
        e = self.body.get_input_embeddings().embedding_dim
        self.cho, self.jung, self.jong = nn.Embedding(20, e), nn.Embedding(22, e), nn.Embedding(29, e)
        for m in (self.cho, self.jung, self.jong):
            nn.init.zeros_(m.weight)
        self.head = nn.Linear(self.body.config.hidden_size, len(out_chars))

    def forward(self, ids, j):
        x = self.body.get_input_embeddings()(ids)
        if args.jamo:
            x = x + self.cho(j[..., 0]) + self.jung(j[..., 1]) + self.jong(j[..., 2])
        h = self.body(inputs_embeds=x, attention_mask=(ids != PAD).long()).last_hidden_state
        return self.head(h)


@torch.no_grad()
def predict(model, xs):
    model.eval()
    out = []
    for s in tqdm(xs, desc="predict", ncols=100, leave=False):
        parts = pieces(s)
        ids, j, _ = collate(parts)
        pred = model(ids, j).argmax(-1).tolist()
        out.append("".join(out_chars[i] for part, row in zip(parts, pred) for i in row[1 : len(part) + 1]))
    return out


def accuracy(P, G):
    n = sum(len(g) for g in G)
    ok = sum(a == b for p_, g in zip(P, G) for a, b in zip(p_, g))
    return f"char acc {ok / n:.4f} sentence acc {sum(p_ == g for p_, g in zip(P, G)) / len(G):.4f}"


# 4) 학습
samples = [(a, b) for x, y in zip(trn.input, trn.output) for a, b in zip(pieces(x), pieces(y))]
model = Tagger().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
for ep in range(args.epochs):
    model.train()
    random.shuffle(samples)
    tot, n = 0.0, 0
    bar = tqdm(range(0, len(samples), args.bs), desc=f"epoch {ep + 1}/{args.epochs}", ncols=100)
    for i in bar:
        xs, ys = zip(*samples[i : i + args.bs])
        ids, j, tgt = collate(list(xs), list(ys))
        loss = F.cross_entropy(model(ids, j).flatten(0, 1), tgt.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot, n = tot + loss.item(), n + 1
        bar.set_postfix(loss=f"{tot / n:.4f}")
    # 5) 에폭마다 검증 점수
    print(f"epoch {ep + 1} 검증", accuracy(predict(model, list(val.input)), list(val.output)), flush=True)

# 6) 제출 파일
pred = predict(model, list(test.input))
pd.DataFrame({"ID": test.ID, "output": pred}).to_csv(args.out, index=False, encoding="utf-8-sig")
print("saved", args.out)
