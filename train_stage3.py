"""난독화된 한글 리뷰 복원: 3단계 (단어를 아는 모델을 덧붙인 교정기)

2단계까지의 남은 오류는 대부분 "아저시들", "케이블가", "채장"처럼 실제로 없는 단어를 만들어 내는 경우였다.
글자 단위 모델(KoCharELECTRA)은 단어 지식이 약하기 때문이다. 그래서 2단계 복원 문장을
단어 조각 단위의 큰 모델(klue/roberta-large)에도 읽히고, 그 출력을 글자 위치로 옮겨 글자 모델 출력과 합쳐서 다시 맞힌다.

train_cascade_v2.py를 다시 돌릴 필요 없이 그 결과 파일만 쓴다.
  oof_v2.csv                         학습 문장의 2단계 복원 결과 (stage2 열)
  submission_cascade_v2_stage2.csv   테스트 문장의 2단계 복원 결과

  python train_stage3.py --folds_run 1     # fold 하나만 돌려 2단계보다 나아지는지 먼저 확인
  python train_stage3.py                   # 전체 (끝난 fold는 캐시에서 이어받는다)
"""
import argparse
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="monologg/kocharelectra-base-discriminator", help="글자 단위 모델")
p.add_argument("--word_model", default="klue/roberta-large", help="단어 조각 단위 모델")
p.add_argument("--hints", default="oof_v2.csv")
p.add_argument("--hint_col", default="stage2")
p.add_argument("--test_hints", default="submission_cascade_v2_stage2.csv")
p.add_argument("--extra", default="", help="재난독화 사본 csv (idx, x, hint). 주면 학습에만 더한다")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--folds_run", type=int, default=0, help="이번에 학습할 fold 수 (0이면 전부)")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--chunk", type=int, default=384)
p.add_argument("--stride", type=int, default=192)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--lr_word", type=float, default=2e-5, help="단어 모델 학습률 (큰 모델이라 낮게)")
p.add_argument("--no_mask", action="store_true", help="제약 디코딩 끄기")
p.add_argument("--no_amp", action="store_true")
p.add_argument("--limit", type=int, default=0, help="동작 확인용: 학습/테스트 문장 수 제한")
p.add_argument("--out", default="submission_stage3.csv")
args = p.parse_args()
args.folds_run = args.folds_run or args.folds

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda" and not args.no_amp
print("device", device, "amp", use_amp)

HANGUL_BASE, N_SYL = 0xAC00, 11172
MAX_SUB = 16  # 단어 조각 안에서 몇 번째 글자인지 (이 값은 "조각에 속하지 않음")


def is_hangul(c):
    return 0 <= ord(c) - HANGUL_BASE < N_SYL


def jamo(c):
    """(초성, 중성, 종성) 인덱스. 한글이 아니면 각 범위의 마지막 인덱스(패딩용)."""
    o = ord(c) - HANGUL_BASE
    if 0 <= o < N_SYL:
        return o // 588, (o % 588) // 28, o % 28
    return 19, 21, 28


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["output"] = train["output"].str.rstrip()
keep = train.input.str.len() == train.output.str.len()
assert keep.all(), "입력과 정답 길이가 다른 행이 있으면 oof 파일과 줄이 어긋난다"
train["hint"] = pd.read_csv(args.hints)[args.hint_col].values
test["hint"] = pd.read_csv(args.test_hints).output.values
if args.limit:
    train, test = train.iloc[: args.limit], test.iloc[: args.limit]
    print(f"[주의] --limit {args.limit}: 테스트도 앞부분만 쓰므로 제출용 파일이 아니다")
X, Y, HX = list(train.input), list(train.output), list(train.hint)
XT, HT = list(test.input), list(test.hint)
assert all(len(a) == len(b) for a, b in zip(X, HX)) and all(len(a) == len(b) for a, b in zip(XT, HT))
# 재난독화 사본: 같은 정답 문장을 새로 난독화한 것과, 그것을 이전 단계 모델이 (본 적 없는 상태로) 복원한 힌트
extra = pd.read_csv(args.extra) if args.extra else pd.DataFrame({"idx": [], "x": [], "hint": []})
extra = extra[extra.idx < len(X)].reset_index(drop=True)
assert all(len(a) == len(b) == len(Y[i]) for i, a, b in zip(extra.idx, extra.x, extra.hint))
print("재난독화 사본", len(extra))


def accuracy(preds, golds):
    n = sum(len(g) for g in golds)
    ok = sum(a == b for p_, g in zip(preds, golds) for a, b in zip(p_, g))
    sent = sum(p_ == g for p_, g in zip(preds, golds)) / len(golds)
    return f"char acc {ok / n:.4f} sentence acc {sent:.4f}"


def pick(lst, idx):
    return [lst[i] for i in idx]


print("[입력 힌트] 학습 문장", accuracy(HX, Y))

tok = AutoTokenizer.from_pretrained(args.model)
wtok = AutoTokenizer.from_pretrained(args.word_model)
in_vocab = tok.get_vocab()
UNK, CLS, SEP, PAD = tok.unk_token_id, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
out_chars = sorted({c for s in Y for c in s if is_hangul(c)})
out_vocab = {c: i for i, c in enumerate(out_chars)}
out_jamo = torch.tensor([jamo(c) for c in out_chars], device=device)

# 제약 디코딩용: feasible[k][난독화 자모, 출력 음절] = 그 음절의 k번째 자모가 이 난독화 자모에서 나온 적이 있는가
seen = [torch.zeros(n, n, dtype=torch.bool) for n in (20, 22, 29)]
for x, y in zip(X, Y):
    for a, b in zip(x, y):
        if is_hangul(a) and is_hangul(b):
            for k, (i, j) in enumerate(zip(jamo(a), jamo(b))):
                seen[k][i, j] = True
feasible = [seen[k][:, out_jamo[:, k].cpu()] for k in range(3)]


def spans(s, n, first=None):
    """긴 문장을 공백 기준으로 n글자 이하 조각으로 나누는 (시작, 끝) 목록. first는 첫 조각의 길이 제한."""
    out, start, lim = [], 0, first or n
    while len(s) - start > lim:
        cut = s.rfind(" ", start + lim // 2, start + lim)
        cut = start + lim if cut == -1 else cut + 1
        out.append((start, cut))
        start, lim = cut, n
    out.append((start, len(s)))
    return out


def windows(length, n, stride):
    """추론용: stride만큼 옮겨가며 겹치는 (시작, 끝) 창. 마지막 창은 문장 끝에 맞춘다."""
    if length <= n:
        return [(0, length)]
    out, a = [], 0
    while a + n < length:
        out.append((a, a + n))
        a += stride
    out.append((length - n, length))
    return out


def collate(items):
    """items: (hint, x, tgt). hint는 2단계가 복원한 문장, x는 실제 난독화 문장."""
    n = max(len(x) for _, x, _ in items) + 2
    ids, cho, jung, jong, tgt = [], [], [], [], []
    for hint, x, t in items:
        pad = n - len(x) - 2
        j = [(19, 21, 28)] + [jamo(c) for c in x] + [(19, 21, 28)] * (pad + 1)
        ids.append([CLS] + [in_vocab.get(c, UNK) for c in hint] + [SEP] + [PAD] * pad)
        cho.append([a for a, _, _ in j])
        jung.append([b for _, b, _ in j])
        jong.append([c for _, _, c in j])
        tgt.append([-100] + t + [-100] * (pad + 1))
    # 단어 모델 입력과, 글자 위치 -> (단어 조각 번호, 조각 안에서 몇 번째 글자) 대응표
    enc = wtok([h for h, _, _ in items], padding=True, truncation=True, max_length=512, return_offsets_mapping=True)
    piece = torch.zeros(len(items), n, dtype=torch.long)  # 어느 조각에도 속하지 않는 글자(공백 등)는 0번([CLS])을 본다
    sub = torch.full((len(items), n), MAX_SUB, dtype=torch.long)
    for b, offsets in enumerate(enc["offset_mapping"]):
        for t, (s, e) in enumerate(offsets):
            if e > s:
                piece[b, s + 1 : e + 1] = t
                sub[b, s + 1 : e + 1] = torch.arange(e - s).clamp(max=MAX_SUB - 1)
    feats = [torch.tensor(v, device=device) for v in (ids, cho, jung, jong)]
    feats += [torch.tensor(enc["input_ids"], device=device), torch.tensor(enc["attention_mask"], device=device)]
    feats += [piece.to(device), sub.to(device)]
    return feats, torch.tensor(tgt, device=device)


class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = AutoModel.from_pretrained(args.model)
        self.word = AutoModel.from_pretrained(args.word_model)
        e = self.body.get_input_embeddings().embedding_dim
        hc, hw = self.body.config.hidden_size, self.word.config.hidden_size
        self.cho = nn.Embedding(20, e)
        self.jung = nn.Embedding(22, e)
        self.jong = nn.Embedding(29, e)
        self.sub = nn.Embedding(MAX_SUB + 1, hw)
        # 처음에는 사전학습된 표현을 흐리지 않도록 0에서 시작한다
        for m in (self.cho, self.jung, self.jong, self.sub):
            nn.init.zeros_(m.weight)
        h = 1024
        self.mix = nn.Linear(hc + hw, h)
        # 같은 단어 조각에 속한 글자들은 단어 모델 출력이 똑같으므로, 글자끼리 한 번 더 섞어 준다
        self.fuse = nn.TransformerEncoderLayer(h, 8, 2 * h, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(0.1)
        self.out_char = nn.Linear(h, len(out_vocab))
        self.out_cho = nn.Linear(h, 19)
        self.out_jung = nn.Linear(h, 21)
        self.out_jong = nn.Linear(h, 28)

    def forward(self, feats):
        c, a, b, d, wid, wmask, piece, sub = feats
        x = self.body.get_input_embeddings()(c) + self.cho(a) + self.jung(b) + self.jong(d)
        hc = self.body(inputs_embeds=x, attention_mask=(c != PAD).long()).last_hidden_state
        hw = self.word(input_ids=wid, attention_mask=wmask).last_hidden_state
        hw = hw.gather(1, piece[..., None].expand(-1, -1, hw.size(-1))) + self.sub(sub)
        h = self.mix(torch.cat([hc, hw], -1))
        h = self.drop(self.norm(self.fuse(h, src_key_padding_mask=c == PAD)))
        return self.out_char(h), self.out_cho(h), self.out_jung(h), self.out_jong(h)

    def scores(self, feats):
        """음절 분류 점수에 자모별 점수를 더해 최종 음절 점수를 만든다."""
        lc, la, lb, ld = (t.float() for t in self(feats))
        s = F.log_softmax(lc, -1)
        s = s + F.log_softmax(la, -1)[..., out_jamo[:, 0]]
        s = s + F.log_softmax(lb, -1)[..., out_jamo[:, 1]]
        s = s + F.log_softmax(ld, -1)[..., out_jamo[:, 2]]
        return s


def make_batches(hints, xs, ys):
    samples = []
    for hint, x, y in zip(hints, xs, ys):
        # 에폭마다 첫 조각 길이를 바꿔서 자르는 위치가 매번 달라지게 한다
        first = random.randint(args.chunk // 2, args.chunk) if len(x) > args.chunk else None
        for a, b in spans(x, args.chunk, first):
            tgt = [out_vocab[g] if is_hangul(c) and is_hangul(g) else -100 for c, g in zip(x[a:b], y[a:b])]
            samples.append((hint[a:b], x[a:b], tgt))
    # 길이가 비슷한 조각끼리 배치로 묶는다
    random.shuffle(samples)
    size = args.bs * 50
    buckets = [sorted(samples[i : i + size], key=lambda s: len(s[1])) for i in range(0, len(samples), size)]
    batches = [b[i : i + args.bs] for b in buckets for i in range(0, len(b), args.bs)]
    random.shuffle(batches)
    return batches


def fit(hints, xs, ys, desc):
    model = Tagger().to(device)
    batches = make_batches(hints, xs, ys)
    total = args.epochs * len(batches)
    warm = max(1, total // 10)

    def lr_at(step):  # 선형 워밍업 뒤 코사인 감쇠
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    word_params = list(model.word.parameters())
    word_ids = {id(q) for q in word_params}
    groups = [
        {"params": [q for q in model.parameters() if id(q) not in word_ids], "lr": args.lr},
        {"params": word_params, "lr": args.lr_word},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    for ep in range(args.epochs):
        if ep:
            batches = make_batches(hints, xs, ys)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        bar = tqdm(batches, desc=f"{desc} epoch {ep + 1}/{args.epochs}", ncols=100)
        for batch in bar:
            feats, tgt = collate(batch)
            mask = tgt != -100
            if not mask.any():
                continue
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                lc, la, lb, ld = model(feats)
            y = tgt[mask]
            yj = out_jamo[y]
            loss = (
                F.cross_entropy(lc[mask].float(), y)
                + F.cross_entropy(la[mask].float(), yj[:, 0])
                + F.cross_entropy(lb[mask].float(), yj[:, 1])
                + F.cross_entropy(ld[mask].float(), yj[:, 2])
            )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            n += 1
            bar.set_postfix(loss=f"{tot / n:.4f}")
        print(f"{desc} epoch {ep + 1}/{args.epochs} loss {tot / max(n, 1):.4f} time {time.time() - t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def score(model, hints, xs, into, scale=1.0):
    """문장별 (길이, 음절 수) 점수를 into에 더한다. 겹치는 창은 창 가운데일수록 크게 반영해 평균한다."""
    pieces, weights = [], []
    for k, x in enumerate(xs):
        ws = windows(len(x), args.chunk, args.stride)
        wsum = torch.zeros(len(x))
        doc_w = []
        for a, b in ws:
            pos = torch.arange(b - a, dtype=torch.float)
            far = torch.full_like(pos, float(args.chunk))
            # 문장의 진짜 시작/끝은 문맥이 잘린 것이 아니므로 깎지 않는다
            w = 1 + torch.minimum(pos if a > 0 else far, (b - a - 1 - pos) if b < len(x) else far)
            wsum[a:b] += w
            doc_w.append(w)
        for (a, b), w in zip(ws, doc_w):
            pieces.append((k, a, b))
            weights.append(w / wsum[a:b] * scale)
    order = sorted(range(len(pieces)), key=lambda i: pieces[i][2] - pieces[i][1])
    model.eval()
    for i in range(0, len(order), 32):
        idx = order[i : i + 32]
        feats, _ = collate([(hints[k][a:b], xs[k][a:b], [-100] * (b - a)) for k, a, b in (pieces[j] for j in idx)])
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            s = model.scores(feats).cpu()
        for row, j in enumerate(idx):
            k, a, b = pieces[j]
            into[k][a:b] += weights[j][:, None] * s[row, 1 : b - a + 1]


def blank(xs):
    return [torch.zeros(len(x), len(out_chars)) for x in xs]


def decode(scores, xs):
    out = []
    for s, x in zip(scores, xs):
        if not args.no_mask and len(x):
            j = torch.tensor([jamo(c) for c in x])
            ok = feasible[0][j[:, 0]] & feasible[1][j[:, 1]] & feasible[2][j[:, 2]]
            ok |= ~ok.any(-1, keepdim=True)  # 가능한 후보가 하나도 없으면 제약을 풀어 준다
            s = s.float().masked_fill(~ok, -1e9)
        out.append("".join(out_chars[i] if is_hangul(c) else c for i, c in zip(s.argmax(-1).tolist(), x)))
    return out


def predict(model, hints, xs):
    """복원 문장 목록. 메모리를 아끼려고 200문장씩 처리한다."""
    preds = []
    for i in range(0, len(xs), 200):
        s = blank(xs[i : i + 200])
        score(model, hints[i : i + 200], xs[i : i + 200], s)
        preds += decode(s, xs[i : i + 200])
    return preds


# 힌트가 이미 "모델이 본 적 없는 상태의 복원 결과"이므로 fold는 이전 단계와 달라도 된다
fold_of = np.random.permutation(len(X)) % args.folds
cache = args.out.replace(".csv", "_cache.pt")
if os.path.exists(cache):
    print("이어하기:", cache)
    st = torch.load(cache)
else:
    st = {"done": [], "oof": [None] * len(X), "test": blank(XT)}
st.setdefault("extra_hint", {})  # 사본 줄 번호 -> 이 단계의 복원 결과 (다음 단계 학습용)
test_s = st.pop("test")
for f in range(args.folds_run):
    if f in st["done"]:
        continue
    tr_i, ho_i = np.where(fold_of != f)[0], np.where(fold_of == f)[0]
    ex = extra[extra.idx.isin(set(tr_i.tolist()))]
    model = fit(
        pick(HX, tr_i) + list(ex.hint), pick(X, tr_i) + list(ex.x), pick(Y, tr_i) + pick(Y, ex.idx.astype(int)),
        f"[3단계 fold {f + 1}/{args.folds}]",
    )
    preds = predict(model, pick(HX, ho_i), pick(X, ho_i))
    for j, i in enumerate(ho_i):
        st["oof"][i] = preds[j]
    # 보류 fold의 재난독화 사본도 복원해 둔다. 이 모델이 본 적 없는 문장이라 다음 단계의 학습 힌트로 쓸 수 있다
    ho_ex = extra[extra.idx.isin(set(ho_i.tolist()))]
    if len(ho_ex):
        st["extra_hint"].update(zip(ho_ex.index.tolist(), predict(model, list(ho_ex.hint), list(ho_ex.x))))
    score(model, HT, XT, test_s)
    print(f"[3단계 fold {f + 1}] 보류 데이터 {accuracy(preds, pick(Y, ho_i))} (입력 힌트 {accuracy(pick(HX, ho_i), pick(Y, ho_i))})", flush=True)
    del model
    torch.cuda.empty_cache()
    st["done"].append(f)
    torch.save({**st, "test": test_s}, cache)

done = [i for i in range(len(X)) if st["oof"][i] is not None]
print("[3단계] OOF", accuracy(pick(st["oof"], done), pick(Y, done)), "| 같은 문장의 입력 힌트", accuracy(pick(HX, done), pick(Y, done)))
oof_out = args.out.replace(".csv", "_oof.csv")
pd.DataFrame({"stage3": [st["oof"][i] or "" for i in range(len(X))]}).to_csv(oof_out, index=False, encoding="utf-8-sig")
if len(extra) and len(st["extra_hint"]) == len(extra):
    extra_out = args.out.replace(".csv", "_extra.csv")
    pd.DataFrame({"idx": extra.idx.astype(int), "x": extra.x, "hint": [st["extra_hint"][i] for i in range(len(extra))]}).to_csv(
        extra_out, index=False, encoding="utf-8-sig"
    )
    print("saved", extra_out, "(다음 단계의 --extra)")
elif len(extra):
    print("[참고] 모든 fold가 끝나야 다음 단계용 사본 힌트 파일이 저장된다")
out = decode(test_s, XT)
assert all(len(a) == len(b) for a, b in zip(out, XT))
pd.DataFrame({"ID": test.ID, "output": out}).to_csv(args.out, index=False, encoding="utf-8-sig")
print("saved", args.out, f"(fold {len(st['done'])}개 앙상블)", "|", oof_out)
