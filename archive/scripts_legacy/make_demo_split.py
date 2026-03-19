#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


src = Path('spam_email_data.log')
out_dir = Path('data/raw')
out_dir.mkdir(parents=True, exist_ok=True)

lines = src.read_text(encoding='utf-8', errors='ignore').splitlines()
rows = []
for ln in lines:
    if '\t' not in ln:
        continue
    rid, js = ln.split('\t', 1)
    try:
        d = json.loads(js)
    except json.JSONDecodeError:
        continue
    rows.append((rid, d))

n = len(rows)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

# demo labels only for pipeline check
for i in idx:
    rows[i][1]['label'] = int(i % 5)

train_end = int(0.7 * n)
valid_end = int(0.85 * n)

splits = {
    'train.log': idx[:train_end],
    'valid.log': idx[train_end:valid_end],
    'test.log': idx[valid_end:],
}

for name, ids in splits.items():
    with (out_dir / name).open('w', encoding='utf-8') as f:
        for i in ids:
            rid, d = rows[int(i)]
            if name == 'test.log':
                d = {k: v for k, v in d.items() if k != 'label'}
            f.write(rid + '\t' + json.dumps(d, ensure_ascii=False) + '\n')

print('created', {k: len(v) for k, v in splits.items()})
