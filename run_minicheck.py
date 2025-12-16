from minicheck.minicheck import MiniCheck
import os

scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False, cache_dir='./ckpts')

import pandas as pd
import sys

fname = sys.argv[1]
df = pd.read_csv(fname)

#df['doc'] = df['Document']
#df['summary'] = df['Gen Summary']
docs = list(df['doc'])
claim = list(df['summary'])
pred_label, raw_prob, _, _ = scorer.score(docs=docs, claims=claim)
df['bespoke_score'] = raw_prob

df.to_csv("outputs/" + fname+".out.csv")
