"""Regenerate all summaries from retained scene counts without prediction access."""
import argparse,json
from pathlib import Path
import numpy as np
from core import summarize

def aggregate(source):
    source=Path(source)
    def read(i,j=None):
        path=source/('distance%d%s.npz'%(i,'' if j is None else '_points%d'%j))
        with np.load(path) as z:return summarize(z['n'],z['fn'])
    return dict(distance=[read(i) for i in range(5)],distance_points=[[read(i,j) for j in range(3)] for i in range(5)])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as f:json.dump(aggregate(a.source),f,indent=2,allow_nan=False)
