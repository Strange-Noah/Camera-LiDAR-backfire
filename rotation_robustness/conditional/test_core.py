import ast
from pathlib import Path
import unittest
import numpy as np
from core import match_sample, summarize, validate_population

def gt(x, i):
    return dict(sample_token='s', annotation_token=str(i), name='car', translation=[x,0,0])

def pred(x,score=.1):
    return dict(sample_token='s', detection_name='car', detection_score=score, translation=[x,0,0])

class Checks(unittest.TestCase):
    def test_boundaries(self):
        self.assertFalse(match_sample([gt(0,0)],[pred(2)])[0])
        self.assertTrue(match_sample([gt(0,0)],[pred(1.999)])[0])
        self.assertFalse(match_sample([gt(0,0)],[pred(0,.09999)])[0])
        self.assertFalse(match_sample([gt(0,0)],[])[0])

    def test_ties(self):
        self.assertEqual(match_sample([gt(0,0),gt(2,1)],[pred(1)]).tolist(),[True,False])
        self.assertEqual(match_sample([gt(0,0),gt(2,1)],[pred(1),pred(-1)]).tolist(),[True,False])
        self.assertEqual(match_sample([gt(0,0),gt(2,1)],[pred(-1),pred(1)]).tolist(),[True,True])

    def test_validation(self):
        with self.assertRaises(ValueError): match_sample([gt(0,0),gt(1,0)],[])
        with self.assertRaises(ValueError): match_sample([gt(0,0)],[pred(float('nan'))])
        with self.assertRaises(ValueError): validate_population(['s','s'],[['s']]*27)
        with self.assertRaises(ValueError): validate_population(['s'],[['x']]*27)
        validate_population(['s'],[['s']]*27)

    def test_rates_and_pairing(self):
        n=[10,20]; row=[2,1]+[3]*25
        a=summarize(n,[row,np.array(row)*2])
        self.assertEqual(a['B'],10)
        self.assertTrue(np.allclose(np.array(a['D'])-a['B'],a['G']))
        self.assertTrue(np.allclose(a['scene_only_G_CI'],10))
        self.assertEqual(a['G_sd'],[0]*5)
        self.assertEqual(a,summarize(n,[row,np.array(row)*2]))
        self.assertEqual(summarize([0],np.zeros((1,27)))['status'],'EMPTY')

    def test_actual_temporal_parity(self):
        # Extract the unmodified authoritative function without running I/O main.
        source=Path(__file__).with_name('temporal_match_reference.py')
        tree=ast.parse(source.read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='match_setting')
        class Frame:
            def __init__(self,*args): pass
        class PD: DataFrame=Frame
        scope=dict(np=np,pd=PD,SCORE_THRESHOLDS=[.1],MATCH_DIST_M=2.,DIST_EDGES=[0,10,float('inf')],DIST_LABELS=['near','far'])
        exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),'exec'),scope)
        rng=np.random.default_rng(4)
        for _ in range(100):
            g=[gt(float(x),i) for i,x in enumerate(rng.integers(0,8,8))]
            ps=[pred(float(x),float(s)) for x,s in zip(rng.integers(0,8,12),rng.choice([.09,.1,.5],12))]
            idx={('s','car'):(np.arange(8),np.array([x['translation'][:2] for x in g],float))}
            hit,_,_=scope['match_setting']({'s':ps},idx,['s'],{'s':(0,0)},8)
            np.testing.assert_array_equal(match_sample(g,ps),hit[.1])

if __name__=='__main__': unittest.main()
