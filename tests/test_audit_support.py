import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from metadata_audit.analyze import support,quantile_bin
CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/audit.json').read_text())

class Support(unittest.TestCase):
    def test_repeated_frames_not_independent(self):
        f=pd.DataFrame({'annotation_token':[str(i) for i in range(1000)],'instance_token':['one']*1000,'scene_token':['one']*1000})
        s=support(f,CFG)
        self.assertEqual((s['N_ann'],s['N_traj'],s['N_scene']),(1000,1,1))
        self.assertEqual(s['status'],'SPARSE');self.assertEqual(s['top1_scene_fraction'],1)
    def test_ties_not_forced_into_three_bins(self):
        bins,spec=quantile_bin(pd.Series([0.]*90+[1.]*10),CFG)
        self.assertTrue(spec['collapsed'])
        self.assertEqual(bins.iloc[:90].nunique(),1)
    def test_missing_is_not_zero(self):
        bins,spec=quantile_bin(pd.Series([np.nan]*10),CFG)
        self.assertTrue(bins.isna().all());self.assertEqual(spec['labels'],['unavailable'])
    def test_empty_coverage(self):
        s=support(pd.DataFrame(columns=['annotation_token','instance_token','scene_token']),CFG)
        self.assertEqual((s['N_ann'],s['N_traj'],s['N_scene']),(0,0,0));self.assertEqual(s['status'],'SPARSE')
