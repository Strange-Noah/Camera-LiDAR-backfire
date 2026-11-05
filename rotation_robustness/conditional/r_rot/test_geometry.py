import unittest
import numpy as np
from geometry import rms_per_radian,choose_camera
class Checks(unittest.TestCase):
 def test_jacobian(self):
  k=np.array([[900,0,800],[0,910,450],[0,0,1.]])
  for p in [np.array([0.,0,10]),np.array([3.,-2,15])]:
   cols=[];e=1e-6
   for axis in np.eye(3):
    # derivative of Q.T P at zero is P cross omega.
    plus=p+e*np.cross(p,axis);minus=p-e*np.cross(p,axis)
    uv=lambda a:(k@a)[:2]/(k@a)[2]
    cols.append((uv(plus)-uv(minus))/(2*e))
   self.assertAlmostEqual(rms_per_radian(p,k),np.linalg.norm(np.stack(cols,axis=1)),places=5)
 def test_scale(self):
  k=np.diag([900.,900,1]);self.assertAlmostEqual(rms_per_radian([0,0,10],k),900*np.sqrt(2));self.assertAlmostEqual(rms_per_radian([1,2,10],k),rms_per_radian([2,4,20],k))
 def test_visibility(self):
  c=dict(channel='CAM_FRONT',center=[0,0,10],corners=[[-1,1,-1,1],[-1,-1,1,1],[10,10,10,10]],intrinsic=[[900,0,800],[0,900,450],[0,0,1]],image_width=1600,image_height=900)
  self.assertTrue(choose_camera([c])['valid']);bad=dict(c,center=[0,0,-10]);self.assertFalse(choose_camera([bad])['valid'])
if __name__=='__main__':unittest.main()
