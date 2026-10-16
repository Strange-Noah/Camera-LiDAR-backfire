import unittest
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from metadata_audit.core import differences, relative_components, corners_world, project_boxes, rotation, yaw

class AuditMath(unittest.TestCase):
    def test_velocity_unequal_and_edges(self):
        t=np.array([0,400000,1100000]);p=np.c_[t/1e6*3,t/1e6*-2]
        v,span,neighbors,method=differences(t,p,1.5)
        np.testing.assert_allclose(v,[[3,-2]]*3)
        self.assertEqual(method.tolist(),['one_sided','central_secant','one_sided'])
        self.assertEqual(neighbors.tolist(),[1,2,1])
    def test_gap_and_singleton_no_imputation(self):
        v,*_=differences([0,2000000],[[0,0],[1,1]],1.5)
        self.assertTrue(np.isnan(v).all())
        v,*_=differences([0],[[0,0]],1.5);self.assertTrue(np.isnan(v).all())
        with self.assertRaises(ValueError):differences([0,0],[[0,0],[1,1]],1.5)
    def test_unwrap_yaw(self):
        y=np.unwrap(np.deg2rad([179,-179,-177]));w,*_=differences([0,500000,1000000],y,1.5)
        np.testing.assert_allclose(w,np.deg2rad(4))
    def test_radial_lateral_rotation_invariance(self):
        p=np.array([[10.,0.],[0.,10.]])
        v=np.array([[3.,4.],[4.,3.]])
        rel,r,l=relative_components(p,np.zeros_like(p),v,np.zeros_like(v))
        np.testing.assert_allclose(r,[3,3]);np.testing.assert_allclose(l,[4,-4])
        np.testing.assert_allclose(r*r+l*l,np.sum(rel*rel,axis=1))
        R=np.array([[0.,-1.],[1.,0.]])
        _,rr,ll=relative_components(p@R.T,np.zeros_like(p),v@R.T,np.zeros_like(v))
        np.testing.assert_allclose(rr,r);np.testing.assert_allclose(ll,l)
        _,r,l=relative_components(np.zeros((1,2)),np.zeros((1,2)),np.ones((1,2)),np.zeros((1,2)))
        self.assertTrue(np.isnan(r).all() and np.isnan(l).all())
    def test_corners_and_full_transform_devkit(self):
        q=Quaternion(axis=[0,0,1],angle=.4)
        centers=np.array([[3.,1.,15.]]);sizes=np.array([[2.,4.,3.]])
        world=corners_world(centers,sizes,[q.elements])
        b=Box(centers[0],sizes[0],q)
        np.testing.assert_allclose(world[0].T,b.corners(),atol=1e-12)
        ego={'translation':[1,2,1],'rotation':Quaternion(axis=[0,0,1],angle=.2).elements}
        cal={'translation':[.3,.1,.2],'rotation':Quaternion(axis=[0,1,0],angle=.1).elements,'camera_intrinsic':[[500,0,800],[0,500,450],[0,0,1]]}
        b.translate(-np.array(ego['translation']));b.rotate(Quaternion(ego['rotation']).inverse)
        b.translate(-np.array(cal['translation']));b.rotate(Quaternion(cal['rotation']).inverse)
        actual=((world-np.array(ego['translation']))@rotation(ego['rotation'])-np.array(cal['translation']))@rotation(cal['rotation'])
        np.testing.assert_allclose(actual[0].T,b.corners(),atol=1e-12)
        result=project_boxes(world,ego,cal,1600,900)
        uv=view_points(b.corners(),np.array(cal['camera_intrinsic']),normalize=True)
        self.assertTrue(result['projection_valid'][0])
        np.testing.assert_allclose(result['projected_width_px'][0],np.ptp(uv[0]),rtol=1e-12)
    def test_projection_rejects_behind_and_crossing(self):
        pose={'translation':[0,0,0],'rotation':[1,0,0,0]}
        cal={**pose,'camera_intrinsic':[[500,0,800],[0,500,450],[0,0,1]]}
        corners=corners_world([[0,0,-10],[0,0,.2]],[[2,2,2]]*2,[[1,0,0,0]]*2)
        p=project_boxes(corners,pose,cal,1600,900)
        self.assertFalse(p['projection_valid'].any());self.assertTrue(np.isnan(p['projected_width_px']).all())
        self.assertTrue(p['near_plane_crossing'][1])

if __name__=='__main__':unittest.main()
