"""First-order distribution RMS for camera-local isotropic Gaussian rotation.

Camera-to-LiDAR R is right-multiplied by Q. Thus clean camera P transforms
to Q.T P. The sign of its first derivative does not affect RMS.
This is a small-angle approximation, not an exact nonlinear pinhole moment.
"""
import numpy as np

CHANNELS=('CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')

def rms_per_radian(point,intrinsic):
    x=np.asarray(point,float);k=np.asarray(intrinsic,float)
    if x.shape!=(3,) or k.shape!=(3,3) or not np.isfinite(x).all() or not np.isfinite(k).all() or x[2]<=0:
        raise ValueError('invalid clean projection')
    h=k@x
    if h[2]<=0:raise ValueError('nonpositive projective depth')
    projection=(k[:2]*h[2]-h[:2,None]*k[2])/h[2]**2
    skew=np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    jacobian=projection@skew
    return float(np.linalg.norm(jacobian,'fro'))

def choose_camera(candidates):
    """Center in raw image, all corners depth>1m, width>1px; largest width.

    Candidates must arrive in CHANNELS order; ties retain that order.
    Visibility is geometric only, with no detection/occlusion conditioning.
    """
    valid=[]
    for c in candidates:
        center=np.asarray(c['center'],float);corners=np.asarray(c['corners'],float);k=np.asarray(c['intrinsic'],float)
        if not np.isfinite(center).all() or not np.isfinite(corners).all() or center[2]<=1 or (corners[2]<=1).any():continue
        uv=k@center;uv=uv[:2]/uv[2]
        if not (0<=uv[0]<c['image_width'] and 0<=uv[1]<c['image_height']):continue
        pixels=k@corners;pixels=pixels[:2]/pixels[2]
        width=float(np.ptp(pixels[0]))
        if not np.isfinite(width) or width<=1:continue
        valid.append((width,c,rms_per_radian(center,k)/width))
    if not valid:return dict(valid=False,reason='no_valid_clean_camera')
    width,c,coef=max(valid,key=lambda v:v[0])
    return dict(valid=True,camera=c['channel'],width_pixels=width,coefficient_per_degree=coef*np.pi/180,center_camera=c['center'])
