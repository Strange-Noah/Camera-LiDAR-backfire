"""Basic MMCV helpers with a fallback for the nuScenes preparation scripts.

The dataset converter only needs file IO and image IO. Keeping the fallback
here lets those scripts run even when importing MMCV triggers an unrelated
MMEngine/MMDet registry error in the user's environment.
"""

try:
    import mmcv as mmcv
except Exception:
    import json
    import os
    import pickle

    import cv2

    class _MMCVFallback:
        @staticmethod
        def dump(obj, file, file_format=None, **kwargs):
            os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
            if file_format == "json" or str(file).lower().endswith(".json"):
                with open(file, "w", encoding="utf-8") as stream:
                    json.dump(obj, stream, **kwargs)
            else:
                with open(file, "wb") as stream:
                    pickle.dump(obj, stream)

        @staticmethod
        def load(file, file_format=None, **kwargs):
            if file_format == "json" or str(file).lower().endswith(".json"):
                with open(file, "r", encoding="utf-8") as stream:
                    return json.load(stream, **kwargs)
            with open(file, "rb") as stream:
                return pickle.load(stream)

        @staticmethod
        def imread(file, flag="color", channel_order="bgr", backend=None):
            mode = cv2.IMREAD_COLOR if flag == "color" else cv2.IMREAD_UNCHANGED
            image = cv2.imread(file, mode)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {file}")
            if channel_order.lower() == "rgb" and image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        @staticmethod
        def imwrite(image, file, **kwargs):
            os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
            return cv2.imwrite(file, image)

        @staticmethod
        def mkdir_or_exist(path):
            os.makedirs(path, exist_ok=True)

        @staticmethod
        def track_iter_progress(iterable, *args, **kwargs):
            return iterable

    mmcv = _MMCVFallback()
    track_iter_progress = mmcv.track_iter_progress
else:
    track_iter_progress = getattr(mmcv, "track_iter_progress", lambda iterable, *a, **k: iterable)
