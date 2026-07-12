import numpy as np


def pose_to_w2c(pose: np.ndarray) -> np.ndarray:
    """Convert a LayerPano/NeRF-style camera-to-world matrix to W2C."""
    w2c = np.linalg.inv(np.asarray(pose, dtype=np.float32))
    w2c[1:3, :3] *= -1
    w2c[:3, 3] *= -1
    return w2c.astype(np.float32)
