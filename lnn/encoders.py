import numpy as np


def encode_agent(obs_vec: np.ndarray):
    """Per-agent 18-D vector pass-through."""
    return obs_vec.astype(np.float32)

def encode_state(obs: dict) -> np.ndarray:
    """Pack environment observation into a feature vector.
    obs keys (single-UAV): 
      - ego: [x,y,vx,vy, E_norm, gdx,gdy]
      - rays: distances to obstacles in 8 directions (normalized 0..1)
      - move_obs: nearest moving obstacle rel (dx,dy) clipped/normalized
      - wind: (wx, wy) normalized
      - nofly_local: binary flag (0/1) whether current cell is no-fly
    Returns 1D float32 feature vector suitable for the LNN policy.
    """
    ego = np.asarray(obs["ego"], dtype=np.float32)
    rays = np.asarray(obs["rays"], dtype=np.float32)
    mo   = np.asarray(obs["move_obs"], dtype=np.float32)
    wind = np.asarray(obs["wind"], dtype=np.float32)
    nf   = np.asarray([obs["nofly_local"]], dtype=np.float32)
    feats = np.concatenate([ego, rays, mo, wind, nf], axis=0)
    return feats
