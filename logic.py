"""Pure calculation logic for the Bududa landslide dashboard.

Only numpy/pandas here, so it can be tested without Streamlit or GIS libraries.
"""
import numpy as np
import pandas as pd

# Factor bands exported from Earth Engine, in band order
FACTORS = ["s_slope", "s_aspect", "s_curv", "s_twi", "s_dist", "s_lc"]
FACTOR_LABELS = {
    "s_slope": "Slope",
    "s_aspect": "Aspect (NE-facing)",
    "s_curv": "Curvature (hollows)",
    "s_twi": "Wetness index (TWI)",
    "s_dist": "Distance to streams",
    "s_lc": "Land cover",
}
DEFAULT_WEIGHTS = {"s_slope": 0.30, "s_aspect": 0.10, "s_curv": 0.15,
                   "s_twi": 0.15, "s_dist": 0.15, "s_lc": 0.15}

CLASS_NAMES = {1: "Very low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very high"}
CLASS_COLORS = {1: "#E5E7EB", 2: "#FDE68A", 3: "#F59E0B", 4: "#D97706", 5: "#7C2D12"}
CLASS_PERCENTILES = (20, 40, 70, 90)   # same breaks as the Earth Engine script

# Rainfall alert rule (from the 2010–2024 CHIRPS analysis)
WATCH = {"r30": 110.0, "r3": 15.0}
WARNING = {"r30": 180.0, "r3": 30.0}


def normalise_weights(w: dict) -> dict:
    total = sum(max(v, 0.0) for v in w.values())
    if total <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: max(v, 0.0) / total for k, v in w.items()}


def box_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter over a (2r+1)x(2r+1) window, ignoring NaNs (like GEE focalMean)."""
    if radius <= 0:
        return arr.copy()
    valid = np.isfinite(arr)
    vals = np.where(valid, arr, 0.0)

    def window_sum(a):
        p = np.pad(a, radius + 1, mode="constant")
        c = p.cumsum(0).cumsum(1)
        k = 2 * radius + 1
        s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
        return s[: a.shape[0], : a.shape[1]]

    s = window_sum(vals)
    n = window_sum(valid.astype(float))
    out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    out[~valid] = np.nan
    return out


def weighted_score(factors: np.ndarray, weights: dict, smooth_radius: int = 2) -> np.ndarray:
    """factors: array (6, rows, cols) of 1–5 scores, 0 = outside Bududa."""
    w = normalise_weights(weights)
    valid = np.all(factors >= 1, axis=0)
    score = np.zeros(factors.shape[1:], dtype="float64")
    for i, name in enumerate(FACTORS):
        score += factors[i].astype("float64") * w[name]
    score[~valid] = np.nan
    return box_mean(score, smooth_radius)


def classify(score: np.ndarray, pcts=CLASS_PERCENTILES):
    """Relative classes 1–5 from percentiles of the score inside Bududa. 0 = no data."""
    valid = np.isfinite(score)
    if not valid.any():
        return np.zeros(score.shape, dtype="uint8"), []
    breaks = np.nanpercentile(score[valid], pcts)
    cls = np.zeros(score.shape, dtype="uint8")
    cls[valid] = 1
    for i, b in enumerate(breaks, start=2):
        cls[valid & (score > b)] = i
    return cls, list(breaks)


def alert_level(r30: float, r3: float) -> str:
    if r30 >= WARNING["r30"] and r3 >= WARNING["r3"]:
        return "WARNING"
    if r30 >= WATCH["r30"] and r3 >= WATCH["r3"]:
        return "WATCH"
    return "NORMAL"


# Which susceptibility classes are flagged for action
CUTOFF_OPTIONS = {
    "Auto (from alert level)": None,
    "Moderate and above": 3,
    "High and above": 4,
    "Very high only": 5,
}
AUTO_CUTOFF = {"NORMAL": None, "WATCH": 5, "WARNING": 4}


def flagged_min_class(cutoff_label: str, alert: str):
    v = CUTOFF_OPTIONS[cutoff_label]
    return AUTO_CUTOFF[alert] if v is None else v


def exposure_by_zone(cls: np.ndarray, pop: np.ndarray, zones: np.ndarray,
                     names: dict, min_class, pixel_km2: float) -> pd.DataFrame:
    """People and area in flagged classes per sub-county zone id."""
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)
    flagged = (cls >= min_class) if min_class else np.zeros(cls.shape, bool)
    inside = cls > 0
    rows = []
    for zid, name in names.items():
        z = (zones == zid) & inside
        if not z.any():
            continue
        p_tot = pop[z].sum()
        p_flag = pop[z & flagged].sum()
        a_tot = z.sum() * pixel_km2
        a_flag = (z & flagged).sum() * pixel_km2
        rows.append({
            "subcounty": name,
            "people_total": round(float(p_tot)),
            "people_flagged": round(float(p_flag)),
            "pct_people_flagged": round(100 * p_flag / p_tot, 1) if p_tot else 0.0,
            "area_km2": round(float(a_tot), 2),
            "pct_area_flagged": round(100 * a_flag / a_tot, 1) if a_tot else 0.0,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("people_flagged", ascending=False).reset_index(drop=True)
    return df


def people_in_buffer(pop: np.ndarray, centre_rc: tuple, radius_m: float, pixel_m: float) -> float:
    """Sum of population within radius of a (row, col) centre on the raster grid."""
    r0, c0 = centre_rc
    rr, cc = np.ogrid[: pop.shape[0], : pop.shape[1]]
    d2 = ((rr - r0) * pixel_m) ** 2 + ((cc - c0) * pixel_m) ** 2
    p = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)
    return float(p[d2 <= radius_m ** 2].sum())


def subcounty_key(text: str) -> str:
    """First sub-county name in an events-table cell, upper-cased, for matching."""
    if not isinstance(text, str):
        return ""
    first = text.replace(";", ",").split(",")[0]
    first = first.split("(")[0].strip()
    return first.upper()
