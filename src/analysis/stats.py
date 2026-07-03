"""Small shared statistics helpers (dependency-free)."""

import math


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def pct(q: float) -> float:
        if n == 1:
            return float(ordered[0])
        pos = q * (n - 1)
        lower = int(math.floor(pos))
        upper = min(lower + 1, n - 1)
        frac = pos - lower
        return float(ordered[lower] * (1 - frac) + ordered[upper] * frac)

    mean = sum(ordered) / n
    variance = sum((v - mean) ** 2 for v in ordered) / n if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": float(ordered[0]),
        "p25": pct(0.25),
        "median": pct(0.5),
        "p75": pct(0.75),
        "max": float(ordered[-1]),
    }


def ols_slope(xs: list[float], ys: list[float]) -> dict[str, float | int | None]:
    """Least-squares slope of y on x. Returns slope=None with n<2 or zero x-variance."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    n = len(pairs)
    if n < 2:
        return {"n": n, "slope": None, "intercept": None}
    mean_x = sum(x for x, _ in pairs) / n
    mean_y = sum(y for _, y in pairs) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in pairs)
    if sxx == 0:
        return {"n": n, "slope": None, "intercept": None}
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    slope = sxy / sxx
    return {"n": n, "slope": slope, "intercept": mean_y - slope * mean_x}
