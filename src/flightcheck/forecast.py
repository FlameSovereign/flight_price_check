"""Long-range outlook: how likely is each leg to reach a target price before a date?

The buy/wait model (model.py) only looks two weeks ahead. This answers the
question behind the whole project: "can two one-ways home and back get under
the round-trip target, and by when?"

Two simulators turn the recent price history into daily price paths:

- markov.py replays real day-to-day moves of routes that were in a similar
  state (how far above the recent low, how far from the route's usual fare,
  how jumpy lately);
- regime.py models a fare as a step function: a daily chance of jumping down
  or up, landing on fare levels the route has been selling.

Each simulates N_PATHS paths per leg; the report averages the two. In a fixed
backtest (two forecast origins, 7/14/30 days ahead) the pair beat a random
walk by 27/46/59% in pinball loss, with 80% intervals covering 85/85/84%.
Nothing beyond ~30 days could be checked: the data only spans a couple of
months, so a forecast for February is "if the coming months move like the
last ones". In particular it cannot know about seasonal sales, e.g. airlines
releasing cheap summer round-trip fares in spring.

The round-trip target is split between the outbound and the return leg at
the split that gives the best chance of buying both (legs simulated
independently), which also gives the two one-way prices worth setting alerts at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import markov, regime
from .config import Watchlist
from .model import build_panel

TRAIN_DAYS = 120  # fit on this many days of observations; moves older than that say little
N_PATHS = 2000
RANGE_DAYS = 30  # "price in a month" range: the longest horizon that was backtested
LEGS_PER_KIND = 4
MIN_ROUTE_DAYS = 14  # a route needs this many observed days before its legs are forecast
CHECKPOINTS = 4  # month starts before the trip to report chances for
SPLIT_STEP = 50
QS = (0.1, 0.5, 0.9)


@dataclass
class Leg:
    series_key: str
    strategy: str
    city: str
    kind: str  # "out" or "back"
    depart: date
    price: float  # today
    range30: tuple[float, float, float]  # 10% / median / 90% price RANGE_DAYS from now
    target: float | None = None
    hit: list[float] = field(default_factory=list)  # P(price <= target on some day before each checkpoint)


@dataclass
class Outlook:
    as_of: date | None
    checkpoints: list[date] = field(default_factory=list)
    legs: list[Leg] = field(default_factory=list)
    target: float | None = None  # two one-ways together at or below this
    split: tuple[float, float] | None = None  # outbound / return prices that add up to the target
    pair_hit: list[float] = field(default_factory=list)  # P(both legs reach their split by each checkpoint)
    status: str = "ok"


def outlook(history: pd.DataFrame, watchlist: Watchlist, n_paths: int | None = None, seed: int = 0) -> Outlook:
    n_paths = n_paths or N_PATHS
    panel = build_panel(history[history["ret"] == ""])
    if panel.empty:
        return Outlook(None, status="还没有单程价格数据")
    last = panel["obs_date"].max()
    as_of = last.date()
    panel = panel[panel["obs_date"] > last - pd.Timedelta(days=TRAIN_DAYS)].copy()
    parts = panel["series_key"].str.split(":")
    panel["strategy"], panel["city"] = parts.str[0], parts.str[1]
    panel["kind"] = np.where(panel["series_key"].str.endswith(":back"), "back", "out")

    query = _pick_legs(panel, last, watchlist)
    if query.empty:
        return Outlook(as_of, status="没有在监控日期范围内、今天有价格的单程")
    checkpoints = _checkpoints(as_of, query, watchlist)
    marks = sorted({RANGE_DAYS} | {(c - as_of).days for c in checkpoints})
    with threadpool_limits(1):
        sims = [markov.Forecaster(n_paths=n_paths, seed=seed), regime.Forecaster(n_paths=n_paths, seed=seed)]
        for sim in sims:
            sim.fit(panel)
        runs = [sim.paths(query, marks) for sim in sims]

    r = marks.index(RANGE_DAYS)
    # average the simulators' log-quantiles level by level (what the backtest scored)
    log_q = np.mean([np.quantile(lev[:, :, r], QS, axis=1).T for lev, _ in runs], axis=0)
    ks = [marks.index((c - as_of).days) for c in checkpoints]
    mins = [mn[:, :, ks] for _, mn in runs]

    def hit(i: int, price: float) -> np.ndarray:
        """P(leg i at or below `price` by each checkpoint), averaged over the simulators."""
        return np.mean([(mn[i] <= np.log(price) + 1e-9).mean(axis=0) for mn in mins], axis=0)

    legs = [
        Leg(row.series_key, row.strategy, row.city, row.kind, row.depart.date(), float(row.price),
            tuple(float(v) for v in np.exp(log_q[i])))
        for i, row in enumerate(query.itertuples())
    ]
    result = Outlook(as_of, checkpoints, legs)
    out = [i for i, leg in enumerate(legs) if leg.kind == "out"]
    back = [i for i, leg in enumerate(legs) if leg.kind == "back"]
    target = watchlist.alerts.get("round_trip")
    if target and out and back and checkpoints:
        # the cheapest leg each way today; picking the best of several simulated legs per path would
        # overstate the chances, because in reality departure dates of a route move together
        o, b = out[0], back[0]
        result.target = target
        result.split, result.pair_hit = _best_split(target, lambda p: hit(o, p), lambda p: hit(b, p))
        targets = {"out": result.split[0], "back": result.split[1]}
    else:
        one_way = watchlist.alerts.get("one_way")
        targets = {"out": one_way, "back": one_way}
    for i, leg in enumerate(legs):
        leg.target = targets[leg.kind]
        if leg.target and checkpoints:
            leg.hit = [float(p) for p in hit(i, leg.target)]
    return result


def _pick_legs(panel: pd.DataFrame, last: pd.Timestamp, watchlist: Watchlist) -> pd.DataFrame:
    """The cheapest departure date of each watched strategy:city per direction, LEGS_PER_KIND per direction."""
    # China-origin searches are dated in China's calendar, often a day ahead of the US ones
    rows = panel[panel["obs_date"] >= last - pd.Timedelta(days=1)]
    rows = rows.sort_values("obs_date").groupby(["series_key", "depart"], sort=False).tail(1)
    rows = rows[rows["strategy"].isin({s.key for s in watchlist.strategies}) & rows["city"].isin(set(watchlist.destinations))]
    # A route added a few days ago has no moves of its own to learn from; its legs would get
    # generic, much wider paths (a new "china" search priced the same fare as "beijing" at 10x the chance).
    days = panel.groupby("series_key")["obs_date"].nunique()
    rows = rows[rows["series_key"].map(days).fillna(0) >= MIN_ROUTE_DAYS]
    min_out = watchlist.model.min_days_out
    windows = {"out": watchlist.outbound, "back": watchlist.inbound}
    picked = []
    for kind, window in windows.items():
        legs = rows[rows["kind"] == kind]
        if window is not None:
            day = legs["depart"].dt.date
            legs = legs[(day >= window.start) & (day <= window.end)]
        legs = legs[legs["days_out"] >= min_out + RANGE_DAYS]
        legs = legs.sort_values(["price", "depart"]).groupby("series_key", sort=False).head(1)
        picked.append(legs.head(LEGS_PER_KIND))
    return pd.concat(picked, ignore_index=True)


def _checkpoints(as_of: date, query: pd.DataFrame, watchlist: Watchlist) -> list[date]:
    """The last CHECKPOINTS month starts before the latest day worth buying the first leg."""
    first = watchlist.outbound.start if watchlist.outbound is not None else query["depart"].min().date()
    last_buy = first - timedelta(days=watchlist.model.min_days_out)
    days = []
    day = date(as_of.year, as_of.month, 1)
    while True:
        day = date(day.year + day.month // 12, day.month % 12 + 1, 1)
        if day > last_buy:
            break
        days.append(day)
    return days[-CHECKPOINTS:]


def _best_split(target: float, hit_out, hit_back) -> tuple[tuple[float, float], list[float]]:
    """Outbound/return prices adding up to `target` with the best chance of buying both by the last checkpoint."""
    lo = SPLIT_STEP * np.ceil(target * 0.25 / SPLIT_STEP)
    grid = np.arange(lo, target - lo + 1, SPLIT_STEP) if target > 2 * lo else np.array([target / 2])
    best = max(grid, key=lambda a: (hit_out(a)[-1] * hit_back(target - a)[-1], -abs(2 * a - target)))
    chances = hit_out(best) * hit_back(target - best)
    return (float(best), float(target - best)), [float(p) for p in chances]
