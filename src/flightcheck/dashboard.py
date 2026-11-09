"""Write the data files behind the fare dashboard (`flightcheck dashboard --out DIR`).

One JSON file of row objects per dataset:

- ``daily``        one row per run: the playbook's call and option, and the day's lows
- ``leg_history``  Google's ~60-day history of the any-airline Shanghai/Beijing one-ways
- ``out_today``    the latest run's cheapest outbound per date and strategy
- ``back_today``   the same for returns
- ``rt_today``     the latest run's cheapest round trip per departure x return date
- ``outlook``      the forecast's split and chance of reaching it by each checkpoint
- ``past_fares``   published US -> mainland China business round trips from past years
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Watchlist
from .report import CITY_NAMES

SHORT_NAMES = {"atl-any": "任意航司", "atl-tk": "土耳其航空", "atl-japan": "经东京一张票", "iah-ana": "休斯顿全日空"}
CALLS = {"buy": "买", "wait": "等几天", "pass": "不买", "ask": "问你", "hold": "本来会出手"}  # hold: not committed yet
HISTORY_CITIES = ("shanghai", "beijing")  # the searches Google returns a price history for
NOT_US = r"all origins|not stated|origin reference|Seoul|Hong Kong|Vancouver|\[HK|\[Korea|\[Can|\[China"


def build(watchlist: Watchlist, data_dir: Path, reports_dir: Path, past_fares: Path, out_dir: Path) -> dict[str, int]:
    """Write one JSON file per dataset into `out_dir`; returns the row count of each."""
    if watchlist.outbound is None or watchlist.inbound is None:
        raise ValueError("the dashboard needs fixed [dates] outbound and return windows")
    out_win = (watchlist.outbound.start.isoformat(), watchlist.outbound.end.isoformat())
    back_win = (watchlist.inbound.start.isoformat(), watchlist.inbound.end.isoformat())
    order = [s.key for s in watchlist.strategies]

    obs = pd.read_csv(data_dir / "observations.csv", dtype={"ret": str, "depart": str})
    obs = obs[obs["leg"].isin(["out", "back", "rt"]) & obs["total_price"].notna()]
    back = obs["leg"] == "back"
    obs = obs[(~back & obs["depart"].between(*out_win)) | (back & obs["depart"].between(*back_win))]
    latest = obs["run_date"].max()
    today = obs[obs["run_date"] == latest]

    files = {
        "daily": _daily(obs, data_dir / "decisions.csv"),
        "leg_history": _leg_history(data_dir / "history.csv", latest, out_win, back_win),
        "out_today": _by_date(today, "out", order),
        "back_today": _by_date(today, "back", order),
        "rt_today": _round_trips(today, order),
        "outlook": _outlook(reports_dir / "forecast.json"),
        "past_fares": _past_fares(past_fares),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        (out_dir / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
    return {name: len(rows) for name, rows in files.items()}


def _cheapest(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.loc[df.groupby(keys)["total_price"].idxmin()] if len(df) else df


def _strategy(key: str, order: list[str]) -> tuple[str, int]:
    return SHORT_NAMES.get(key, key), order.index(key) if key in order else len(order)


def _daily(obs: pd.DataFrame, decisions: Path) -> list[dict]:
    if not decisions.exists() or obs.empty:
        return []
    lows = obs.groupby(["run_date", "leg"])["total_price"].min().unstack()
    rows = []
    for d in pd.read_csv(decisions, dtype=str, keep_default_na=False).itertuples():
        if d.date not in lows.index or not d.total:
            continue
        lo = lows.loc[d.date]
        out, ret = _low(lo, "out"), _low(lo, "back")
        rows.append({
            "day": d.date, "call": CALLS.get(d.action, d.action), "kind": d.kind, "tier": d.tier,
            "playbook_total": int(float(d.total)), "pair_bare": out + ret if out and ret else None,
            "best_out": out, "best_back": ret, "best_rt": _low(lo, "rt"),
            "option": d.desc, "depart": d.depart, "ret": d.ret,
        })
    return rows


def _low(lows: pd.Series, leg: str) -> int | None:
    return int(lows[leg]) if leg in lows and pd.notna(lows[leg]) else None


def _leg_history(history: Path, latest: str, out_win: tuple, back_win: tuple) -> list[dict]:
    """Each day's cheapest date in the window, as Google's embedded history saw it."""
    hist = pd.read_csv(history, dtype={"ret": str, "depart": str})
    hist = hist[hist["ret"].isna() & (hist["obs_date"] <= latest)]
    rows = []
    for idx, city in enumerate(HISTORY_CITIES):
        for leg, key, win in (("out", f"atl-any:{city}", out_win), ("back", f"atl-any:{city}:back", back_win)):
            h = hist[(hist["series_key"] == key) & hist["depart"].between(*win)]
            for day, price in h.groupby("obs_date")["price"].min().items():
                rows.append({"day": day, "leg": leg, "city": CITY_NAMES[city], "city_idx": idx, "price": int(price)})
    return sorted(rows, key=lambda r: (r["leg"], r["city_idx"], r["day"]))


def _by_date(today: pd.DataFrame, leg: str, order: list[str]) -> list[dict]:
    rows = []
    for r in _cheapest(today[today["leg"] == leg], ["depart", "strategy"]).sort_values(["depart", "strategy"]).itertuples():
        name, idx = _strategy(r.strategy, order)
        rows.append({"depart": r.depart, "strategy": name, "strategy_idx": idx, "price": int(r.total_price),
                     "city": CITY_NAMES.get(r.city, r.city), "route": r.route, "airlines": r.airlines})
    return rows


def _round_trips(today: pd.DataFrame, order: list[str]) -> list[dict]:
    rows = []
    for r in _cheapest(today[today["leg"] == "rt"], ["depart", "ret"]).sort_values(["depart", "ret"]).itertuples():
        rows.append({"depart": r.depart, "ret": r.ret, "price": int(r.total_price), "strategy": _strategy(r.strategy, order)[0],
                     "city": CITY_NAMES.get(r.city, r.city), "route": r.route, "airlines": r.airlines})
    return rows


def _outlook(path: Path) -> list[dict]:
    if not path.exists():
        return []
    fc = json.loads(path.read_text())
    if fc.get("status") != "ok" or not fc.get("split"):
        return []
    return [{"checkpoint": c, "pair_chance": round(p, 4), "out_target": int(fc["split"][0]), "back_target": int(fc["split"][1])}
            for c, p in zip(fc["checkpoints"], fc["pair_hit"])]


def _past_fares(path: Path) -> list[dict]:
    """US-origin business round trips to mainland China with a known posting month."""
    if not path.exists():
        return []
    past = pd.read_csv(path, dtype=str, keep_default_na=False)
    past["month"] = past["month_posted"].str.extract(r"^(\d{4}-\d{2})")[0]
    keep = (
        past["trip"].str.startswith("round-trip") & ~past["trip"].str.contains("mixed")
        & past["month"].notna() & ~past["origin"].str.contains(NOT_US)
        & ~past["destination"].str.contains(r"Hong Kong|\[HK")
    )
    return [{"month": r.month, "price": int(float(r.price_usd)), "origin": r.origin, "destination": r.destination,
             "airline": r.airline, "travel": r.travel_window, "source": r.source_url}
            for r in past[keep].sort_values("month").itertuples()]
