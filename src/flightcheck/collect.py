"""Run every search in the watchlist and append the results to `data/`.

Files written:

* ``observations.csv``  one row per search per run: cheapest itinerary,
  positioning leg, Google's typical range.
* ``history.csv``       daily price series per (strategy:city, departure date).
  Seeded from the ~60-day history embedded in each Google result, then
  extended by every run. This is the model's training data.
* ``itineraries/<run date>.csv.gz``  the cheapest few itineraries per search.
"""

from __future__ import annotations

import io
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import google
from .config import Job, Watchlist

LOCAL_TZ = ZoneInfo("America/New_York")
HISTORY_KEY = ["series_key", "depart", "ret", "obs_date"]
TOP_ITINERARIES = 5
MAX_CONSECUTIVE_FAILURES = 8


def local_today() -> date:
    return datetime.now(LOCAL_TZ).date()


def _filter_via(itineraries: list[google.Itinerary], via: tuple[str, ...]) -> list[google.Itinerary]:
    if not via:
        return itineraries
    wanted = set(via)
    return [it for it in itineraries if wanted & set(it.route.split("-")[1:-1])]


def run_job(job: Job, today: date, fetch=google.search) -> tuple[dict, list[dict], list[dict]]:
    """Run one job. Returns (observation row, history rows, itinerary rows)."""
    obs = {
        "run_date": today.isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series_key": job.series_key,
        "strategy": job.strategy.key,
        "city": job.city,
        "depart": job.depart.isoformat(),
        "ret": job.ret.isoformat() if job.ret else "",
        "days_out": (job.depart - today).days,
        "url": job.search.url(),
    }
    result = fetch(job.search)
    itineraries = sorted(_filter_via(result.itineraries, job.strategy.via), key=lambda it: it.price)
    best = itineraries[0] if itineraries else None
    insight = result.insight
    obs.update(
        price=best.price if best else None,
        airlines=" + ".join(best.airlines) if best else "",
        route=best.route if best else "",
        flights=" ".join(best.flight_numbers) if best else "",
        stops=best.stops if best else None,
        duration_min=best.duration_min if best else None,
        n_itineraries=len(itineraries),
        g_current=insight.current if insight else None,
        g_typical=insight.typical if insight else None,
        g_low=insight.typical_low if insight else None,
        g_high=insight.typical_high if insight else None,
        g_level=insight.level if insight else None,
        history_points=len(insight.history) if insight else 0,
    )

    position_price = None
    if job.positioning is not None and best is not None:
        try:
            pos = fetch(job.positioning).cheapest
        except google.FetchError:
            pos = None
        if pos is not None:
            position_price = pos.price
            obs.update(position_route=pos.route, position_airlines=" + ".join(pos.airlines))
    obs["position_price"] = position_price
    # Without the positioning leg the total would look cheaper than the trip
    # really is, so leave it empty rather than understate it.
    if best is not None and (job.positioning is None or position_price is not None):
        obs["total_price"] = best.price + (position_price or 0)
    else:
        obs["total_price"] = None

    base = {"series_key": job.series_key, "depart": obs["depart"], "ret": obs["ret"]}
    history = []
    # The embedded history tracks Google's unfiltered cheapest price, so it only
    # matches our number when no local `via` filter changed the answer.
    if insight and insight.history and not job.strategy.via:
        history = [{**base, "obs_date": d.isoformat(), "price": p, "source": "google"} for d, p in insight.history]
    elif best is not None:
        history = [{**base, "obs_date": today.isoformat(), "price": best.price, "source": "observed"}]

    itinerary_rows = [
        {
            **base,
            "run_date": obs["run_date"],
            "rank": rank,
            "price": it.price,
            "airlines": " + ".join(it.airlines),
            "route": it.route,
            "flights": " ".join(it.flight_numbers),
            "stops": it.stops,
            "duration_min": it.duration_min,
            "dep_time": it.depart,
            "arr_time": it.arrive,
        }
        for rank, it in enumerate(itineraries[:TOP_ITINERARIES], start=1)
    ]
    return obs, history, itinerary_rows


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Keep integer columns integer even when some values are missing."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind == "f" and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype("Int64")
    return df


def _merge_csv(path: Path, new: pd.DataFrame, key: list[str]) -> None:
    new = pd.read_csv(io.StringIO(_tidy(new).to_csv(index=False)), dtype=str, keep_default_na=False)
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=key, keep="last").sort_values(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def _save(data_dir: Path, today: date, observations: list, history: list, itineraries: list) -> None:
    if observations:
        _merge_csv(data_dir / "observations.csv", pd.DataFrame(observations), ["run_date", "series_key", "depart", "ret"])
    if history:
        _merge_csv(data_dir / "history.csv", pd.DataFrame(history), HISTORY_KEY)
    if itineraries:
        path = data_dir / "itineraries" / f"{today.isoformat()}.csv.gz"
        new = pd.read_csv(io.StringIO(_tidy(pd.DataFrame(itineraries)).to_csv(index=False)), dtype=str, keep_default_na=False)
        if path.exists():
            # Same-day rerun: replace only the searches this run fetched again.
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            refetched = set(map(tuple, new[["series_key", "depart", "ret"]].to_numpy()))
            keep = [tuple(r) not in refetched for r in old[["series_key", "depart", "ret"]].to_numpy()]
            new = pd.concat([old[keep], new], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(path, index=False)


def collect(
    watchlist: Watchlist,
    data_dir: Path,
    *,
    today: date | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
    fetch=google.search,
    log=print,
) -> pd.DataFrame:
    today = today or local_today()
    jobs = watchlist.jobs(today)
    if only:
        jobs = [j for j in jobs if j.strategy.key in only or j.city in only]
    if limit:
        jobs = jobs[:limit]
    log(f"{len(jobs)} searches for {today} ({len(watchlist.departure_dates(today))} departure dates)")

    cache: dict[str, google.Result] = {}
    requests = 0

    def polite_fetch(search):
        # Positioning legs repeat across destinations, so fetch each URL once.
        nonlocal requests
        url = search.url()
        if url not in cache:
            if requests and watchlist.request_delay:
                time.sleep(watchlist.request_delay)
            requests += 1
            cache[url] = fetch(search)
        return cache[url]

    observations, history, itineraries = [], [], []
    failures = consecutive = 0
    try:
        for i, job in enumerate(jobs, start=1):
            try:
                obs, hist, its = run_job(job, today, fetch=polite_fetch)
            except Exception as exc:  # one bad page must not cost the whole day's data
                failures += 1
                consecutive += 1
                log(f"[{i}/{len(jobs)}] {job.series_key} {job.depart}: FAILED {type(exc).__name__}: {exc}")
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    log("too many consecutive failures; Google is probably blocking us, stopping early")
                    break
                continue
            consecutive = 0
            observations.append(obs)
            history.extend(hist)
            itineraries.extend(its)
            price = f"${obs['total_price']:,}" if obs.get("total_price") else "no total"
            log(f"[{i}/{len(jobs)}] {job.series_key} {job.depart}: {price} {obs['route']}")
    finally:
        # Keep whatever was collected even if the loop is interrupted.
        _save(data_dir, today, observations, history, itineraries)

    if not observations:
        raise RuntimeError(f"all {len(jobs)} searches failed")
    log(f"done: {len(observations)} ok, {failures} failed")
    if failures:
        print(f"warning: {failures} searches failed", file=sys.stderr)
    obs_df = pd.DataFrame(observations)
    return obs_df
