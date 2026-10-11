from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from flightcheck import collect, config, google, model

ROOT = Path(__file__).resolve().parents[1]


def test_default_watchlist_loads():
    wl = config.load(ROOT / "config" / "watchlist.toml")
    assert wl.cabin == "business"
    assert {s.key for s in wl.strategies} >= {"atl-any", "atl-tk"}
    jobs = wl.jobs(date(2026, 10, 11))
    assert len(jobs) == len(wl.departure_dates(date(2026, 10, 11))) * len(wl.strategies) * len(wl.destinations)


def test_departure_dates_are_stable_across_days():
    wl = config.load(ROOT / "config" / "watchlist.toml")
    a = set(wl.departure_dates(date(2026, 10, 11)))
    b = set(wl.departure_dates(date(2026, 10, 12)))
    # Moving one day forward drops at most one date at the front and adds at most one at the end.
    assert len(a & b) >= len(a) - 1
    assert all((d - config.DATE_ANCHOR).days % wl.step_days == 0 for d in a)


def test_windows_add_every_day():
    wl = config.load(ROOT / "config" / "watchlist.toml")
    wl.windows = [(date(2027, 1, 28), date(2027, 2, 3))]
    dates = wl.departure_dates(date(2026, 10, 11))
    assert all(date(2027, 1, 28) + timedelta(days=i) in dates for i in range(7))


def _fake_fetch(search):
    s = search.slices[0]
    if "IAH" in s.destinations:  # positioning leg
        it = google.Itinerary(99, ["United"], "ATL-IAH", ["UA1"], 0, 140, None, None)
        return google.Result([it], None)
    price = 5000 + (s.date.toordinal() % 7) * 100
    via = "HND" if s.connecting else "ICN"
    its = [
        google.Itinerary(price, ["X"], f"{s.origins[0]}-{via}-{s.destinations[0]}", ["XX1", "XX2"], 1, 900, None, None),
        google.Itinerary(price - 400, ["Y"], f"{s.origins[0]}-LAX-{s.destinations[0]}", ["YY1", "YY2"], 1, 900, None, None),
    ]
    hist = [(date(2026, 10, 11) - timedelta(days=i), price + i) for i in range(5)][::-1]
    insight = None if s.connecting else google.Insight(price - 400, price, price - 800, price + 800, 3, hist)
    return google.Result(its, insight)


def test_collect_writes_data_and_applies_filters(tmp_path):
    wl = config.load(ROOT / "config" / "watchlist.toml")
    wl.request_delay = 0
    obs = collect.collect(wl, tmp_path, today=date(2026, 10, 11), limit=8, fetch=_fake_fetch, log=lambda *_: None)
    assert len(obs) == 8
    japan = obs[obs["strategy"] == "atl-japan"].iloc[0]
    assert "HND" in japan["route"]  # `via` dropped the cheaper LAX itinerary
    iah = obs[obs["strategy"] == "iah-ana"].iloc[0]
    assert iah["total_price"] == iah["price"] + 99
    hist = pd.read_csv(tmp_path / "history.csv")
    assert set(hist["source"]) == {"google", "observed"}
    assert (tmp_path / "itineraries" / "2026-10-11.csv.gz").exists()
    # Running again on the same day must not duplicate rows.
    collect.collect(wl, tmp_path, today=date(2026, 10, 11), limit=8, fetch=_fake_fetch, log=lambda *_: None)
    assert len(pd.read_csv(tmp_path / "observations.csv")) == 8
    assert len(pd.read_csv(tmp_path / "history.csv")) == len(hist)


def _synthetic_history(n_series=24, days=70, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-08-01")
    for k in range(n_series):
        depart = start + pd.Timedelta(days=90 + 6 * k)
        price = 4000.0
        for d in range(days):
            # Spikes that revert: after a jump the price tends to come back down.
            if rng.random() < 0.08:
                price *= 1.15
            elif price > 4100 and rng.random() < 0.25:
                price = 4000.0
            rows.append(
                {"series_key": f"s{k % 3}:city", "depart": depart, "ret": "", "obs_date": start + pd.Timedelta(days=d),
                 "price": round(price), "source": "google"}
            )
    return pd.DataFrame(rows)


def test_targets_look_forward_only():
    hist = pd.DataFrame(
        {"series_key": "a:b", "depart": pd.Timestamp("2027-01-01"), "ret": "",
         "obs_date": pd.date_range("2026-10-01", periods=6), "price": [100, 100, 98, 120, 80, 100], "source": "google"}
    )
    panel = model.add_targets(model.build_panel(hist), horizon=2, threshold=0.05)
    assert list(panel["future_min"].iloc[:4]) == [98, 98, 80, 80]
    assert panel["y"].iloc[:4].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert panel["y"].iloc[4:].isna().all()  # not enough future observed yet


def test_train_learns_mean_reversion_and_backtests():
    settings = config.ModelSettings(horizon_days=7, min_days_out=5)
    result = model.train(_synthetic_history(), settings)
    m = result.metrics
    assert m["status"] == "ok"
    assert m["auc"] > 0.7
    assert m["backtest"]["episodes"] > 0
    signals = model.latest_signals(result, settings)
    assert len(signals) == 24
    assert set(signals["signal"]) <= {"买", "等", "观望", "数据不足"}


def test_report_renders_and_alerts_only_on_new_lows(tmp_path):
    from flightcheck import report

    wl = config.load(ROOT / "config" / "watchlist.toml")
    wl.request_delay = 0
    wl.alerts = {"shanghai": 99999}
    data, reports = tmp_path / "data", tmp_path / "reports"
    collect.collect(wl, data, today=date(2026, 10, 10), limit=16, fetch=_fake_fetch, log=lambda *_: None)
    info = report.build(wl, data, reports)
    md = (reports / "latest.md").read_text()
    assert "回国商务舱价格监控 · 2026-10-10" in md and "### 上海" in md
    assert info["alerts"] and (reports / "alerts.md").exists()

    # Same prices the next day: nothing new under the limit, so no alert.
    collect.collect(wl, data, today=date(2026, 10, 11), limit=16, fetch=_fake_fetch, log=lambda *_: None)
    info = report.build(wl, data, reports)
    assert not info["alerts"] and not (reports / "alerts.md").exists()
