from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from flightcheck import collect, config, forecast, google, markov, model, regime

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "watchlist.toml"


def _watchlist():
    wl = config.load(FIXTURE)
    wl.trips = ("two-one-ways",)
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 9), 4)
    wl.inbound = config.DateRange(date(2027, 8, 1), date(2027, 8, 7), 6)
    wl.alerts = {"round_trip": 5000, "one_way": 2800}
    return wl


def _leg_history(seed=0, last="2026-10-11"):
    """Step-function fares that hop between a route's fare levels, like business fares do."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-08-13", last)
    routes = (
        ("atl-any:shanghai", pd.date_range("2027-05-01", "2027-05-09", freq="4D"), (2600, 3000, 3400)),
        ("atl-any:beijing", pd.date_range("2027-05-01", "2027-05-09", freq="4D"), (2900, 3300, 3700)),
        ("atl-tk:shanghai", pd.date_range("2027-05-01", "2027-05-09", freq="4D"), (4200, 4600, 5000)),
        ("atl-any:shanghai:back", pd.date_range("2027-08-01", "2027-08-07", freq="6D"), (2100, 2500, 2900)),
        ("atl-any:beijing:back", pd.date_range("2027-08-01", "2027-08-07", freq="6D"), (2300, 2700, 3100)),
        ("atl-any:beijing:back", pd.date_range("2027-09-20", "2027-09-20"), (900, 900, 900)),  # outside the window
    )
    rows = []
    for key, departs, ladder in routes:
        for depart in departs:
            level = 1
            for day in days:
                if rng.random() < 0.08:
                    level = int(rng.integers(0, len(ladder)))
                rows.append({"series_key": key, "depart": depart, "ret": "", "obs_date": day,
                             "price": float(ladder[level]), "source": "google"})
    return pd.DataFrame(rows)


def test_outlook_splits_the_round_trip_target_and_reports_chances():
    wl = _watchlist()
    o = forecast.outlook(_leg_history(), wl, n_paths=300)
    assert o.status == "ok" and o.as_of == date(2026, 10, 11)
    # the last four month starts before the latest day worth buying (May 1 - 21 days)
    assert o.checkpoints == [date(2027, 1, 1), date(2027, 2, 1), date(2027, 3, 1), date(2027, 4, 1)]

    out = [leg for leg in o.legs if leg.kind == "out"]
    back = [leg for leg in o.legs if leg.kind == "back"]
    assert {leg.series_key for leg in out} == {"atl-any:shanghai", "atl-any:beijing", "atl-tk:shanghai"}
    assert all(date(2027, 5, 1) <= leg.depart <= date(2027, 5, 9) for leg in out)
    assert all(date(2027, 8, 1) <= leg.depart <= date(2027, 8, 7) for leg in back)  # the $900 September fare is ignored
    assert out == sorted(out, key=lambda leg: leg.price) and back == sorted(back, key=lambda leg: leg.price)

    assert o.target == 5000 and sum(o.split) == 5000
    assert len(o.pair_hit) == len(o.checkpoints)
    assert all(0 <= p <= 1 for p in o.pair_hit) and o.pair_hit == sorted(o.pair_hit)  # more time, more chances
    for leg in o.legs:
        assert leg.target == (o.split[0] if leg.kind == "out" else o.split[1])
        assert leg.hit == sorted(leg.hit) and all(0 <= p <= 1 for p in leg.hit)
        if leg.price <= leg.target:
            assert leg.hit == [1.0] * len(o.checkpoints)
        lo, mid, hi = leg.range30
        assert lo <= mid <= hi
    # the joint chance uses the cheapest leg each way
    p_out, p_back = out[0].hit[-1], back[0].hit[-1]
    assert abs(o.pair_hit[-1] - p_out * p_back) < 1e-9

    again = forecast.outlook(_leg_history(), wl, n_paths=300)
    assert again == o  # seeded: the daily report does not flicker on its own


def test_split_maximises_the_chance_of_buying_both_legs():
    hit_out = lambda p: np.array([min(1.0, max(0.0, (p - 2000) / 1000))])  # noqa: E731
    hit_back = lambda p: np.array([min(1.0, max(0.0, (p - 2400) / 200))])  # noqa: E731
    (a, b), chances = forecast._best_split(5000, hit_out, hit_back)
    # out at $2,400 (40%) with the return at $2,600 (certain) beats any even split
    assert (a, b) == (2400, 2600) and np.allclose(chances, [0.4])


def test_checkpoints_roll_over_the_year_and_stop_before_the_trip():
    wl = _watchlist()
    wl.outbound = config.DateRange(date(2027, 2, 10), date(2027, 2, 20), 1)
    query = pd.DataFrame({"depart": [pd.Timestamp("2027-02-10")]})
    assert forecast._checkpoints(date(2026, 11, 15), query, wl) == [date(2026, 12, 1), date(2027, 1, 1)]
    assert forecast._checkpoints(date(2027, 1, 25), query, wl) == []


def test_outlook_without_usable_legs_says_why():
    wl = _watchlist()
    assert forecast.outlook(_leg_history().iloc[:0], wl).status != "ok"
    wl.outbound = config.DateRange(date(2028, 5, 1), date(2028, 5, 9), 4)
    wl.inbound = config.DateRange(date(2028, 8, 1), date(2028, 8, 7), 6)
    assert forecast.outlook(_leg_history(), wl, n_paths=50).status != "ok"


def test_simulated_minima_never_rise_and_start_at_todays_price():
    panel = model.build_panel(_leg_history())
    parts = panel["series_key"].str.split(":")
    panel["strategy"], panel["city"] = parts.str[0], parts.str[1]
    panel["kind"] = np.where(panel["series_key"].str.endswith(":back"), "back", "out")
    query = panel[panel["obs_date"] == panel["obs_date"].max()].head(4)
    for sim in (markov.Forecaster(n_paths=100), regime.Forecaster(n_paths=100)):
        sim.fit(panel)
        lev, mins = sim.paths(query, [0, 10, 40])
        today = np.log(query["price"].to_numpy())[:, None]
        assert lev.shape == mins.shape == (4, 100, 3)
        assert np.allclose(lev[:, :, 0], today) and np.allclose(mins[:, :, 0], today)
        assert (np.diff(mins, axis=2) <= 1e-12).all() and (mins <= lev + 1e-12).all()


def _fetch(search):
    s = search.slices[0]
    it = google.Itinerary(3000 + s.date.day, ["X"], f"{s.origins[0]}-ICN-{s.destinations[0]}", ["XX1"], 1, 900, None, None)
    return google.Result([it], None)


def test_report_shows_the_outlook_and_survives_a_forecast_failure(tmp_path, monkeypatch):
    from flightcheck import report

    wl = _watchlist()
    wl.request_delay = 0
    data, reports = tmp_path / "data", tmp_path / "reports"
    collect.collect(wl, data, today=date(2026, 10, 11), fetch=_fetch, log=lambda *_: None)
    hist = _leg_history()
    hist.assign(depart=hist["depart"].dt.date, obs_date=hist["obs_date"].dt.date).to_csv(data / "history.csv", index=False)
    monkeypatch.setattr(forecast, "N_PATHS", 100)
    report.build(wl, data, reports)
    md = (reports / "latest.md").read_text()
    assert "## 远期预测：两张单程能不能凑到 $5,000？" in md and "胜算最大的分法" in md
    assert (reports / "forecast.json").exists()

    def broken(*_, **__):
        raise ValueError("boom")

    monkeypatch.setattr(forecast, "outlook", broken)
    report.build(wl, data, reports)
    md = (reports / "latest.md").read_text()
    assert "预测出错" in md and "## 模型：现在买还是再等等？" in md


def test_outlook_works_before_any_fare_has_moved():
    flat = _leg_history().assign(price=3000.0)  # a brand-new watchlist: not a single jump yet
    o = forecast.outlook(flat, _watchlist(), n_paths=50)
    assert o.status == "ok" and all(np.allclose(leg.range30, 3000) for leg in o.legs)


def test_routes_without_enough_history_are_left_out():
    hist = _leg_history()
    new = hist[(hist["series_key"] == "atl-any:beijing:back") & (hist["obs_date"] >= hist["obs_date"].max() - pd.Timedelta(days=2))]
    hist = pd.concat([hist, new.assign(series_key="atl-any:china:back")], ignore_index=True)
    wl = _watchlist()
    wl.destinations = dict(wl.destinations, china=("PVG", "PEK"))
    o = forecast.outlook(hist, wl, n_paths=50)
    assert "atl-any:china:back" not in {leg.series_key for leg in o.legs}
