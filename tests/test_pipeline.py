from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from flightcheck import collect, config, google, model

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "watchlist.toml"


def test_real_watchlist_loads_and_expands():
    wl = config.load(ROOT / "config" / "watchlist.toml")
    jobs = wl.jobs(date(2026, 10, 11))
    assert jobs
    keys = [(j.series_key, j.depart, j.ret) for j in jobs]
    assert len(keys) == len(set(keys))


def test_departure_dates_are_stable_across_days():
    wl = config.load(FIXTURE)
    a = set(wl.departure_dates(date(2026, 10, 11)))
    b = set(wl.departure_dates(date(2026, 10, 12)))
    # Moving one day forward drops at most one date at the front and adds at most one at the end.
    assert len(a & b) >= len(a) - 1
    assert all((d - config.DATE_ANCHOR).days % wl.step_days == 0 for d in a)


def test_windows_add_every_day():
    wl = config.load(FIXTURE)
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
    wl = config.load(FIXTURE)
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

    wl = config.load(FIXTURE)
    wl.request_delay = 0
    wl.alerts = {"one_way": 99999}
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


def test_one_unexpected_error_does_not_lose_the_run(tmp_path):
    wl = config.load(FIXTURE)
    wl.request_delay = 0
    calls = []

    def flaky(search):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("garbled page")
        return _fake_fetch(search)

    obs = collect.collect(wl, tmp_path, today=date(2026, 10, 11), limit=8, fetch=flaky, log=lambda *_: None)
    assert len(obs) == 7
    assert len(pd.read_csv(tmp_path / "observations.csv")) == 7


def test_missing_positioning_leaves_total_empty(tmp_path):
    wl = config.load(FIXTURE)
    wl.request_delay = 0

    def no_positioning(search):
        if "IAH" in search.slices[0].destinations:
            return google.Result([], None)
        return _fake_fetch(search)

    obs = collect.collect(wl, tmp_path, today=date(2026, 10, 11), limit=8, fetch=no_positioning, log=lambda *_: None)
    iah = obs[obs["strategy"] == "iah-ana"].iloc[0]
    assert iah["price"] > 0 and pd.isna(iah["total_price"])


def test_positioning_flies_the_day_before():
    wl = config.load(FIXTURE)
    job = next(j for j in wl.jobs(date(2026, 10, 11)) if j.positioning is not None)
    assert job.positioning.slices[0].date == job.depart - timedelta(days=1)


def test_same_day_rerun_merges_itineraries(tmp_path):
    wl = config.load(FIXTURE)
    wl.request_delay = 0
    today = date(2026, 10, 11)
    collect.collect(wl, tmp_path, today=today, limit=8, fetch=_fake_fetch, log=lambda *_: None)
    path = tmp_path / "itineraries" / "2026-10-11.csv.gz"
    before = pd.read_csv(path)
    collect.collect(wl, tmp_path, today=today, limit=8, only={"atl-tk"}, fetch=_fake_fetch, log=lambda *_: None)
    after = pd.read_csv(path)
    key = ["series_key", "depart", "rank"]
    assert not after.duplicated(subset=key).any()  # re-fetched searches replaced, not appended
    kept = before.merge(after[key], on=key, how="left", indicator=True)
    assert (kept["_merge"] == "both").all()  # nothing from the first run was lost


def test_label_window_does_not_span_a_gap():
    days = list(pd.date_range("2026-09-01", "2026-09-21")) + list(pd.date_range("2026-09-28", "2026-10-11"))
    prices = [5000] * len(days)
    prices[days.index(pd.Timestamp("2026-10-01"))] = 4000
    hist = pd.DataFrame({"series_key": "a:b", "depart": pd.Timestamp("2027-03-01"), "ret": "", "obs_date": days,
                         "price": prices, "source": "observed"})
    panel = model.add_targets(model.build_panel(hist), horizon=14, threshold=0.03)
    row = panel[panel["obs_date"] == pd.Timestamp("2026-09-14")].iloc[0]
    # 09-22..09-24 are forward-filled, then the series breaks; the 10-01 drop is 17 days away.
    assert pd.isna(row["y"]) or row["y"] == 0.0
    assert panel["seg"].nunique() == 2


def test_cross_section_works_for_round_trips():
    rows = []
    for k in range(10):
        depart = pd.Timestamp("2027-01-01") + pd.Timedelta(days=6 * k)
        for d in pd.date_range("2026-09-01", periods=20):
            rows.append({"series_key": "a:b", "depart": depart, "ret": str((depart + pd.Timedelta(days=30)).date()),
                         "obs_date": d, "price": 4000 + 50 * k, "source": "google"})
    panel = model.add_features(model.build_panel(pd.DataFrame(rows)))
    assert panel["r_xsec"].notna().mean() > 0.5


def test_train_survives_features_that_are_all_missing():
    # 20 days of history: the 14-day change is empty for every training row of the first fold.
    settings = config.ModelSettings(horizon_days=3, min_days_out=0)
    result = model.train(_synthetic_history(n_series=30, days=20), settings)
    assert result.metrics["status"] == "ok"
    assert result.metrics["policy"] in model.POLICIES


def test_cli_accepts_shared_options_before_or_after_the_subcommand(tmp_path, monkeypatch):
    from flightcheck import cli

    seen = {}
    monkeypatch.setattr("flightcheck.report.build", lambda wl, data, reports: seen.update(data=data, reports=reports) or {"metrics": {}})
    cli.main(["--config", str(FIXTURE), "report", "--data", str(tmp_path / "d"), "--reports", str(tmp_path / "r")])
    assert seen == {"data": tmp_path / "d", "reports": tmp_path / "r"}
    cli.main(["--config", str(FIXTURE), "--data", str(tmp_path / "d2"), "report"])
    assert seen["data"] == tmp_path / "d2"


def test_alerts_survive_partial_runs_and_reruns(tmp_path):
    from flightcheck import report

    wl = config.load(FIXTURE)
    wl.request_delay = 0
    wl.alerts = {"one_way": 99999}
    data, reports = tmp_path / "data", tmp_path / "reports"
    run = lambda day, **kw: collect.collect(wl, data, today=day, fetch=_fake_fetch, log=lambda *_: None, **kw)  # noqa: E731
    run(date(2026, 10, 10), limit=16)
    assert report.build(wl, data, reports)["alerts"]
    assert not report.build(wl, data, reports)["alerts"]  # same-day rerun of the report
    run(date(2026, 10, 11), limit=2)  # partial run: most combos missing today
    assert not report.build(wl, data, reports)["alerts"]
    run(date(2026, 10, 12), limit=16)  # full run, unchanged prices
    assert not report.build(wl, data, reports)["alerts"]


def test_report_without_any_history(tmp_path):
    from flightcheck import report

    wl = config.load(FIXTURE)
    wl.request_delay = 0
    nothing = lambda search: google.Result([], None)  # noqa: E731
    collect.collect(wl, tmp_path / "data", today=date(2026, 10, 11), limit=4, fetch=nothing, log=lambda *_: None)
    assert not (tmp_path / "data" / "history.csv").exists()
    info = report.build(wl, tmp_path / "data", tmp_path / "reports")
    assert info["metrics"]["status"] == "no price history yet"


def test_round_trip_report_shows_return_dates_and_ignores_one_way_rows(tmp_path):
    from flightcheck import report

    wl = config.load(FIXTURE)
    wl.request_delay = 0
    data, reports = tmp_path / "data", tmp_path / "reports"
    collect.collect(wl, data, today=date(2026, 10, 11), limit=8, fetch=_fake_fetch, log=lambda *_: None)  # one-way
    wl.trips, wl.return_after_days = ("round-trip",), 45
    collect.collect(wl, data, today=date(2026, 10, 11), limit=8, fetch=_fake_fetch, log=lambda *_: None)
    report.build(wl, data, reports)
    md = (reports / "latest.md").read_text()
    assert "回程 = 出发后 45 天" in md and "回 2026-12-07" in md
    obs, _, missing = report.load_latest_run(data, wl)
    assert len(obs) == 8 and (obs["ret"] != "").all()
    assert missing["rt"] > 0  # only 8 of the round trips were searched (limit=8)


def test_two_one_ways_pairs_outbound_and_return_legs(tmp_path):
    from flightcheck import report

    wl = config.load(FIXTURE)
    wl.request_delay = 0
    wl.trips, wl.return_after_days = ("two-one-ways",), 45
    jobs = wl.jobs(date(2026, 10, 11))
    back = [j for j in jobs if j.leg == "back"]
    assert len(back) == len(jobs) // 2
    j = next(j for j in back if j.strategy.key == "iah-ana")
    assert j.depart == j.trip_depart + timedelta(days=45)
    assert j.search.slices[0].origins == ("PVG", "SHA") and j.search.slices[0].destinations == ("IAH",)
    assert j.positioning.slices[0].origins == ("IAH",) and j.positioning.slices[0].destinations == ("ATL",)

    data, reports = tmp_path / "data", tmp_path / "reports"
    collect.collect(wl, data, today=date(2026, 10, 11), limit=16, fetch=_fake_fetch, log=lambda *_: None)
    obs = pd.read_csv(data / "observations.csv")
    assert set(obs["leg"]) == {"out", "back"}
    assert obs["series_key"].str.endswith(":back").sum() == 8
    report.build(wl, data, reports)
    md = (reports / "latest.md").read_text()
    assert "两张单程最低合计" in md and "按日期：两张单程合计" in md


def test_time_budget_stops_early_and_keeps_data(tmp_path):
    wl = config.load(FIXTURE)
    wl.request_delay = 0
    ticks = iter(range(0, 10_000, 30))  # every clock() call advances 30 seconds
    obs = collect.collect(wl, tmp_path, today=date(2026, 10, 11), limit=40, max_minutes=2, fetch=_fake_fetch,
                          log=lambda *_: None, clock=lambda: next(ticks))
    assert 0 < len(obs) < 40
    assert len(pd.read_csv(tmp_path / "observations.csv")) == len(obs)


def test_fixed_windows_pair_every_outbound_with_every_return(tmp_path):
    from flightcheck import report

    wl = config.load(FIXTURE)
    wl.request_delay = 0
    wl.trips = ("round-trip", "two-one-ways")
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 9), 4)  # 3 dates
    wl.inbound = config.DateRange(date(2027, 8, 1), date(2027, 8, 7), 6)  # 2 dates
    wl.alerts = {"round_trip": 99999, "one_way": 1}
    jobs = wl.jobs(date(2026, 10, 11))
    per_place = len(wl.strategies) * len(wl.destinations)
    kinds = [("rt" if j.ret else j.leg) for j in jobs]
    assert kinds.count("out") == 3 * per_place and kinds.count("back") == 2 * per_place
    assert kinds.count("rt") == 3 * 2 * per_place
    assert kinds.index("rt") > max(i for i, k in enumerate(kinds) if k != "rt")  # pairs run last

    data, reports = tmp_path / "data", tmp_path / "reports"
    collect.collect(wl, data, today=date(2026, 10, 11), fetch=_fake_fetch, log=lambda *_: None)
    info = report.build(wl, data, reports)
    md = (reports / "latest.md").read_text()
    assert "去程 \\ 回程" in md and "现在更便宜的买法" in md and "两张单程最低合计" in md
    assert info["alerts"] and "往返票" in (reports / "alerts.md").read_text()
    # Windows drop dates that have passed.
    assert all(j.depart > date(2027, 5, 5) for j in wl.jobs(date(2027, 5, 5)) if j.leg == "out")


def test_round_trip_pairs_can_be_sparser_than_single_legs():
    wl = config.load(FIXTURE)
    wl.trips = ("round-trip", "two-one-ways")
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 9), 2)  # 5 dates
    wl.inbound = config.DateRange(date(2027, 8, 1), date(2027, 8, 7), 3)  # 3 dates
    wl.round_trip_every = 2
    jobs = wl.jobs(date(2026, 10, 11))
    rt = {(j.depart, j.ret) for j in jobs if j.ret}
    assert {d for d, _ in rt} == {date(2027, 5, 1), date(2027, 5, 5), date(2027, 5, 9)}
    assert {r for _, r in rt} == {date(2027, 8, 1), date(2027, 8, 7)}
    assert len({j.depart for j in jobs if j.leg == "out" and not j.ret}) == 5


def test_resume_skips_searches_already_done_today(tmp_path):
    wl = config.load(FIXTURE)
    wl.request_delay = 0
    calls = []
    counting = lambda s: calls.append(1) or _fake_fetch(s)  # noqa: E731
    collect.collect(wl, tmp_path, today=date(2026, 10, 11), limit=8, fetch=counting, log=lambda *_: None)
    first = len(calls)
    # With resume, --limit counts the searches still to do.
    collect.collect(wl, tmp_path, today=date(2026, 10, 11), limit=4, resume=True, fetch=counting, log=lambda *_: None)
    assert len(calls) - first <= 4 + 4  # 4 new searches (+ at most their positioning)
    obs = pd.read_csv(tmp_path / "observations.csv")
    assert len(obs) == 12 and not obs.duplicated(subset=["series_key", "depart"]).any()


def _windows(wl, out=(date(2027, 5, 1), date(2027, 5, 9), 4), back=(date(2027, 8, 1), date(2027, 8, 7), 6)):
    wl.request_delay = 0
    wl.trips = ("round-trip", "two-one-ways")
    wl.outbound, wl.inbound = config.DateRange(*out), config.DateRange(*back)
    return wl


def test_config_rejects_settings_that_would_be_silently_ignored(tmp_path):
    base = FIXTURE.read_text()
    for extra, message in (
        ('[dates]\nreturn = { from = "2027-08-01", to = "2027-08-25" }\n', "only works together"),
        ('[alerts]\nshanghai = 3800\n', "unknown [alerts] keys"),
    ):
        text = base.replace("[dates]\n", "[dates_unused]\n") if extra.startswith("[dates]") else base.replace("[alerts]\n", "[alerts_unused]\n")
        path = tmp_path / "w.toml"
        path.write_text(text + "\n" + extra)
        try:
            config.load(path)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected an error for {extra!r}")
    try:
        config._range({"from": "2027-05-01", "to": "2027-05-09", "step_days": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("step_days = 0 must be rejected")


def test_round_trip_pairs_rotate_daily_and_only_filters_by_leg():
    wl = _windows(config.load(FIXTURE))
    first = [(j.depart, j.ret) for j in wl.jobs(date(2026, 10, 11)) if j.ret]
    second = [(j.depart, j.ret) for j in wl.jobs(date(2026, 10, 12)) if j.ret]
    assert sorted(first) == sorted(second) and first[0] != second[0]
    assert all(j.leg == "rt" for j in wl.jobs(date(2026, 10, 11)) if j.ret)


def test_nothing_left_to_search_is_not_an_error(tmp_path):
    from flightcheck import report

    wl = _windows(config.load(FIXTURE))
    obs = collect.collect(wl, tmp_path / "d", today=date(2027, 9, 1), fetch=_fake_fetch, log=lambda *_: None)
    assert obs.empty
    collect.collect(wl, tmp_path / "d", today=date(2026, 10, 11), limit=4, fetch=_fake_fetch, log=lambda *_: None)
    pd.read_csv(tmp_path / "d" / "observations.csv").assign(run_date="2027-09-01").to_csv(tmp_path / "d" / "observations.csv", index=False)
    info = report.build(wl, tmp_path / "d", tmp_path / "r")
    assert info["run_date"] is None and "passed" in (tmp_path / "r" / "latest.md").read_text()


def test_overlapping_windows_never_pair_a_return_before_the_outbound(tmp_path):
    from flightcheck import report

    wl = _windows(config.load(FIXTURE), out=(date(2027, 5, 1), date(2027, 5, 20), 1), back=(date(2027, 5, 2), date(2027, 5, 20), 1))

    def fetch(search):
        s = search.slices[0]
        if "IAH" in s.destinations or "IAH" in s.origins and "ATL" in s.destinations:
            return _fake_fetch(search)
        back = s.origins[0] != "ATL" and s.origins[0] != "IAH"
        # The cheapest return is on May 2, before most outbound dates.
        price = 1000 if back and s.date == date(2027, 5, 2) else 3000 + s.date.day * 10
        it = google.Itinerary(price, ["X"], f"{s.origins[0]}-{s.destinations[0]}", ["X1"], 0, 600, None, None)
        return google.Result([it], None)

    data = tmp_path / "data"
    collect.collect(wl, data, today=date(2026, 10, 11), only={"out", "back"}, fetch=fetch, log=lambda *_: None)
    obs, _, _ = report.load_latest_run(data, wl)
    priced = obs.dropna(subset=["total_price"])
    o, b, total = report._two_one_ways(wl, priced[priced["kind"] == "out"], priced[priced["kind"] == "back"], "shanghai")
    assert b["depart"] > o["depart"]


def test_two_one_ways_under_the_round_trip_price_alert(tmp_path):
    from flightcheck import report

    wl = _windows(config.load(FIXTURE))
    wl.alerts = {"round_trip": 99999}
    data, reports = tmp_path / "data", tmp_path / "reports"
    collect.collect(wl, data, today=date(2026, 10, 11), only={"out", "back"}, fetch=_fake_fetch, log=lambda *_: None)
    assert report.build(wl, data, reports)["alerts"]
    assert "两张单程合计" in (reports / "alerts.md").read_text()
    assert not report.build(wl, data, reports)["alerts"]  # same pair again: no repeat


def test_return_positioning_flies_on_the_landing_day(tmp_path):
    wl = _windows(config.load(FIXTURE))
    seen = []

    def fetch(search):
        s = search.slices[0]
        if s.origins == ("IAH",) and s.destinations == ("ATL",):
            seen.append(s.date)
            return google.Result([google.Itinerary(99, ["UA"], "IAH-ATL", ["UA1"], 0, 120, None, None)], None)
        landed = s.date + timedelta(days=1)
        it = google.Itinerary(5000, ["NH"], f"{s.origins[0]}-HND-{s.destinations[0]}", ["NH1"], 1, 900,
                              f"{s.date} 10:00", f"{landed} 08:30")
        return google.Result([it], None)

    collect.collect(wl, tmp_path, today=date(2026, 10, 11), only={"back"}, fetch=fetch, log=lambda *_: None)
    assert seen and all(d.day in (2, 8) for d in seen)  # Aug 1 / Aug 7 departures land Aug 2 / Aug 8
