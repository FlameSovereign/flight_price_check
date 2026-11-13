import json
from datetime import date, timedelta
from pathlib import Path

from flightcheck import collect, config, dashboard, forecast, google, report

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "watchlist.toml"
TODAY = date(2026, 10, 11)


def _watchlist():
    wl = config.load(FIXTURE)
    wl.request_delay = 0
    wl.trips = ("round-trip", "two-one-ways")
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 9), 4)
    wl.inbound = config.DateRange(date(2027, 8, 1), date(2027, 8, 7), 6)
    return wl


def _fetch(search):
    s = search.slices[0]
    if "IAH" in s.destinations:  # positioning leg
        return google.Result([google.Itinerary(99, ["United"], "ATL-IAH", ["UA1"], 0, 140, None, None)], None)
    price = 3000 + (s.date.toordinal() % 7) * 100 + (2500 if len(search.slices) > 1 else 0)
    it = google.Itinerary(price, ["Korean Air"], f"{s.origins[0]}-ICN-{s.destinations[0]}", ["KE1", "KE2"], 1, 900, None, None)
    hist = [(TODAY - timedelta(days=i), price + 10 * i) for i in range(5)]
    return google.Result([it], None if s.connecting else google.Insight(price, price, price - 500, price + 500, 3, hist))


def _past_fares(path):
    path.write_text(
        "year,month_posted,origin,destination,trip,price_usd,airline,travel_window,max_stay,source_url,quote\n"
        "2026,2026-02-09 (deal),Los Angeles (LAX),Beijing (PEK),round-trip,3681,Air China,May 2026,,https://a,q\n"
        "2026,2026-03,US (all origins) [aggregate],Shanghai (PVG),round-trip,5038,various,March,,https://b,q\n"
        "2026,2026-01-14,Boston (BOS),Hong Kong (HKG) [HK],round-trip,4144,Cathay,Spring,,https://c,q\n"
        "2026,2026-02-10,Seattle (SEA),Beijing (PEK),one-way,2100,Hainan,Summer,,https://d,q\n"
        "2024,~late Nov 2024,New York (JFK),Beijing (PEK),round-trip,5204,Air China,Spring,,https://e,q\n"
    )


def test_dashboard_files_follow_the_collected_data(tmp_path, monkeypatch):
    wl = _watchlist()
    data, reports, out = tmp_path / "data", tmp_path / "reports", tmp_path / "dash"
    collect.collect(wl, data, today=TODAY, fetch=_fetch, log=lambda *_: None)
    monkeypatch.setattr(forecast, "N_PATHS", 50)
    report.build(wl, data, reports)
    _past_fares(tmp_path / "past.csv")

    counts = dashboard.build(wl, data, reports, tmp_path / "past.csv", out)
    files = {name: json.loads((out / f"{name}.json").read_text()) for name in counts}
    assert counts == {name: len(rows) for name, rows in files.items()}

    (day,) = files["daily"]
    assert day["day"] == TODAY.isoformat() and day["call"] in dashboard.CALLS.values()
    assert day["pair_bare"] == day["best_out"] + day["best_back"] and day["best_rt"] > day["best_out"]

    order = [s.key for s in wl.strategies]
    for leg, (lo, hi) in (("out_today", ("2027-05-01", "2027-05-09")), ("back_today", ("2027-08-01", "2027-08-07"))):
        rows = files[leg]
        assert rows and all(lo <= r["depart"] <= hi for r in rows)
        # one row per date and strategy, the strategy's place in the watchlist kept for the legend colors
        assert len({(r["depart"], r["strategy"]) for r in rows}) == len(rows)
        assert {r["strategy_idx"] for r in rows} <= set(range(len(order)))
        assert all(r["strategy"] == dashboard.SHORT_NAMES[order[r["strategy_idx"]]] for r in rows)

    assert files["rt_today"] and all(r["depart"] < r["ret"] for r in files["rt_today"])
    history = files["leg_history"]
    assert {r["city"] for r in history} == {"上海", "北京"} and max(r["day"] for r in history) <= TODAY.isoformat()
    assert all(r["city_idx"] == (0 if r["city"] == "上海" else 1) for r in history)

    # only US-origin round trips to mainland China with a posting month
    assert [(r["month"], r["price"]) for r in files["past_fares"]] == [("2026-02", 3681)]
