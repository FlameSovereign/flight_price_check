from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from flightcheck import collect, config, google, playbook, report

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "watchlist.toml"
TODAY = date(2026, 10, 11)
ALLOW = config.PlaybookSettings().allow


def _watchlist(**trip):
    wl = config.load(FIXTURE)
    wl.request_delay = 0
    wl.trips = ("round-trip", "two-one-ways")
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 9), 4)
    wl.inbound = config.DateRange(date(2027, 8, 1), date(2027, 8, 7), 6)
    wl.alerts = {"round_trip": 5000, "one_way": 2800}
    wl.trip = config.TripSettings(**trip)
    return wl


def test_tiers_follow_one_ticket_one_airline():
    assert playbook.tier("KE36 KE851", "ATL-ICN-PEK", ALLOW) == "A1"
    assert playbook.tier("AA1812 CX875 CX346", "ATL-DFW-HKG-SHA", ALLOW) == "A2"  # oneworld, DFW feeder
    assert playbook.tier("AC8504 AC33 AC29", "ATL-YYZ-YVR-PEK", ALLOW) == "A2"  # one airline, Toronto feeder
    assert playbook.tier("AC8502 AC61 KE863", "ATL-YYZ-ICN-PEK", ALLOW) == "B"  # two alliances
    assert playbook.tier("WS1601 CI31 CI517", "ATL-YVR-TPE-PEK", ALLOW) == "B"  # WestJet is in no alliance
    assert playbook.tier("PR127 PR336", "LAX-MNL-PEK", ALLOW) == "B"  # not on the allow list
    assert playbook.tier("KE36 KE851 KE111", "ATL-ICN-PUS-PEK", ALLOW) == "B"  # second stop is not a feeder
    assert playbook.tier("", "", ALLOW) == "B"


def test_phases_count_back_from_departure():
    dep = date(2027, 5, 19)
    cases = {131: "watch", 130: "window", 96: "window", 95: "last_call", 81: "last_call", 80: "fallback", 64: "fallback", 63: "out"}
    for days, expected in cases.items():
        assert playbook.phase(dep, dep - timedelta(days=days)) == expected
    assert playbook.milestones(dep)["fallback"] == date(2027, 2, 28)


def _opt(total, kind="rt", carriers=("KE",), grade="A1", at_floor=None, lower=False):
    return playbook.Option(kind, total, grade, carriers, date(2027, 5, 19), date(2027, 8, 20), "shanghai", "x",
                           at_floor=at_floor, lower_lately=lower)


def test_decision_table():
    pb = config.PlaybookSettings()
    judge = lambda o, ph, pair=None, best=None, cap=None: playbook._judge(o, ph, 5000, pb, cap, pair, best)[0]  # noqa: E731
    assert judge(_opt(4990), "watch") == "buy"  # under the target: any phase
    assert judge(_opt(5400, carriers=("TK", "TK")), "watch") == "buy"  # Turkish line
    assert judge(_opt(5400, carriers=("TK", "TK"), at_floor=False, lower=True), "window") == "buy"
    assert judge(_opt(5400, carriers=("TK", "KE"), at_floor=False, lower=True), "window") == "wait"  # no line for mixed carriers
    assert judge(_opt(5300, at_floor=True, lower=True), "window") == "buy"
    assert judge(_opt(5300, at_floor=False, lower=True), "window") == "wait"  # a lower step was just seen
    assert judge(_opt(5300, at_floor=False, lower=False), "window") == "buy"  # nothing to wait for
    assert judge(_opt(5300, at_floor=None), "watch") == "buy"  # no history: the 24-hour hold is the wait
    assert judge(_opt(5300, at_floor=False, lower=True), "last_call") == "buy"  # from D-95 stop waiting
    assert judge(_opt(5800), "window") == "pass"
    assert judge(_opt(5800), "last_call", pair=6000) == "buy"
    assert judge(_opt(5800), "last_call", pair=5000) == "pass"  # two one-ways are $800 cheaper
    big = _opt(6500)
    assert judge(big, "window") == "pass"
    assert judge(big, "fallback", pair=6400, best=big) == "buy"
    assert judge(big, "fallback", pair=5000, best=big) == "pass"
    pair = _opt(5600, kind="pair")
    assert judge(pair, "window") == "hold"
    assert judge(pair, "fallback", best=big) == "buy"
    assert judge(_opt(4800, kind="pair"), "watch") == "buy"  # two one-ways under the target
    assert judge(_opt(4500, grade="B"), "watch") == "pass"  # tier B never triggers
    assert judge(_opt(5300), "window", cap=5200) == "ask"


def _obs(rows):
    base = {"kind": "rt", "series_key": "atl-any:shanghai", "strategy": "atl-any", "city": "shanghai", "ret": "",
            "url": "", "airlines": "Korean Air", "route": "ATL-ICN-PVG", "flights": "KE36 KE893"}
    df = pd.DataFrame([base | r for r in rows], columns=[*base, "depart", "total_price"])
    df["price"] = df["total_price"]
    return df


EMPTY_HISTORY = pd.DataFrame(columns=["series_key", "depart", "ret", "obs_date", "price", "source"])


def test_nothing_is_bought_before_the_trip_is_committed():
    obs = _obs([{"depart": "2027-05-05", "ret": "2027-08-07", "total_price": 4900.0}])
    d = playbook.decide(obs, EMPTY_HISTORY, _watchlist(committed=False, decide_by=date(2026, 12, 1)), TODAY)
    assert d.options[0].action == "buy" and d.action == "hold"
    assert any("本来会买" in n for n in d.notes) and any("2026-12-01" in n for n in d.notes)

    d = playbook.decide(obs, EMPTY_HISTORY, _watchlist(committed=True), TODAY)
    assert d.action == "buy" and d.headline.startswith("买：$4,900")
    assert [o.total for o in d.buys] == [4900.0]

    d = playbook.decide(obs, EMPTY_HISTORY, _watchlist(committed=True, purchased=True), TODAY)
    assert d.action == "pass" and not d.buys


def test_floor_comes_from_the_round_trip_history():
    days = pd.date_range("2026-09-28", periods=14)
    prices = [5300.0] * 10 + [5000.0, 5400.0, 5400.0, 5400.0]  # a lower step 3 days ago
    hist = pd.DataFrame({"series_key": "atl-any:shanghai", "depart": pd.Timestamp("2027-05-05"), "ret": "2027-08-07",
                         "obs_date": days, "price": prices, "source": "google"})
    obs = _obs([{"depart": "2027-05-05", "ret": "2027-08-07", "total_price": 5400.0}])
    d = playbook.decide(obs, hist, _watchlist(committed=True), TODAY)
    o = d.options[0]
    assert o.at_floor is False and o.lower_lately and o.action == "wait"

    hist.loc[hist.index[-4], "price"] = 5300.0  # without the dip the floor is $5,300, and $5,400 is within one step of it
    d = playbook.decide(obs, hist, _watchlist(committed=True), TODAY)
    assert d.options[0].at_floor is True and d.options[0].action == "buy"


def test_pairs_need_a_tier_a_outbound_and_a_later_return():
    obs = _obs([
        {"kind": "out", "depart": "2027-05-05", "total_price": 2500.0, "flights": "PR127 PR336", "route": "ATL-MNL-PVG"},
        {"kind": "out", "depart": "2027-05-09", "total_price": 2900.0},
        {"kind": "back", "depart": "2027-08-07", "total_price": 2700.0, "flights": "QR726 QR755", "route": "PVG-DOH-ATL"},
        {"kind": "back", "depart": "2027-05-07", "total_price": 900.0},  # before the outbound: never paired with it
    ])
    pair = playbook._best_pair(obs, ALLOW, {})
    assert pair.total == 2900 + 2700 + playbook.PAIR_ALLOWANCE
    assert pair.depart == date(2027, 5, 9) and pair.ret == date(2027, 8, 7)


def test_hand_checked_quotes_count_while_fresh(tmp_path):
    (tmp_path / "manual_quotes.csv").write_text(
        "date,carrier,depart,ret,city,price,stops,source\n"
        "2026-10-10,TK,2027-05-20,2027-08-15,shanghai,5450,1,turkishairlines.com\n"
        "2026-10-01,QR,2027-05-20,2027-08-15,shanghai,4300,1,qatarairways.com\n"  # stale
        "2026-10-11,KE,May 20,2027-08-15,shanghai,4500,1,koreanair.com\n"  # unreadable date
    )
    d = playbook.decide(_obs([]), EMPTY_HISTORY, _watchlist(committed=True), TODAY, playbook.load_quotes(tmp_path))
    assert [(o.kind, o.total, o.action) for o in d.options] == [("manual", 5450.0, "buy")]  # Turkish line
    assert any("1 行读不了" in n for n in d.notes)


def test_checklist_starts_on_its_date():
    wl = _watchlist(committed=True)
    obs = _obs([{"depart": "2027-05-05", "ret": "2027-08-07", "total_price": 7000.0}])
    assert playbook.decide(obs, EMPTY_HISTORY, wl, TODAY).checklist == []
    wl.playbook = config.PlaybookSettings(checks_from=date(2026, 10, 1))
    monday = date(2026, 10, 12)
    lines = playbook.decide(obs, EMPTY_HISTORY, wl, monday).checklist
    assert lines[0].startswith("今天是手查日") and "5/5 去 8/7 回" in lines[0]


def test_decisions_are_logged_once_per_day(tmp_path):
    obs = _obs([{"depart": "2027-05-05", "ret": "2027-08-07", "total_price": 7000.0}])
    for _ in range(2):
        playbook.log_decision(playbook.decide(obs, EMPTY_HISTORY, _watchlist(), TODAY), tmp_path)
    rows = pd.read_csv(tmp_path / "decisions.csv")
    assert len(rows) == 1 and rows.loc[0, "total"] == 7000 and rows.loc[0, "action"] == "pass"


def test_config_reads_trip_and_playbook(tmp_path):
    text = FIXTURE.read_text() + '\n[trip]\ncommitted = true\ncap = 6500\n\n[playbook]\ngood = 5400\nallow = ["tk", "qr"]\n'
    path = tmp_path / "w.toml"
    path.write_text(text)
    wl = config.load(path)
    assert wl.trip.committed and wl.trip.cap == 6500 and wl.playbook.good == 5400 and wl.playbook.allow == ("TK", "QR")
    for bad in ('\n[trip]\nsoon = true\n', '\n[playbook]\ngood = 4000\n'):
        path.write_text(FIXTURE.read_text() + bad)
        try:
            config.load(path)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def _fetch(search):
    s = search.slices[0]
    if s.destinations == ("IAH",) or s.origins == ("ATL",) and s.destinations == ("IAH",):
        return google.Result([google.Itinerary(99, ["United"], "ATL-IAH", ["UA1"], 0, 140, None, None)], None)
    price = 5300 if len(search.slices) == 2 else 2900
    it = google.Itinerary(price, ["Korean Air"], f"{s.origins[0]}-ICN-{s.destinations[0]}", ["KE36", "KE893"], 1, 900, None, None)
    return google.Result([it], None)


def test_report_alerts_on_playbook_buys_only_once_committed(tmp_path):
    for committed, expected in ((False, False), (True, True)):
        wl = _watchlist(committed=committed)
        wl.strategies = [s for s in wl.strategies if s.key == "atl-any"]
        data, reports = tmp_path / str(committed) / "data", tmp_path / str(committed) / "reports"
        collect.collect(wl, data, today=TODAY, fetch=_fetch, log=lambda *_: None)
        report.build(wl, data, reports)
        md = (reports / "latest.md").read_text()
        assert "## 今天该做什么（按你的买法）" in md
        alerts = reports / "alerts.md"
        assert alerts.exists() == expected
        if expected:
            assert "按你的买法：现在该买" in alerts.read_text()
            report.build(wl, data, reports)  # same prices the next run: no repeat
            assert not alerts.exists()
        assert (data / "decisions.csv").exists()
