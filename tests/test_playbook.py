from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flightcheck import collect, config, google, playbook, report

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "watchlist.toml"
TODAY = date(2026, 10, 11)
ALLOW = config.PlaybookSettings().allow


def _watchlist(**trip):
    wl = config.load(FIXTURE)
    wl.request_delay = 0
    wl.trips = ("round-trip", "two-one-ways")
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 31), 2)
    wl.inbound = config.DateRange(date(2027, 8, 1), date(2027, 8, 25), 3)
    wl.alerts = {"round_trip": 5000, "one_way": 2800}
    wl.trip = config.TripSettings(**trip)
    return wl


def _obs(rows):
    base = {"kind": "rt", "series_key": "atl-any:shanghai", "strategy": "atl-any", "city": "shanghai", "ret": "",
            "url": "", "airlines": "Korean Air", "route": "ATL-ICN-PVG", "flights": "KE36 KE893",
            "position_price": np.nan, "trip_depart": ""}
    df = pd.DataFrame([base | r for r in rows], columns=[*base, "depart", "total_price"])
    df["price"] = df["total_price"]
    return df


def _its(rows):
    base = {"series_key": "atl-any:shanghai", "ret": "", "rank": "1", "airlines": "Korean Air",
            "route": "ATL-ICN-PVG", "flights": "KE36 KE893", "run_date": TODAY.isoformat()}
    df = pd.DataFrame([base | r for r in rows])
    df["price"] = df["price"].astype(float)
    return df


def _rt(depart, ret, total, **kw):
    return {"kind": "rt", "depart": depart, "ret": ret, "total_price": float(total)} | kw


def _decide(obs, wl, today=TODAY, its=None, quotes=None, bad=0):
    return playbook.decide(obs, its, wl, today, quotes, {}, bad)


def test_tiers_follow_one_ticket_one_airline():
    assert playbook.tier("KE36 KE851", "ATL-ICN-PEK", ALLOW) == "A1"
    assert playbook.tier("AA1812 CX875 CX346", "ATL-DFW-HKG-SHA", ALLOW) == "A2"  # oneworld, DFW feeder
    assert playbook.tier("AC8504 AC33 AC29", "ATL-YYZ-YVR-PEK", ALLOW) == "A2"  # one airline, Toronto feeder
    assert playbook.tier("UA100 NH11", "ATL-ORD-HND", ALLOW) == "A2"  # one stop, but two airlines: feeder needed
    assert playbook.tier("DL27 KE2051", "ATL-ICN-PEK", ALLOW) == "B"  # same alliance, connection not a feeder
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


def _opt(total, kind="rt", carriers=("KE",), grade="A1", at_floor=None, lower=False, legs=None, back_rose=False):
    return playbook.Option(kind, total, grade, carriers, date(2027, 5, 19), date(2027, 8, 20), "shanghai", "x",
                           at_floor=at_floor, lower_lately=lower, legs=legs, back_rose=back_rose)


def _judge(o, ph, days=200, rt=None, pair=None, cap=None):
    return playbook._judge(o, ph, days, 5000, config.PlaybookSettings(), cap, rt, pair)[0]


def test_decision_table():
    assert _judge(_opt(4990), "watch") == "buy"  # under the target: any phase
    assert _judge(_opt(5400, carriers=("TK", "TK"), at_floor=False, lower=True), "window") == "buy"  # Turkish line
    assert _judge(_opt(5400, carriers=("TK", "KE"), at_floor=False, lower=True), "window") == "wait"  # no line for mixed
    assert _judge(_opt(5300, at_floor=True, lower=True), "window") == "buy"
    assert _judge(_opt(5300, at_floor=False, lower=True), "window") == "wait"  # a lower step was just seen
    assert _judge(_opt(5300, at_floor=False, lower=False), "window") == "buy"  # nothing to wait for
    assert _judge(_opt(5300), "watch") == "buy"  # no history: the 24-hour hold is the wait
    assert _judge(_opt(5300, at_floor=False, lower=True), "last_call") == "buy"  # from D-95 stop waiting
    assert _judge(_opt(5800), "window") == "pass"
    assert _judge(_opt(5800), "last_call", pair=_opt(5600, "pair")) == "buy"
    assert _judge(_opt(5800), "last_call", pair=_opt(5000, "pair")) == "pass"  # two one-ways $800 cheaper
    assert _judge(_opt(4500, grade="B"), "watch") == "pass"  # tier B never triggers


def test_fallback_buys_the_cheaper_of_round_trip_and_pair():
    rt = _opt(6400)
    pair = _opt(6200, "pair", legs=6050)
    cap = 10000
    assert _judge(rt, "fallback", 75, rt=rt, pair=pair, cap=cap) == "pass"
    assert _judge(pair, "fallback", 75, rt=rt, pair=pair, cap=cap) == "buy"  # above `ok` no single-ticket premium applies
    other = _opt(6500)
    assert _judge(other, "fallback", 75, rt=rt, pair=pair, cap=cap) == "pass"  # the date has a cheaper round trip
    pricey_pair = _opt(6600, "pair", legs=6450)
    assert _judge(rt, "fallback", 75, rt=rt, pair=pricey_pair, cap=cap) == "buy"
    assert _judge(pricey_pair, "fallback", 75, rt=rt, pair=pricey_pair, cap=cap) == "pass"
    within_ok = _opt(5900)
    assert _judge(_opt(5700, "pair", legs=5550), "fallback", 75, rt=within_ok, cap=cap) == "pass"  # one ticket worth $200 more


def test_pairs_before_the_fallback():
    assert _judge(_opt(5050, "pair", legs=4900), "watch") == "buy"  # legs under the target; the allowance is not a price
    assert _judge(_opt(5400, "pair", legs=5250, back_rose=True), "window", days=120) == "hold"  # before D-110
    assert _judge(_opt(5400, "pair", legs=5250, back_rose=True), "window", days=109) == "buy"
    assert _judge(_opt(5400, "pair", legs=5250, back_rose=False), "window", days=109) == "hold"


def test_cap_blocks_buys_and_asks_only_at_the_fallback():
    over = _opt(5300)
    assert _judge(over, "window", cap=5200) == "pass"
    assert _judge(over, "fallback", 75, rt=over, cap=5200) == "ask"
    big = _opt(9800)
    assert _judge(big, "fallback", 75, rt=big) == "ask"  # no cap set: a fallback buy above `ok` is your call
    assert _judge(big, "fallback", 75, rt=big, cap=10000) == "buy"
    obs = _obs([_rt("2027-05-21", "2027-08-20", 5800), _rt("2027-05-25", "2027-08-20", 7000, flights="TK32 TK26", route="ATL-IST-PVG")])
    d = _decide(obs, _watchlist(committed=True, cap=6500), today=date(2026, 12, 1))
    assert d.action == "pass" and "$5,800" in d.headline  # the over-cap fare does not take over months early


def test_nothing_is_bought_before_the_trip_is_committed():
    obs = _obs([_rt("2027-05-05", "2027-08-07", 4900)])
    d = _decide(obs, _watchlist(committed=False, decide_by=date(2026, 12, 1)))
    assert d.options[0].action == "buy" and d.action == "hold"
    assert any("本来会" in n for n in d.notes) and any("2026-12-01" in n for n in d.notes)

    d = _decide(obs, _watchlist(committed=True))
    assert d.action == "buy" and d.headline.startswith("买：$4,900")
    assert [o.total for o in d.buys] == [4900.0]
    assert any("cap" in n for n in d.notes)  # committed without a cap: say so

    d = _decide(obs, _watchlist(committed=True, purchased=True))
    assert d.action == "pass" and not d.buys


def test_a_buy_on_a_dearer_date_is_not_lost_among_cheaper_passes():
    today = date(2027, 2, 1)  # May 1 is at D-89, the later dates still before D-95
    rows = [_rt(f"2027-05-{d:02d}", "2027-08-07", 5600 + i * 30) for i, d in enumerate(range(9, 27, 2))]
    rows.append(_rt("2027-05-01", "2027-08-07", 5950))
    d = _decide(_obs(rows), _watchlist(committed=True), today=today)
    assert [(o.depart, o.action) for o in d.buys] == [(date(2027, 5, 1), "buy")]
    assert d.action == "buy" and d.subject.depart == date(2027, 5, 1)
    assert any(o.depart == date(2027, 5, 1) for o in d.shown())


def test_tier_a_fare_behind_a_cheaper_tier_b_one_counts():
    obs = _obs([_rt("2027-05-05", "2027-08-07", 4600, flights="PR127 PR336", route="ATL-MNL-PVG", airlines="Philippine")])
    its = _its([
        {"depart": "2027-05-05", "ret": "2027-08-07", "rank": "1", "price": 4600, "flights": "PR127 PR336", "route": "ATL-MNL-PVG"},
        {"depart": "2027-05-05", "ret": "2027-08-07", "rank": "2", "price": 4900, "flights": "KE36 KE893", "route": "ATL-ICN-PVG"},
    ])
    d = _decide(obs, _watchlist(committed=True), its=its)
    assert [(o.total, o.tier, o.action) for o in d.options] == [(4900.0, "A1", "buy")]


def _days(n, end=TODAY):
    return [(end - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


def test_floor_uses_tier_a_prices_and_calendar_days():
    key = {"depart": "2027-05-05", "ret": "2027-08-07"}
    obs = _obs([_rt("2027-05-05", "2027-08-07", 5400)])
    wl = _watchlist(committed=True)

    prices = [5300] * 10 + [5000, 5400, 5400, 5400]  # a lower step 3 days ago
    its = _its([key | {"run_date": d, "price": p} for d, p in zip(_days(14), prices)])
    o = _decide(obs, wl, its=its).options[0]
    assert o.at_floor is False and o.lower_lately and o.action == "wait"

    prices[10] = 5300  # without the dip the floor is $5,300, and $5,400 is within one step of it
    its = _its([key | {"run_date": d, "price": p} for d, p in zip(_days(14), prices)])
    assert _decide(obs, wl, its=its).options[0].action == "buy"

    # every other day observed: the $5,000 low was 12 days ago, past the 7-day wait
    days = [(TODAY - timedelta(days=2 * (13 - i))).isoformat() for i in range(14)]
    prices = [5300] * 7 + [5000] + [5400] * 6
    its = _its([key | {"run_date": d, "price": p} for d, p in zip(days, prices)])
    o = _decide(obs, wl, its=its).options[0]
    assert not o.lower_lately and o.action == "buy"

    # a cheaper tier-B fare on other days is not a tier-A floor
    its = _its([key | {"run_date": d, "price": 5400} for d in _days(14)]
               + [key | {"run_date": d, "price": 5000, "flights": "B61 MU1", "route": "ATL-JFK-PVG"} for d in _days(13)[:-1]])
    o = _decide(obs, wl, its=its).options[0]
    assert o.at_floor is True and not o.lower_lately


def test_pairs_are_per_outbound_date_and_skip_dates_already_out():
    today = date(2027, 3, 17)  # May 3 is at D-47 (out), May 31 at D-75 (fallback)
    obs = _obs([
        {"kind": "out", "depart": "2027-05-03", "total_price": 2500.0},
        {"kind": "out", "depart": "2027-05-31", "total_price": 3200.0},
        {"kind": "out", "depart": "2027-05-29", "total_price": 2400.0, "flights": "PR127 PR336", "route": "ATL-MNL-PVG"},
        {"kind": "back", "series_key": "atl-any:shanghai:back", "depart": "2027-08-04", "total_price": 2700.0,
         "flights": "QR871 QR755", "route": "PVG-DOH-ATL"},
        _rt("2027-05-31", "2027-08-07", 6500),
    ])
    d = _decide(obs, _watchlist(committed=True, cap=8000), today=today)
    pairs = [o for o in d.options if o.kind == "pair"]
    assert [(o.depart, o.total) for o in pairs] == [(date(2027, 5, 31), 3200 + 2700 + playbook.PAIR_ALLOWANCE)]
    assert {o.kind: o.action for o in d.options} == {"pair": "buy", "rt": "pass"}


def test_pairs_add_the_transfer_when_cities_differ():
    obs = _obs([
        {"kind": "out", "depart": "2027-05-11", "total_price": 3000.0},
        {"kind": "back", "series_key": "atl-any:beijing:back", "city": "beijing", "depart": "2027-08-04",
         "total_price": 2700.0, "flights": "QR871 QR755", "route": "PKX-DOH-ATL"},
    ])
    (pair,) = playbook._pairs(obs, playbook._tier_a_daily(None, ALLOW), None, _watchlist(), TODAY, {})
    assert pair.legs == 3000 + 2700 + playbook.TRANSFER_COST and "换城市" in pair.desc


def test_rolling_grid_pairs_only_with_their_own_return():
    wl = _watchlist()
    wl.outbound = wl.inbound = None
    obs = _obs([
        {"kind": "out", "depart": "2027-01-10", "total_price": 2300.0},
        {"kind": "back", "series_key": "atl-any:shanghai:back", "depart": "2027-01-12", "trip_depart": "2026-11-28", "total_price": 2400.0},
        {"kind": "back", "series_key": "atl-any:shanghai:back", "depart": "2027-02-24", "trip_depart": "2027-01-10", "total_price": 2900.0},
    ])
    (pair,) = playbook._pairs(obs, playbook._tier_a_daily(None, ALLOW), None, wl, TODAY, {})
    assert pair.ret == date(2027, 2, 24)


def test_hand_checked_quotes(tmp_path):
    (tmp_path / "manual_quotes.csv").write_text(
        "date,carrier,depart,ret,city,price,stops,via,source\n"
        "2026-10-10,TK,2027-05-20,2027-08-15,shanghai,5450,1,,turkishairlines.com\n"
        "2026-10-11,AA+CX,2027-05-20,2027-08-16,shanghai,4800,2,DFW,cathaypacific.com\n"
        "2026-10-11,NH,2027-05-20,2027-08-17,shanghai,4850,2,,ana.co.jp\n"
        "2026-10-11,DL+KE,2027-05-20,2027-08-18,shanghai,4700,1,ICN,koreanair.com\n"
        "2026-10-01,QR,2027-05-20,2027-08-15,shanghai,4300,1,,qatarairways.com\n"  # stale
        "2026-10-11,KE,May 20,2027-08-15,shanghai,4500,1,,koreanair.com\n"  # unreadable date
        "2026-10-11,QR,2027-05-20,2027-08-15,shanghai,5,300,1,,qatarairways.com\n"  # thousands separator
    )
    quotes, bad = playbook.load_quotes(tmp_path)
    d = _decide(_obs([]), _watchlist(committed=True), quotes=quotes, bad=bad)
    got = {(o.total, o.tier, o.action) for o in d.options}
    assert got == {(5450.0, "A1", "buy"), (4800.0, "A2", "buy"), (4850.0, "A2", "buy"), (4700.0, "B", "pass")}
    assert any("2 行读不了" in n for n in d.notes)


def test_checklist_names_three_outbound_dates_from_its_start():
    wl = _watchlist(committed=True)
    obs = _obs([{"kind": "out", "depart": f"2027-05-{d:02d}", "total_price": 3000.0 + d} for d in (5, 9, 13, 17)]
               + [{"kind": "back", "series_key": "atl-any:shanghai:back", "depart": "2027-08-07", "total_price": 2700.0,
                   "flights": "QR871 QR755", "route": "PVG-DOH-ATL"}])
    assert _decide(obs, wl).checklist == []
    wl.playbook = config.PlaybookSettings(checks_from=date(2026, 10, 1))
    lines = _decide(obs, wl, today=date(2026, 10, 12)).checklist  # a Monday
    assert lines[0].startswith("今天是手查日") and "5/5 去 8/7 回、5/9 去 8/7 回、5/13 去 8/7 回" in lines[0]


def test_after_the_last_date_leaves_the_call_is_yours():
    obs = _obs([_rt("2027-05-21", "2027-08-20", 6200)])
    wl = _watchlist(committed=True)
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 21), 4)
    d = _decide(obs, wl, today=date(2027, 4, 1))
    assert d.action == "ask" and "$6,200" in d.headline and "你来拍板" in d.headline


def test_decisions_log_the_recommended_option_once_per_day(tmp_path):
    today = date(2027, 2, 1)
    obs = _obs([_rt("2027-05-21", "2027-08-20", 5700), _rt("2027-05-01", "2027-08-07", 5900)])
    for _ in range(2):
        playbook.log_decision(_decide(obs, _watchlist(committed=True), today=today), tmp_path)
    rows = pd.read_csv(tmp_path / "decisions.csv")
    assert len(rows) == 1 and rows.loc[0, "action"] == "buy" and rows.loc[0, "total"] == 5900


def test_config_reads_trip_and_playbook(tmp_path):
    path = tmp_path / "w.toml"
    path.write_text(FIXTURE.read_text() + '\n[trip]\ncommitted = true\ncap = 6500\n\n[playbook]\ngood = 5400\nallow = ["tk", "qr"]\n')
    wl = config.load(path)
    assert wl.trip.committed and wl.trip.cap == 6500 and wl.playbook.good == 5400 and wl.playbook.allow == ("TK", "QR")

    path.write_text(FIXTURE.read_text().replace("round_trip = 5000", "round_trip = 6000"))
    wl = config.load(path)  # no [playbook]: the default lines follow a higher target
    assert (wl.playbook.good, wl.playbook.ok) == (6000, 6000)

    for bad in ('\n[trip]\nsoon = true\n', '\n[trip]\ncommitted = "false"\n', '\n[playbook]\ngood = 4000\n',
                '\n[playbook]\ngood = 6500\nok = 6000\n'):
        path.write_text(FIXTURE.read_text() + bad)
        with pytest.raises(ValueError):
            config.load(path)


def _fetch(price_rt):
    def fetch(search):
        s = search.slices[0]
        price = price_rt if len(search.slices) == 2 else 2900
        it = google.Itinerary(price, ["Korean Air"], f"{s.origins[0]}-ICN-{s.destinations[0]}", ["KE36", "KE893"], 1, 900, None, None)
        return google.Result([it], None)
    return fetch


def _small(wl):
    wl.strategies = [s for s in wl.strategies if s.key == "atl-any"]
    wl.destinations = {"shanghai": wl.destinations["shanghai"]}
    wl.outbound = config.DateRange(date(2027, 5, 1), date(2027, 5, 9), 4)
    wl.inbound = config.DateRange(date(2027, 8, 1), date(2027, 8, 7), 6)
    return wl


def test_alerts_follow_the_trip_state(tmp_path):
    def run(name, price, **trip):
        wl = _small(_watchlist(**trip))
        data, reports = tmp_path / name / "data", tmp_path / name / "reports"
        collect.collect(wl, data, today=TODAY, fetch=_fetch(price), log=lambda *_: None)
        report.build(wl, data, reports)
        alerts = reports / "alerts.md"
        return wl, data, reports, (alerts.read_text() if alerts.exists() else "")

    # not committed: plain price alerts, no playbook calls
    _, _, reports, text = run("prices", 4800)
    assert "往返票" in text and "按你的买法" not in text
    assert "## 今天该做什么（按你的买法）" in (reports / "latest.md").read_text()

    # committed: playbook buys only (the $4,800 fare is not listed twice), with their own header
    wl, data, reports, text = run("live", 5300, committed=True)
    assert "按你的买法：现在该买" in text and "### 往返票" not in text and "跌破提醒价" not in text
    report.build(wl, data, reports)  # same prices on the next run: nothing new
    assert not (reports / "alerts.md").exists()
    assert (data / "decisions.csv").exists()

    # purchased: nothing at all
    _, _, _, text = run("done", 4800, committed=True, purchased=True)
    assert text == ""


def test_a_buy_that_goes_away_and_comes_back_alerts_again(tmp_path):
    wl = _small(_watchlist(committed=True))
    state = tmp_path / "alerted.csv"
    obs = _obs([_rt("2027-05-05", "2027-08-07", 5300)])
    for price, expected in ((5300, True), (6500, False), (5300, True)):
        obs["total_price"] = obs["price"] = float(price)
        d = _decide(obs, wl)
        assert bool(report._alerts(obs, wl, {}, state, TODAY, d)) == expected


def test_price_alerts_skip_tier_b(tmp_path):
    wl = _small(_watchlist())
    obs = _obs([_rt("2027-05-05", "2027-08-07", 4600, flights="PR127 PR336", route="ATL-MNL-PVG")])
    assert report._alerts(obs, wl, {}, tmp_path / "a.csv", TODAY, _decide(obs, wl)) == ""
