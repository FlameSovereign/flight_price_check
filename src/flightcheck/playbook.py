"""Your buying habits as rules: one action per day for the trip.

The rules come from the traveller's own bookings and Google Flights price
tracking (kept out of the repo), checked against the tracker's data and past
public fares (reference/past_fares.csv):

- Judge the target only against one ticket on one airline (tier A1), or one
  alliance where the extra stop is a North American feeder (A2), on an allowed
  long-haul carrier. Everything else (tier B) is context, never a trigger.
  Each search keeps its top itineraries, so a tier-A fare counts even when a
  cheaper tier-B one tops the same search.
- Count back from each candidate departure date D: the main buying window opens
  at D-130; from D-95 stop waiting for old lows; at D-80 compare the round trip
  with two one-way tickets for the same outbound date and buy the cheaper; at
  D-63 the date leaves the system. When every date has left, the call is yours.
- One decision table on the all-in price (positioning included):
    <= target (alerts.round_trip)  buy the same day, any phase
    <= a carrier line (TK 5,500)   buy the same day (those fares barely move)
    <= good                        before D-95: buy at the 14-day tier-A floor, otherwise
                                   wait up to WAIT_DAYS for it to return; from D-95: buy
    <= ok                          from D-95: buy unless that date's two one-ways are
                                   cheaper by more than SINGLE_TICKET_PREMIUM
    more                           at D-80: buy the cheaper of the round trip and two one-ways
  Two one-ways (outbound tier A, plus a PAIR_ALLOWANCE for the separate-ticket
  risk and a TRANSFER_COST when they use different cities) are bought before
  D-80 only when the legs come to the target or less, or, from D-110, to
  EARLY_PAIR or less after the return has risen a step.
  Nothing over [trip] cap is bought; at D-80 it becomes a question for you, and
  so does any fallback buy above `ok` while no cap is set.
- Nothing is "buy" until the trip is committed ([trip] committed = true); until
  then the report shows what it would have done.
- From [playbook] checks_from, hand-check airline sites on Mondays and
  Thursdays: Google Flights often shows no long-stay round trips, and
  data/manual_quotes.csv feeds those quotes into the same table.
- Every day's call and the option it is about are logged to data/decisions.csv
  so the lines can be re-fitted once real spring data exists.

Buying a US-origin ticket directly from the airline at least 7 days before
departure comes with a free 24-hour cancellation or hold (US DOT), so a "buy"
here means "book today, recheck tomorrow".
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import Watchlist

# Days before departure (counted from each candidate outbound date).
WINDOW, EARLY_CHECK, LAST_CALL, FALLBACK, OUT = 130, 110, 95, 80, 63
FLOOR_OBS = 14  # the floor is the lowest tier-A price of the last 14 observed days
STEP = 0.03  # moves under 3% are the same fare step
WAIT_DAYS = 7  # a lower step seen within this many days may come back
PAIR_ALLOWANCE = 150  # two separate tickets carry misconnect risk
TRANSFER_COST = 150  # landing in one city and flying home from another (train, maybe a night)
EARLY_PAIR = 5300  # from D-110, two one-ways up to this once the return has risen a step
SINGLE_TICKET_PREMIUM = 300  # worth paying for one ticket over two
QUOTE_DAYS = 2  # hand-checked quotes older than this are stale
HISTORY_DAYS = 45  # itinerary files read for floors
SHOW_A, SHOW_B = 8, 3  # rows in the report table besides every buy / wait / ask
CHECKLIST_SITES = (
    ("土耳其航空", "https://www.turkishairlines.com/"),
    ("卡塔尔航空", "https://www.qatarairways.com/"),
    ("大韩航空", "https://www.koreanair.com/"),
    ("达美", "https://www.delta.com/"),
    ("美联航", "https://www.united.com/"),
    ("加航", "https://www.aircanada.com/"),
    ("国泰", "https://www.cathaypacific.com/"),
    ("全日空（美国站）", "https://www.ana.co.jp/en/us/"),
    ("日航", "https://www.jal.co.jp/en/"),
    ("Kayak", "https://www.kayak.com/flights"),
)
ALLIANCES = {
    "star": {"A3", "AC", "AI", "AV", "BR", "CA", "CM", "ET", "LH", "LO", "LX", "MS", "NH", "NZ", "OS", "OZ", "SA", "SQ", "TK", "TP", "UA", "ZH"},
    "oneworld": {"AA", "AS", "AY", "BA", "CX", "FJ", "IB", "JL", "MH", "QF", "QR", "RJ", "UL", "WY"},
    "skyteam": {"AF", "AM", "AR", "CI", "DL", "GA", "KE", "KL", "KQ", "ME", "MF", "MU", "OK", "RO", "SK", "SV", "UX", "VN"},
}
NA_HUBS = {"ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAD", "IAH", "JFK", "LAX", "MIA", "MSP", "ORD", "PHL", "SEA", "SFO", "YUL", "YVR", "YYZ"}
CITY_OF = {"PVG": "SHA", "SHA": "SHA", "PEK": "BJS", "PKX": "BJS"}  # airports of one city
FLIGHT = re.compile(r"^([A-Z0-9]{2})\d+[A-Z]?$")
PHASES = {"watch": "观望", "window": "主窗口", "last_call": "最后一档", "fallback": "兜底", "out": "已出局"}
ACTIONS = {"buy": "买", "wait": "等", "pass": "不买", "ask": "问你", "hold": "先别动"}
KEY = ["series_key", "depart", "ret"]


@dataclass
class Option:
    """One way to buy the whole trip, all-in."""

    kind: str  # "rt" (one round-trip ticket), "manual" (hand-checked round trip) or "pair" (two one-ways)
    total: float  # all-in, incl. positioning (and for pairs the allowance and any transfer)
    tier: str  # "A1", "A2" or "B"
    carriers: tuple[str, ...]
    depart: date
    ret: date | None
    city: str
    desc: str
    url: str = ""
    at_floor: bool | None = None  # within one step of its 14-day tier-A floor (None: not enough history)
    lower_lately: bool = False  # a lower tier-A step was seen within WAIT_DAYS
    legs: float | None = None  # pairs: outbound + return + positioning + transfer, without the allowance
    back_rose: bool = False  # pairs: the return is a step above its own recent floor
    action: str = ""
    reason: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        """Alert-state key: the same fare found by several city searches shares it."""
        return (f"playbook:{self.kind}:{'/'.join(self.carriers)}", self.depart.isoformat(), self.ret.isoformat() if self.ret else "")

    @property
    def same_ticket(self) -> tuple:
        """Identity of the fare regardless of which city search found it."""
        return self.key + (round(self.total),)


@dataclass
class Decision:
    as_of: date
    action: str  # overall: buy / wait / pass / ask / hold
    headline: str
    options: list[Option] = field(default_factory=list)  # every judged option, cheapest first
    milestones: list[tuple[date, dict[str, date], str]] = field(default_factory=list)  # (outbound date, milestones, phase)
    checklist: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    subject: Option | None = None  # the option the overall action is about

    @property
    def buys(self) -> list[Option]:
        """Options to buy today, one per fare (the same fare can come from several city searches)."""
        seen, out = set(), []
        for o in self.options:
            if o.action == "buy" and o.same_ticket not in seen:
                seen.add(o.same_ticket)
                out.append(o)
        return out

    def shown(self) -> list[Option]:
        """Rows for the report: every buy / wait / ask, then the cheapest tier-A and tier-B options."""
        rows = [o for o in self.options if o.action in ("buy", "wait", "ask")]
        rows += [o for o in self.options if o.tier != "B" and o not in rows][:SHOW_A]
        rows += [o for o in self.options if o.tier == "B" and o not in rows][:SHOW_B]
        return sorted(rows, key=lambda o: o.total)


def carriers_of(flights: str) -> tuple[str, ...]:
    """Airline codes in flight order, e.g. "AA2403 CX875 CX334" -> ("AA", "CX", "CX")."""
    return tuple(m.group(1) for f in str(flights).split() if (m := FLIGHT.match(f)))


def grade(codes: tuple[str, ...], stops: int, first_stop: str | None, allow) -> str:
    """A1: one airline, at most one stop. A2: one airline or alliance whose first connection is a
    North American feeder, at most two stops. The long-haul carriers must be allowed; anything else is B."""
    if not codes:
        return "B"
    one_airline = len(set(codes)) == 1
    if one_airline and stops <= 1:
        return "A1" if codes[0] in allow else "B"
    one_alliance = one_airline or any(set(codes) <= members for members in ALLIANCES.values())
    long_haul = codes[1:] if len(codes) > 1 else codes
    if one_alliance and stops <= 2 and first_stop in NA_HUBS and set(long_haul) <= set(allow):
        return "A2"
    return "B"


def tier(flights: str, route: str, allow) -> str:
    codes = carriers_of(flights)
    stops = str(route).split("-")[1:-1]
    return grade(codes, len(codes) - 1, stops[0] if stops else None, allow)


def phase(depart: date, today: date) -> str:
    days = (depart - today).days
    if days <= OUT:
        return "out"
    if days <= FALLBACK:
        return "fallback"
    if days <= LAST_CALL:
        return "last_call"
    if days <= WINDOW:
        return "window"
    return "watch"


def milestones(depart: date) -> dict[str, date]:
    return {name: depart - timedelta(days=d) for name, d in
            (("window", WINDOW), ("last_call", LAST_CALL), ("fallback", FALLBACK), ("out", OUT))}


def load_itineraries(data_dir: Path, today: date) -> pd.DataFrame | None:
    """The top itineraries of every search from the last HISTORY_DAYS days of runs."""
    files = []
    for path in sorted((Path(data_dir) / "itineraries").glob("*.csv.gz")):
        try:
            day = date.fromisoformat(path.name.split(".")[0])
        except ValueError:
            continue
        if today - timedelta(days=HISTORY_DAYS) <= day <= today:
            files.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    if not files:
        return None
    its = pd.concat(files, ignore_index=True)
    its["price"] = pd.to_numeric(its["price"], errors="coerce")
    return its.dropna(subset=["price"])


def load_quotes(data_dir: Path) -> tuple[pd.DataFrame | None, int]:
    """Hand-checked round-trip quotes from data/manual_quotes.csv, and how many lines could not be read.

    Columns: date, carrier, depart, ret, city, price[, stops, via, source, note]. `carrier` is the airline
    code or codes ("TK", "AA+CX"); `via` is the first connection (e.g. DFW), needed for a two-airline quote
    to count as tier A2; a two-stop quote without `via` is assumed to start with a North American feeder.
    """
    path = Path(data_dir) / "manual_quotes.csv"
    if not path.exists():
        return None, 0
    with open(path, newline="", encoding="utf-8-sig") as fh:
        lines = [row for row in csv.reader(fh) if any(cell.strip() for cell in row)]
    if not lines:
        return None, 0
    header = [h.strip() for h in lines[0]]
    good = [row for row in lines[1:] if len(row) == len(header)]  # e.g. "5,300" unquoted adds a field
    return pd.DataFrame(good, columns=header), len(lines) - 1 - len(good)


def decide(obs: pd.DataFrame, itineraries: pd.DataFrame | None, watchlist: Watchlist, today: date,
           quotes: pd.DataFrame | None = None, names: dict | None = None, bad_quotes: int = 0) -> Decision:
    """Today's action for the trip from the latest observations (report.load_latest_run rows) and the
    recent top itineraries of every search (load_itineraries)."""
    names = names or {}
    trip, pb = watchlist.trip, watchlist.playbook
    target = watchlist.alerts.get("round_trip")
    daily = _tier_a_daily(itineraries, pb.allow)
    manual, bad = _quotes(quotes, today, pb.allow)
    bad += bad_quotes
    every = _round_trips(obs, daily, today, pb.allow, names) + manual + _pairs(obs, daily, itineraries, watchlist, today, names)
    every.sort(key=lambda o: o.total)
    options = [o for o in every if phase(o.depart, today) != "out"]

    best_rt, best_pair = {}, {}
    for o in options:
        if o.tier == "B":
            continue
        group = best_pair if o.kind == "pair" else best_rt
        group.setdefault(o.depart, o)  # cheapest first
    for o in options:
        ph = phase(o.depart, today)
        o.action, o.reason = _judge(o, ph, (o.depart - today).days, target, pb, trip.cap,
                                    best_rt.get(o.depart), best_pair.get(o.depart))

    dates = _candidate_dates(watchlist, obs, today)
    marks = [(d, milestones(d), phase(d, today)) for d in dates]
    d = Decision(today, "pass", "", options, marks, _checklist(watchlist, options, today))
    _summarise(d, trip, target, every)
    if bad:
        d.notes.append(f"data/manual_quotes.csv 里有 {bad} 行读不了（日期写成 2027-05-19 这种格式，价格只写数字、不加逗号），已跳过。")
    return d


def _tier_a_daily(its: pd.DataFrame | None, allow) -> pd.DataFrame:
    """The cheapest tier-A itinerary of every search on every run day."""
    cols = KEY + ["run_date", "price", "flights", "route", "airlines", "tier"]
    if its is None or its.empty:
        return pd.DataFrame(columns=cols)
    its = its.assign(tier=[tier(f, r, allow) for f, r in zip(its["flights"], its["route"])])
    a = its[its["tier"] != "B"]
    if a.empty:
        return pd.DataFrame(columns=cols)
    return a.loc[a.groupby(KEY + ["run_date"])["price"].idxmin(), cols].reset_index(drop=True)


def _floor_flags(series: pd.DataFrame, price: float, today: date) -> tuple[bool | None, bool]:
    """(at the 14-day floor, a lower step seen within WAIT_DAYS) from one search's daily prices."""
    series = series.sort_values("run_date")
    if len(series) < FLOOR_OBS:
        return None, False
    at_floor = bool(price <= series["price"].tail(FLOOR_OBS).min() * (1 + STEP))
    recent = series[series["run_date"] >= (today - timedelta(days=WAIT_DAYS)).isoformat()]
    return at_floor, bool((recent["price"] < price / (1 + STEP)).any())


def _today_rows(obs: pd.DataFrame, daily: pd.DataFrame, kind: str, allow, today: date) -> pd.DataFrame:
    """Today's tier-A fare of each search of a kind, joined with its observation (positioning, city, url).

    Uses the stored top itineraries when there are any for today; otherwise the observation row itself."""
    rows = obs[obs["kind"] == kind]
    rows = rows[rows["total_price"].notna()]  # unpriced, or the positioning ticket could not be priced
    if rows.empty:
        return rows
    todays = daily[daily["run_date"] == today.isoformat()] if not daily.empty else daily
    if todays.empty:
        flagged = rows.assign(tier=[tier(f, r, allow) for f, r in zip(rows["flights"], rows["route"])])
        return flagged[flagged["tier"] != "B"]
    joined = rows.drop(columns=["price", "flights", "route", "airlines"]).merge(todays.drop(columns="run_date"), on=KEY)
    joined["total_price"] = joined["price"] + pd.to_numeric(joined["position_price"], errors="coerce").fillna(0)
    return joined


def _round_trips(obs, daily, today, allow, names) -> list[Option]:
    rows = _today_rows(obs, daily, "rt", allow, today)
    history = {k: g for k, g in daily.groupby(KEY)} if not daily.empty else {}
    out = []
    for _, r in rows.iterrows():
        at_floor, lower = _floor_flags(history[(r["series_key"], r["depart"], r["ret"])], r["price"], today) \
            if (r["series_key"], r["depart"], r["ret"]) in history else (None, False)
        out.append(Option(
            "rt", float(r["total_price"]), r["tier"], carriers_of(r["flights"]),
            date.fromisoformat(r["depart"]), date.fromisoformat(r["ret"]), r["city"],
            f"{names.get(r['strategy'], r['strategy'])} · {r['route']}（{r['airlines']}）", str(r.get("url", "")),
            at_floor, lower,
        ))
    return out


def _quotes(quotes, today, allow) -> tuple[list[Option], int]:
    """Fresh hand-checked quotes as options, and how many rows could not be read."""
    if quotes is None or quotes.empty:
        return [], 0
    out, bad = [], 0
    for _, q in quotes.iterrows():
        try:
            seen = date.fromisoformat(q["date"])
            codes = tuple(c.strip().upper() for c in q["carrier"].replace("+", " ").split() if c.strip())
            stops = int(q.get("stops") or 1)
            via = (q.get("via") or "").strip().upper() or ("ATL" if stops == 2 else None)  # assume a feeder
            opt = Option("manual", float(q["price"]), grade(codes, stops, via, allow), codes,
                         date.fromisoformat(q["depart"]), date.fromisoformat(q["ret"]), q.get("city", ""),
                         f"{q.get('source') or '手查'} · {'+'.join(codes)}（{seen} 手查）")
        except (KeyError, ValueError, IndexError, AttributeError):
            bad += 1
            continue
        if 0 <= (today - seen).days <= QUOTE_DAYS:
            out.append(opt)
    return out, bad


def _pairs(obs, daily, its, watchlist, today, names) -> list[Option]:
    """Best outbound + return one-way pair for each outbound date still in the system.

    Outbound legs must be tier A; the return is the cheapest of its search. Fixed windows pair
    an outbound with any later return; the rolling grid with the return searched for it."""
    allow = watchlist.playbook.allow
    outs = _today_rows(obs, daily, "out", allow, today)
    backs = obs[(obs["kind"] == "back") & obs["total_price"].notna()]
    if outs.empty or backs.empty:
        return []
    outs = outs[[phase(date.fromisoformat(d), today) != "out" for d in outs["depart"]]]
    back_days = _daily_any(its)
    best: dict[date, Option] = {}
    for _, o in outs.sort_values("total_price").iterrows():
        if watchlist.outbound is not None:
            later = backs[backs["depart"] > o["depart"]]
        else:
            later = backs[backs["trip_depart"] == o["depart"]]
        if later.empty:
            continue
        b = later.loc[later["total_price"].idxmin()]
        transfer = TRANSFER_COST if _city(o["route"].split("-")[-1]) != _city(b["route"].split("-")[0]) else 0
        legs = float(o["total_price"] + b["total_price"] + transfer)
        out_day = date.fromisoformat(o["depart"])
        if out_day in best and best[out_day].legs <= legs:
            continue
        rose = False
        key = (b["series_key"], b["depart"], "")
        if key in back_days:
            floor = back_days[key].sort_values("run_date")["price"].tail(FLOOR_OBS)
            rose = len(floor) >= FLOOR_OBS and b["price"] > floor.min() * (1 + STEP)
        city = o["city"] if o["city"] == b["city"] else f"{o['city']}/{b['city']}"
        desc = (f"去 {o['route']}（{o['airlines']}）${int(o['total_price']):,} + 回 {b['route']}（{b['airlines']}）"
                f"${int(b['total_price']):,}" + (f" + 换城市 ${transfer}" if transfer else "") + f" + 分票风险 ${PAIR_ALLOWANCE}")
        best[out_day] = Option("pair", legs + PAIR_ALLOWANCE, o["tier"], carriers_of(o["flights"]) + carriers_of(b["flights"]),
                               out_day, date.fromisoformat(b["depart"]), city, desc, legs=legs, back_rose=bool(rose))
    return list(best.values())


def _daily_any(its: pd.DataFrame | None) -> dict:
    """Cheapest itinerary of each search per run day, any tier (for one-way returns)."""
    if its is None or its.empty:
        return {}
    daily = its.groupby(KEY + ["run_date"], as_index=False)["price"].min()
    return {k: g for k, g in daily.groupby(KEY)}


def _city(airport: str) -> str:
    return CITY_OF.get(airport, airport)


def _judge(o: Option, ph: str, days_out: int, target, pb, cap, best_rt: Option | None,
           best_pair: Option | None) -> tuple[str, str]:
    action, reason = _table(o, ph, days_out, target, pb, best_rt, best_pair)
    if action in ("buy", "wait"):
        if cap and o.total > cap:
            if ph == "fallback":
                return "ask", f"按规则该买，但超过你的封顶价 ${int(cap):,}：你来拍板（D-63 前）"
            return "pass", f"超过你的封顶价 ${int(cap):,}，不买"
        if not cap and ph == "fallback" and o.total > pb.ok:
            return "ask", "到了兜底点（D-80），但 [trip] cap 还没定：你来拍板，定了 cap 就按它判断"
    return action, reason


def _table(o: Option, ph, days_out, target, pb, best_rt, best_pair) -> tuple[str, str]:
    p = o.total
    if o.tier == "B":
        return "pass", "不是一张票一家航司（或不在白名单），只作参考"
    if o.kind == "pair":
        if target and o.legs <= target:
            return "buy", f"两张单程合计 ≤ 目标 ${int(target):,}：当天买"
        if days_out <= EARLY_CHECK and o.back_rose and o.legs <= EARLY_PAIR:
            return "buy", f"过了 D-110，回程已经涨了一档，两张单程合计 ≤ ${EARLY_PAIR:,}：先买"
        if ph == "fallback":
            if best_rt is None:
                return "buy", "到了兜底点（D-80），这个日期没有 A 类往返：买两张单程"
            premium = SINGLE_TICKET_PREMIUM if best_rt.total <= pb.ok else 0
            if p < best_rt.total - premium:
                return "buy", f"到了兜底点（D-80），比同日期最便宜的往返 ${int(best_rt.total):,} 便宜：买两张单程"
            return "pass", "到了兜底点（D-80），同日期的往返更划算"
        return "hold", "两张单程只在兜底点（D-80）才考虑，之前先别动"
    if target and p <= target:
        return "buy", f"≤ 目标 ${int(target):,}：当天买"
    line = pb.carrier_lines.get(o.carriers[0]) if len(set(o.carriers)) == 1 else None
    if line and p <= line:
        return "buy", f"≤ 这家航司的线 ${int(line):,}（它的价很少回落）：当天买"
    early = ph in ("watch", "window")
    pair_total = best_pair.total if best_pair is not None else None
    if p <= pb.good:
        if not early:
            return "buy", f"≤ ${int(pb.good):,}，已过 D-95：买"
        if o.at_floor is None:
            return "buy", f"≤ ${int(pb.good):,}，还没有 14 天价格历史，用 24 小时取消代替“等”：买"
        if o.at_floor or not o.lower_lately:
            return "buy", f"≤ ${int(pb.good):,}，在 14 天最低档：买"
        return "wait", f"≤ ${int(pb.good):,}，但 {WAIT_DAYS} 天内见过更低一档，最多等 {WAIT_DAYS} 天，回不来就买"
    if p <= pb.ok:
        if early:
            return "pass", f"${int(pb.good):,}–{int(pb.ok):,} 之间，D-95 之前不买"
        if pair_total is None or p <= pair_total + SINGLE_TICKET_PREMIUM:
            return "buy", f"已过 D-95，比同日期两张单程贵不到 ${SINGLE_TICKET_PREMIUM}：买"
        return "pass", "同日期两张单程明显更便宜，等兜底点（D-80）比较"
    if ph == "fallback":
        if best_rt is not o:
            return "pass", "同日期有更便宜的往返"
        if pair_total is None or p <= pair_total:
            return "buy", "到了兜底点（D-80），这是这个日期最便宜的整趟走法：买"
        return "pass", "到了兜底点（D-80），同日期两张单程更便宜"
    return "pass", f"> ${int(pb.ok):,}，兜底点（D-80）之前不买"


def _candidate_dates(watchlist: Watchlist, obs: pd.DataFrame, today: date) -> list[date]:
    if watchlist.outbound is not None:
        return watchlist.outbound.dates(today)
    days = obs.loc[obs["kind"].isin(["rt", "out"]), "depart"].unique()
    return sorted(date.fromisoformat(d) for d in days if date.fromisoformat(d) > today)


def _checklist(watchlist: Watchlist, options: list[Option], today: date) -> list[str]:
    """Twice-weekly hand checks for the three cheapest tier-A date pairs, one per outbound date."""
    start = watchlist.playbook.checks_from
    if start is None or today < start:
        return []
    picks: list[tuple[date, date]] = []
    for o in options:
        if o.tier != "B" and o.ret and o.depart not in {d for d, _ in picks}:
            picks.append((o.depart, o.ret))
        if len(picks) == 3:
            break
    when = "、".join(f"{d.month}/{d.day} 去 {r.month}/{r.day} 回" for d, r in picks) or "5 月去、8 月回的几组日期"
    sites = "、".join(f"[{n}]({u})" for n, u in CHECKLIST_SITES)
    day = "今天是手查日。" if today.weekday() in (0, 3) else "每周一、周四手查。"
    return [
        f"{day}按 {when} 查一张票的往返：{sites}",
        "顺手看票规：最长停留、提前购票、改期费。看到的价记进 `data/manual_quotes.csv`"
        "（date,carrier,depart,ret,city,price,stops,via,source；价格只写数字）。",
    ]


def _summarise(d: Decision, trip, target, every: list[Option]) -> None:
    if trip.purchased:
        d.action, d.headline = "pass", "已经买好了：不再给建议，也不再提醒。"
        for o in d.options:
            o.action, o.reason = "pass", "已买"
        return
    tier_a = [o for o in d.options if o.tier != "B"]
    if d.milestones and all(ph == "out" for _, _, ph in d.milestones):
        left = [o for o in every if o.tier != "B"]
        d.action, d.subject = "ask", left[0] if left else None
        what = f"最便宜的 A 类走法是 ${int(left[0].total):,}（{_option_line(left[0])}）" if left else "眼下没有 A 类走法"
        cap = f"，你的封顶价 ${int(trip.cap):,}" if trip.cap else "，封顶价还没定"
        d.headline = f"所有候选出发日都过了 D-63，自动窗口关了：{what}{cap}。买、换日期还是不去，你来拍板。"
        return
    for wanted in ("buy", "wait", "ask"):
        hit = next((o for o in d.options if o.action == wanted), None)
        if hit is not None:
            d.action, d.subject = wanted, hit
            d.headline = f"{ACTIONS[wanted]}：{_option_line(hit)}。{hit.reason}。"
            break
    else:
        d.action = "pass"
        d.subject = tier_a[0] if tier_a else None
        if tier_a:
            o = tier_a[0]
            what = "两张单程" if o.kind == "pair" else "往返"
            d.headline = f"今天不买。最便宜的 {o.tier} 类走法是{what} ${int(o.total):,}：{o.reason}。"
        else:
            d.headline = "今天不买：还没有 A 类（一张票、一家航司或同一联盟）的走法，继续盯。"
    if not trip.committed:
        if d.action in ("buy", "wait", "ask"):
            d.notes.append("你还没确认明年夏天一定回（[trip] committed = false），所以上面只是“本来会”，不会提醒。")
            d.action = "hold"
        deadline = f"（{trip.decide_by} 前）" if trip.decide_by else ""
        d.notes.append(f"先定三件事{deadline}：回不回、候选出发日和最晚回程、整趟封顶价。定了就在 config/watchlist.toml 的 [trip] 里写上。")
    elif not trip.cap:
        d.notes.append("还没定整趟封顶价（[trip] cap）：到了兜底点（D-80），比 ok 线贵的走法会先问你。")
    if d.action == "buy":
        d.notes.append("美国出发、出发前 ≥7 天在航司官网下单，24 小时内可免费取消或保留：今天下单，明天再看一眼。")


def _option_line(o: Option) -> str:
    when = f"{o.depart.month}/{o.depart.day} 去" + (f" {o.ret.month}/{o.ret.day} 回" if o.ret else "")
    return f"${int(o.total):,} · {when} · {o.desc}"


def log_decision(d: Decision, data_dir: Path) -> None:
    """Record today's call and the option it is about in data/decisions.csv (one row per day)."""
    path = Path(data_dir) / "decisions.csv"
    cols = ["date", "action", "kind", "tier", "total", "depart", "ret", "city", "reason", "desc"]
    rows = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=cols)
    rows = rows[rows["date"] != d.as_of.isoformat()]
    row = {"date": d.as_of.isoformat(), "action": d.action}
    o = d.subject
    if o is not None:
        row |= {"kind": o.kind, "tier": o.tier, "total": int(o.total), "depart": o.depart.isoformat(),
                "ret": o.ret.isoformat() if o.ret else "", "city": o.city, "reason": o.reason, "desc": o.desc}
    pd.concat([rows, pd.DataFrame([row])], ignore_index=True).reindex(columns=cols).to_csv(path, index=False)
