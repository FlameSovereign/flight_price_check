"""Your buying habits as rules: one action per day for the trip.

The rules come from the traveller's own bookings and Google Flights price
tracking (kept out of the repo), checked against the tracker's data and past
public fares (reference/past_fares.csv):

- Judge the target only against one ticket on one airline (tier A1), or one
  alliance where the extra stop is a North American feeder (A2), on an allowed
  long-haul carrier. Everything else (tier B) is context, never a trigger.
- Count back from each candidate departure date D: the main buying window opens
  at D-130; from D-95 stop waiting for old lows; at D-80 fall back to two
  one-way tickets if they beat the round trip; at D-63 the date leaves the
  system and the decision is yours.
- One decision table on the all-in price (positioning included):
    <= target (alerts.round_trip)  buy the same day, any phase
    <= a carrier line (TK 5,500)   buy the same day (those fares barely move)
    <= good                        before D-95: buy at the 14-day floor step, otherwise
                                   wait up to WAIT_DAYS for it to return; from D-95: buy
    <= ok                          from D-95: buy unless two one-ways are clearly cheaper
    more                           from D-80: buy the cheaper of the round trip and two one-ways
  Over the walk-away cap nothing is bought automatically.
- Nothing is "buy" until the trip is committed ([trip] committed = true); until
  then the report shows what it would have done.
- From [playbook] checks_from, hand-check airline sites on Mondays and
  Thursdays: Google Flights often shows no long-stay round trips, and
  data/manual_quotes.csv feeds those quotes into the same table.
- Every day's best option and action is logged to data/decisions.csv so the
  lines can be re-fitted once real spring data exists.

Buying a US-origin ticket directly from the airline at least 7 days before
departure comes with a free 24-hour cancellation or hold (US DOT), so a "buy"
here means "book today, recheck tomorrow".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import Watchlist

# Days before departure (counted from each candidate outbound date).
WINDOW, LAST_CALL, FALLBACK, OUT = 130, 95, 80, 63
FLOOR_OBS = 14  # the floor is the lowest price of the last 14 daily observations
STEP = 0.03  # moves under 3% are the same fare step
WAIT_DAYS = 7  # a lower step seen this recently may come back
PAIR_ALLOWANCE = 150  # two separate tickets carry misconnect risk
SINGLE_TICKET_PREMIUM = 300  # worth paying for one ticket over two
QUOTE_DAYS = 2  # hand-checked quotes older than this are stale
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
FLIGHT = re.compile(r"^([A-Z0-9]{2})\d+[A-Z]?$")
PHASES = {
    "watch": "观望",
    "window": "主窗口",
    "last_call": "最后一档",
    "fallback": "兜底",
    "out": "已出局",
}
ACTIONS = {"buy": "买", "wait": "等", "pass": "不买", "ask": "问你", "hold": "先别动"}


@dataclass
class Option:
    """One way to buy the whole trip, all-in."""

    kind: str  # "rt" (one round-trip ticket), "manual" (hand-checked round trip) or "pair" (two one-ways)
    total: float
    tier: str  # "A1", "A2" or "B"
    carriers: tuple[str, ...]
    depart: date
    ret: date | None
    city: str
    desc: str
    url: str = ""
    at_floor: bool | None = None  # price within one step of its 14-day floor (None: no history)
    lower_lately: bool = False  # a lower step was seen within WAIT_DAYS
    action: str = ""
    reason: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (f"playbook:{self.kind}:{self.city}", self.depart.isoformat(), self.ret.isoformat() if self.ret else "")


@dataclass
class Decision:
    as_of: date
    action: str  # overall: buy / wait / pass / ask / hold
    headline: str
    options: list[Option] = field(default_factory=list)  # tier-A round trips, quotes and the best pair, cheapest first
    milestones: list[tuple[date, dict[str, date], str]] = field(default_factory=list)  # (outbound date, milestones, phase)
    checklist: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def buys(self) -> list[Option]:
        return [o for o in self.options if o.action == "buy"]


def carriers_of(flights: str) -> tuple[str, ...]:
    """Airline codes in flight order, e.g. "AA2403 CX875 CX334" -> ("AA", "CX", "CX")."""
    return tuple(m.group(1) for f in str(flights).split() if (m := FLIGHT.match(f)))


def tier(flights: str, route: str, allow: tuple[str, ...]) -> str:
    """A1: one airline, at most one stop. A2: one alliance, the extra stop a North American feeder.
    The long-haul carriers must be on the allow list; anything else is B."""
    codes = carriers_of(flights)
    if not codes:
        return "B"
    stops = len(codes) - 1
    one_airline = len(set(codes)) == 1
    if one_airline and stops <= 1:
        return "A1" if codes[0] in allow else "B"
    one_alliance = one_airline or any(set(codes) <= members for members in ALLIANCES.values())
    first_stop = str(route).split("-")[1:2]
    feeder = stops <= 1 or (stops == 2 and first_stop and first_stop[0] in NA_HUBS)
    if one_alliance and feeder and set(codes[1:]) <= set(allow):
        return "A2"
    return "B"


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


def decide(obs: pd.DataFrame, history: pd.DataFrame, watchlist: Watchlist, today: date,
           quotes: pd.DataFrame | None = None, names: dict | None = None) -> Decision:
    """Today's action for the trip from the latest observations (report.load_latest_run rows)."""
    names = names or {}
    trip, pb = watchlist.trip, watchlist.playbook
    target = watchlist.alerts.get("round_trip")
    manual, bad_quotes = _quotes(quotes, today, pb.allow)
    options = _round_trips(obs, history, pb.allow, names) + manual
    pair = _best_pair(obs, pb.allow, names)
    pair_total = pair.total if pair is not None else None
    if pair is not None:
        options.append(pair)
    options = [o for o in options if phase(o.depart, today) != "out"]
    options.sort(key=lambda o: o.total)

    tier_a = [o for o in options if o.tier != "B"]
    best_rt = next((o for o in tier_a if o.kind != "pair"), None)
    for o in options:
        o.action, o.reason = _judge(o, phase(o.depart, today), target, pb, trip.cap, pair_total, best_rt)

    dates = _candidate_dates(watchlist, obs, today)
    marks = [(d, milestones(d), phase(d, today)) for d in dates]
    checklist = _checklist(watchlist, options, today)
    shown = sorted([o for o in options if o.tier != "B"][:8] + [o for o in options if o.tier == "B"][:3], key=lambda o: o.total)
    d = Decision(today, "pass", "", shown, marks, checklist)
    _summarise(d, trip, target, today)
    if bad_quotes:
        d.notes.append(f"data/manual_quotes.csv 里有 {bad_quotes} 行读不了（日期写成 2027-05-19 这种格式，价格只写数字），已跳过。")
    return d


def _round_trips(obs, history, allow, names) -> list[Option]:
    rt = obs[(obs["kind"] == "rt")].dropna(subset=["total_price"])
    if rt.empty:
        return []
    hist = history[history["ret"] != ""] if not history.empty else history
    by_series = {k: g.sort_values("obs_date")["price"].to_numpy() for k, g in hist.groupby(["series_key", "depart", "ret"])} if not hist.empty else {}
    out = []
    for _, r in rt.iterrows():
        series = by_series.get((r["series_key"], pd.Timestamp(r["depart"]), r["ret"]))
        at_floor, lower = None, False
        if series is not None and len(series) >= FLOOR_OBS:
            recent = series[-FLOOR_OBS:]
            at_floor = bool(r["price"] <= recent.min() * (1 + STEP))
            lower = bool((series[-WAIT_DAYS:] < r["price"] / (1 + STEP)).any())
        out.append(Option(
            "rt", float(r["total_price"]), tier(r["flights"], r["route"], allow), carriers_of(r["flights"]),
            date.fromisoformat(r["depart"]), date.fromisoformat(r["ret"]), r["city"],
            f"{names.get(r['strategy'], r['strategy'])} · {r['route']}（{r['airlines']}）", str(r.get("url", "")),
            at_floor, lower,
        ))
    return out


def load_quotes(data_dir: Path) -> pd.DataFrame | None:
    """Hand-checked round-trip quotes: data/manual_quotes.csv with columns
    date, carrier, depart, ret, city, price[, stops, source, note]."""
    path = Path(data_dir) / "manual_quotes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False)


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
            opt = Option("manual", float(q["price"]), "A1" if len(set(codes)) == 1 and stops <= 1 and codes[0] in allow else "B",
                         codes, date.fromisoformat(q["depart"]), date.fromisoformat(q["ret"]), q.get("city", ""),
                         f"{q.get('source') or '手查'} · {'+'.join(codes)}（{seen} 手查）")
        except (KeyError, ValueError, IndexError):
            bad += 1
            continue
        if 0 <= (today - seen).days <= QUOTE_DAYS:
            out.append(opt)
    return out, bad


def _best_pair(obs, allow, names) -> Option | None:
    """Best outbound + return one-way pair (return after outbound), tier-A outbound only."""
    priced = obs.dropna(subset=["total_price"])
    out = priced[priced["kind"] == "out"]
    back = priced[priced["kind"] == "back"]
    out = out[[tier(f, r, allow) != "B" for f, r in zip(out["flights"], out["route"])]]
    if out.empty or back.empty:
        return None
    best = None
    for _, o in out.sort_values("total_price").head(20).iterrows():
        later = back[back["depart"] > o["depart"]]
        if later.empty:
            continue
        b = later.loc[later["total_price"].idxmin()]
        total = o["total_price"] + b["total_price"] + PAIR_ALLOWANCE
        if best is None or total < best[2]:
            best = (o, b, total)
    if best is None:
        return None
    o, b, total = best
    city = o["city"] if o["city"] == b["city"] else f"{o['city']}/{b['city']}"
    desc = (f"去 {o['route']}（{o['airlines']}）${int(o['total_price']):,} + 回 {b['route']}（{b['airlines']}）"
            f"${int(b['total_price']):,} + 分票风险 ${PAIR_ALLOWANCE}")
    return Option("pair", float(total), tier(o["flights"], o["route"], allow), carriers_of(o["flights"]) + carriers_of(b["flights"]),
                  date.fromisoformat(o["depart"]), date.fromisoformat(b["depart"]), city, desc)


def _judge(o: Option, ph: str, target, pb, cap, pair_total, best_rt) -> tuple[str, str]:
    p = o.total
    if o.tier == "B":
        return "pass", "不是一张票一家航司（或不在白名单），只作参考"
    if cap and p > cap:
        return "ask", f"超过你的封顶价 ${int(cap):,}，不自动买；D-63 前你自己拍板"
    if target and p <= target:
        return "buy", f"≤ 目标 ${int(target):,}：当天买"
    if o.kind == "pair":
        if ph == "fallback" and (best_rt is None or p + SINGLE_TICKET_PREMIUM < best_rt.total):
            return "buy", "到了兜底点（D-80），两张单程比往返票便宜：买"
        return "hold", "两张单程只在兜底点（D-80）才考虑，之前先别动"
    line = next((pb.carrier_lines[c] for c in set(o.carriers) if c in pb.carrier_lines and len(set(o.carriers)) == 1), None)
    if line and p <= line:
        return "buy", f"≤ 这家航司的线 ${int(line):,}（它的价很少回落）：当天买"
    early = ph in ("watch", "window")
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
            return "buy", "已过 D-95，比两张单程贵不到 $300：买"
        return "pass", "两张单程明显更便宜，等兜底点（D-80）比较"
    if ph == "fallback" and best_rt is o and (pair_total is None or p <= pair_total + SINGLE_TICKET_PREMIUM):
        return "buy", "到了兜底点（D-80），这是最便宜的整趟走法：买"
    return "pass", f"> ${int(pb.ok):,}，兜底点（D-80）之前不买"


def _candidate_dates(watchlist: Watchlist, obs: pd.DataFrame, today: date) -> list[date]:
    if watchlist.outbound is not None:
        return watchlist.outbound.dates(today)
    days = obs.loc[obs["kind"].isin(["rt", "out"]), "depart"].unique()
    return sorted(date.fromisoformat(d) for d in days)


def _checklist(watchlist: Watchlist, options: list[Option], today: date) -> list[str]:
    start = watchlist.playbook.checks_from
    if start is None or today < start:
        return []
    picks = []
    for o in options:
        pair = (o.depart, o.ret)
        if o.tier != "B" and o.ret and pair not in picks:
            picks.append(pair)
        if len(picks) == 3:
            break
    when = "、".join(f"{d.month}/{d.day} 去 {r.month}/{r.day} 回" for d, r in picks) or "5 月去、8 月回的几组日期"
    sites = "、".join(f"[{n}]({u})" for n, u in CHECKLIST_SITES)
    day = "今天是手查日。" if today.weekday() in (0, 3) else "每周一、周四手查。"
    return [
        f"{day}按 {when} 查一张票的往返：{sites}",
        "顺手看票规：最长停留、提前购票、改期费；看到的价记进 `data/manual_quotes.csv`（date,carrier,depart,ret,city,price,stops,source）。",
    ]


def _summarise(d: Decision, trip, target, today: date) -> None:
    buys = d.buys
    waits = [o for o in d.options if o.action == "wait"]
    asks = [o for o in d.options if o.action == "ask"]
    if trip.purchased:
        d.action, d.headline = "pass", "已经买好了：不再给建议，也不再提醒。"
        for o in d.options:
            o.action, o.reason = "pass", "已买"
        return
    best = buys[0] if buys else waits[0] if waits else asks[0] if asks else None
    if best is None:
        d.action = "pass"
        tier_a = [o for o in d.options if o.tier != "B"]
        if tier_a:
            o = tier_a[0]
            what = "两张单程" if o.kind == "pair" else "往返"
            d.headline = f"今天不买。最便宜的 {o.tier} 类走法是{what} ${int(o.total):,}：{o.reason}。"
        else:
            d.headline = "今天不买：还没有 A 类（一张票、一家航司或同一联盟）的走法，继续盯。"
    else:
        d.action = best.action
        verb = {"buy": "买", "wait": "等", "ask": "问你"}[best.action]
        d.headline = f"{verb}：{_option_line(best)}。{best.reason}。"
    if not trip.committed:
        if d.action in ("buy", "wait"):
            d.notes.append("你还没确认明年夏天一定回（[trip] committed = false），所以上面只是“本来会买”，不会提醒。")
            d.action = "hold"
        deadline = f"（{trip.decide_by} 前）" if trip.decide_by else ""
        d.notes.append(f"先定三件事{deadline}：回不回、候选出发日和最晚回程、整趟封顶价。定了就在 config/watchlist.toml 的 [trip] 里写上。")
    if d.action == "buy":
        d.notes.append("美国出发、出发前 ≥7 天在航司官网下单，24 小时内可免费取消或保留：今天下单，明天再看一眼。")


def _option_line(o: Option) -> str:
    when = f"{o.depart.month}/{o.depart.day} 去" + (f" {o.ret.month}/{o.ret.day} 回" if o.ret else "")
    return f"${int(o.total):,} · {when} · {o.desc}"


def log_decision(d: Decision, data_dir: Path) -> None:
    """Append today's best option and action to data/decisions.csv (one row per day), for re-fitting the lines."""
    path = Path(data_dir) / "decisions.csv"
    cols = ["date", "action", "kind", "tier", "total", "depart", "ret", "city", "desc"]
    rows = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=cols)
    rows = rows[rows["date"] != d.as_of.isoformat()]
    best = next((o for o in d.options if o.tier != "B"), None)
    row = {"date": d.as_of.isoformat(), "action": d.action}
    if best is not None:
        row |= {"kind": best.kind, "tier": best.tier, "total": int(best.total), "depart": best.depart.isoformat(),
                "ret": best.ret.isoformat() if best.ret else "", "city": best.city, "desc": best.desc}
    pd.concat([rows, pd.DataFrame([row])], ignore_index=True).reindex(columns=cols).to_csv(path, index=False)
