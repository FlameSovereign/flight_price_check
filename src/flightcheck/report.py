"""Render the daily Markdown report (and price alerts) from collected data."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import forecast, model, playbook
from .config import Watchlist

WEEKDAYS = "一二三四五六日"
CITY_NAMES = {
    "china": "全国任意",
    "shanghai": "上海",
    "beijing": "北京",
    "guangzhou": "广州",
    "shenzhen": "深圳",
    "chengdu": "成都",
    "hangzhou": "杭州",
}
TOP_N = 15


def _money(v) -> str:
    return "—" if v is None or v == "" or pd.isna(v) else f"${int(float(v)):,}"


def _pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.0%}"


def _level(row) -> str:
    price, low, high = row.get("g_current"), row.get("g_low"), row.get("g_high")
    if any(v is None or pd.isna(v) for v in (price, low, high)):
        return "—"
    word = "偏低" if price < low else "偏高" if price > high else "正常"
    return f"{word}（{_money(low)}–{_money(high)}）"


def _city(c: str) -> str:
    return CITY_NAMES.get(c, c)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


ALERT_STATE = "alerted.csv"
ALERT_KEY = ["series_key", "depart", "ret"]
KIND_NAMES = {"rt": "往返票", "out": "去程单程", "back": "回程单程"}


def _kind(row) -> str:
    if row["ret"]:
        return "rt"
    return "back" if row["leg"] == "back" else "out"


class NothingToReport(RuntimeError):
    pass


def _job_key(j) -> tuple[str, str, str, str]:
    return (j.series_key, j.depart.isoformat(), j.ret.isoformat() if j.ret else "", (j.trip_depart or j.depart).isoformat())


def load_latest_run(data_dir: Path, watchlist: Watchlist) -> tuple[pd.DataFrame, date, dict[str, int]]:
    """The latest run's observations for exactly the searches the watchlist asks for,
    the run date, and how many wanted searches of each kind have no row that day.

    Rows from an older setup (other dates, trip type, cities or strategies) stay in
    the data for the model but not in the report.
    """
    obs = pd.read_csv(Path(data_dir) / "observations.csv", dtype={"ret": str}, keep_default_na=False)
    for col in ("price", "total_price", "position_price", "g_current", "g_low", "g_high", "days_out"):
        obs[col] = pd.to_numeric(obs[col], errors="coerce")
    obs["leg"] = obs["leg"].replace("", "out") if "leg" in obs else "out"
    obs["trip_depart"] = obs["trip_depart"].where(obs["trip_depart"] != "", obs["depart"]) if "trip_depart" in obs else obs["depart"]
    run_date = max(obs["run_date"])
    jobs = watchlist.jobs(date.fromisoformat(run_date))
    if not jobs:
        raise NothingToReport("every date in the watchlist has passed; update [dates] in config/watchlist.toml")
    wanted = {_job_key(j): ("rt" if j.ret else j.leg) for j in jobs}
    latest = obs[obs["run_date"] == run_date]
    keys = list(zip(latest["series_key"], latest["depart"], latest["ret"], latest["trip_depart"]))
    latest = latest[[k in wanted for k in keys]].copy()
    if latest.empty:
        raise RuntimeError("the latest run does not match the current watchlist; run `flightcheck collect` first")
    latest["kind"] = [_kind(r) for _, r in latest.iterrows()]
    present = set(zip(latest["series_key"], latest["depart"], latest["ret"], latest["trip_depart"]))
    missing = {k: 0 for k in ("rt", "out", "back")}
    for key, kind in wanted.items():
        if key not in present:
            missing[kind] += 1
    return latest, date.fromisoformat(run_date), missing


def build(watchlist: Watchlist, data_dir: Path, reports_dir: Path) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        obs, run_date, missing = load_latest_run(data_dir, watchlist)
    except NothingToReport as exc:
        (reports_dir / "latest.md").write_text(f"# 回国{_cabin(watchlist.cabin)}价格监控\n\n> {exc}\n", encoding="utf-8")
        (reports_dir / "alerts.md").unlink(missing_ok=True)
        return {"run_date": None, "alerts": False, "metrics": {"status": str(exc)}}
    history = model.load_history(data_dir)
    result = model.train(history, watchlist.model)
    try:
        outlook = forecast.outlook(history, watchlist)
    except Exception as exc:  # an experimental section must not cost the day's report (and its data commit)
        print(f"forecast failed: {exc!r}", file=sys.stderr)
        outlook = forecast.Outlook(None, status=f"预测出错：{exc!r}")
    signals = model.latest_signals(result, watchlist.model)
    obs = obs.merge(
        signals[["series_key", "depart", "ret", "p_drop", "signal", "hist_min", "hist_max", "hist_len"]],
        on=["series_key", "depart", "ret"],
        how="left",
    )
    names = {s.key: s.name for s in watchlist.strategies}
    try:
        quotes, bad_quotes = None, 0
        try:
            quotes, bad_quotes = playbook.load_quotes(data_dir)
        except Exception as exc:  # a hand-edited file must not cost the day's call
            print(f"manual quotes unreadable: {exc!r}", file=sys.stderr)
            bad_quotes = 1
        decision = playbook.decide(obs, playbook.load_itineraries(data_dir, run_date), watchlist, run_date,
                                   quotes, names, bad_quotes)
        playbook.log_decision(decision, data_dir)
    except Exception as exc:  # never lose the day's report (and its data commit) over it
        print(f"playbook failed: {exc!r}", file=sys.stderr)
        decision = playbook.Decision(run_date, "pass", f"买票规则出错：{exc!r}")

    alerts = _alerts(obs, watchlist, names, Path(data_dir) / ALERT_STATE, run_date, decision)
    alert_path = reports_dir / "alerts.md"
    if alerts:
        alert_path.write_text(alerts, encoding="utf-8")
    elif alert_path.exists():
        alert_path.unlink()

    md = _render(watchlist, run_date, obs, names, result.metrics, alerts, missing, outlook, decision)
    (reports_dir / "latest.md").write_text(md, encoding="utf-8")
    (reports_dir / "model.json").write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n")
    (reports_dir / "forecast.json").write_text(json.dumps(asdict(outlook), ensure_ascii=False, indent=2, default=str) + "\n")
    return {"run_date": str(run_date), "alerts": bool(alerts), "metrics": result.metrics}


def _wd(day: str) -> str:
    return "周" + WEEKDAYS[date.fromisoformat(day).weekday()]


def _day(day: str) -> str:
    return f"{day}（{_wd(day)}）"


def _when(r) -> str:
    if r["kind"] == "rt":
        return f"{_day(r['depart'])}<br>回 {_day(r['ret'])}"
    if r["kind"] == "back":
        paired = r.get("trip_depart") and r["trip_depart"] != r["depart"]
        return f"回 {_day(r['depart'])}" + (f"<br>去程 {r['trip_depart']}" if paired else "")
    return _day(r["depart"])


def _signal(r) -> str:
    return r["signal"] if isinstance(r.get("signal"), str) else "—"


def _deal_row(r, names) -> list[str]:
    pos_price = r.get("position_price")
    pos = f"<br>含定位 {_money(pos_price)}" if pos_price is not None and pd.notna(pos_price) else ""
    return [
        _when(r),
        _city(r["city"]),
        names.get(r["strategy"], r["strategy"]),
        f"**{_money(r['total_price'])}**{pos}",
        f"{r['route']}<br>{r['airlines']}",
        r["flights"],
        _level(r),
        _money(r.get("hist_min")),
        _pct(r.get("p_drop")),
        _signal(r),
    ]


def _deal_headers(watchlist: Watchlist, kind: str = "out") -> list[str]:
    m = watchlist.model
    leg = "去程" if kind == "rt" else ""
    return ["日期", "城市", "策略", "价格", f"{leg}航线 / 航司", f"{leg}航班", "Google 判断（常见区间）", "历史最低（不含定位）",
            f"P({m.horizon_days}天内降≥{m.drop_threshold:.0%})", "建议"]


def _describe_dates(w: Watchlist) -> str:
    if w.outbound is not None:
        text = f"去程 {w.outbound.start}–{w.outbound.end}（每 {w.outbound.step} 天）"
        if w.inbound is not None:
            text += f"，回程 {w.inbound.start}–{w.inbound.end}（每 {w.inbound.step} 天）"
        return text
    text = f"未来 {w.start_days}–{w.end_days} 天出发（每 {w.step_days} 天）"
    if set(w.trips) & {"round-trip", "two-one-ways"}:
        text += f"，回程 = 出发后 {w.return_after_days} 天"
    return text


def _describe_trips(w: Watchlist) -> str:
    names = {"round-trip": "一张往返票", "two-one-ways": "两张单程（去程、回程分开买）", "one-way": "去程单程"}
    return " + ".join(names[t] for t in w.trips)


def _best_legs(legs: pd.DataFrame, city: str, by: str | None):
    """Cheapest leg per `by` value (or overall when by is None) for a city."""
    c = legs[legs["city"] == city]
    if c.empty:
        return None
    if by is None:
        return c.sort_values("total_price").iloc[0]
    return c.loc[c.groupby(by)["total_price"].idxmin()].set_index(by)


def _two_one_ways(watchlist: Watchlist, out: pd.DataFrame, back: pd.DataFrame, city: str):
    """(outbound row, return row, total) for the cheapest pairing, or None.

    Fixed windows: any outbound date with any return date. Rolling grid: each
    outbound with the return leg searched for it."""
    if watchlist.outbound is not None:
        o, b = _best_legs(out, city, "depart"), _best_legs(back, city, "depart")
        if o is None or b is None:
            return None
        pairs = [(o.loc[d], b.loc[r], o.loc[d, "total_price"] + b.loc[r, "total_price"])
                 for d in o.index for r in b.index if r > d]
        if not pairs:
            return None
        best = min(pairs, key=lambda p: p[2])
        return best[0].rename(None).to_dict() | {"depart": best[0].name}, best[1].rename(None).to_dict() | {"depart": best[1].name}, best[2]
    o, b = _best_legs(out, city, "trip_depart"), _best_legs(back, city, "trip_depart")
    if o is None or b is None:
        return None
    both = o.index.intersection(b.index)
    if both.empty:
        return None
    totals = o.loc[both, "total_price"] + b.loc[both, "total_price"]
    k = totals.idxmin()
    return o.loc[k], b.loc[k], totals[k]


def _leg_line(r, names) -> str:
    return (f"{_money(r['total_price'])} · {_when(r).replace('<br>', ' ')} · {names.get(r['strategy'], r['strategy'])} · "
            f"{r['route']}（{r['airlines']}）· 建议：{_signal(r)}")


def _render(watchlist, run_date, obs, names, metrics, alerts, missing=None, outlook=None, decision=None) -> str:
    missing = missing or {}
    order = [s.key for s in watchlist.strategies]
    priced = obs.dropna(subset=["total_price"]).sort_values("total_price")
    rt, out, back = (priced[priced["kind"] == k] for k in ("rt", "out", "back"))
    lines = [
        f"# 回国{_cabin(watchlist.cabin)}价格监控 · {run_date}",
        "",
        f"> Google Flights · {_describe_trips(watchlist)} · {_describe_dates(watchlist)} · {watchlist.adults} 成人 · "
        f"{watchlist.currency}。今天共 {len(obs)} 次搜索，{obs['total_price'].notna().sum()} 次有价格。"
        "价格含定位航段（如有：去程提前一天飞到出发机场，回程当天飞回；分开出票的经济舱，不含酒店；查不到定位航段时不给价格）。"
        + ("往返票的航线和航班是去程，回程在 Google Flights 上再选。" if "round-trip" in watchlist.trips else ""),
        "",
    ]
    if alerts:
        lines += ["## 🔔 价格提醒", "", alerts.split("\n", 2)[2].strip(), ""]
    skipped = sum(missing.values())
    if skipped:
        parts = "，".join(f"{KIND_NAMES[k]} {n} 个" for k, n in missing.items() if n)
        lines += [f"> ⚠️ 今天有 {skipped} 个搜索没有结果记录（{parts}），多半是时间预算用完或搜索失败；往返票的配对每天轮换起点，明天会先跑别的。", ""]

    if decision is not None:
        lines += _playbook_section(decision)
    lines += ["## 今日速览", ""]
    for city in watchlist.destinations:
        lines.append(f"- **{_city(city)}**")
        best_rt = _best_legs(rt, city, None)
        pair = _two_one_ways(watchlist, out, back, city) if "two-one-ways" in watchlist.trips else None
        if "round-trip" in watchlist.trips:
            searched = not obs[(obs["city"] == city) & (obs["kind"] == "rt")].empty
            if best_rt is not None:
                lines.append(f"  - 往返票最低 **{_leg_line(best_rt, names)}**")
            elif searched:
                lines.append("  - 往返票：Google 上还没有这些日期组合的往返票价（便宜的往返票往往要等航司放出夏季运价）")
            else:
                lines.append("  - 往返票：今天没有搜到这一项（没跑到或搜索失败）")
        if pair is not None:
            o, b, total = pair
            lines += [f"  - 两张单程最低合计 **{_money(total)}**",
                      f"    - 去程 {_leg_line(o, names)}",
                      f"    - 回程 {_leg_line(b, names)}"]
        elif "one-way" in watchlist.trips or "two-one-ways" in watchlist.trips:
            for kind, legs in (("out", out), ("back", back)):
                leg = _best_legs(legs, city, None)
                if leg is not None:
                    lines.append(f"  - {KIND_NAMES[kind]}最低 {_leg_line(leg, names)}")
        if best_rt is not None and pair is not None:
            cheaper = "往返票" if best_rt["total_price"] <= pair[2] else "两张单程"
            lines.append(f"  - 现在更便宜的买法：**{cheaper}**（差 {_money(abs(best_rt['total_price'] - pair[2]))}）")
    lines.append("")

    kinds = [k for k in ("rt", "out", "back") if not priced[priced["kind"] == k].empty]
    if kinds:
        summary_rows = []
        for key in order:
            cells = [names.get(key, key)]
            for city in watchlist.destinations:
                for k in kinds:
                    c = priced[(priced["kind"] == k) & (priced["strategy"] == key) & (priced["city"] == city)]["total_price"]
                    cells.append(f"{_money(c.min())}（{_money(c.median())}）" if len(c) else "—")
            summary_rows.append(cells)
        headers = ["策略"] + [f"{_city(c)} {KIND_NAMES[k]}" for c in watchlist.destinations for k in kinds]
        lines += ["### 策略对比：最低（中位数）", "", _table(headers, summary_rows), ""]

    if "round-trip" in watchlist.trips:
        lines += ["## 往返票", ""]
        for city in watchlist.destinations:
            c = obs[(obs["city"] == city) & (obs["kind"] == "rt")]
            if c.empty:
                continue
            lines += [f"### {_city(city)}", ""]
            if watchlist.outbound is not None:
                lines += ["各策略里最便宜的往返票：行 = 去程日期，列 = 回程日期，“—” = 还没有票价。最低价加粗。", "", _rt_matrix(c), ""]
            else:
                lines += [_strategy_pivot(c, names, order), ""]

    if "two-one-ways" in watchlist.trips and watchlist.outbound is None:
        lines += _paired_legs(watchlist, out, back, names)
    for kind in ("out", "back"):
        legs = obs[obs["kind"] == kind]
        if legs.empty:
            continue
        lines += [f"## {KIND_NAMES[kind]}：按日期对比策略", "", "每行最低价加粗。“—” = 那天这种走法没搜到航班。", ""]
        for city in watchlist.destinations:
            c = legs[legs["city"] == city]
            if not c.empty:
                lines += [f"### {_city(city)}", "", _strategy_pivot(c, names, order), ""]

    for kind, rows in (("rt", rt), ("out", out), ("back", back)):
        if not rows.empty:
            lines += [f"## 最便宜的 {TOP_N} 张{KIND_NAMES[kind]}", "",
                      _table(_deal_headers(watchlist, kind), [_deal_row(r, names) for _, r in rows.head(TOP_N).iterrows()]), ""]

    if outlook is not None:
        lines += _forecast_section(outlook, names)
    lines += _model_section(metrics, watchlist)
    return "\n".join(lines).rstrip() + "\n"


def _rt_matrix(rt: pd.DataFrame) -> str:
    best = rt.groupby(["depart", "ret"])["total_price"].min()
    outs, rets = sorted(rt["depart"].unique()), sorted(rt["ret"].unique())
    cheapest = best.min()
    rows = []
    for o in outs:
        cells = [_day(o)]
        for r in rets:
            v = best.get((o, r), np.nan)
            cells.append(f"**{_money(v)}**" if pd.notna(v) and v == cheapest else _money(v))
        rows.append(cells)
    return _table(["去程 \\ 回程"] + [_day(r) for r in rets], rows)


def _strategy_pivot(obs: pd.DataFrame, names: dict, order: list[str]) -> str:
    pivot = obs.pivot_table(index=["depart", "ret"], columns="strategy", values="total_price", aggfunc="min")
    pivot = pivot.reindex(sorted({(d, r) for d, r in zip(obs["depart"], obs["ret"])}))
    rows = []
    for (depart, ret), prices in pivot.iterrows():
        best = prices.min()
        cells = [_day(depart) + (f"<br>回 {_day(ret)}" if ret else "")]
        for key in order:
            v = prices.get(key, np.nan)
            cells.append(f"**{_money(v)}**" if pd.notna(v) and v == best else _money(v))
        rows.append(cells)
    return _table(["日期"] + [names.get(k, k) for k in order], rows)


def _paired_legs(watchlist: Watchlist, out: pd.DataFrame, back: pd.DataFrame, names: dict) -> list[str]:
    """Rolling grid with two-one-ways: each outbound date with its own return leg."""
    lines = ["## 按日期：两张单程合计", "", "每段取当天所有策略里最便宜的那张；合计最低的一行加粗。", ""]
    for city in watchlist.destinations:
        o, b = _best_legs(out, city, "trip_depart"), _best_legs(back, city, "trip_depart")
        if o is None or b is None:
            continue
        both = o.index.intersection(b.index).sort_values()
        totals = o.loc[both, "total_price"] + b.loc[both, "total_price"]
        rows = []
        for k in both:
            total = _money(totals[k])
            rows.append([_day(k), _day(b.loc[k, "depart"]),
                         f"{_money(o.loc[k, 'total_price'])} {names.get(o.loc[k, 'strategy'], '')}",
                         f"{_money(b.loc[k, 'total_price'])} {names.get(b.loc[k, 'strategy'], '')}",
                         f"**{total}**" if totals[k] == totals.min() else total])
        lines += [f"### {_city(city)}", "", _table(["去程", "回程", "去程最低", "回程最低", "合计"], rows), ""]
    return lines


def _cabin(cabin: str) -> str:
    return {"business": "商务舱", "first": "头等舱", "premium-economy": "超经舱", "economy": "经济舱"}.get(cabin, cabin)


def _trip_dates(o) -> str:
    return f"{_day(o.depart.isoformat())}" + (f"<br>回 {_day(o.ret.isoformat())}" if o.ret else "")


def _playbook_section(d: playbook.Decision) -> list[str]:
    lines = ["## 今天该做什么（按你的买法）", "", f"**{d.headline}**" if d.headline else "", ""]
    lines += [f"- {n}" for n in d.notes]
    if d.notes:
        lines.append("")
    shown = d.shown()
    if shown:
        lines += [_table(
            ["动作", "类别", "价格（含定位）", "日期", "走法", "依据"],
            [[f"**{playbook.ACTIONS[o.action]}**" if o.action == "buy" else playbook.ACTIONS[o.action], o.tier,
              _money(o.total), _trip_dates(o), o.desc.replace("|", "/"), o.reason] for o in shown],
        ), "", "类别：A1 = 一张票、一家航司、每程最多转一次；A2 = 一张票、同一联盟，多的一站是北美国内接驳；B = 其他，只作参考。", ""]
    if d.milestones:
        picks = d.milestones if len(d.milestones) <= 5 else [d.milestones[round(i * (len(d.milestones) - 1) / 4)] for i in range(5)]
        lines += ["按出发日倒推的时间点（D-130 主窗口 / D-95 不再等旧低价 / D-80 兜底比两张单程 / D-63 出局）：", "", _table(
            ["出发日", "D-130", "D-95", "D-80", "D-63", "今天在"],
            [[_day(dep.isoformat()), *(f"{m.month}/{m.day}" for m in marks.values()), playbook.PHASES[ph]] for dep, marks, ph in picks],
        ), ""]
    if d.checklist:
        lines += ["**手查清单**", ""] + [f"- {c}" for c in d.checklist] + [""]
    return lines


def _chance(p: float) -> str:
    return "<1%" if p < 0.005 else ">99%" if p > 0.995 else f"{p:.0%}"


def _forecast_section(o: forecast.Outlook, names: dict) -> list[str]:
    title = f"两张单程能不能凑到 {_money(o.target)}？" if o.target else "单程价格走势"
    lines = [f"## 远期预测：{title}（实验性）", ""]
    if o.status != "ok":
        return lines + [f"- {o.status}", ""]
    when = [f"{c.month}/{c.day} 前" for c in o.checkpoints]
    lines += [
        f"用最近 {forecast.TRAIN_DAYS} 天的价格变化，给每段单程模拟 {forecast.N_PATHS:,} 条未来的每日价格走势"
        "（两个模拟器各算一遍取平均：一个照搬相似状态下真实发生过的涨跌，一个把票价当成在几个舱位档位之间跳）。",
        "",
        f"> ⚠️ 回测只验证到 {forecast.RANGE_DAYS} 天以内（两个检验时点，误差比随机游走小约三到六成）。更远的概率只是“如果接下来几个月和最近一样”的推演："
        "它不知道航司春季放出便宜夏季往返价这类事件。那种票一出现，每日任务会直接提醒。"
        "往年参考（README）：2026 年 1–3 月见到的 5–8 月出发商务舱往返，16 条里 14 条低于 $5,000。",
        "",
    ]
    if o.split is not None:
        out, back = o.split
        lines += [
            f"**凑到 {_money(o.target)} 胜算最大的分法：去程 ≤ {_money(out)}、回程 ≤ {_money(back)}**"
            "（两段分开买，按两段走势互不相关估算；可以照这两个价设提醒）",
            "",
            _table([""] + when, [["两段都买到"] + [_chance(p) for p in o.pair_hit]]),
            "",
        ]
    rows = []
    for leg in o.legs:
        lo, mid, hi = leg.range30
        rows.append(
            [KIND_NAMES[leg.kind], f"{names.get(leg.strategy, leg.strategy)} · {_city(leg.city)}", _day(str(leg.depart)),
             _money(leg.price), f"{_money(lo)}–{_money(hi)}（中位 {_money(mid)}）",
             f"≤ {_money(leg.target)}" if leg.target else "—"]
            + ([_chance(p) for p in leg.hit] if leg.hit else ["—"] * len(when))
        )
    headers = ["航段", "走法", "日期", "今天", f"{forecast.RANGE_DAYS} 天后（80% 区间）", "目标"] + when
    lines += [f"每个走法今天最便宜的日期，以及在各日期前至少有一天跌到目标价的概率（数据截至 {o.as_of}）：", "",
              _table(headers, rows), ""]
    return lines


POLICY_NAMES = {"model": "模型", "rule": "规则", "buy-now": "直接买"}


def _model_section(m: dict, watchlist: Watchlist) -> list[str]:
    s = watchlist.model
    lines = [
        "## 模型：现在买还是再等等？",
        "",
        f"问题：从今天起 {s.horizon_days} 天内，这个日期/策略的最低价会不会比现在再便宜至少 {s.drop_threshold:.0%}？",
        "",
        "三种下单策略，用过去几周的数据逐周回放（每周都只用当时之前的数据训练），比较“开始盯价后按策略下单”比“立刻买”平均省多少：",
        "",
        f"- **模型**：P(降价) ≥ {s.wait_above:.0%} 就等；",
        f"- **规则**：价格比 28 天最低价贵 {s.rule_margin:.0%} 以上就等（便宜的舱位最近开过，可能再开）；",
        "- **直接买**：永远不等。",
        "",
        f"离出发不足 {s.min_days_out} 天一律买。报告里的“建议”列跟随过去几周省得最多的那个策略。",
        "",
        f"- 数据：{m.get('series', 0)} 条价格序列，{m.get('samples', 0):,} 个带标签样本，观测日期 {m.get('first_obs', '?')} → {m.get('last_obs', '?')}",
    ]
    if m.get("status") != "ok":
        lines += [f"- 状态：{m.get('status')}；建议列暂按“直接买”。", ""]
        return lines
    lines.append(f"- 整体基准：{m.get('base_rate', 0):.0%} 的样本在 {s.horizon_days} 天内降价 ≥{s.drop_threshold:.0%}")
    folds = m.get("folds") or []
    if not folds:
        lines += [f"- 验证：{m.get('evaluation', '—')}", ""]
        return lines
    bt = m.get("backtest", {})
    policy = m.get("policy", "buy-now")
    lines += [
        f"- 当前建议跟随：**{POLICY_NAMES.get(policy, policy)}**",
        f"- 预测能力：平均 AUC {m['auc'] if m.get('auc') is not None else '—'}，"
        f"Brier {m['brier']}（只猜训练期基准率是 {m['brier_base']}，越低越好）",
        "",
        _table(
            ["测试周", "回放次数", "当周降价比例", "AUC", "模型省", "规则省", "事后最优"],
            [
                [
                    f"{f['start']} → {f['end']}",
                    f.get("episodes", 0),
                    f"{f['test_rate']:.0%}",
                    f.get("auc", "—"),
                    f"{f.get('model_saving_pct', 0)}%",
                    f"{f.get('rule_saving_pct', 0)}%",
                    f"{f.get('oracle_saving_pct', 0)}%",
                ]
                for f in folds
            ]
            + [["**合计**", bt.get("episodes", 0), f"{m['test_base_rate']:.0%}", m.get("auc", "—"),
                f"**{bt.get('model_saving_pct', 0)}%**", f"**{bt.get('rule_saving_pct', 0)}%**", f"{bt.get('oracle_saving_pct', 0)}%"]],
        ),
        "",
        "注意：同一航线相邻日期的价格高度相关（同一个舱位开关会同时影响几十个日期），有效样本远少于表面数字。"
        "每天的定时任务都会多一周可回放的数据，建议会自动跟随表现更好的策略。",
        "",
    ]
    return lines


def _alerts(obs: pd.DataFrame, watchlist: Watchlist, names: dict, state_path: Path, run_date: date,
            decision: playbook.Decision | None = None) -> str:
    """What to open an issue about today; empty when nothing is new.

    Before the trip is committed: price alerts. [alerts] round_trip applies to round-trip
    tickets and to two one-ways together, one_way to single legs, and only tier-A tickets
    (one ticket, one airline or alliance) count. Once committed: the playbook's "buy" calls
    instead. Once purchased: nothing.
    The alerted prices live in a state file, so a partial run, a failed search or a
    same-day rerun cannot make old alerts look new. A ticket that climbs back above
    the limit (or stops being a buy) is forgotten, so its next drop alerts again.
    """
    trip = watchlist.trip
    if trip.purchased:
        return ""
    state: dict[tuple, float] = {}
    if state_path.exists():
        old = pd.read_csv(state_path, dtype=str, keep_default_na=False)
        state = {tuple(r[k] for k in ALERT_KEY): float(r["total_price"]) for _, r in old.iterrows()}
    hits, pair_hits, buy_hits = [], [], []
    if trip.committed:
        buy_hits = _playbook_hits(decision, state)
    else:
        hits, pair_hits = _price_hits(obs, watchlist, state)

    today = run_date.isoformat()
    rows = [dict(zip(ALERT_KEY, k), total_price=int(v)) for k, v in sorted(state.items()) if not k[1] or k[1] >= today]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ALERT_KEY + ["total_price"]).to_csv(state_path, index=False)
    if not hits and not pair_hits and not buy_hits:
        return ""
    labels = {"round_trip": "往返票", "one_way": "单程"}
    limits = "，".join(f"{labels.get(k, k)} ≤ {_money(v)}" for k, v in watchlist.alerts.items())
    sections, summary = [], []
    if buy_hits:
        summary.append(f"按你的买法，{len(buy_hits)} 张票现在该买（不一定低于提醒价）。")
        sections += ["### 按你的买法：现在该买", "", _table(
            ["走法", "日期", "价格", "为什么"],
            [[o.desc, _trip_dates(o), f"**{_money(o.total)}**", o.reason] for o in buy_hits],
        ), "", "美国出发、出发前 ≥7 天在航司官网下单，24 小时内可免费取消或保留。", ""]
    if hits or pair_hits:
        summary.append(f"{len(hits) + len(pair_hits)} 项新跌破提醒价，或比上次提醒时更便宜（{limits}；两张单程合计按往返价算；只算 A 类票）：")
    if pair_hits:
        sections += ["### 两张单程合计", "", _table(
            ["城市", "去程", "回程", "合计"],
            [[_city(c), _leg_line(o, names), _leg_line(b, names), f"**{_money(t)}**"] for c, (o, b, t) in pair_hits],
        ), ""]
    for kind in ("rt", "out", "back"):
        rows_k = [r for r in hits if r["kind"] == kind]
        if rows_k:
            sections += [f"### {KIND_NAMES[kind]}", "", _table(_deal_headers(watchlist, kind), [_deal_row(r, names) for r in rows_k]), ""]
    return "# 价格提醒\n\n" + " ".join(summary) + "\n\n" + "\n".join(sections).rstrip() + "\n"


def _playbook_hits(decision: playbook.Decision | None, state: dict) -> list:
    """Today's buy calls not alerted before at this price or lower. Calls that went away
    (priced today, no longer a buy) are forgotten, so they alert again if they come back."""
    if decision is None:
        return []
    buying = {o.key for o in decision.options if o.action == "buy"}
    for o in decision.options:
        if o.key in state and o.key not in buying:
            state.pop(o.key)
    hits = []
    for o in decision.buys:
        if o.key not in state or o.total < state[o.key]:
            hits.append(o)
            state[o.key] = float(o.total)
    return hits


def _price_hits(obs: pd.DataFrame, watchlist: Watchlist, state: dict) -> tuple[list, list]:
    allow = watchlist.playbook.allow
    obs = obs[[playbook.tier(f, r, allow) != "B" for f, r in zip(obs["flights"], obs["route"])]] if not obs.empty else obs
    hits = []
    for _, r in obs.sort_values("total_price").iterrows():
        limit = watchlist.alerts.get("round_trip" if r["kind"] == "rt" else "one_way")
        total = r["total_price"]
        if limit is None or pd.isna(total):
            continue  # no alert configured, or no price today: keep what we knew
        key = tuple(str(r[k]) for k in ALERT_KEY)
        # The same flights on the same dates found by several city searches (e.g. "china"
        # and "beijing") are one fare: alert it once.
        fare = (f"fare:{r['kind']}:{r['flights']}", str(r["depart"]), str(r["ret"]))
        if total <= limit:
            new_low = key not in state or total < state[key]
            if new_low and not (fare in state and state[fare] <= total):
                hits.append(r)
            if new_low:
                state[key] = float(total)
            state[fare] = min(state.get(fare, float(total)), float(total))
        else:
            state.pop(key, None)
            state.pop(fare, None)

    # Two one-ways count as a round trip: alert when together they beat round_trip.
    pair_hits = []
    limit = watchlist.alerts.get("round_trip")
    if limit is not None and "two-one-ways" in watchlist.trips:
        priced = obs.dropna(subset=["total_price"])
        out, back = priced[priced["kind"] == "out"], priced[priced["kind"] == "back"]
        for city in watchlist.destinations:
            pair = _two_one_ways(watchlist, out, back, city)
            if pair is None:
                continue
            key = (f"pair:{city}", "", "")
            if pair[2] <= limit:
                if key not in state or pair[2] < state[key]:
                    pair_hits.append((city, pair))
                    state[key] = float(pair[2])
            else:
                state.pop(key, None)

    return hits, pair_hits
