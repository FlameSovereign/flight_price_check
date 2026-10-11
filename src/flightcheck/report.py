"""Render the daily Markdown report (and price alerts) from collected data."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import model
from .config import Watchlist

WEEKDAYS = "一二三四五六日"
CITY_NAMES = {
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


def load_latest_run(data_dir: Path) -> tuple[pd.DataFrame, date]:
    obs = pd.read_csv(Path(data_dir) / "observations.csv", dtype={"ret": str}, keep_default_na=False)
    for col in ("price", "total_price", "position_price", "g_current", "g_low", "g_high", "days_out"):
        obs[col] = pd.to_numeric(obs[col], errors="coerce")
    run_date = max(obs["run_date"])
    return obs[obs["run_date"] == run_date].copy(), date.fromisoformat(run_date)


def build(watchlist: Watchlist, data_dir: Path, reports_dir: Path) -> dict:
    obs, run_date = load_latest_run(data_dir)
    result = model.train(model.load_history(data_dir), watchlist.model)
    signals = model.latest_signals(result, watchlist.model)
    obs = obs.merge(
        signals[["series_key", "depart", "ret", "p_drop", "signal", "hist_min", "hist_max", "hist_len"]],
        on=["series_key", "depart", "ret"],
        how="left",
    )
    names = {s.key: s.name for s in watchlist.strategies}
    order = [s.key for s in watchlist.strategies]
    priced = obs.dropna(subset=["total_price"]).sort_values("total_price")

    reports_dir.mkdir(parents=True, exist_ok=True)
    alerts = _alerts(obs, watchlist, names, Path(data_dir) / ALERT_STATE, run_date)
    alert_path = reports_dir / "alerts.md"
    if alerts:
        alert_path.write_text(alerts, encoding="utf-8")
    elif alert_path.exists():
        alert_path.unlink()

    md = _render(watchlist, run_date, obs, priced, names, order, result.metrics, alerts)
    (reports_dir / "latest.md").write_text(md, encoding="utf-8")
    (reports_dir / "model.json").write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n")
    return {"run_date": str(run_date), "alerts": bool(alerts), "metrics": result.metrics}


def _deal_row(r, names, with_city=True) -> list[str]:
    weekday = WEEKDAYS[date.fromisoformat(r["depart"]).weekday()]
    pos_price = r.get("position_price")
    pos = f"<br>含定位 {_money(pos_price)}" if pos_price is not None and pd.notna(pos_price) else ""
    cells = [f"{r['depart']}（{weekday}）"]
    if with_city:
        cells.append(_city(r["city"]))
    cells += [
        names.get(r["strategy"], r["strategy"]),
        f"**{_money(r['total_price'])}**{pos}",
        f"{r['route']}<br>{r['airlines']}",
        r["flights"],
        _level(r),
        _money(r.get("hist_min")),
        _pct(r.get("p_drop")),
        r["signal"] if isinstance(r.get("signal"), str) else "—",
    ]
    return cells


def _deal_headers(watchlist: Watchlist) -> list[str]:
    m = watchlist.model
    return ["出发", "城市", "策略", "总价", "航线 / 航司", "航班", "Google 判断（常见区间）", "历史最低（不含定位）",
            f"P({m.horizon_days}天内降≥{m.drop_threshold:.0%})", "建议"]


def _render(watchlist, run_date, obs, priced, names, order, metrics, alerts) -> str:
    trip = "往返" if watchlist.trip == "round-trip" else "单程"
    lines = [
        f"# 回国{_cabin(watchlist.cabin)}价格监控 · {run_date}",
        "",
        f"> 数据：Google Flights · {trip} · {watchlist.adults} 成人 · {watchlist.currency} · "
        f"今天共 {len(obs)} 次搜索，{obs['total_price'].notna().sum()} 次有结果。"
        "总价 = 主航段 + 定位航段（如有：提前一天、分开出票的经济舱，不含酒店；查不到定位航段时不给总价）。",
        "",
    ]
    if alerts:
        lines += ["## 🔔 价格提醒", "", alerts.split("\n", 2)[2].strip(), ""]

    lines += ["## 今日速览", ""]
    for city in watchlist.destinations:
        c = priced[priced["city"] == city]
        if c.empty:
            continue
        best = c.iloc[0]
        lines.append(
            f"- **{_city(city)}** 最低 {_money(best['total_price'])}：{best['depart']} · "
            f"{names.get(best['strategy'], best['strategy'])} · {best['route']}（{best['airlines']}）"
        )
    lines.append("")

    summary_rows = []
    for key in order:
        cells = [names.get(key, key)]
        for city in watchlist.destinations:
            c = priced[(priced["strategy"] == key) & (priced["city"] == city)]["total_price"]
            cells += [_money(c.min()) if len(c) else "—", _money(c.median()) if len(c) else "—"]
        summary_rows.append(cells)
    headers = ["策略"] + [f"{_city(c)} {k}" for c in watchlist.destinations for k in ("最低", "中位数")]
    lines += ["### 策略对比（所有扫描日期）", "", _table(headers, summary_rows), ""]

    lines += [f"## 最便宜的 {TOP_N} 个组合", "", _table(_deal_headers(watchlist), [_deal_row(r, names) for _, r in priced.head(TOP_N).iterrows()]), ""]

    lines += ["## 按出发日期对比策略（总价）", "", "每行最低价加粗。“—” = 当天该策略没搜到航班。", ""]
    for city in watchlist.destinations:
        c = obs[obs["city"] == city]
        if c.empty:
            continue
        pivot = c.pivot_table(index="depart", columns="strategy", values="total_price", aggfunc="min")
        rows = []
        for depart, prices in pivot.iterrows():
            best = prices.min()
            weekday = WEEKDAYS[date.fromisoformat(depart).weekday()]
            cells = [f"{depart}（{weekday}）"]
            for key in order:
                v = prices.get(key, np.nan)
                cells.append(f"**{_money(v)}**" if v == best else _money(v))
            rows.append(cells)
        lines += [f"### {_city(city)}", "", _table(["出发"] + [names.get(k, k) for k in order], rows), ""]

    lines += _model_section(metrics, watchlist)
    return "\n".join(lines).rstrip() + "\n"


def _cabin(cabin: str) -> str:
    return {"business": "商务舱", "first": "头等舱", "premium-economy": "超经舱", "economy": "经济舱"}.get(cabin, cabin)


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


def _alerts(obs: pd.DataFrame, watchlist: Watchlist, names: dict, state_path: Path, run_date: date) -> str:
    """Combos under their city's alert price that were never alerted, or are cheaper than when last alerted.

    The alerted prices live in a state file, so a partial run, a failed search or a
    same-day rerun cannot make old alerts look new. A combo that climbs back above
    the limit is forgotten, so its next drop under the limit alerts again.
    """
    state: dict[tuple, float] = {}
    if state_path.exists():
        old = pd.read_csv(state_path, dtype=str, keep_default_na=False)
        state = {tuple(r[k] for k in ALERT_KEY): float(r["total_price"]) for _, r in old.iterrows()}
    hits = []
    for _, r in obs.sort_values("total_price").iterrows():
        limit = watchlist.alerts.get(r["city"])
        total = r["total_price"]
        if limit is None or pd.isna(total):
            continue  # no alert configured, or no price today: keep what we knew
        key = tuple(str(r[k]) for k in ALERT_KEY)
        if total <= limit:
            if key not in state or total < state[key]:
                hits.append(r)
                state[key] = float(total)
        else:
            state.pop(key, None)
    today = run_date.isoformat()
    rows = [dict(zip(ALERT_KEY, k), total_price=int(v)) for k, v in sorted(state.items()) if k[1] >= today]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ALERT_KEY + ["total_price"]).to_csv(state_path, index=False)
    if not hits:
        return ""
    limits = "，".join(f"{_city(c)} ≤ {_money(v)}" for c, v in watchlist.alerts.items())
    table = _table(_deal_headers(watchlist), [_deal_row(r, names) for r in hits])
    return f"# 价格提醒\n\n{len(hits)} 个组合新跌破提醒价，或比上次提醒时更便宜（{limits}）：\n\n" + table + "\n"
