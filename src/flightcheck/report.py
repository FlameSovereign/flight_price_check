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


def load_runs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, date]:
    """Observations of the latest run, those of the run before it, and the run date."""
    obs = pd.read_csv(Path(data_dir) / "observations.csv", dtype={"ret": str}, keep_default_na=False)
    for col in ("price", "total_price", "position_price", "g_current", "g_low", "g_high", "days_out"):
        obs[col] = pd.to_numeric(obs[col], errors="coerce")
    runs = sorted(obs["run_date"].unique())
    latest = obs[obs["run_date"] == runs[-1]].copy()
    previous = obs[obs["run_date"] == runs[-2]].copy() if len(runs) > 1 else obs.iloc[0:0]
    return latest, previous, date.fromisoformat(runs[-1])


def build(watchlist: Watchlist, data_dir: Path, reports_dir: Path) -> dict:
    obs, previous, run_date = load_runs(data_dir)
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
    alerts = _alerts(priced, previous, watchlist, names)
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


DEAL_HEADERS = ["出发", "城市", "策略", "总价", "航线 / 航司", "航班", "Google 判断（常见区间）", "历史最低", "P(14天内降≥3%)", "建议"]


def _render(watchlist, run_date, obs, priced, names, order, metrics, alerts) -> str:
    trip = "往返" if watchlist.trip == "round-trip" else "单程"
    lines = [
        f"# 回国{_cabin(watchlist.cabin)}价格监控 · {run_date}",
        "",
        f"> 数据：Google Flights · {trip} · {watchlist.adults} 成人 · {watchlist.currency} · "
        f"今天共 {len(obs)} 次搜索，{obs['total_price'].notna().sum()} 次有结果。"
        "总价 = 主航段 + 定位航段（如有，分开出票的经济舱）。",
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

    lines += [f"## 最便宜的 {TOP_N} 个组合", "", _table(DEAL_HEADERS, [_deal_row(r, names) for _, r in priced.head(TOP_N).iterrows()]), ""]

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


def _model_section(m: dict, watchlist: Watchlist) -> list[str]:
    s = watchlist.model
    lines = [
        "## 模型：现在买还是再等等？",
        "",
        f"问题定义：从今天起 {s.horizon_days} 天内，这个日期/策略的最低价会不会比现在再便宜至少 {s.drop_threshold:.0%}？"
        f"P ≥ {s.wait_above:.0%} 建议**等**，P ≤ {s.buy_below:.0%} 建议**买**，中间是**观望**；"
        f"离出发不足 {s.min_days_out} 天一律建议买。",
        "",
        f"- 训练数据：{m.get('series', 0)} 条价格序列，{m.get('samples', 0)} 个带标签样本，"
        f"观测日期 {m.get('first_obs', '?')} → {m.get('last_obs', '?')}",
    ]
    if m.get("status") != "ok":
        lines += [f"- 状态：{m.get('status')}", ""]
        return lines
    lines.append(f"- 整体基准：{m.get('base_rate', 0):.0%} 的样本在 {s.horizon_days} 天内降价 ≥{s.drop_threshold:.0%}")
    if "auc" in m:
        lines += [
            f"- 时间切分验证（{m['test_start']} 之后为测试集，训练集 {m['train_samples']} / 测试集 {m['test_samples']}）："
            f"AUC {m['auc']}，Brier {m['brier']}（只猜基准率是 {m['brier_base']}）",
        ]
        bt = m.get("backtest", {})
        if bt.get("episodes"):
            lost = bt.get("model_wait_lost_share")
            lines += [
                f"- 回测 {bt['episodes']} 次“开始盯价→按建议下单”：相比立刻买，"
                f"模型平均省 **{bt['model_saving_pct']}%**，简单规则（比 28 天最低贵 5% 就等）省 {bt['rule_saving_pct']}%，"
                f"事后诸葛亮的理论上限 {bt['oracle_saving_pct']}%。"
                f"模型选择等待的占 {bt['model_wait_share']:.0%}"
                + (f"，其中 {lost:.0%} 等完反而更贵。" if lost is not None else "。"),
            ]
    else:
        lines.append(f"- 验证：{m.get('evaluation', '—')}")
    lines += [
        "",
        "注意：同一航线相邻日期的价格高度相关，有效样本远少于表面数字；数据越攒越多（每天 +1 天）结果才越可信。",
        "",
    ]
    return lines


def _alerts(priced: pd.DataFrame, previous: pd.DataFrame, watchlist: Watchlist, names: dict) -> str:
    """Combos under the alert price that are new or cheaper than in the previous run."""
    before = previous.set_index(["series_key", "depart", "ret"])["total_price"].to_dict()
    hits = []
    for city, limit in watchlist.alerts.items():
        for _, r in priced[(priced["city"] == city) & (priced["total_price"] <= limit)].iterrows():
            last = before.get((r["series_key"], r["depart"], r["ret"]))
            if last is None or pd.isna(last) or last > limit or r["total_price"] < last:
                hits.append(r)
    if not hits:
        return ""
    rows = [_deal_row(r, names) for r in sorted(hits, key=lambda r: r["total_price"])]
    limits = "，".join(f"{_city(c)} ≤ {_money(v)}" for c, v in watchlist.alerts.items())
    return f"# 价格提醒\n\n{len(hits)} 个组合新跌破提醒价（{limits}）：\n\n" + _table(DEAL_HEADERS, rows) + "\n"
