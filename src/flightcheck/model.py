"""Buy-now-or-wait model.

Every (series, departure date) has a daily price series. For each day `t` of
each series we build features from prices up to `t` only, and the label

    y = min(price[t+1 .. t+H]) <= price[t] * (1 - drop_threshold)

i.e. "would waiting up to H days have saved at least X%?". One gradient
boosted classifier is trained on all routes pooled together; features are
relative (ratios, changes, days to departure) so routes with very different
price levels can share what they teach.

Evaluation is a strict time split: the model is trained only on samples whose
whole label window ends before the test period starts, then replayed as a
buying policy on the test period and compared with buying immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import ModelSettings

SERIES = ["series_key", "depart", "ret"]
FEATURES = [
    "days_out",
    "r_min28",
    "r_med28",
    "r_max28",
    "chg3",
    "chg7",
    "chg14",
    "vol14",
    "n_chg28",
    "days_since_change",
    "r_xsec",
    "dep_month",
    "dep_dow",
    "strategy_code",
]
MIN_HISTORY = 7
MAX_FILL_DAYS = 3
TEST_DAYS = 14


def load_history(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "history.csv"
    df = pd.read_csv(path, dtype={"ret": str}, keep_default_na=False)
    df["depart"] = pd.to_datetime(df["depart"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price"])


def build_panel(history: pd.DataFrame) -> pd.DataFrame:
    """Daily grid per series, short gaps forward-filled, only days before departure."""
    frames = []
    for (key, depart, ret), g in history.groupby(SERIES, sort=False):
        s = g.set_index("obs_date")["price"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill(limit=MAX_FILL_DAYS)
        frame = s.rename("price").rename_axis("obs_date").reset_index()
        frame["series_key"], frame["depart"], frame["ret"] = key, depart, ret
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["days_out"] = (panel["depart"] - panel["obs_date"]).dt.days
    panel = panel[panel["days_out"] >= 0].dropna(subset=["price"])
    return panel.sort_values(SERIES + ["obs_date"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    g = panel.groupby(SERIES, sort=False)["price"]
    roll = lambda w, f: g.transform(lambda s: getattr(s.rolling(w, min_periods=MIN_HISTORY), f)())  # noqa: E731
    panel["min28"] = roll(28, "min")
    panel["r_min28"] = panel["price"] / panel["min28"] - 1
    panel["r_med28"] = panel["price"] / roll(28, "median") - 1
    panel["r_max28"] = panel["price"] / roll(28, "max") - 1
    for k in (3, 7, 14):
        panel[f"chg{k}"] = panel["price"] / g.shift(k) - 1
    log_ret = np.log(panel["price"]).groupby([panel[c] for c in SERIES], sort=False).diff()
    panel["vol14"] = log_ret.groupby([panel[c] for c in SERIES], sort=False).transform(
        lambda s: s.rolling(14, min_periods=MIN_HISTORY).std()
    )
    changed = (log_ret.fillna(0) != 0).astype(int)
    panel["n_chg28"] = changed.groupby([panel[c] for c in SERIES], sort=False).transform(
        lambda s: s.rolling(28, min_periods=MIN_HISTORY).sum()
    )
    panel["days_since_change"] = changed.groupby([panel[c] for c in SERIES], sort=False).transform(_days_since)
    panel["hist_len"] = g.cumcount() + 1

    # How this departure date compares with neighbouring dates seen the same day.
    panel = panel.sort_values(["series_key", "ret", "obs_date", "depart"])
    xsec = panel.groupby(["series_key", "ret", "obs_date"], sort=False)["price"].transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median()
    )
    panel["r_xsec"] = panel["price"] / xsec - 1
    panel = panel.sort_values(SERIES + ["obs_date"])

    panel["dep_month"] = panel["depart"].dt.month
    panel["dep_dow"] = panel["depart"].dt.dayofweek
    panel["strategy"] = panel["series_key"].str.split(":").str[0]
    panel["strategy_code"] = panel["strategy"].astype("category").cat.codes
    return panel.reset_index(drop=True)


def _days_since(changed: pd.Series) -> pd.Series:
    out, run = [], np.nan
    for flag in changed.to_numpy():
        run = 0 if flag else (run + 1 if run == run else np.nan)  # run == run: not NaN
        out.append(run)
    return pd.Series(out, index=changed.index)


def add_targets(panel: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    panel = panel.copy()
    fut_min = panel.groupby(SERIES, sort=False)["price"].transform(
        lambda s: s[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    )
    panel["future_min"] = fut_min
    panel["gain"] = 1 - fut_min / panel["price"]
    panel["y"] = np.where(fut_min.isna(), np.nan, (panel["gain"] >= threshold).astype(float))
    return panel


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=250,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=0,
    )


def recommend(p_drop: float, days_out: int, hist_len: int, s: ModelSettings) -> str:
    if hist_len < MIN_HISTORY or p_drop != p_drop:
        return "数据不足"
    if days_out <= s.min_days_out:
        return "买"
    if p_drop >= s.wait_above:
        return "等"
    if p_drop <= s.buy_below:
        return "买"
    return "观望"


@dataclass
class TrainResult:
    model: HistGradientBoostingClassifier | None
    panel: pd.DataFrame
    metrics: dict


def train(history: pd.DataFrame, s: ModelSettings) -> TrainResult:
    panel = add_targets(add_features(build_panel(history)), s.horizon_days, s.drop_threshold)
    usable = panel["hist_len"] >= MIN_HISTORY
    labeled = panel[usable & panel["y"].notna()]
    metrics: dict = {
        "samples": int(len(labeled)),
        "series": int(panel.groupby(SERIES).ngroups),
        "first_obs": str(panel["obs_date"].min().date()),
        "last_obs": str(panel["obs_date"].max().date()),
        "horizon_days": s.horizon_days,
        "drop_threshold": s.drop_threshold,
    }
    if labeled["y"].nunique() < 2 or len(labeled) < 200:
        metrics["status"] = "not enough labelled data yet"
        panel["p_drop"] = np.nan
        return TrainResult(None, panel, metrics)

    metrics.update(_evaluate(panel, s))
    model = _classifier().fit(labeled[FEATURES], labeled["y"].astype(int))
    panel["p_drop"] = np.where(usable, model.predict_proba(panel[FEATURES])[:, 1], np.nan)
    metrics["base_rate"] = round(float(labeled["y"].mean()), 3)
    metrics["status"] = "ok"
    return TrainResult(model, panel, metrics)


def _evaluate(panel: pd.DataFrame, s: ModelSettings) -> dict:
    h = s.horizon_days
    last = panel["obs_date"].max()
    test_start = last - pd.Timedelta(days=h + TEST_DAYS)
    usable = panel["hist_len"] >= MIN_HISTORY
    labeled = panel[usable & panel["y"].notna()]
    train = labeled[labeled["obs_date"] + pd.Timedelta(days=h) < test_start]
    test = labeled[labeled["obs_date"] >= test_start]
    out: dict = {"test_start": str(test_start.date()), "train_samples": int(len(train)), "test_samples": int(len(test))}
    if train["y"].nunique() < 2 or test["y"].nunique() < 2 or len(train) < 200:
        out["evaluation"] = "skipped: not enough data on one side of the time split"
        return out

    model = _classifier().fit(train[FEATURES], train["y"].astype(int))
    p = model.predict_proba(test[FEATURES])[:, 1]
    y = test["y"].astype(int)
    out.update(
        auc=round(float(roc_auc_score(y, p)), 3),
        brier=round(float(brier_score_loss(y, p)), 4),
        brier_base=round(float(brier_score_loss(y, np.full(len(y), train["y"].mean()))), 4),
        test_base_rate=round(float(y.mean()), 3),
    )

    # Replay as a buying policy: start watching on day t0, buy on the first day
    # the policy says buy (or at t0 + H at the latest).
    sim = panel[usable & (panel["obs_date"] >= test_start)].copy()
    sim["p_drop"] = model.predict_proba(sim[FEATURES])[:, 1]
    out["backtest"] = _backtest(sim, s, last)
    return out


def _backtest(sim: pd.DataFrame, s: ModelSettings, last: pd.Timestamp) -> dict:
    h = s.horizon_days
    rows = []
    for _, g in sim.groupby(SERIES, sort=False):
        g = g.set_index("obs_date").sort_index()
        prices, p_drop, days_out = g["price"], g["p_drop"], g["days_out"]
        min28 = g["min28"]
        for t0 in g.index:
            end = t0 + pd.Timedelta(days=h)
            if end > last or end not in g.index:
                continue
            window = prices.loc[t0:end]
            if len(window) != h + 1:
                continue
            model_wait = (p_drop.loc[t0:end] >= s.wait_above) & (days_out.loc[t0:end] > s.min_days_out)
            rule_wait = (prices.loc[t0:end] > min28.loc[t0:end] * 1.05) & (days_out.loc[t0:end] > s.min_days_out)
            rows.append(
                {
                    "buy_now": window.iloc[0],
                    "model": _first_buy(window, model_wait),
                    "rule": _first_buy(window, rule_wait),
                    "oracle": window.min(),
                    "waited": bool(model_wait.iloc[0]),
                }
            )
    if not rows:
        return {"episodes": 0}
    df = pd.DataFrame(rows)
    saving = lambda col: round(float(((df["buy_now"] - df[col]) / df["buy_now"]).mean() * 100), 2)  # noqa: E731
    waited = df[df["waited"]]
    return {
        "episodes": int(len(df)),
        "model_saving_pct": saving("model"),
        "rule_saving_pct": saving("rule"),
        "oracle_saving_pct": saving("oracle"),
        "model_wait_share": round(float(df["waited"].mean()), 3),
        "model_wait_lost_share": round(float((waited["model"] > waited["buy_now"]).mean()), 3) if len(waited) else None,
    }


def _first_buy(window: pd.Series, wait: pd.Series) -> float:
    for (day, price), w in zip(window.items(), wait.to_numpy()):
        if not w:
            return price
    return window.iloc[-1]


def latest_signals(result: TrainResult, s: ModelSettings) -> pd.DataFrame:
    """One row per series: today's price, its context and the recommendation."""
    panel = result.panel
    idx = panel.groupby(SERIES, sort=False)["obs_date"].idxmax()
    latest = panel.loc[idx].copy()
    stats = panel.groupby(SERIES, sort=False)["price"].agg(hist_min="min", hist_max="max", hist_median="median")
    latest = latest.join(stats, on=SERIES)
    latest["signal"] = [
        recommend(p, d, n, s) for p, d, n in zip(latest["p_drop"], latest["days_out"], latest["hist_len"])
    ]
    keep = SERIES + ["obs_date", "price", "days_out", "hist_len", "hist_min", "hist_max", "hist_median", "p_drop", "signal"]
    out = latest[keep].copy()
    out["depart"] = out["depart"].dt.date.astype(str)
    out["obs_date"] = out["obs_date"].dt.date
    return out.reset_index(drop=True)


def today_only(signals: pd.DataFrame, today: date) -> pd.DataFrame:
    return signals[signals["obs_date"] == today]
