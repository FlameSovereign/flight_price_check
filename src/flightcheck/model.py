"""Buy-now-or-wait model.

Every (series, departure date) has a daily price series. For each day `t` of
each series we build features from prices up to `t` only, and the label

    y = min(price[t+1 .. t+H]) <= price[t] * (1 - drop_threshold)

i.e. "would waiting up to H days have saved at least X%?". One gradient
boosted classifier is trained on all routes pooled together; features are
relative (ratios, changes, days to departure) so routes with very different
price levels can share what they teach.

Evaluation is walk-forward: the last weeks are split into 7-day folds, and
for each fold the model is trained only on samples whose whole label window
ends before the fold starts. Each fold is replayed as a buying policy and
compared with buying immediately, with a simple rule ("wait while the price is
well above its recent low", i.e. a cheaper fare class was open recently and
may reopen) and with the hindsight optimum.

The recommendation shown in the report follows whichever policy (model, rule
or plain buy-now) has saved the most over those past folds, so the model only
drives advice once it has earned it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import ModelSettings

SERIES = ["series_key", "depart", "ret"]
# A series is cut into segments wherever a gap was too long to fill, so every
# row-based window (rolling features, the H-day label) spans consecutive days.
GROUP = SERIES + ["seg"]
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
FOLD_DAYS = 7
MAX_FOLDS = 8
MIN_TRAIN = 200
POLICIES = ("buy-now", "rule", "model")


def load_history(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "history.csv"
    if not path.exists():
        return pd.DataFrame(columns=SERIES + ["obs_date", "price", "source"])
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
        frame = frame.dropna(subset=["price"])
        frame["seg"] = (frame["obs_date"].diff().dt.days.fillna(1) > 1).cumsum()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=GROUP + ["obs_date", "price", "days_out"])
    panel = pd.concat(frames, ignore_index=True)
    panel["days_out"] = (panel["depart"] - panel["obs_date"]).dt.days
    panel = panel[panel["days_out"] >= 0]
    return panel.sort_values(GROUP + ["obs_date"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    g = panel.groupby(GROUP, sort=False)["price"]
    roll = lambda w, f: g.transform(lambda s: getattr(s.rolling(w, min_periods=MIN_HISTORY), f)())  # noqa: E731
    panel["min28"] = roll(28, "min")
    panel["r_min28"] = panel["price"] / panel["min28"] - 1
    panel["r_med28"] = panel["price"] / roll(28, "median") - 1
    panel["r_max28"] = panel["price"] / roll(28, "max") - 1
    for k in (3, 7, 14):
        panel[f"chg{k}"] = panel["price"] / g.shift(k) - 1
    log_ret = np.log(panel["price"]).groupby([panel[c] for c in GROUP], sort=False).diff()
    panel["vol14"] = log_ret.groupby([panel[c] for c in GROUP], sort=False).transform(
        lambda s: s.rolling(14, min_periods=MIN_HISTORY).std()
    )
    changed = (log_ret.fillna(0) != 0).astype(int)
    panel["n_chg28"] = changed.groupby([panel[c] for c in GROUP], sort=False).transform(
        lambda s: s.rolling(28, min_periods=MIN_HISTORY).sum()
    )
    panel["days_since_change"] = changed.groupby([panel[c] for c in GROUP], sort=False).transform(_days_since)
    panel["hist_len"] = g.cumcount() + 1

    # How this departure date compares with neighbouring dates seen the same day
    # (for round trips: neighbouring dates with the same length of stay).
    ret = pd.to_datetime(panel["ret"].replace("", None), errors="coerce")
    panel["stay"] = (ret - panel["depart"]).dt.days.fillna(0)
    panel = panel.sort_values(["series_key", "stay", "obs_date", "depart"])
    xsec = panel.groupby(["series_key", "stay", "obs_date"], sort=False)["price"].transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median()
    )
    panel["r_xsec"] = panel["price"] / xsec - 1
    panel = panel.sort_values(GROUP + ["obs_date"])

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
    fut_min = panel.groupby(GROUP, sort=False)["price"].transform(
        lambda s: s[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    )
    panel["future_min"] = fut_min
    panel["gain"] = 1 - fut_min / panel["price"]
    panel["y"] = np.where(fut_min.isna(), np.nan, (panel["gain"] >= threshold).astype(float))
    return panel


class _Model:
    """Gradient boosting on the features that have at least one value in training.

    scikit-learn refuses a column that is entirely NaN, which happens early on
    (e.g. a 14-day change needs 14 days of history).
    """

    def __init__(self, train: pd.DataFrame):
        self.columns = [c for c in FEATURES if train[c].notna().any()]
        self.clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=250,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=0,
        ).fit(train[self.columns], train["y"].astype(int))

    def p_drop(self, frame: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(frame[self.columns])[:, 1]


def _trainable(labeled: pd.DataFrame) -> bool:
    return len(labeled) >= MIN_TRAIN and labeled["y"].nunique() == 2


def rule_waits(price, min28, days_out, s: ModelSettings):
    return (price > min28 * (1 + s.rule_margin)) & (days_out > s.min_days_out)


def model_waits(p_drop, days_out, s: ModelSettings):
    return (p_drop >= s.wait_above) & (days_out > s.min_days_out)


def recommend(row, policy: str, s: ModelSettings) -> str:
    if row["hist_len"] < MIN_HISTORY:
        return "数据不足"
    if row["days_out"] <= s.min_days_out or policy == "buy-now":
        return "买"
    if policy == "rule":
        return "等" if rule_waits(row["price"], row["min28"], row["days_out"], s) else "买"
    if row["p_drop"] != row["p_drop"]:
        return "数据不足"
    return "等" if model_waits(row["p_drop"], row["days_out"], s) else "买"


@dataclass
class TrainResult:
    model: _Model | None
    panel: pd.DataFrame
    metrics: dict
    policy: str = "buy-now"


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
    panel["p_drop"] = np.nan
    if not _trainable(labeled):
        metrics["status"] = "not enough labelled data yet"
        metrics["policy"] = "buy-now"
        return TrainResult(None, panel, metrics)

    metrics.update(walk_forward(panel, s))
    model = _Model(labeled)
    panel.loc[usable, "p_drop"] = model.p_drop(panel[usable])
    metrics["base_rate"] = round(float(labeled["y"].mean()), 3)
    metrics["status"] = "ok"
    policy = choose_policy(metrics.get("backtest", {}))
    metrics["policy"] = policy
    return TrainResult(model, panel, metrics, policy)


def choose_policy(backtest: dict) -> str:
    """The policy with the highest average saving over past folds; ties go to buying now."""
    if not backtest.get("episodes"):
        return "buy-now"
    savings = {"buy-now": 0.0, "rule": backtest["rule_saving_pct"], "model": backtest["model_saving_pct"]}
    best = max(savings.values())
    return next(p for p in POLICIES if savings[p] == best)


def walk_forward(panel: pd.DataFrame, s: ModelSettings) -> dict:
    h = s.horizon_days
    last = panel["obs_date"].max()
    usable = panel["hist_len"] >= MIN_HISTORY
    labeled = panel[usable & panel["y"].notna()]
    folds, episodes = [], []
    probs, outcomes, base = [], [], []
    for k in range(MAX_FOLDS):
        end = last - pd.Timedelta(days=h + k * FOLD_DAYS)
        start = end - pd.Timedelta(days=FOLD_DAYS - 1)
        train = labeled[labeled["obs_date"] + pd.Timedelta(days=h) < start]
        test = labeled[(labeled["obs_date"] >= start) & (labeled["obs_date"] <= end)]
        if not _trainable(train) or test.empty:
            break
        model = _Model(train)
        p = model.p_drop(test)
        fold = {"start": str(start.date()), "end": str(end.date()), "train": int(len(train)), "test": int(len(test)),
                "test_rate": round(float(test["y"].mean()), 3)}
        if test["y"].nunique() == 2:
            fold["auc"] = round(float(roc_auc_score(test["y"].astype(int), p)), 3)
        probs.append(p)
        outcomes.append(test["y"].to_numpy())
        base.append(np.full(len(test), train["y"].mean()))

        sim = panel[usable & (panel["obs_date"] >= start) & (panel["obs_date"] <= end + pd.Timedelta(days=h))].copy()
        sim["p_drop"] = model.p_drop(sim)
        ep = _episodes(sim, s, start, end)
        fold.update(_summarise(ep))
        episodes.append(ep)
        folds.append(fold)

    if not folds:
        return {"evaluation": "skipped: not enough history before the most recent weeks"}
    y, p, b = np.concatenate(outcomes), np.concatenate(probs), np.concatenate(base)
    aucs = [f["auc"] for f in folds if "auc" in f]
    return {
        "folds": folds[::-1],
        "auc": round(float(np.mean(aucs)), 3) if aucs else None,
        "brier": round(float(brier_score_loss(y, p)), 4),
        "brier_base": round(float(brier_score_loss(y, b)), 4),
        "test_samples": int(len(y)),
        "test_base_rate": round(float(y.mean()), 3),
        "backtest": _summarise(pd.concat(episodes, ignore_index=True)),
    }


def _episodes(sim: pd.DataFrame, s: ModelSettings, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Start watching on each day t0 in [start, end]; buy on the first day the policy
    says buy, or on t0 + H at the latest. Every decision uses only that day's data."""
    h = s.horizon_days
    rows = []
    for _, g in sim.groupby(GROUP, sort=False):
        g = g.set_index("obs_date").sort_index()
        model_wait = model_waits(g["p_drop"], g["days_out"], s)
        rule_wait = rule_waits(g["price"], g["min28"], g["days_out"], s)
        for t0 in g.index[(g.index >= start) & (g.index <= end)]:
            stop = t0 + pd.Timedelta(days=h)
            window = g["price"].loc[t0:stop]
            if len(window) != h + 1:
                continue  # gap in the series or departure inside the window
            rows.append(
                {
                    "buy_now": window.iloc[0],
                    "model": _first_buy(window, model_wait.loc[t0:stop]),
                    "rule": _first_buy(window, rule_wait.loc[t0:stop]),
                    "oracle": window.min(),
                    "model_waited": bool(model_wait.loc[t0]),
                    "rule_waited": bool(rule_wait.loc[t0]),
                }
            )
    return pd.DataFrame(rows, columns=["buy_now", "model", "rule", "oracle", "model_waited", "rule_waited"])


def _summarise(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"episodes": 0}

    def saving(col):
        return round(float(((df["buy_now"] - df[col]) / df["buy_now"]).mean() * 100), 2)

    return {
        "episodes": int(len(df)),
        "model_saving_pct": saving("model"),
        "rule_saving_pct": saving("rule"),
        "oracle_saving_pct": saving("oracle"),
        "model_wait_share": round(float(df["model_waited"].mean()), 3),
        "rule_wait_share": round(float(df["rule_waited"].mean()), 3),
    }


def _first_buy(window: pd.Series, wait: pd.Series) -> float:
    for price, w in zip(window.to_numpy(), wait.to_numpy()):
        if not w:
            return price
    return window.iloc[-1]


def latest_signals(result: TrainResult, s: ModelSettings) -> pd.DataFrame:
    """One row per series: today's price, its context and the recommendation."""
    panel = result.panel
    idx = panel.groupby(SERIES, sort=False)["obs_date"].idxmax()  # latest segment
    latest = panel.loc[idx].copy()
    stats = panel.groupby(SERIES, sort=False)["price"].agg(hist_min="min", hist_max="max", hist_median="median")
    latest = latest.join(stats, on=SERIES)
    latest["signal"] = [recommend(row, result.policy, s) for _, row in latest.iterrows()]
    keep = SERIES + ["obs_date", "price", "days_out", "hist_len", "hist_min", "hist_max", "hist_median", "min28", "p_drop", "signal"]
    out = latest[keep].copy()
    out["depart"] = out["depart"].dt.date.astype(str)
    out["obs_date"] = out["obs_date"].dt.date
    return out.reset_index(drop=True)
