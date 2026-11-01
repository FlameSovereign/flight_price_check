"""Fare-level (regime) forecaster for business-class airfares.

Model
-----
A fare series is a step function: it sits on a fare level for days, then jumps when an airline opens or
closes a fare class.  Fare classes are shared across departure dates of a route, so the levels a series
can jump to are mostly levels that the series itself, or departures near it on the same route, have shown
recently (its "fare ladder").  The model is a daily Markov chain with two parts:

1. When to jump (daily hazard).  A 3-class multinomial logistic regression (stay / jump down / jump up,
   a jump being a log move > 2%) fitted on every observed day-to-day transition in the training data
   (L2, C=0.03, chosen on a temporal split of the training window).  Features: route (series_key),
   kind, days_out (piecewise linear, interacting with kind), the gap between the current level and the
   lowest / highest level the series has shown (L / H: a cheaper class that may reopen, a dearer class
   it may return to), the deviation from the median price of departures within +-7 days on the same
   route, the series' own past jump rate (shrunk to the route rate), days since its last jump, history
   length, and the share of neighbouring departures (+-14 days) that jumped down / up over the last
   3 days (route-wide shocks spread over a few days).

2. Where to jump (destination).  With probability 1-EPS the new level is drawn from the fare ladder:
   the series' own past levels (weight A_OWN * exp(-age/AGE_TAU) per day seen) and the levels shown by
   departures within +-LADDER_DEP days over the last LADDER_DAYS days (weight exp(-|dep diff|/NB_DECAY)
   per observation), restricted to levels below (down jump) or above (up jump) the current level,
   times exp(-|level - x|/TAU).  With probability EPS, or when the ladder has nothing in that direction,
   the move is a generic jump drawn from the empirical jump sizes of the training data (each route-day
   weighted equally so that one route-wide shock does not dominate).  Ladder parameters were chosen on
   the out-of-sample log-likelihood of observed jump landings.  Non-jump days add the empirical small
   day-to-day noise (taxes / FX wiggles).

Forecast: simulate n_paths paths of the chain for h days per query series, updating the level, L, H,
days_out, days since the last jump and the history length each day; quantiles are read off the
simulated prices at T + h.  Route rates rest on a handful of route-wide shocks, so each path also gets
its own random shift of the down / up log-odds (sd RATE_SD, mean-corrected): the forecast integrates over
plausible rates instead of trusting the point estimate.  The same paths answer the first-passage
question "how likely is a fare <= X at some point before date D" via prob_at_or_below(query, h, X), and
paths(query, marks) returns prices and running minima on several days from one run.

Long horizons (h >> 30, e.g. 60-220 days for May 2027 outbound / Aug 2027 return legs)
------------------------------------------------------------------------------------
The same daily chain simply runs longer, so the forecast spreads out at the fitted per-day rates:
  * days_out falls along each path and the fitted days_out curve (clipped to the training range,
    ~12-356 days) moves the jump rates towards the near-departure regime as departure approaches.
  * Origin snapshots fade: the neighbour-shock shares fade to the route average over 3 days, and the
    pull towards the neighbours' median halves every NB_HALFLIFE days (the neighbours move too).
  * The fare ladder stays the one seen at the origin, so destinations remain the fare classes this
    route has been selling recently.  Generic jumps may create levels outside the ladder, but a path can
    overshoot the ladder's range by at most OVERSHOOT (log) before further generic jumps in that
    direction are refused (the day counts as "stay").  Fare classes are finite, and without this bound
    the up/down asymmetry of a two-month window compounds into an unbounded random walk.
  * Once days_out reaches 0 a path is frozen.
So for long h the distribution tends towards "where on its recent fare ladder the series will sit",
tilted by the fitted up/down asymmetry and days_out curve.  Caveat: training covers only ~2 months of
observations (Aug-Oct 2026, a period in which many fares rose), so seasonal effects outside that window
(fare releases, holiday peaks, sales) are not learned; long-horizon intervals mean "if the coming months
behave like the last two".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

THR = 0.02          # |log change| above this is a jump (fare-class change), below is noise
NB_LEVEL_WIN = 7    # neighbours for the level anchor: departures within +-7 days
NB_SHOCK_WIN = 14   # neighbours for common shocks: departures within +-14 days
SHOCK_DAYS = 3      # neighbour jumps over the last 3 observed days
OWN_PRIOR = 10.0    # pseudo-days shrinking a series' own jump rate to its route rate
DO_KNOTS = (30, 60, 120, 200, 300)
NB_HALFLIFE = 30.0  # days over which the pull towards the neighbours' median halves in simulation
C_REG = 0.03        # L2 strength of the hazard model
# fare-ladder destination model (chosen on out-of-sample landing log-likelihood of training jumps)
EPS = 0.2
A_OWN = 10.0
AGE_TAU = 7.0
TAU = 2.0
NB_DECAY = 7.0
LADDER_DAYS = 14
LADDER_DEP = 60
OVERSHOOT = 0.3     # max log distance a path may move beyond its ladder's range via generic jumps
RATE_SD = 0.5       # sd of a per-path random effect on the down / up log-odds (rate uncertainty)


def _prep(train: pd.DataFrame) -> pd.DataFrame:
    t = train.sort_values(["series_key", "depart", "seg", "obs_date"]).reset_index(drop=True).copy()
    t["lp"] = np.log(t["price"].astype(float))
    t["gid"] = t.groupby(["series_key", "depart", "seg"], sort=False).ngroup()
    t["di"] = t["obs_date"].values.astype("datetime64[D]").astype(np.int64)
    t["depi"] = t["depart"].values.astype("datetime64[D]").astype(np.int64)
    g = t.groupby("gid", sort=False)
    t["d"] = g["lp"].diff()                  # change INTO this row
    t["L"] = g["lp"].cummin()                # lowest level up to and including this row
    t["H"] = g["lp"].cummax()
    t["n_hist"] = g.cumcount()               # number of previous rows in this segment
    jump = (t["d"].abs() > THR).astype(float)
    t["jdn"] = (t["d"] < -THR).astype(float)
    t["jup"] = (t["d"] > THR).astype(float)
    t["n_jumps"] = jump.groupby(t["gid"]).cumsum()
    idx = np.arange(len(t))
    last_j = pd.Series(np.where(jump.values > 0, idx, np.nan)).groupby(t["gid"].values).ffill()
    start = idx - g.cumcount().values
    t["dsj"] = np.where(np.isnan(last_j.values), idx - start, idx - last_j.values)
    t["never"] = np.isnan(last_j.values).astype(float)
    return t


def _neighbour_features(t: pd.DataFrame) -> pd.DataFrame:
    """Neighbour level anchor and recent common-shock shares, per row (state on that obs date)."""
    nb_med = np.full(len(t), np.nan)
    nb_dn = np.full(len(t), np.nan)
    nb_up = np.full(len(t), np.nan)
    for _, df in t.groupby("series_key", sort=False):
        ix = df.index.values
        dates, deps = df["di"].values, df["depi"].values
        d0, p0 = dates.min(), deps.min()
        nd, npd = dates.max() - d0 + 1, deps.max() - p0 + 1
        valid = np.zeros((nd, npd))
        dn = np.zeros((nd, npd))
        up = np.zeros((nd, npd))
        ri, ci = dates - d0, deps - p0
        hp = df["d"].notna().values
        np.add.at(valid, (ri[hp], ci[hp]), 1.0)
        np.add.at(dn, (ri[hp], ci[hp]), df["jdn"].values[hp])
        np.add.at(up, (ri[hp], ci[hp]), df["jup"].values[hp])

        def box(a):
            c = np.cumsum(np.pad(a, ((1, 0), (0, 0))), axis=0)
            lo = np.maximum(np.arange(nd) - SHOCK_DAYS + 1, 0)
            a2 = c[np.arange(nd) + 1] - c[lo]
            c2 = np.cumsum(np.pad(a2, ((0, 0), (1, 0))), axis=1)
            hi = np.minimum(np.arange(npd) + NB_SHOCK_WIN + 1, npd)
            lo2 = np.maximum(np.arange(npd) - NB_SHOCK_WIN, 0)
            return c2[:, hi] - c2[:, lo2]

        vb, db, ub = box(valid), box(dn), box(up)
        own = df.groupby("gid", sort=False)
        own_v = own["d"].transform(lambda s: s.notna().astype(float).rolling(SHOCK_DAYS, min_periods=1).sum()).values
        own_d = own["jdn"].transform(lambda s: s.rolling(SHOCK_DAYS, min_periods=1).sum()).values
        own_u = own["jup"].transform(lambda s: s.rolling(SHOCK_DAYS, min_periods=1).sum()).values
        cnt = vb[ri, ci] - own_v
        with np.errstate(invalid="ignore", divide="ignore"):
            nb_dn[ix] = np.where(cnt > 0, (db[ri, ci] - own_d) / cnt, np.nan)
            nb_up[ix] = np.where(cnt > 0, (ub[ri, ci] - own_u) / cnt, np.nan)
        for _, dd in df.groupby("di", sort=False):
            dep = dd["depi"].values
            lp = dd["lp"].values
            diff = np.abs(dep[:, None] - dep[None, :])
            m = (diff <= NB_LEVEL_WIN) & (diff > 0)
            has = m.any(axis=1)
            if has.any():
                lpm = np.where(m, lp[None, :], np.nan)
                nb_med[dd.index.values[has]] = np.nanmedian(lpm[has], axis=1)
    return pd.DataFrame({"nb_med": nb_med, "nb_dn": nb_dn, "nb_up": nb_up}, index=t.index)


class Forecaster:
    def __init__(self, n_paths: int = 2000, C: float = C_REG, seed: int = 0, nb_halflife: float = NB_HALFLIFE,
                 use_nb: bool = True, use_shock: bool = True, use_own: bool = True, ladder: bool = True,
                 eps: float = EPS, a_own: float = A_OWN, age_tau: float = AGE_TAU, tau: float = TAU,
                 nb_decay: float = NB_DECAY, overshoot: float = OVERSHOOT, rate_sd: float = RATE_SD):
        self.n_paths = n_paths
        self.C = C
        self.seed = seed
        self.nb_halflife = nb_halflife
        self.use_nb, self.use_shock, self.use_own = use_nb, use_shock, use_own
        self.ladder = ladder
        self.eps, self.a_own, self.age_tau, self.tau, self.nb_decay = eps, a_own, age_tau, tau, nb_decay
        self.overshoot = overshoot
        self.rate_sd = rate_sd

    # ------------------------------------------------------------------ features
    def _blocks(self, back, do, gap, gapH, gnb, has_nb, own, dsj, never, nbdn, nbup, nh):
        """Non-route feature blocks (list of 2-D arrays); the route one-hot is handled separately."""
        do = np.clip(do, self.do_lo, self.do_hi).astype(float)
        bas = np.column_stack([np.log(do) - 4.5] + [np.maximum(do - k, 0.0) / 100.0 for k in DO_KNOTS])
        out = [bas, bas * back[:, None]]
        atL = (gap <= THR).astype(float)
        atH = (gapH <= THR).astype(float)
        gapc = np.minimum(gap, 0.8)
        gapHc = np.minimum(gapH, 0.8)
        out.append(np.column_stack([atL, gapc, np.sqrt(gapc), atH, gapHc, np.sqrt(gapHc), atL * atH]))
        if self.use_nb:
            g = np.where(has_nb > 0, gnb, 0.0)
            out.append(np.column_stack([np.clip(g, 0, 0.6), np.clip(-g, 0, 0.6),
                                        (g > THR).astype(float), (g < -THR).astype(float), 1.0 - has_nb]))
        if self.use_own:
            out.append(np.column_stack([np.log(own), np.log1p(np.minimum(dsj, 60)), never,
                                        np.log1p(np.minimum(nh, 60))]))
        if self.use_shock:
            out.append(np.column_stack([nbdn, nbup, np.sqrt(nbdn), np.sqrt(nbup)]))
        return out

    def _onehot(self, ri):
        R = np.zeros((len(ri), self.n_routes))
        R[np.arange(len(ri)), ri] = 1.0
        return R

    def _own_rate(self, n_jumps, n_hist, ri):
        return (n_jumps + OWN_PRIOR * self.route_rate[ri]) / (n_hist + OWN_PRIOR)

    def _state_blocks(self, s):
        ri = s["route_idx"].values.astype(int)
        own = self._own_rate(s["n_jumps"].values, s["n_hist"].values, ri)
        nbdn = np.where(np.isnan(s["nb_dn"].values), self.route_nbdn[ri], s["nb_dn"].values)
        nbup = np.where(np.isnan(s["nb_up"].values), self.route_nbup[ri], s["nb_up"].values)
        gnb = s["lp"].values - s["nb_med"].values
        has_nb = (~np.isnan(gnb)).astype(float)
        return ri, self._blocks(s["back"].values, s["days_out"].values,
                                s["lp"].values - s["L"].values, s["H"].values - s["lp"].values,
                                np.nan_to_num(gnb), has_nb, own, s["dsj"].values, s["never"].values,
                                nbdn, nbup, s["n_hist"].values)

    def fit(self, train: pd.DataFrame) -> None:
        t = _prep(train)
        t = t.join(_neighbour_features(t))
        self.routes = sorted(t["series_key"].unique())
        self.route_map = {r: i for i, r in enumerate(self.routes)}
        self.n_routes = len(self.routes)
        t["route_idx"] = t["series_key"].map(self.route_map).astype(int)
        t["back"] = (t["kind"] == "back").astype(float)
        self.do_lo, self.do_hi = float(max(t["days_out"].min(), 1)), float(t["days_out"].max())

        is_tr = t["d"].notna().values
        s = t.shift(1)[is_tr]       # state = previous row of the same segment
        o = t[is_tr]                # outcome row
        jump = (o["d"].abs() > THR).astype(float)
        rr = jump.groupby(o["series_key"]).mean()
        overall = float(jump.mean()) if len(jump) else 0.05
        self.route_rate = np.array([rr.get(r, overall) for r in self.routes]) + 1e-3
        mdn = o.groupby("series_key")["jdn"].mean()
        mup = o.groupby("series_key")["jup"].mean()
        self.route_nbdn = np.array([mdn.get(r, 0.0) for r in self.routes])
        self.route_nbup = np.array([mup.get(r, 0.0) for r in self.routes])

        ri, blocks = self._state_blocks(s)
        X = np.hstack([self._onehot(ri)] + blocks)
        y = o["d"].values
        cls = np.where(y < -THR, 1, np.where(y > THR, 2, 0))
        self.clf = None
        if len(np.unique(cls)) >= 2:
            self.clf = LogisticRegression(C=self.C, max_iter=5000)
            self.clf.fit(X, cls)
            self.classes_ = self.clf.classes_
        else:  # no jumps at all yet (e.g. a brand-new watchlist): every day is a stay
            self.classes_ = np.unique(cls)
        self.full3 = len(self.classes_) == 3
        if self.full3:
            self.coef_route = self.clf.coef_[:, :self.n_routes]          # (3, n_routes)
            self.coef_rest = self.clf.coef_[:, self.n_routes:]           # (3, k)
            self.intercept = self.clf.intercept_

        # pools: noise on stay days; generic jump sizes, each route-day event weighted equally
        self.noise = y[cls == 0] if (cls == 0).any() else np.zeros(1)
        ev = o["series_key"].astype(str) + "|" + o["di"].astype(str)
        self.dn_sizes, self.dn_w = self._pool(y, cls == 1, ev, np.array([-0.05, -0.1, -0.2, -0.3]))
        self.up_sizes, self.up_w = self._pool(y, cls == 2, ev, np.array([0.05, 0.1, 0.2, 0.3]))
        self.t = t
        self.state = t.set_index(["series_key", "depart", "seg", "obs_date"])
        self._ladder_cache = {}

    @staticmethod
    def _pool(y, m, ev, default):
        if m.sum() < 5:
            return default, np.cumsum(np.full(len(default), 1.0 / len(default)))
        v = y[m]
        e = pd.Series(ev.values[m])
        w = 1.0 / e.map(e.value_counts()).values
        return v, np.cumsum(w / w.sum())

    # ------------------------------------------------------------------ fare ladder
    def _ladders(self, keys):
        """For each query (series_key, depart, seg, origin date): candidate levels and base weights."""
        out = []
        t = self.t
        per_T = {}
        for (key, dep, seg, T) in keys:
            Ti = int(np.datetime64(T, "D").astype(np.int64))
            if Ti not in per_T:
                recent = t[(t["di"] <= Ti) & (t["di"] > Ti - LADDER_DAYS)]
                own_hist = t[t["di"] <= Ti]
                per_T[Ti] = ({k: df for k, df in recent.groupby("series_key", sort=False)},
                             {k: (df["lp"].values, df["di"].values)
                              for k, df in own_hist.groupby(["series_key", "depart", "seg"], sort=False)})
            by_route, own_g = per_T[Ti]
            depi = np.datetime64(dep, "D").astype(np.int64)
            lv, wt = [], []
            og = own_g.get((key, dep, seg))
            if og is not None:
                lv.append(og[0])
                wt.append(self.a_own * np.exp(-(Ti - og[1]) / self.age_tau))
            df = by_route.get(key)
            if df is not None:
                m = (np.abs(df["depi"].values - depi) <= LADDER_DEP) & (df["depi"].values != depi)
                lv.append(df["lp"].values[m])
                wt.append(np.exp(-np.abs(df["depi"].values[m] - depi) / self.nb_decay))
            if lv:
                lv, wt = np.concatenate(lv), np.concatenate(wt)
                b = np.round(lv / 0.005).astype(np.int64)        # merge levels within ~0.5%
                _, inv = np.unique(b, return_inverse=True)
                ws = np.bincount(inv, weights=wt)
                ls = np.bincount(inv, weights=lv * wt) / np.maximum(ws, 1e-300)
                keep = ws > 1e-12
                out.append((ls[keep], ws[keep]))
            else:
                out.append((np.zeros(0), np.zeros(0)))
        return out

    # ------------------------------------------------------------------ forecast
    def _probs_from(self, ri, blocks, re=None):
        if not self.full3:
            X = np.hstack([self._onehot(ri)] + blocks)
            P = np.zeros((len(X), 3))
            if self.clf is None:
                P[:, self.classes_] = 1.0
            else:
                P[:, self.classes_] = self.clf.predict_proba(X)
            return P
        z = self.coef_route[:, ri].T + self.intercept[None, :]
        if re is not None:
            z = z + re
        j = 0
        for b in blocks:
            k = b.shape[1]
            z = z + b @ self.coef_rest[:, j:j + k].T
            j += k
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def quantiles(self, query: pd.DataFrame, h: int, qs) -> np.ndarray:
        """Quantiles of each query series' price at T + h (dollars), shape (len(query), len(qs))."""
        if len(query) == 0:
            return np.zeros((0, len(qs)))
        final, _ = self._simulate(query, h)
        return np.exp(np.quantile(final, qs, axis=1).T)

    def prob_at_or_below(self, query: pd.DataFrame, h: int, price) -> np.ndarray:
        """P(the price is at or below `price` on at least one day in (T, T + h]) per query series --
        the first-passage question "how likely is a fare <= X before date D" (h = D - T in days).
        `price` is a scalar or an array aligned with `query`.  Not used by the harness."""
        if len(query) == 0:
            return np.zeros(0)
        _, run_min = self._simulate(query, h)
        thr = np.log(np.broadcast_to(np.asarray(price, dtype=float), (len(query),)))
        return (run_min <= thr[:, None] + 1e-9).mean(axis=1)

    def paths(self, query: pd.DataFrame, marks) -> tuple[np.ndarray, np.ndarray]:
        """Log price and running minimum of log price over [T, T + m] (today included) for each day count m
        in `marks`; two arrays of shape (len(query), n_paths, len(marks))."""
        marks = [max(int(m), 0) for m in marks]
        if len(query) == 0:
            return np.zeros((0, self.n_paths, len(marks))), np.zeros((0, self.n_paths, len(marks)))
        lev, run_min = self._simulate(query, max(marks, default=0), marks)
        today = np.log(query["price"].astype(float).to_numpy())[:, None, None]
        return lev, np.minimum(run_min, today)

    def _simulate(self, query: pd.DataFrame, h: int, marks=()):
        """Simulate n_paths paths per query row; returns (log price at T+h, running min over (T, T+h]),
        or with `marks` both on each of those days, shape (n, n_paths, len(marks))."""
        rng = np.random.default_rng(self.seed + 7919 * int(h))
        q = query.reset_index(drop=True)
        n = len(q)
        keys4 = list(zip(q["series_key"], q["depart"], q["seg"], q["obs_date"]))
        st = self.state.reindex(keys4)
        miss = st["lp"].isna().values
        lp0 = np.log(q["price"].astype(float).values)
        L = np.where(miss, lp0, np.fmin(st["L"].values, lp0))
        H = np.where(miss, lp0, np.fmax(st["H"].values, lp0))
        ri = np.array([self.route_map.get(k, -1) for k in q["series_key"]])
        ri = np.where(ri < 0, 0, ri)
        back = (q["kind"].values == "back").astype(float)
        do0 = q["days_out"].values.astype(float)
        nb_med = st["nb_med"].values
        has_nb = (~np.isnan(nb_med)).astype(float)
        nb_med = np.where(has_nb > 0, nb_med, lp0)
        n_hist = np.nan_to_num(st["n_hist"].values, nan=0.0)
        n_jumps = np.nan_to_num(st["n_jumps"].values, nan=0.0)
        own = self._own_rate(n_jumps, n_hist, ri)
        dsj0 = np.nan_to_num(st["dsj"].values, nan=0.0)
        never = np.nan_to_num(st["never"].values, nan=1.0)
        nbdn0 = np.where(np.isnan(st["nb_dn"].values), self.route_nbdn[ri], st["nb_dn"].values)
        nbup0 = np.where(np.isnan(st["nb_up"].values), self.route_nbup[ri], st["nb_up"].values)

        # fare ladders as a padded matrix (n x K) plus their range
        LV = np.full((n, 1), np.nan)
        LW = np.zeros((n, 1))
        if self.ladder:
            ck = tuple(keys4)
            if ck not in self._ladder_cache:
                self._ladder_cache = {ck: self._ladders(keys4)}
            lad = self._ladder_cache[ck]
            K = max([len(a) for a, _ in lad] + [1])
            LV = np.full((n, K), np.nan)
            LW = np.zeros((n, K))
            for i, (a, w) in enumerate(lad):
                LV[i, :len(a)] = a
                LW[i, :len(a)] = w
        lad_lo = np.fmin(np.nanmin(np.where(np.isnan(LV), np.inf, LV), axis=1), lp0)
        lad_hi = np.fmax(np.nanmax(np.where(np.isnan(LV), -np.inf, LV), axis=1), lp0)

        P = self.n_paths
        rep = lambda a: np.repeat(np.asarray(a, dtype=float), P)
        qi = np.repeat(np.arange(n), P)
        x = rep(lp0)
        Lp, Hp = rep(L), rep(H)
        do = rep(do0)
        dsj, nev, nh = rep(dsj0), rep(never), rep(n_hist)
        ri_r = np.repeat(ri, P)
        back_r, own_r, hasnb_r, nbmed_r = rep(back), rep(own), rep(has_nb), rep(nb_med)
        nbdn_r, nbup_r = rep(nbdn0), rep(nbup0)
        rdn, rup = self.route_nbdn[ri_r], self.route_nbup[ri_r]
        lo_r, hi_r = rep(lad_lo) - self.overshoot, rep(lad_hi) + self.overshoot
        re = None
        if self.rate_sd > 0:
            # rate uncertainty: each path gets its own down / up log-odds shift (mean-corrected so that the
            # average odds are unchanged), so long horizons integrate over plausible route rates
            sd = self.rate_sd
            re = np.zeros((len(x), 3))
            re[:, 1:] = sd * rng.standard_normal((len(x), 2)) - 0.5 * sd * sd
        run_min = np.full(len(x), np.inf)
        at = {m: i for i, m in enumerate(marks)}
        lev_at = np.empty((len(x), len(marks)))
        min_at = np.empty((len(x), len(marks)))
        if 0 in at:
            lev_at[:, at[0]], min_at[:, at[0]] = x, np.inf
        for k in range(int(h)):
            alive = do > 0
            w = max(0.0, (SHOCK_DAYS - k) / SHOCK_DAYS)
            fade = 0.5 ** (k / self.nb_halflife) if self.nb_halflife > 0 else 1.0
            blocks = self._blocks(back_r, do, x - Lp, Hp - x, (x - nbmed_r) * fade, hasnb_r, own_r, dsj, nev,
                                  w * nbdn_r + (1 - w) * rdn, w * nbup_r + (1 - w) * rup, nh)
            Pr = self._probs_from(ri_r, blocks, re)
            u = rng.random(len(x))
            c = np.where(u < Pr[:, 1], 1, np.where(u < Pr[:, 1] + Pr[:, 2], 2, 0))
            c[~alive] = 0
            nx = x + np.where(alive & (c == 0), self.noise[rng.integers(0, len(self.noise), len(x))], 0.0)
            moved = np.zeros(len(x), dtype=bool)
            for cc, sizes, cw, sgn in ((1, self.dn_sizes, self.dn_w, -1), (2, self.up_sizes, self.up_w, 1)):
                j = np.flatnonzero(c == cc)
                if len(j) == 0:
                    continue
                xj = x[j]
                gen = xj + sizes[np.minimum(np.searchsorted(cw, rng.random(len(j))), len(sizes) - 1)]
                # generic jumps may overshoot the ladder's range by at most `overshoot`
                gen_ok = (gen >= lo_r[j]) if sgn < 0 else (gen <= hi_r[j])
                dest = np.where(gen_ok, gen, xj)
                ok_move = gen_ok.copy()
                if self.ladder:
                    lv = LV[qi[j]]
                    lw = LW[qi[j]]
                    xs = xj[:, None]
                    cand = (lv < xs - THR) if sgn < 0 else (lv > xs + THR)
                    ww = np.where(cand, lw * np.exp(-np.abs(np.nan_to_num(lv) - xs) / self.tau), 0.0)
                    tot = ww.sum(axis=1)
                    use = (tot > 0) & (rng.random(len(j)) >= self.eps)
                    if use.any():
                        cdf = np.cumsum(ww[use], axis=1) / tot[use][:, None]
                        pick = np.minimum((rng.random(use.sum())[:, None] > cdf).sum(axis=1), lv.shape[1] - 1)
                        dest[use] = lv[use][np.arange(use.sum()), pick]
                        ok_move[use] = True
                nx[j] = np.where(ok_move, dest, x[j])
                moved[j] = ok_move
            x = nx
            run_min = np.minimum(run_min, x)
            Lp = np.minimum(Lp, x)
            Hp = np.maximum(Hp, x)
            dsj = np.where(moved, 0.0, dsj + 1)
            nev = np.where(moved, 0.0, nev)
            nh = nh + alive
            do = do - 1
            if k + 1 in at:
                lev_at[:, at[k + 1]], min_at[:, at[k + 1]] = x, run_min
        if marks:
            return lev_at.reshape(n, P, -1), min_at.reshape(n, P, -1)
        if h <= 0:
            run_min = x.copy()
        return x.reshape(n, P), run_min.reshape(n, P)
