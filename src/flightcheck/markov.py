"""Markov fare-ladder simulator: a lead-time-aware, state-conditioned block bootstrap of daily fare moves.

How it works
------------
fit(train) turns every observed one-day transition (price_t -> price_t+1 of a series) into a "move"
record and files it under the *state* the series was in on day t:

  sk   route key (strategy:city[:back])         grp  strategy|kind        kind  out/back
  dto  days-to-departure bucket (<30, 30-60, 60-90, 90-150, 150-240, 240+)
  gap  log(price / min over the last 14 days)  -> "is a cheaper fare that just closed likely to reopen?"
  lvl  log(price / route anchor), anchor = median log price of the route in training
  act  number of >= 1% moves in the last 14 days (flickering fares vs. sticky ones)

Each move record is one of
  * small / relative move:  lp_{t+1} = lp_t + r        (FX noise on CNY-priced return legs, small steps)
  * revert:                 lp_{t+1} = recent_min + v  (an elevated price falls back to its 14-day low)
  * jump (|r| >= 3%):       lp_{t+1} = route_anchor + z_dest  (snap to a fare level seen on this route)

quantiles(query, h, qs) starts every query series from its real recent history (14-day window, route
anchor, activity count) and simulates n_paths daily paths for h days. Whenever a path needs a new
"donor", it computes its current state and draws a training transition filed under that state; the
path then replays that donor's next `block` = 3 consecutive daily moves (same series, same days), each
re-applied through the encodings above against the path's own recent minimum / route anchor, and then
draws a new donor from its updated state. Replaying 3-day blocks keeps short-lived dynamics intact
(a fare that closes and reopens the next day, or a spike that sticks), which a pure 1-day chain smears
out; 1-day steps over-predicted reversion on the internal holdout.
Sparse states back off hierarchically:
  (sk,dto,lvl,gap) -> (sk,lvl,gap,act) -> (grp,dto,lvl,gap) -> (grp,lvl,gap,act) -> (kind,lvl,gap,act)
  -> (lvl,gap,act) -> (gap,act) -> all moves,
with a cell used with probability n_eff / (n_eff + k), k = 10. n_eff counts distinct (route,
observation day) clusters rather than rows, because one fare-class change moves many departure dates of
a route at once. Moves are recency weighted (half-life 14 days).

Long horizons (h = 60..220 days)
--------------------------------
Nothing is extrapolated in closed form; the same chain simply runs longer:
  * days-to-departure falls along each path, so the near-departure behaviour learned from the closest
    departures (dto < 30, e.g. Houston/ANA fares jumping to ~$15k) switches on automatically late in
    the path,
  * big moves land on fare levels actually observed on the route (relative to its anchor), so repeated
    jumps do not compound into a runaway random walk: the distribution widens with h and then
    saturates toward the route's empirical fare-level mix (mean reversion to the route's typical fares),
  * the Aug -> Oct 2026 upward drift is only carried as far as it is encoded in state-conditional moves;
    it is not compounded as a calendar trend.
Caveats: moves come from ~2 months of data (Aug-Oct 2026) weighted toward the last few weeks, so
seasonal sales / holiday repricing outside that window are not represented; the dto < 30 behaviour is
learned from a handful of Oct/Nov departures per route; route-wide repricing events are rare and
clustered, so tail probabilities at long horizons are rough; a route that never moved in the sample
(Turkish-only outbound, flat at $7,477 for two months) gets an essentially zero-width forecast even at
220 days, which means "no evidence of movement", not certainty.

Extra API: simulate(query, h, return_min=True) returns the simulated end log-prices and running minima;
paths(query, marks) returns both on several days in one run; hit_probability(query, threshold, h) =
P(price <= threshold on some day in [T, T+h]), which answers "how likely is a price at or below X
before date D" (h = D - T).

Built and scored in a fixed backtest (two forecast origins, 7/14/30-day horizons, pinball loss on log
price against a random walk); see forecast.py for how the report uses it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["series_key", "depart", "seg"]

DEFAULT_LEVELS = (
    ("sk", "dto", "lvl", "gap"),
    ("sk", "lvl", "gap", "act"),
    ("grp", "dto", "lvl", "gap"),
    ("grp", "lvl", "gap", "act"),
    ("kind", "lvl", "gap", "act"),
    ("lvl", "gap", "act"),
    ("gap", "act"),
    (),
)


class Forecaster:
    def __init__(self, window=14, eps_ref=0.01, gap_edges=(0.01, 0.05, 0.15, 0.4),
                 dto_edges=(30, 60, 90, 150, 240), lvl_edges=(-0.2, 0.2, 0.6),
                 dur_edges=(1, 3, 7, 14), dur_thr=0.01, dur_censor=14, act_edges=(1, 2, 4), block=3,
                 levels=DEFAULT_LEVELS, k=10.0, half_life=14.0, jump_thr=0.03, level_mix=1.0,
                 n_paths=2000, max_batch=400_000, seed=0):
        self.window = int(window)              # days in the rolling "recent minimum" window
        self.eps_ref = float(eps_ref)          # within 1% of the recent min counts as "at the min"
        self.gap_edges = np.asarray(gap_edges, float)
        self.dto_edges = np.asarray(dto_edges, float)
        self.lvl_edges = np.asarray(lvl_edges, float)
        self.dur_edges = np.asarray(dur_edges, float)
        self.dur_thr = float(dur_thr)
        self.dur_censor = float(dur_censor)
        self.act_edges = np.asarray(act_edges, float)
        self.block = int(block)                # days a sampled donor path keeps driving a simulated path
        self.levels = [tuple(l) for l in levels]
        self.k = float(k)
        self.half_life = half_life
        self.jump_thr = float(jump_thr)
        self.level_mix = float(level_mix)      # share of big jumps applied as "snap to fare level"
        self.n_paths = int(n_paths)
        self.max_batch = int(max_batch)
        self.seed = seed

    # ------------------------------------------------------------------ state features
    def _cat_features(self, df):
        sk = df["series_key"].astype(str)
        grp = df["strategy"].astype(str) + "|" + df["kind"].astype(str)
        return {
            "sk": sk.map(self.sk_ids).fillna(-1).astype(np.int64).to_numpy(),
            "grp": grp.map(self.grp_ids).fillna(-1).astype(np.int64).to_numpy(),
            "kind": (df["kind"].astype(str).to_numpy() == "back").astype(np.int64),
        }

    def _card(self):
        return {"sk": len(self.sk_ids), "grp": len(self.grp_ids), "kind": 2,
                "dto": len(self.dto_edges) + 1, "gap": len(self.gap_edges) + 1,
                "lvl": len(self.lvl_edges) + 1, "dur": len(self.dur_edges) + 1,
                "act": len(self.act_edges) + 1}

    def _cells(self, feats, lev, n):
        """Mixed-radix cell id of one back-off level; -1 where a categorical is unknown."""
        card = self._card()
        cid = np.zeros(n, dtype=np.int64)
        bad = np.zeros(n, dtype=bool)
        for f in lev:
            v = feats[f]
            bad |= v < 0
            cid = cid * card[f] + np.maximum(v, 0)
        return np.where(bad, -1, cid)

    def _ncell(self, lev):
        card = self._card()
        return int(np.prod([card[f] for f in lev])) if lev else 1

    def _anchor_of(self, df):
        return df["series_key"].astype(str).map(self.anchor).to_numpy(dtype=float)

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> None:
        t = train.sort_values(KEYS + ["obs_date"]).reset_index(drop=True)
        t["lp"] = np.log(t["price"].astype(float))
        g = t.groupby(KEYS, sort=False)
        t["ref"] = g["lp"].transform(lambda s: s.rolling(self.window, min_periods=1).min())
        t["nxt"] = g["lp"].shift(-1)
        t["nxt_day"] = g["obs_date"].shift(-1)
        # days since the last move >= dur_thr (a series with no move since its first observation is
        # treated as having been stable for dur_censor days before it)
        pos = g.cumcount().to_numpy()
        chg = (g["lp"].diff().abs() >= self.dur_thr).to_numpy()
        lastchg = (pd.Series(np.where(chg, pos, np.nan))
                   .groupby([t[c] for c in KEYS], sort=False).ffill().to_numpy())
        t["dur"] = np.where(np.isnan(lastchg), pos + self.dur_censor, pos - np.nan_to_num(lastchg))
        t["chg"] = chg.astype(float)
        t["act"] = (t["chg"].groupby([t[c] for c in KEYS], sort=False)
                    .transform(lambda s: s.rolling(self.window, min_periods=1).sum()).to_numpy())
        self.hist = t[KEYS + ["obs_date", "lp", "dur", "chg"]]

        self.sk_ids = {s: i for i, s in enumerate(sorted(t["series_key"].astype(str).unique()))}
        grp = t["strategy"].astype(str) + "|" + t["kind"].astype(str)
        self.grp_ids = {s: i for i, s in enumerate(sorted(grp.unique()))}
        self.anchor = t.groupby(t["series_key"].astype(str))["lp"].median().to_dict()

        d = t[(t["nxt_day"] - t["obs_date"]).dt.days == 1].reset_index(drop=True)
        lp, ref, nxt = d["lp"].to_numpy(), d["ref"].to_numpy(), d["nxt"].to_numpy()
        anc = self._anchor_of(d)
        gapb = np.searchsorted(self.gap_edges, lp - ref, side="right")
        r = nxt - lp
        rev = (gapb >= 1) & (nxt <= ref + self.eps_ref)
        jump = (~rev) & (np.abs(r) >= self.jump_thr)
        typ = np.where(rev, 1, np.where(jump, 2, 0)).astype(np.int8)
        val = np.where(rev, nxt - ref, r)
        zdst = nxt - anc
        feats = self._cat_features(d)
        feats["gap"] = gapb
        feats["dto"] = np.searchsorted(self.dto_edges, d["days_out"].to_numpy(), side="right")
        feats["lvl"] = np.searchsorted(self.lvl_edges, lp - anc, side="right")
        feats["dur"] = np.searchsorted(self.dur_edges, d["dur"].to_numpy(), side="right")
        feats["act"] = np.searchsorted(self.act_edges, d["act"].to_numpy(), side="right")
        last = t["obs_date"].max()
        if self.half_life:
            w = 0.5 ** ((last - d["obs_date"]).dt.days.to_numpy() / float(self.half_life))
        else:
            w = np.ones(len(d))
        clus = pd.factorize(d["series_key"].astype(str) + "|" + d["obs_date"].astype(str))[0]
        nclus = int(clus.max()) + 1 if len(clus) else 1

        # base arrays in (series, day) order; CONT marks that the next record is the same series' next day
        self.TYP, self.VAL, self.ZDST = typ, val, zdst
        same = np.zeros(len(d), dtype=bool)
        if len(d) > 1:
            nd = d["obs_date"].to_numpy()
            same[:-1] = ((d[KEYS].iloc[1:].to_numpy() == d[KEYS].iloc[:-1].to_numpy()).all(axis=1)
                         & ((nd[1:] - nd[:-1]) == np.timedelta64(1, "D")))
        self.CONT = same
        O_, W_ = [], []
        self.lev_tab = []
        off = 0
        for lev in self.levels:
            n = self._ncell(lev)
            c = self._cells(feats, lev, len(d))
            order = np.argsort(c, kind="stable")
            cs, wo = c[order], w[order]
            cnt = np.bincount(cs, minlength=n)
            wsum = np.bincount(cs, weights=wo, minlength=n)
            _, first = np.unique(cs.astype(np.int64) * nclus + clus[order], return_index=True)
            neff = np.bincount(cs[first], weights=wo[first], minlength=n)
            start = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64) + off
            O_.append(order); W_.append(wo)
            self.lev_tab.append((start, wsum, neff))
            off += len(c)
        self.ORD = np.concatenate(O_)
        wc = np.concatenate(W_)
        self.WCUM = np.cumsum(wc)
        self.WBASE = self.WCUM - wc

    # ------------------------------------------------------------------ simulation
    def _init_state(self, q):
        L = self.window
        lp0 = np.log(q["price"].astype(float).to_numpy())
        buf = np.full((L, len(q)), np.nan)
        buf[L - 1] = lp0
        dur = np.full(len(q), self.dur_censor)
        key = q[KEYS + ["obs_date"]].rename(columns={"obs_date": "T"}).copy()
        key["qi"] = np.arange(len(q))
        m = self.hist.merge(key, on=KEYS)
        lag = (m["T"] - m["obs_date"]).dt.days.to_numpy()
        ok = (lag >= 1) & (lag < L)
        buf[L - 1 - lag[ok], m["qi"].to_numpy()[ok]] = m["lp"].to_numpy()[ok]
        # slots older than the series' first observation repeat its oldest observed price
        buf = pd.DataFrame(buf[::-1]).ffill().to_numpy()[::-1].copy()
        cbuf = np.zeros((L, len(q)))
        ok0 = (lag >= 0) & (lag < L)
        cbuf[L - 1 - lag[ok0], m["qi"].to_numpy()[ok0]] = m["chg"].to_numpy()[ok0]
        z = m[lag == 0]
        dur[z["qi"].to_numpy()] = z["dur"].to_numpy()
        anc = self._anchor_of(q)
        anc = np.where(np.isfinite(anc), anc, lp0)  # unseen route: anchor at today's price
        return lp0, buf, dur, anc, cbuf

    def _draw(self, feats, P, rng):
        """Pick a back-off level per path, then a recency-weighted training transition from that cell."""
        nlev, nW = len(self.levels), len(self.WCUM)
        u, u2 = rng.random(P), rng.random(P)
        chosen = np.zeros(P, dtype=bool)
        target = np.zeros(P)
        remaining = np.ones(P)
        for li, lev in enumerate(self.levels):
            start, wsum, neff = self.lev_tab[li]
            c = self._cells(feats, lev, P)
            okc = c >= 0
            cc = np.where(okc, c, 0)
            ne = np.where(okc, neff[cc], 0.0)
            ws = np.where(okc, wsum[cc], 0.0)
            wl = ne / (ne + self.k) if li < nlev - 1 else (ws > 0).astype(float)
            pl = remaining * wl
            take = (~chosen) & (ws > 0) & ((u < pl) | (li == nlev - 1))
            tg = self.WBASE[np.minimum(start[cc], nW - 1)] + u2 * ws
            target = np.where(take, tg, target)
            chosen |= take
            u = np.where(chosen, u, u - pl)
            remaining = remaining - pl
        pos = np.minimum(np.searchsorted(self.WCUM, target, side="right"), nW - 1)
        return self.ORD[pos]

    def _simulate_batch(self, q, h, M, rng, marks=()):
        """Simulate M paths per row for h days. With `marks` (day counts <= h) also return the log price
        and the running minimum on each of those days, shape (N, M, len(marks))."""
        lp0, buf0, dur0, anc0, cbuf0 = self._init_state(q)
        N, L = len(q), self.window
        P = N * M
        lp = np.repeat(lp0, M)
        buf = np.repeat(buf0, M, axis=1)
        dur = np.repeat(dur0, M)
        cbuf = np.repeat(cbuf0, M, axis=1)
        act = cbuf.sum(axis=0)
        anc = np.repeat(anc0, M)
        feats = {k_: np.repeat(v, M) for k_, v in self._cat_features(q).items()}
        dto = np.repeat(q["days_out"].to_numpy().astype(float), M)
        runmin = lp.copy()
        at = {m: i for i, m in enumerate(marks)}
        lev_at = np.empty((P, len(marks)))
        min_at = np.empty((P, len(marks)))
        if 0 in at:
            lev_at[:, at[0]] = min_at[:, at[0]] = lp
        ptr = np.zeros(P, dtype=np.int64)   # index of the donor transition driving each path today
        rem = np.zeros(P, dtype=np.int64)   # days left in the current donor block
        for step in range(h):
            ref = buf.min(axis=0)
            need = np.flatnonzero(rem <= 0)
            if len(need):
                sub = {
                    "gap": np.searchsorted(self.gap_edges, lp[need] - ref[need], side="right"),
                    "dto": np.searchsorted(self.dto_edges, dto[need] - step, side="right"),
                    "lvl": np.searchsorted(self.lvl_edges, lp[need] - anc[need], side="right"),
                    "dur": np.searchsorted(self.dur_edges, dur[need], side="right"),
                    "act": np.searchsorted(self.act_edges, act[need], side="right"),
                }
                for k_ in ("sk", "grp", "kind"):
                    sub[k_] = feats[k_][need]
                ptr[need] = self._draw(sub, len(need), rng)
                rem[need] = self.block
            typ = self.TYP[ptr]
            val = self.VAL[ptr]
            new = np.where(typ == 1, ref + val, lp + val)
            if self.level_mix > 0:
                use_lvl = (typ == 2) & (rng.random(P) < self.level_mix)
                new = np.where(use_lvl, anc + self.ZDST[ptr], new)
            moved = (np.abs(new - lp) >= self.dur_thr).astype(float)
            dur = np.where(moved > 0, 0.0, dur + 1.0)
            act += moved - cbuf[step % L]
            cbuf[step % L] = moved
            lp = new
            buf[step % L] = lp
            runmin = np.minimum(runmin, lp)
            if step + 1 in at:
                lev_at[:, at[step + 1]], min_at[:, at[step + 1]] = lp, runmin
            # continue the donor's own next day if it exists, else draw a fresh donor tomorrow
            rem = np.where(self.CONT[ptr], rem - 1, 0)
            ptr = np.minimum(ptr + 1, len(self.TYP) - 1)
        if marks:
            return lev_at.reshape(N, M, -1), min_at.reshape(N, M, -1)
        return lp.reshape(N, M), runmin.reshape(N, M)

    def paths(self, query, marks, n_paths=None):
        """Log price and running minimum of log price over [T, T + m] for each day count m in `marks`.

        Returns two arrays of shape (len(query), n_paths, len(marks)).
        """
        q = query.reset_index(drop=True)
        M = int(n_paths or self.n_paths)
        marks = [max(int(m), 0) for m in marks]
        rng = np.random.default_rng(self.seed)
        levs, mins = [np.zeros((0, M, len(marks)))], [np.zeros((0, M, len(marks)))]
        step = max(1, self.max_batch // M)
        for i in range(0, len(q), step):
            lv, mn = self._simulate_batch(q.iloc[i:i + step].reset_index(drop=True), max(marks, default=0), M, rng, marks)
            levs.append(lv); mins.append(mn)
        return np.concatenate(levs), np.concatenate(mins)

    def simulate(self, query, h, n_paths=None, return_min=False):
        """Simulate daily log-price paths h days ahead.

        Returns an (N, M) array of log prices at T + h (and, with return_min, the running minimum of the
        log price over days T..T+h).
        """
        q = query.reset_index(drop=True)
        M = int(n_paths or self.n_paths)
        h = max(int(h), 0)
        rng = np.random.default_rng(self.seed)
        ends, mins = [], []
        step = max(1, self.max_batch // M)
        for i in range(0, len(q), step):
            e, mn = self._simulate_batch(q.iloc[i:i + step].reset_index(drop=True), h, M, rng)
            ends.append(e); mins.append(mn)
        if not ends:
            ends = mins = [np.zeros((0, M))]
        lp, mn = np.concatenate(ends), np.concatenate(mins)
        return (lp, mn) if return_min else lp

    def quantiles(self, query, h, qs):
        qs = np.asarray(qs, float)
        if len(query) == 0:
            return np.zeros((0, len(qs)))
        lp = self.simulate(query, h)
        out = np.exp(np.quantile(lp, qs, axis=1).T)
        return np.maximum.accumulate(out, axis=1)

    def hit_probability(self, query, threshold, h):
        """P(price <= threshold on at least one day in [T, T+h]) for each query row (today counts).

        threshold: scalar or per-row array of dollar prices.
        """
        if len(query) == 0:
            return np.zeros(0)
        _, mn = self.simulate(query, h, return_min=True)
        thr = np.log(np.broadcast_to(np.asarray(threshold, float), (len(query),)))
        return (mn <= thr[:, None]).mean(axis=1)
