"""Load the watchlist and expand it into concrete searches."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .tfs import Search, Slice

# Departure dates are multiples of `step_days` from this fixed day, so the
# same dates come back on every run and each one builds a continuous series.
DATE_ANCHOR = date(2024, 1, 1)


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    origin: tuple[str, ...]
    airlines: tuple[str, ...] = ()
    connecting: tuple[str, ...] = ()
    via: tuple[str, ...] = ()
    position_from: tuple[str, ...] = ()
    position_days_before: int = 1  # fly the separate positioning ticket this many days early


@dataclass(frozen=True)
class ModelSettings:
    horizon_days: int = 14
    drop_threshold: float = 0.03
    wait_above: float = 0.55
    min_days_out: int = 21
    rule_margin: float = 0.05


@dataclass(frozen=True)
class Job:
    """One search to run: a strategy, a destination and a departure date."""

    strategy: Strategy
    city: str
    depart: date
    ret: date | None
    search: Search
    positioning: Search | None

    @property
    def series_key(self) -> str:
        return f"{self.strategy.key}:{self.city}"


@dataclass
class Watchlist:
    cabin: str
    trip: str
    return_after_days: int
    adults: int
    currency: str
    request_delay: float
    start_days: int
    end_days: int
    step_days: int
    windows: list[tuple[date, date]]
    destinations: dict[str, tuple[str, ...]]
    strategies: list[Strategy]
    model: ModelSettings
    alerts: dict[str, float] = field(default_factory=dict)

    def departure_dates(self, today: date) -> list[date]:
        first = today + timedelta(days=self.start_days)
        last = today + timedelta(days=self.end_days)
        dates = set()
        offset = (first - DATE_ANCHOR).days % self.step_days
        day = first + timedelta(days=(self.step_days - offset) % self.step_days)
        while day <= last:
            dates.add(day)
            day += timedelta(days=self.step_days)
        for start, end in self.windows:
            day = max(start, today + timedelta(days=1))
            while day <= end:
                dates.add(day)
                day += timedelta(days=1)
        return sorted(dates)

    def jobs(self, today: date) -> list[Job]:
        out = []
        for depart in self.departure_dates(today):
            ret = depart + timedelta(days=self.return_after_days) if self.trip == "round-trip" else None
            for strategy in self.strategies:
                for city, airports in self.destinations.items():
                    out.append(self._job(strategy, city, airports, depart, ret))
        return out

    def _job(self, strategy: Strategy, city: str, airports, depart: date, ret: date | None) -> Job:
        slices = [
            Slice(depart, strategy.origin, airports, strategy.airlines, strategy.connecting),
        ]
        if ret is not None:
            slices.append(Slice(ret, airports, strategy.origin, strategy.airlines, strategy.connecting))
        common = dict(trip=self.trip, adults=self.adults, currency=self.currency)
        search = Search(tuple(slices), cabin=self.cabin, **common)
        positioning = None
        if strategy.position_from:
            pos = [Slice(depart - timedelta(days=strategy.position_days_before), strategy.position_from, strategy.origin)]
            if ret is not None:
                pos.append(Slice(ret, strategy.origin, strategy.position_from))
            positioning = Search(tuple(pos), cabin="economy", **common)
        return Job(strategy, city, depart, ret, search, positioning)


def _codes(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(str(v).strip().upper() for v in value)


def load(path: str | Path) -> Watchlist:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    search = raw.get("search", {})
    dates = raw.get("dates", {})
    windows = [
        (date.fromisoformat(str(w["from"])), date.fromisoformat(str(w["to"])))
        for w in dates.get("windows", [])
    ]
    destinations = {city: _codes(codes) for city, codes in raw.get("destinations", {}).items()}
    if not destinations:
        raise ValueError("watchlist needs at least one [destinations] entry")
    strategies = []
    for s in raw.get("strategy", []):
        strategies.append(
            Strategy(
                key=s["key"],
                name=s.get("name", s["key"]),
                origin=_codes(s["origin"]),
                airlines=_codes(s.get("airlines")),
                connecting=_codes(s.get("connecting")),
                via=_codes(s.get("via")),
                position_from=_codes(s.get("position_from")),
                position_days_before=int(s.get("position_days_before", 1)),
            )
        )
    if not strategies:
        raise ValueError("watchlist needs at least one [[strategy]]")
    keys = [s.key for s in strategies]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate strategy keys in {keys}")
    step = int(dates.get("step_days", 7))
    if step < 1:
        raise ValueError("step_days must be >= 1")
    return Watchlist(
        cabin=search.get("cabin", "business"),
        trip=search.get("trip", "one-way"),
        return_after_days=int(search.get("return_after_days", 30)),
        adults=int(search.get("adults", 1)),
        currency=search.get("currency", "USD"),
        request_delay=float(search.get("request_delay_seconds", 1.0)),
        start_days=int(dates.get("start_days", 7)),
        end_days=int(dates.get("end_days", 240)),
        step_days=step,
        windows=windows,
        destinations=destinations,
        strategies=strategies,
        model=ModelSettings(**{k: v for k, v in raw.get("model", {}).items() if k in ModelSettings.__dataclass_fields__}),
        alerts={k: float(v) for k, v in raw.get("alerts", {}).items()},
    )
