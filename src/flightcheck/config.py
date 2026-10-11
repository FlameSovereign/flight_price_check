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
# one-way:      outbound only
# round-trip:   one round-trip ticket, return = departure + return_after_days
# two-one-ways: an outbound one-way plus a separate return one-way on
#               departure + return_after_days (cheaper for stays over a month,
#               where discounted round-trip business fares are not sold)
TRIP_MODES = ("one-way", "round-trip", "two-one-ways")
# round_trip: a round-trip ticket, or two one-ways together, at or below this
# one_way: a single one-way leg at or below this
ALERT_KINDS = {"round_trip", "one_way"}


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
class DateRange:
    start: date
    end: date
    step: int = 1

    def dates(self, today: date) -> list[date]:
        out, day = [], self.start
        while day <= self.end:
            if day > today:
                out.append(day)
            day += timedelta(days=self.step)
        return out


@dataclass(frozen=True)
class Job:
    """One search to run: a strategy, a destination and a departure date."""

    strategy: Strategy
    city: str
    depart: date  # the date this search flies (the return date for a return leg)
    ret: date | None  # round-trip tickets only
    search: Search
    positioning: Search | None
    leg: str = "out"  # "out" (one-way there), "back" (one-way home) or "rt" (round-trip ticket)
    trip_depart: date | None = None  # outbound date of the trip this leg belongs to

    @property
    def series_key(self) -> str:
        key = f"{self.strategy.key}:{self.city}"
        return key + ":back" if self.leg == "back" else key


@dataclass
class Watchlist:
    cabin: str
    trips: tuple[str, ...]
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
    alerts: dict[str, float] = field(default_factory=dict)  # "round_trip" / "one_way" -> price
    outbound: DateRange | None = None  # fixed travel windows; None = rolling grid
    inbound: DateRange | None = None
    round_trip_every: int = 1  # pair only every n-th window date for round trips (fewer searches)

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
        """Every search for `today`, deduplicated, in the order they should run.

        With fixed windows ([dates] outbound/return) every outbound date pairs
        with every return date; otherwise the rolling grid is used and the
        return is `return_after_days` after each departure.
        """
        legs = "one-way" in self.trips or "two-one-ways" in self.trips
        jobs: dict[tuple, Job] = {}

        def add(job: Job) -> None:
            jobs.setdefault((job.series_key, job.depart, job.ret), job)

        places = [(s, c, a) for s in self.strategies for c, a in self.destinations.items()]
        if self.outbound is None:
            stay = timedelta(days=self.return_after_days)
            for depart in self.departure_dates(today):
                for strategy, city, airports in places:
                    if legs:
                        add(self._job(strategy, city, airports, depart, None))
                    if "round-trip" in self.trips:
                        add(self._job(strategy, city, airports, depart, depart + stay))
                    if "two-one-ways" in self.trips:
                        add(self._return_job(strategy, city, airports, depart + stay, depart))
            return list(jobs.values())

        # Fixed windows: single legs first (always on sale), round-trip pairs last,
        # so a time budget that runs out drops pairs rather than whole directions.
        out_dates = self.outbound.dates(today)
        in_dates = self.inbound.dates(today) if self.inbound else []
        if legs:
            for depart in out_dates:
                for strategy, city, airports in places:
                    add(self._job(strategy, city, airports, depart, None))
        if "two-one-ways" in self.trips:
            for day in in_dates:
                for strategy, city, airports in places:
                    add(self._return_job(strategy, city, airports, day, None))
        if "round-trip" in self.trips:
            k = self.round_trip_every
            pairs = [(d, r) for d in out_dates[::k] for r in in_dates[::k] if r > d]
            if pairs:
                # Start at a different pair each day, so if a time budget cuts the
                # run short it is not always the same pairs that go unsearched.
                start = today.toordinal() % len(pairs)
                for depart, ret in pairs[start:] + pairs[:start]:
                    for strategy, city, airports in places:
                        add(self._job(strategy, city, airports, depart, ret))
        return list(jobs.values())

    def _job(self, strategy: Strategy, city: str, airports, depart: date, ret: date | None) -> Job:
        slices = [
            Slice(depart, strategy.origin, airports, strategy.airlines, strategy.connecting),
        ]
        if ret is not None:
            slices.append(Slice(ret, airports, strategy.origin, strategy.airlines, strategy.connecting))
        common = dict(trip="round-trip" if ret else "one-way", adults=self.adults, currency=self.currency)
        search = Search(tuple(slices), cabin=self.cabin, **common)
        positioning = None
        if strategy.position_from:
            pos = [Slice(depart - timedelta(days=strategy.position_days_before), strategy.position_from, strategy.origin)]
            if ret is not None:
                pos.append(Slice(ret, strategy.origin, strategy.position_from))
            positioning = Search(tuple(pos), cabin="economy", **common)
        return Job(strategy, city, depart, ret, search, positioning, leg="rt" if ret else "out", trip_depart=depart)

    def _return_job(self, strategy: Strategy, city: str, airports, day: date, trip_depart: date | None) -> Job:
        common = dict(trip="one-way", adults=self.adults, currency=self.currency)
        search = Search((Slice(day, airports, strategy.origin, strategy.airlines, strategy.connecting),), cabin=self.cabin, **common)
        positioning = None
        if strategy.position_from:
            # Home from the gateway on the day you land there.
            positioning = Search((Slice(day, strategy.origin, strategy.position_from),), cabin="economy", **common)
        return Job(strategy, city, day, None, search, positioning, leg="back", trip_depart=trip_depart)


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
    trips = search.get("trip", "one-way")
    trips = tuple([trips] if isinstance(trips, str) else trips)
    for trip in trips:
        if trip not in TRIP_MODES:
            raise ValueError(f"trip must be one of {TRIP_MODES}, got {trip!r}")
    outbound, inbound = _range(dates.get("outbound")), _range(dates.get("return"))
    if outbound and set(trips) & {"round-trip", "two-one-ways"} and not inbound:
        raise ValueError("[dates] outbound is set, so round trips also need [dates] return")
    if inbound and not outbound:
        raise ValueError("[dates] return only works together with [dates] outbound")
    alerts = {k: float(v) for k, v in raw.get("alerts", {}).items()}
    unknown = set(alerts) - ALERT_KINDS
    if unknown:
        raise ValueError(f"unknown [alerts] keys {sorted(unknown)}; use {sorted(ALERT_KINDS)}")
    step = int(dates.get("step_days", 7))
    if step < 1:
        raise ValueError("step_days must be >= 1")
    return Watchlist(
        cabin=search.get("cabin", "business"),
        trips=trips,
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
        alerts=alerts,
        outbound=outbound,
        inbound=inbound,
        round_trip_every=max(1, int(dates.get("round_trip_every", 1))),
    )


def _range(raw) -> DateRange | None:
    if not raw:
        return None
    start, end = date.fromisoformat(str(raw["from"])), date.fromisoformat(str(raw["to"]))
    if end < start:
        raise ValueError(f"date range ends before it starts: {raw}")
    step = int(raw.get("step_days", 1))
    if step < 1:
        raise ValueError(f"step_days must be >= 1 in {raw}")
    return DateRange(start, end, step)
