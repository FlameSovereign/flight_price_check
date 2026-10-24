"""Encode a Google Flights search into the `tfs` URL parameter.

`tfs` is a base64-encoded protobuf message. Only the handful of fields this
project needs are encoded by hand, so no protobuf dependency is required.
Field numbers follow the schema reverse-engineered by the fast-flights project
(https://github.com/AWeirdDev/flights); `to_airport` and `from_airport` are
repeated, which is what lets one search cover a whole metro area (PVG + SHA).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlencode

SEATS = {"economy": 1, "premium-economy": 2, "business": 3, "first": 4}
TRIPS = {"round-trip": 1, "one-way": 2}
SEARCH_URL = "https://www.google.com/travel/flights/search"


@dataclass(frozen=True)
class Slice:
    """One direction of travel."""

    date: date
    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    airlines: tuple[str, ...] = ()
    connecting: tuple[str, ...] = ()
    max_stops: int | None = None


@dataclass(frozen=True)
class Search:
    slices: tuple[Slice, ...]
    cabin: str = "business"
    trip: str = "one-way"
    adults: int = 1
    currency: str = "USD"
    language: str = "en-US"

    def tfs(self) -> str:
        return encode(self.slices, self.cabin, self.trip, self.adults)

    def url(self) -> str:
        params = {"tfs": self.tfs(), "hl": self.language, "curr": self.currency}
        return f"{SEARCH_URL}?{urlencode(params)}"


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _len_field(num: int, payload: bytes) -> bytes:
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _int_field(num: int, value: int) -> bytes:
    return _varint(num << 3) + _varint(value)


def _airport(code: str) -> bytes:
    return _len_field(2, code.encode())


def _slice(s: Slice) -> bytes:
    out = _len_field(2, s.date.isoformat().encode())
    if s.max_stops is not None:
        out += _int_field(5, s.max_stops)
    for airline in s.airlines:
        out += _len_field(6, airline.encode())
    for code in s.origins:
        out += _len_field(13, _airport(code))
    for code in s.destinations:
        out += _len_field(14, _airport(code))
    for code in s.connecting:
        out += _len_field(15, code.encode())
    return out


def encode(slices: tuple[Slice, ...] | list[Slice], cabin: str, trip: str, adults: int = 1) -> str:
    if cabin not in SEATS:
        raise ValueError(f"unknown cabin {cabin!r}; expected one of {sorted(SEATS)}")
    if trip not in TRIPS:
        raise ValueError(f"unknown trip {trip!r}; expected one of {sorted(TRIPS)}")
    if not 1 <= adults <= 9:
        raise ValueError("adults must be between 1 and 9")
    msg = b"".join(_len_field(3, _slice(s)) for s in slices)
    msg += _len_field(8, bytes([1]) * adults)  # packed repeated Passenger.ADULT
    msg += _int_field(9, SEATS[cabin])
    msg += _int_field(19, TRIPS[trip])
    return base64.b64encode(msg).decode()
