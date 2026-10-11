"""Fetch a Google Flights results page and parse the data embedded in it.

The page carries a `<script class="ds:1">` block whose `data:` array holds:

* ``[2][0]`` the "best" itineraries and ``[3][0]`` the remaining ones;
* ``[5]`` the price insight: Google's typical price range for the search plus
  a ~60-day history of the cheapest price, one point per day.

The history is what makes a model trainable on day one: every search returns
two months of past prices for that exact route, date and cabin.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from .tfs import Search

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
_SCRIPT_MARKER = 'class="ds:1"'


class FetchError(RuntimeError):
    """The page could not be fetched or did not contain flight data."""


@dataclass
class Itinerary:
    price: int
    airlines: list[str]
    route: str  # e.g. "ATL-SEA-ICN-PVG"
    flight_numbers: list[str]
    stops: int
    duration_min: int | None
    depart: str | None  # local "YYYY-MM-DD HH:MM"
    arrive: str | None


@dataclass
class Insight:
    current: int | None
    typical: int | None
    typical_low: int | None
    typical_high: int | None
    level: int | None  # raw 1-5 bucket Google uses for "low / typical / high"
    history: list[tuple[date, int]] = field(default_factory=list)


@dataclass
class Result:
    itineraries: list[Itinerary]
    insight: Insight | None

    @property
    def cheapest(self) -> Itinerary | None:
        return min(self.itineraries, key=lambda it: it.price, default=None)


def fetch_html(url: str, *, timeout: float = 30.0, retries: int = 3) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        if attempt:
            time.sleep(2**attempt + random.random())
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            continue
        if _SCRIPT_MARKER in html:
            return html
        last_error = FetchError("response has no flight data (consent page or block?)")
    raise FetchError(f"failed to fetch {url}: {last_error}")


def search(s: Search, **kwargs) -> Result:
    return parse_html(fetch_html(s.url(), **kwargs))


def parse_html(html: str) -> Result:
    start = html.find(_SCRIPT_MARKER)
    if start < 0:
        raise FetchError("no ds:1 script in page")
    end = html.find("</script>", start)
    script = html[start:end]
    if "data:" not in script:
        raise FetchError("ds:1 script has no data")
    data = script.split("data:", 1)[1].rsplit(",", 1)[0]
    if data.rstrip().endswith("errorHasStatus: true"):
        return Result(itineraries=[], insight=None)
    return parse_payload(json.loads(data))


def parse_payload(payload: list) -> Result:
    itineraries: list[Itinerary] = []
    for block_index in (2, 3):
        block = _get(payload, block_index, 0)
        for entry in block or []:
            itinerary = _parse_itinerary(entry)
            if itinerary is not None:
                itineraries.append(itinerary)
    return Result(itineraries=itineraries, insight=_parse_insight(_get(payload, 5)))


def _get(obj, *path):
    for key in path:
        if not isinstance(obj, list) or key >= len(obj):
            return None
        obj = obj[key]
    return obj


def _fmt_dt(day, hm) -> str | None:
    if not day or len(day) < 3:
        return None
    hm = list(hm or []) + [None, None]
    return f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d} {hm[0] or 0:02d}:{hm[1] or 0:02d}"


def _parse_itinerary(entry: list) -> Itinerary | None:
    info = _get(entry, 0)
    price = _get(entry, 1, 0, 1)
    if not isinstance(info, list) or not isinstance(price, int):
        return None
    segments = _get(info, 2) or []
    if not segments:
        return None
    codes = [_get(segments[0], 3)] + [_get(seg, 6) for seg in segments]
    flight_numbers = []
    for seg in segments:
        carrier = _get(seg, 22, 0)
        number = _get(seg, 22, 1)
        if carrier and number:
            flight_numbers.append(f"{carrier}{number}")
    return Itinerary(
        price=price,
        airlines=list(_get(info, 1) or []),
        route="-".join(c or "?" for c in codes),
        flight_numbers=flight_numbers,
        stops=len(segments) - 1,
        duration_min=_get(info, 9),
        depart=_fmt_dt(_get(info, 4), _get(info, 5)),
        arrive=_fmt_dt(_get(info, 7), _get(info, 8)),
    )


def _history_date(ms: int) -> date:
    # Points are stamped at local midnight of Google's chosen region; shifting by
    # half a day before truncating makes the date robust to any UTC offset.
    return (datetime.fromtimestamp(ms / 1000, tz=timezone.utc) + timedelta(hours=12)).date()


def _parse_insight(block) -> Insight | None:
    if not isinstance(block, list) or not block:
        return None
    history = []
    for point in _get(block, 10, 0) or []:
        if isinstance(point, list) and len(point) >= 2 and isinstance(point[1], int):
            history.append((_history_date(point[0]), point[1]))
    return Insight(
        current=_get(block, 1, 1),
        typical=_get(block, 2, 1),
        typical_low=_get(block, 4, 1),
        typical_high=_get(block, 5, 1),
        level=_get(block, 0),
        history=history,
    )
