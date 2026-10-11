import json
from datetime import date

from flightcheck import google
from flightcheck.tfs import Search, Slice, encode


def test_encode_matches_reference_query():
    # Same query encoded by fast-flights 3.1: ATL -> PVG, 2026-12-10, business, one-way.
    slices = [Slice(date(2026, 12, 10), ("ATL",), ("PVG",))]
    assert encode(slices, "business", "one-way") == "GhoSCjIwMjYtMTItMTBqBRIDQVRMcgUSA1BWR0IBAUgDmAEC"


def test_url_carries_currency_and_language():
    s = Search((Slice(date(2026, 12, 10), ("ATL",), ("PVG", "SHA"), airlines=("TK",)),))
    url = s.url()
    assert url.startswith("https://www.google.com/travel/flights/search?tfs=")
    assert "curr=USD" in url and "hl=en-US" in url


def _segment(frm, to, carrier, number, day, dep, arr):
    seg = [None] * 23
    seg[3], seg[6] = frm, to
    seg[8], seg[10] = dep, arr
    seg[20], seg[21] = day, day
    seg[22] = [carrier, number, None, carrier]
    return seg


def _entry(price, airlines, segments, dep_day, dep_time, arr_day, arr_time, duration):
    info = [None] * 23
    info[0], info[1], info[2] = "multi", airlines, segments
    info[4], info[5], info[7], info[8], info[9] = dep_day, dep_time, arr_day, arr_time, duration
    return [info, [[None, price], "token"]]


def _payload():
    best = _entry(
        4290,
        ["Alaska", "Asiana Airlines"],
        [
            _segment("ATL", "SEA", "AS", "348", [2026, 12, 10], [7], [9, 44]),
            _segment("SEA", "ICN", "AS", "119", [2026, 12, 10], [12, 5], [17, 0]),
            _segment("ICN", "PVG", "OZ", "367", [2026, 12, 11], [19, 30], [21, 55]),
        ],
        [2026, 12, 10], [7], [2026, 12, 11], [21, 55], 1555,
    )
    other = _entry(
        4903, ["Korean Air"],
        [_segment("ATL", "ICN", "KE", "36", [2026, 12, 10], [12, 25], [17, 0]),
         _segment("ICN", "PVG", "KE", "893", [2026, 12, 11], [19, 0], [20, 10])],
        [2026, 12, 10], [12, 25], [2026, 12, 11], [20, 10], 1505,
    )
    insight = [4, [None, 4290], [None, 3607], [None, -684], [None, 3000], [None, 4900], 3, None, None, None,
               [[[1786420800000, 3769], [1786507200000, 4319]]]]
    payload = [None] * 8
    payload[2] = [[best]]
    payload[3] = [[other]]
    payload[5] = insight
    return payload


def test_parse_payload_reads_both_blocks_and_history():
    result = google.parse_payload(_payload())
    assert [it.price for it in result.itineraries] == [4290, 4903]
    best = result.cheapest
    assert best.route == "ATL-SEA-ICN-PVG"
    assert best.flight_numbers == ["AS348", "AS119", "OZ367"]
    assert best.stops == 2
    assert best.depart == "2026-12-10 07:00"
    assert result.insight.typical_low == 3000 and result.insight.typical_high == 4900
    assert result.insight.history == [(date(2026, 8, 11), 3769), (date(2026, 8, 12), 4319)]


def test_parse_html_extracts_ds1_script():
    data = json.dumps(_payload())
    html = (
        '<html><script class="ds:0">AF_initDataCallback({key: \'ds:0\', data:[1], sideChannel: {}});</script>'
        f'<script class="ds:1" nonce="x">AF_initDataCallback({{key: \'ds:1\', hash: \'1\', data:{data}, sideChannel: {{}}}});</script></html>'
    )
    result = google.parse_html(html)
    assert result.cheapest.price == 4290


def test_parse_payload_tolerates_missing_blocks():
    result = google.parse_payload([None] * 4)
    assert result.itineraries == [] and result.insight is None and result.cheapest is None
