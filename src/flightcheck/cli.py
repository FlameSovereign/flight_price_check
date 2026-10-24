"""Command line entry point: `flightcheck collect | report | run | search`."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from . import config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flightcheck", description=__doc__)
    parser.add_argument("--config", default="config/watchlist.toml")
    parser.add_argument("--data", default="data")
    parser.add_argument("--reports", default="reports")
    # The same options are accepted after the subcommand too. They have no
    # default there, so they never overwrite a value given before it.
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--config")
    common.add_argument("--data")
    common.add_argument("--reports")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("collect", "run all searches and append to data/"), ("run", "collect, then report")):
        p = sub.add_parser(name, help=help_text, parents=[common])
        p.add_argument("--limit", type=int, help="only run the first N searches")
        p.add_argument("--only", nargs="*", help="only these strategy keys, cities or ticket kinds (out, back, rt)")
        p.add_argument("--today", type=date.fromisoformat, help="pretend today is this date")
        p.add_argument("--max-minutes", type=float, help="stop searching after this long and keep what was collected")
        p.add_argument("--resume", action="store_true", help="skip searches already recorded today")
    sub.add_parser("report", help="train the model and write reports/latest.md", parents=[common])
    p_search = sub.add_parser("search", help="one ad-hoc search, printed to the terminal")
    p_search.add_argument("origin", help="e.g. ATL or ATL,IAH")
    p_search.add_argument("destination", help="e.g. PVG,SHA")
    p_search.add_argument("date", type=date.fromisoformat)
    p_search.add_argument("--cabin", default="business")
    p_search.add_argument("--airlines", default="")
    p_search.add_argument("--connecting", default="")
    p_search.add_argument("--return-date", type=date.fromisoformat)
    args = parser.parse_args(argv)

    if args.cmd == "search":
        _search(args)
        return

    watchlist = config.load(args.config)
    data_dir = Path(args.data)
    if args.cmd in ("collect", "run"):
        from .collect import collect

        collect(watchlist, data_dir, today=args.today, limit=args.limit, only=set(args.only or []),
                max_minutes=args.max_minutes, resume=args.resume)
    if args.cmd in ("report", "run"):
        from .report import build

        info = build(watchlist, data_dir, Path(args.reports))
        print(f"report written to {args.reports}/latest.md (model: {info['metrics'].get('status')})")


def _codes(text: str) -> tuple[str, ...]:
    return tuple(c.strip().upper() for c in text.split(",") if c.strip())


def _search(args) -> None:
    from . import google
    from .tfs import Search, Slice

    out = Slice(args.date, _codes(args.origin), _codes(args.destination), _codes(args.airlines), _codes(args.connecting))
    slices = [out]
    if args.return_date:
        slices.append(Slice(args.return_date, out.destinations, out.origins, out.airlines, out.connecting))
    s = Search(tuple(slices), cabin=args.cabin, trip="round-trip" if args.return_date else "one-way")
    result = google.search(s)
    print(s.url())
    if result.insight:
        i = result.insight
        print(f"Google: now ${i.current}, typical ${i.typical_low}-${i.typical_high}, {len(i.history)} days of history")
        if i.history:
            print("  last 14 days:", " ".join(str(p) for _, p in i.history[-14:]))
    for it in sorted(result.itineraries, key=lambda it: it.price)[:15]:
        print(f"${it.price:>6,}  {it.route:<22} {' '.join(it.flight_numbers):<24} {it.depart}  {' + '.join(it.airlines)}")


if __name__ == "__main__":
    main()
