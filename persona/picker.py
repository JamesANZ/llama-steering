"""Interactive multi-select over feature search results.

Default-checks the top N (default 3). Falls back to printing + auto-picking
top N when running non-interactive (no TTY, --noninteractive flag, or
questionary import failure).
"""

from __future__ import annotations

import sys
from typing import Iterable

from rich.console import Console
from rich.table import Table

from .feature_search import FeatureMatch

console = Console()


def _print_table(matches: list[FeatureMatch]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("feature", justify="right")
    table.add_column("explanation")
    table.add_column("top tokens")
    table.add_column("score", justify="right")
    for i, m in enumerate(matches):
        table.add_row(
            str(i + 1),
            str(m.feature_id),
            m.short_snippet(80),
            m.top_tokens(5)[:48],
            f"{m.rerank_score:.2f}",
        )
    console.print(table)


def _format_choice_label(idx: int, m: FeatureMatch) -> str:
    expl = m.short_snippet(60)
    toks = m.top_tokens(4)
    label = f"#{m.feature_id:>6}  {expl}"
    if toks:
        label += f"  | tokens: {toks}"
    return label


def auto_pick(matches: list[FeatureMatch], n: int = 3) -> list[FeatureMatch]:
    chosen = matches[: max(1, n)]
    print(f"Auto-picked top {len(chosen)} features:")
    for m in chosen:
        print(f"  #{m.feature_id}  {m.short_snippet(70)}")
    return chosen


def pick_features(
    matches: list[FeatureMatch],
    default_n: int = 3,
    noninteractive: bool = False,
) -> list[FeatureMatch]:
    """Show candidates, let the user check the ones they want.

    Returns the picked subset (may be empty if the user picks none and accepts).
    """
    if not matches:
        print("No candidates to pick from.")
        return []

    _print_table(matches)

    if noninteractive or not sys.stdin.isatty():
        return auto_pick(matches, default_n)

    try:
        import questionary
    except ImportError:
        print("questionary not installed; auto-picking top features.")
        return auto_pick(matches, default_n)

    choices = []
    for i, m in enumerate(matches):
        choices.append(
            questionary.Choice(
                title=_format_choice_label(i, m),
                value=i,
                checked=(i < default_n),
            )
        )

    print(
        f"Use space to toggle, enter to confirm. Default: top {default_n} checked. "
        f"Pick 1-{len(matches)} features."
    )
    try:
        picked_idxs: list[int] | None = questionary.checkbox(
            "Select features for this persona:", choices=choices
        ).ask()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return []

    if picked_idxs is None:
        # ESC / cancel
        print("Cancelled; no features picked.")
        return []
    if not picked_idxs:
        print("No features picked. Falling back to auto-pick top 1.")
        return matches[:1]

    return [matches[i] for i in picked_idxs]
