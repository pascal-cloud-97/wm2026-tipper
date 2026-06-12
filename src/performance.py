from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BettingPerformance:
    total_bets: int
    pending_bets: int
    settled_bets: int
    wins: int
    total_stake: float
    total_profit: float
    roi: float
    hit_rate: float
    current_bankroll: float
    pending_stake: float
    available_bankroll: float
    average_closing_line_value: float
    ledger: pd.DataFrame


def add_closing_line_value(
    paper_bets: pd.DataFrame,
    odds: pd.DataFrame | None,
    matches: pd.DataFrame,
    bookmaker: str = "Swisslos",
) -> pd.DataFrame:
    ledger = paper_bets.copy()
    ledger["closing_odds"] = np.nan
    ledger["closing_line_value"] = np.nan
    if ledger.empty or odds is None or odds.empty:
        return ledger
    kickoff_lookup = matches.copy()
    kickoff_lookup["match_id"] = kickoff_lookup["match_id"].astype(str)
    kickoff_lookup = kickoff_lookup.set_index("match_id")["date"].to_dict()
    bookmaker_rows = odds[
        odds["bookmaker"].astype(str).str.lower() == bookmaker.lower()
    ].copy()
    for index, bet in ledger.iterrows():
        match_id = str(bet["match_id"])
        kickoff = kickoff_lookup.get(match_id)
        if kickoff is None:
            continue
        rows = bookmaker_rows[
            (bookmaker_rows["match_id"].astype(str) == match_id)
            & (bookmaker_rows["collected_at"] <= pd.Timestamp(kickoff))
        ]
        if rows.empty:
            continue
        closing = rows.sort_values("collected_at").iloc[-1]
        column = {
            "1": "home_odds",
            "X": "draw_odds",
            "2": "away_odds",
        }.get(str(bet["outcome"]))
        if column is None or pd.isna(closing[column]):
            continue
        closing_odds = float(closing[column])
        ledger.at[index, "closing_odds"] = closing_odds
        ledger.at[index, "closing_line_value"] = (
            float(bet["odds"]) / closing_odds - 1.0
        )
    return ledger


def calculate_betting_performance(
    paper_bets: pd.DataFrame,
    starting_bankroll: float,
    odds: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
) -> BettingPerformance:
    matches = matches if matches is not None else pd.DataFrame()
    ledger = add_closing_line_value(paper_bets, odds, matches)
    if ledger.empty:
        return BettingPerformance(
            total_bets=0,
            pending_bets=0,
            settled_bets=0,
            wins=0,
            total_stake=0.0,
            total_profit=0.0,
            roi=0.0,
            hit_rate=0.0,
            current_bankroll=float(starting_bankroll),
            pending_stake=0.0,
            available_bankroll=float(starting_bankroll),
            average_closing_line_value=float("nan"),
            ledger=ledger,
        )
    settled = ledger[ledger["status"].isin(["won", "lost"])]
    total_stake = float(settled["stake"].sum())
    total_profit = float(settled["profit"].fillna(0.0).sum())
    wins = int((settled["status"] == "won").sum())
    pending_stake = float(
        ledger.loc[ledger["status"] == "pending", "stake"].sum()
    )
    current_bankroll = float(starting_bankroll) + total_profit
    closing_values = ledger["closing_line_value"].dropna()
    return BettingPerformance(
        total_bets=len(ledger),
        pending_bets=int((ledger["status"] == "pending").sum()),
        settled_bets=len(settled),
        wins=wins,
        total_stake=total_stake,
        total_profit=total_profit,
        roi=total_profit / total_stake if total_stake > 0 else 0.0,
        hit_rate=wins / len(settled) if len(settled) else 0.0,
        current_bankroll=current_bankroll,
        pending_stake=pending_stake,
        available_bankroll=max(0.0, current_bankroll - pending_stake),
        average_closing_line_value=(
            float(closing_values.mean()) if not closing_values.empty else float("nan")
        ),
        ledger=ledger,
    )
