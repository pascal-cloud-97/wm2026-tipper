from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from .prediction import MatchPrediction


OUTCOME_LABELS = {
    "1": "Heimsieg",
    "X": "Unentschieden",
    "2": "Auswärtssieg",
}


@dataclass(frozen=True)
class BetAssessment:
    outcome: str
    label: str
    model_probability: float
    decimal_odds: float
    implied_probability: float
    edge: float
    expected_return: float
    full_kelly_fraction: float
    recommended_fraction: float
    stake: float
    decision: str
    reason: str


@dataclass(frozen=True)
class MarketBetRecommendation:
    decision: str
    outcome: str
    label: str
    decimal_odds: float
    model_probability: float
    implied_probability: float
    edge: float
    expected_return: float
    recommended_fraction: float
    stake: float
    reason: str
    assessments: tuple[BetAssessment, ...]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "outcome": item.outcome,
                    "label": item.label,
                    "model_probability": item.model_probability,
                    "decimal_odds": item.decimal_odds,
                    "implied_probability": item.implied_probability,
                    "edge": item.edge,
                    "expected_return": item.expected_return,
                    "full_kelly_fraction": item.full_kelly_fraction,
                    "recommended_fraction": item.recommended_fraction,
                    "stake": item.stake,
                    "decision": item.decision,
                    "reason": item.reason,
                }
                for item in self.assessments
            ]
        )


def allocate_match_portfolio(
    recommendations: pd.DataFrame,
    bankroll: float,
    config: dict,
    pending_bets: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = recommendations.copy()
    frame["raw_bet_decision"] = frame["bet_decision"]
    frame["raw_bet_stake"] = pd.to_numeric(
        frame["bet_stake"], errors="coerce"
    ).fillna(0.0)
    frame["portfolio_priority"] = (
        pd.to_numeric(frame["bet_expected_return"], errors="coerce").fillna(-1.0)
        * pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
        * (
            1.0
            - pd.to_numeric(frame["data_uncertainty"], errors="coerce")
            .fillna(1.0)
            .clip(0.0, 1.0)
        )
    )
    frame["bet_stake"] = 0.0

    pending = pending_bets.copy() if pending_bets is not None else pd.DataFrame()
    if not pending.empty:
        pending = pending[pending["status"] == "pending"].copy()
    pending_total = (
        float(pd.to_numeric(pending["stake"], errors="coerce").fillna(0.0).sum())
        if not pending.empty
        else 0.0
    )
    pending_keys = (
        set(
            zip(
                pending["match_id"].astype(str),
                pending["outcome"].astype(str),
            )
        )
        if not pending.empty
        else set()
    )

    match_dates: dict[str, pd.Timestamp] = {}
    if matches is not None and not matches.empty:
        match_dates = (
            matches.assign(match_id=matches["match_id"].astype(str))
            .set_index("match_id")["date"]
            .map(pd.Timestamp)
            .to_dict()
        )
    pending_by_day: dict[pd.Timestamp, float] = {}
    if not pending.empty and match_dates:
        for row in pending.to_dict("records"):
            kickoff = match_dates.get(str(row["match_id"]))
            if kickoff is None:
                continue
            day = pd.Timestamp(kickoff).normalize()
            pending_by_day[day] = pending_by_day.get(day, 0.0) + float(row["stake"])

    total_cap = max(0.0, float(bankroll)) * float(
        config.get("max_total_exposure_fraction", 0.10)
    )
    daily_cap = max(0.0, float(bankroll)) * float(
        config.get("max_daily_exposure_fraction", 0.05)
    )
    minimum_stake = float(config.get("minimum_portfolio_stake", 0.50))
    max_bets_per_day = int(config.get("max_bets_per_day", 3))
    remaining_total = max(0.0, total_cap - pending_total)
    allocated_by_day: dict[pd.Timestamp, float] = {}
    bets_by_day: dict[pd.Timestamp, int] = {}

    candidates = frame[frame["raw_bet_decision"] == "WETTEN"].sort_values(
        ["portfolio_priority", "bet_expected_return", "bet_edge"],
        ascending=False,
    )
    for index, row in candidates.iterrows():
        key = (str(row["match_id"]), str(row["bet_outcome"]))
        day = pd.Timestamp(row["date"]).normalize()
        reason = ""
        if key in pending_keys:
            reason = "Diese Paper-Wette ist bereits offen."
        elif remaining_total < minimum_stake:
            reason = "Das gesamte Portfolio-Risikolimit ist ausgeschöpft."
        elif bets_by_day.get(day, 0) >= max_bets_per_day:
            reason = "Die maximale Anzahl Wetten für diesen Spieltag ist erreicht."
        else:
            day_used = pending_by_day.get(day, 0.0) + allocated_by_day.get(day, 0.0)
            day_remaining = max(0.0, daily_cap - day_used)
            stake = min(
                float(row["raw_bet_stake"]),
                remaining_total,
                day_remaining,
            )
            if stake >= minimum_stake:
                frame.at[index, "bet_stake"] = stake
                frame.at[index, "bet_decision"] = "WETTEN"
                frame.at[index, "bet_reason"] = (
                    str(row["bet_reason"])
                    + " Portfolio- und Tageslimit sind eingehalten."
                )
                remaining_total -= stake
                allocated_by_day[day] = allocated_by_day.get(day, 0.0) + stake
                bets_by_day[day] = bets_by_day.get(day, 0) + 1
                continue
            reason = "Das verbleibende Tagesbudget liegt unter dem Mindesteinsatz."
        frame.at[index, "bet_decision"] = "KEINE WETTE"
        frame.at[index, "bet_stake"] = 0.0
        frame.at[index, "bet_reason"] = reason

    frame["portfolio_total_cap"] = total_cap
    frame["portfolio_pending_stake"] = pending_total
    frame["portfolio_new_stake"] = float(frame["bet_stake"].sum())
    frame["portfolio_remaining_capacity"] = max(
        0.0,
        total_cap - pending_total - float(frame["bet_stake"].sum()),
    )
    return frame


def _finite(value: float | None) -> bool:
    try:
        return value is not None and isfinite(float(value))
    except (TypeError, ValueError):
        return False


def assess_bet(
    outcome: str,
    model_probability: float,
    decimal_odds: float | None,
    implied_probability: float | None,
    confidence: float,
    data_uncertainty: float,
    bankroll: float,
    config: dict,
    odds_age_hours: float | None = None,
    market_open: bool = True,
    label: str | None = None,
) -> BetAssessment:
    label = label or OUTCOME_LABELS.get(outcome, outcome)
    probability = float(model_probability)
    odds = float(decimal_odds) if _finite(decimal_odds) else float("nan")
    implied = (
        float(implied_probability)
        if _finite(implied_probability)
        else (1.0 / odds if _finite(odds) and odds > 1.0 else float("nan"))
    )
    edge = probability - implied if _finite(implied) else float("nan")
    expected_return = (
        probability * odds - 1.0
        if _finite(odds) and odds > 1.0
        else float("nan")
    )
    full_kelly = (
        max(0.0, expected_return / (odds - 1.0))
        if _finite(expected_return) and odds > 1.0
        else 0.0
    )

    fractional_kelly = float(config.get("fractional_kelly", 0.25))
    max_stake_fraction = float(config.get("max_stake_fraction", 0.02))
    confidence_factor = max(0.0, min(1.0, float(confidence)))
    data_quality = max(0.0, min(1.0, 1.0 - float(data_uncertainty)))
    recommended_fraction = min(
        max_stake_fraction,
        full_kelly * fractional_kelly * confidence_factor * data_quality,
    )

    reason = ""
    if not market_open:
        reason = "Das Spiel ist nicht mehr für eine neue Wette offen."
    elif not _finite(odds) or odds <= 1.0:
        reason = "Keine gültige Swisslos-Quote geladen."
    elif (
        _finite(odds_age_hours)
        and float(odds_age_hours) > float(config.get("max_odds_age_hours", 24.0))
    ):
        reason = "Die Swisslos-Quote ist älter als die erlaubte Aktualitätsgrenze."
    elif confidence < float(config.get("min_confidence", 0.45)):
        reason = "Die Modell-Confidence ist für eine Wette zu niedrig."
    elif data_uncertainty > float(config.get("max_data_uncertainty", 0.55)):
        reason = "Die Datenunsicherheit ist für eine Wette zu hoch."
    elif edge < float(config.get("min_edge", 0.04)):
        reason = "Der Modellvorteil gegenüber der Swisslos-Wahrscheinlichkeit ist zu klein."
    elif expected_return < float(config.get("min_expected_return", 0.03)):
        reason = "Der modellierte erwartete Ertrag liegt unter der Mindestgrenze."
    elif recommended_fraction <= 0.0 or bankroll <= 0.0:
        reason = "Die konservative Einsatzberechnung ergibt keinen Einsatz."

    decision = "WETTEN" if not reason else "KEINE WETTE"
    if decision == "WETTEN":
        reason = (
            "Positive Modellabweichung und positiver Expected Return; "
            "Einsatz durch Fractional Kelly und Höchstgrenze reduziert."
        )
    else:
        recommended_fraction = 0.0

    return BetAssessment(
        outcome=outcome,
        label=label,
        model_probability=probability,
        decimal_odds=odds,
        implied_probability=implied,
        edge=edge,
        expected_return=expected_return,
        full_kelly_fraction=full_kelly,
        recommended_fraction=recommended_fraction,
        stake=max(0.0, float(bankroll)) * recommended_fraction,
        decision=decision,
        reason=reason,
    )


def assess_market(
    prediction: MatchPrediction,
    decimal_odds: dict[str, float | None],
    implied_probabilities: dict[str, float | None],
    bankroll: float,
    config: dict,
    odds_age_hours: float | None = None,
    market_open: bool = True,
) -> MarketBetRecommendation:
    probabilities = {
        "1": prediction.home_win,
        "X": prediction.draw,
        "2": prediction.away_win,
    }
    assessments = tuple(
        assess_bet(
            outcome=outcome,
            model_probability=probabilities[outcome],
            decimal_odds=decimal_odds.get(outcome),
            implied_probability=implied_probabilities.get(outcome),
            confidence=prediction.confidence,
            data_uncertainty=prediction.data_uncertainty,
            bankroll=bankroll,
            config=config,
            odds_age_hours=odds_age_hours,
            market_open=market_open,
        )
        for outcome in ("1", "X", "2")
    )
    eligible = [item for item in assessments if item.decision == "WETTEN"]
    priced = [item for item in assessments if _finite(item.expected_return)]
    candidate = max(
        eligible or priced or assessments,
        key=lambda item: (
            item.expected_return if _finite(item.expected_return) else float("-inf")
        ),
    )
    if eligible:
        decision = "WETTEN"
        reason = candidate.reason
    else:
        decision = "KEINE WETTE"
        reason = candidate.reason
        if not priced:
            reason = "Keine gültigen Swisslos-Quoten für dieses Spiel geladen."

    return MarketBetRecommendation(
        decision=decision,
        outcome=candidate.outcome if priced else "",
        label=candidate.label if priced else "Keine Wette",
        decimal_odds=candidate.decimal_odds,
        model_probability=candidate.model_probability,
        implied_probability=candidate.implied_probability,
        edge=candidate.edge,
        expected_return=candidate.expected_return,
        recommended_fraction=candidate.recommended_fraction,
        stake=candidate.stake,
        reason=reason,
        assessments=assessments,
    )


def evaluate_outright_market(
    team_probabilities: pd.DataFrame,
    outright_odds: pd.DataFrame | None,
    bankroll: float,
    config: dict,
    bookmaker: str = "Swisslos",
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    columns = [
        "team_id",
        "team",
        "champion_probability",
        "decimal_odds",
        "implied_probability",
        "market_probability",
        "edge",
        "expected_return",
        "decision",
        "stake",
        "reason",
        "odds_collected_at",
        "odds_age_hours",
        "market_coverage",
    ]
    if outright_odds is None or outright_odds.empty:
        result = team_probabilities.copy()
        for column in columns:
            if column not in result:
                result[column] = np.nan if column not in {"decision", "reason"} else ""
        result["decision"] = "KEINE WETTE"
        result["reason"] = "Keine Swisslos-Weltmeisterquoten geladen."
        result["market_coverage"] = 0.0
        return result[columns]

    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    rows = outright_odds[
        (outright_odds["bookmaker"].astype(str).str.lower() == bookmaker.lower())
        & (outright_odds["market"].astype(str).str.lower() == "champion")
        & (outright_odds["collected_at"] <= now)
    ].copy()
    if rows.empty:
        return evaluate_outright_market(
            team_probabilities,
            None,
            bankroll,
            config,
            bookmaker=bookmaker,
            as_of=now,
        )
    latest = rows.sort_values("collected_at").groupby("team_id", as_index=False).tail(1)
    latest["team_id"] = latest["team_id"].astype(str)
    market = team_probabilities.copy()
    market["team_id"] = market["team_id"].astype(str)
    market = market.merge(
        latest[["team_id", "decimal_odds", "collected_at"]],
        on="team_id",
        how="left",
    )
    market["implied_probability"] = 1.0 / market["decimal_odds"]
    available = int(market["decimal_odds"].notna().sum())
    coverage = available / len(market) if len(market) else 0.0
    implied_sum = market["implied_probability"].sum(skipna=True)
    complete_market = available == len(market) and implied_sum > 0
    market["market_probability"] = (
        market["implied_probability"] / implied_sum
        if complete_market
        else market["implied_probability"]
    )

    assessments = []
    outright_config = dict(config)
    outright_config["max_odds_age_hours"] = float(
        config.get("outright_max_odds_age_hours", 168.0)
    )
    for row in market.to_dict("records"):
        collected_at = row.get("collected_at")
        age_hours = (
            max(
                0.0,
                (now - pd.Timestamp(collected_at)).total_seconds() / 3600,
            )
            if pd.notna(collected_at)
            else np.nan
        )
        assessment = assess_bet(
            outcome=str(row["team_id"]),
            label=str(row.get("team", row["team_id"])),
            model_probability=float(row["champion_probability"]),
            decimal_odds=row.get("decimal_odds"),
            implied_probability=row.get("market_probability"),
            confidence=1.0 - float(row.get("simulation_uncertainty", 0.20)),
            data_uncertainty=float(row.get("simulation_uncertainty", 0.20)),
            bankroll=bankroll,
            config=outright_config,
            odds_age_hours=age_hours,
        )
        decision = assessment.decision
        stake = assessment.stake
        reason = assessment.reason
        if coverage < float(config.get("outright_min_market_coverage", 0.80)):
            decision = "KEINE WETTE"
            stake = 0.0
            reason = (
                "Zu wenige Teams mit Swisslos-Weltmeisterquote geladen, "
                "um die Buchmachermarge belastbar einzuordnen."
            )
        assessments.append(
            {
                **row,
                "edge": assessment.edge,
                "expected_return": assessment.expected_return,
                "decision": decision,
                "stake": stake,
                "reason": reason,
                "odds_collected_at": collected_at,
                "odds_age_hours": age_hours,
                "market_coverage": coverage,
            }
        )
    result = pd.DataFrame(assessments)[columns]
    result["_decision_rank"] = (result["decision"] == "WETTEN").astype(int)
    return result.sort_values(
        ["_decision_rank", "expected_return"],
        ascending=[False, False],
    ).drop(columns=["_decision_rank"])
