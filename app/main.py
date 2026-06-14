from __future__ import annotations

import copy
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(
    os.environ.get(
        "WM_TIPPER_DB_PATH",
        str(ROOT / "data" / "wm2026_tipper.sqlite"),
    )
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    DataBundle,
    DataValidationError,
    load_dataset,
    load_uploaded_bytes,
    validate_dataframe,
    validate_references,
)
from src.betting import (  # noqa: E402
    allocate_match_portfolio,
    assess_market,
    evaluate_outright_market,
)
from src.backtesting import fit_and_validate_calibrator, run_backtest  # noqa: E402
from src.explainability import build_explanation  # noqa: E402
from src.exporting import to_excel_bytes, to_markdown, to_srf_text  # noqa: E402
from src.feature_engineering import build_match_features  # noqa: E402
from src.form_analysis import team_form_curve  # noqa: E402
from src.fifa_rankings import update_local_fifa_rankings  # noqa: E402
from src.history_updater import update_history  # noqa: E402
from src.matchday import apply_match_results  # noqa: E402
from src.optimizer import optimize_all, optimize_tip  # noqa: E402
from src.performance import calculate_betting_performance  # noqa: E402
from src.prediction import predict_all  # noqa: E402
from src.ratings import current_elo_ratings, teams_with_current_elo  # noqa: E402
from src.scoring import ScoringRules  # noqa: E402
from src.simulation import simulate_tournament, simulate_world_cup  # noqa: E402
from src.swisslos_odds import (  # noqa: E402
    SWISSLOS_SOURCE,
    SwisslosOddsError,
    fetch_swisslos_odds,
)
from src.world_cup_updater import update_world_cup_files  # noqa: E402
from src.storage import (  # noqa: E402
    load_paper_bets,
    load_match_odds,
    load_match_results,
    load_outright_odds,
    save_match_odds,
    save_match_result,
    save_outright_odds,
    save_paper_bets,
    save_recommendations,
    settle_paper_bets,
)


def apply_website_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 85% 5%, rgba(57, 217, 138, .12), transparent 28rem),
                linear-gradient(180deg, #07111f 0%, #0a1524 55%, #07111f 100%);
        }
        [data-testid="stHeader"] { background: rgba(7, 17, 31, .78); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1728 0%, #0d1c2e 100%);
            border-right: 1px solid rgba(255, 255, 255, .08);
        }
        .block-container { max-width: 1500px; padding-top: 1.4rem; }
        .wm-hero {
            padding: 2.1rem 2.3rem;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 24px;
            background:
                linear-gradient(120deg, rgba(16, 30, 49, .96), rgba(15, 48, 54, .90));
            box-shadow: 0 24px 70px rgba(0, 0, 0, .24);
            margin-bottom: 1.2rem;
        }
        .wm-kicker {
            color: #70efb3;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .wm-hero h1 {
            margin: .35rem 0 .5rem;
            font-size: clamp(2rem, 5vw, 4.1rem);
            line-height: 1.02;
            letter-spacing: -.045em;
        }
        .wm-hero p {
            max-width: 820px;
            color: #c6d2df;
            font-size: 1.04rem;
            margin: 0;
        }
        .wm-badges {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.15rem;
        }
        .wm-badge {
            border: 1px solid rgba(112, 239, 179, .25);
            border-radius: 999px;
            padding: .38rem .7rem;
            color: #d9fbed;
            background: rgba(57, 217, 138, .08);
            font-size: .82rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,.09);
            border-radius: 16px;
            padding: .9rem 1rem;
            background: rgba(16, 30, 49, .76);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 14px;
            overflow: hidden;
        }
        .wm-footer {
            margin-top: 2.2rem;
            padding: 1.1rem 0 .4rem;
            border-top: 1px solid rgba(255,255,255,.08);
            color: #8091a5;
            font-size: .82rem;
        }
        @media (max-width: 700px) {
            .block-container { padding-left: .8rem; padding-right: .8rem; }
            .wm-hero { padding: 1.35rem; border-radius: 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_website_header() -> None:
    st.markdown(
        """
        <section class="wm-hero">
          <div class="wm-kicker">Football Analytics · World Cup 2026</div>
          <h1>WM 2026 Prognosezentrum</h1>
          <p>
            Datenbasierte 1/X/2-Wahrscheinlichkeiten, hypothetische Endstände,
            Turniersimulation und Expected-Value-Analyse für das Tippspiel.
            Modellwerte sind keine Gewinnzusage.
          </p>
          <div class="wm-badges">
            <span class="wm-badge">48 Teams</span>
            <span class="wm-badge">72 Gruppenspiele</span>
            <span class="wm-badge">Aktuelle Form 5/10</span>
            <span class="wm-badge">Monte Carlo</span>
            <span class="wm-badge">Transparente Unsicherheit</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="wm-footer">
          WM 2026 Tipper · Rein statistische Modellrechnung. Wetten können
          zum vollständigen Verlust des Einsatzes führen. Keine Finanz- oder
          Gewinnberatung.
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_config() -> dict:
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@st.cache_data
def load_world_cup_bundle() -> DataBundle:
    bundle = load_dataset(ROOT / "data" / "world_cup_2026")
    latest = (
        bundle.historical_results["date"].max().strftime("%d.%m.%Y")
        if not bundle.historical_results.empty
        else "keine Historie"
    )
    bundle.source_label = f"FIFA WM 2026 - Resultate bis {latest}"
    return bundle


def _merge_snapshots(
    base: pd.DataFrame,
    stored: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    if stored.empty:
        return base.copy()
    if base.empty:
        return stored.copy()
    return (
        pd.concat([base, stored], ignore_index=True)
        .drop_duplicates(subset=keys, keep="last")
        .reset_index(drop=True)
    )


def merge_persisted_market_data(bundle: DataBundle) -> DataBundle:
    database = DATABASE_PATH
    bundle.odds = _merge_snapshots(
        bundle.odds,
        load_match_odds(database),
        ["match_id", "bookmaker", "collected_at"],
    )
    bundle.outright_odds = _merge_snapshots(
        bundle.outright_odds,
        load_outright_odds(database),
        ["team_id", "bookmaker", "market", "collected_at"],
    )
    bundle = apply_match_results(bundle, load_match_results(database))
    settle_paper_bets(database, bundle.matches)
    return bundle


def get_bundle() -> DataBundle:
    if "bundle" not in st.session_state:
        st.session_state.bundle = merge_persisted_market_data(
            copy.deepcopy(load_world_cup_bundle())
        )
    return st.session_state.bundle


def get_scoring(config: dict) -> ScoringRules:
    if "scoring" not in st.session_state:
        st.session_state.scoring = dict(config["scoring"])
    return ScoringRules.from_dict(st.session_state.scoring)


def get_model_config(config: dict) -> dict:
    if "model_config" not in st.session_state:
        st.session_state.model_config = copy.deepcopy(config["model"])
    return st.session_state.model_config


def build_analysis(
    bundle: DataBundle,
    config: dict,
    model_config: dict,
    scoring: ScoringRules,
    strategy: str,
    use_joker: bool,
    betting_config: dict | None = None,
    bankroll: float = 0.0,
    pending_bets: pd.DataFrame | None = None,
):
    elo_ratings = current_elo_ratings(bundle.historical_results)
    analysis_ratings = pd.concat(
        [bundle.ratings, elo_ratings],
        ignore_index=True,
    )
    features = build_match_features(
        bundle.matches,
        bundle.teams,
        bundle.historical_results,
        analysis_ratings,
        bundle.availability,
        bundle.lineups,
        bundle.odds,
    )
    predictions = predict_all(features, model_config)
    recommendations = optimize_all(
        predictions,
        scoring,
        strategy,
        config.get("optimizer", {}),
        use_joker=use_joker,
    )
    names = bundle.teams.set_index("team_id")["team_name"].to_dict()
    betting_rules = betting_config or config.get("betting", {})
    analysis_time = pd.Timestamp.now(tz="UTC").tz_localize(None)
    rows = []
    for match in bundle.matches.to_dict("records"):
        match_id = str(match["match_id"])
        prediction = predictions[match_id]
        recommendation = recommendations[match_id]
        outcome_code, predicted_winner, winner_probability = (
            prediction.most_likely_outcome()
        )
        hypothetical_score, hypothetical_score_probability = (
            prediction.representative_score()
        )
        modal_score, modal_score_probability = prediction.modal_score()
        feature_row = features.loc[
            features["match_id"].astype(str) == match_id
        ].iloc[0]
        swisslos_quote = {
            "1": feature_row.get("swisslos_home_odds"),
            "X": feature_row.get("swisslos_draw_odds"),
            "2": feature_row.get("swisslos_away_odds"),
        }[outcome_code]
        swisslos_implied = {
            "1": feature_row.get("swisslos_home_probability"),
            "X": feature_row.get("swisslos_draw_probability"),
            "2": feature_row.get("swisslos_away_probability"),
        }[outcome_code]
        swisslos_collected_at = feature_row.get("swisslos_collected_at")
        swisslos_current_age = (
            max(
                0.0,
                (analysis_time - pd.Timestamp(swisslos_collected_at)).total_seconds()
                / 3600.0,
            )
            if pd.notna(swisslos_collected_at)
            else pd.NA
        )
        bet = assess_market(
            prediction,
            decimal_odds={
                "1": feature_row.get("swisslos_home_odds"),
                "X": feature_row.get("swisslos_draw_odds"),
                "2": feature_row.get("swisslos_away_odds"),
            },
            implied_probabilities={
                "1": feature_row.get("swisslos_home_probability"),
                "X": feature_row.get("swisslos_draw_probability"),
                "2": feature_row.get("swisslos_away_probability"),
            },
            bankroll=bankroll,
            config=betting_rules,
            odds_age_hours=swisslos_current_age,
            market_open=str(match.get("status", "scheduled")).lower()
            not in {"completed", "finished", "final"},
        )
        rows.append(
            {
                "match_id": match_id,
                "official_match_number": match.get("official_match_number"),
                "date": pd.Timestamp(match["date"]).strftime("%Y-%m-%d %H:%M"),
                "group": match.get("group", ""),
                "stage": match.get("stage", ""),
                "home_team": names.get(str(match["home_team"]), match["home_team"]),
                "away_team": names.get(str(match["away_team"]), match["away_team"]),
                "predicted_outcome": outcome_code,
                "predicted_winner": predicted_winner,
                "winner_probability": winner_probability,
                "hypothetical_score": hypothetical_score,
                "hypothetical_score_probability": hypothetical_score_probability,
                "modal_score": modal_score,
                "modal_score_probability": modal_score_probability,
                "swisslos_home_odds": feature_row.get("swisslos_home_odds"),
                "swisslos_draw_odds": feature_row.get("swisslos_draw_odds"),
                "swisslos_away_odds": feature_row.get("swisslos_away_odds"),
                "swisslos_quote": swisslos_quote,
                "swisslos_implied_probability": swisslos_implied,
                "swisslos_margin": feature_row.get("swisslos_margin"),
                "swisslos_age_hours": feature_row.get("swisslos_age_hours"),
                "swisslos_collected_at": swisslos_collected_at,
                "swisslos_current_age_hours": swisslos_current_age,
                "model_swisslos_edge": (
                    winner_probability - swisslos_implied
                    if pd.notna(swisslos_implied)
                    else pd.NA
                ),
                "bet_decision": bet.decision,
                "bet_outcome": bet.outcome,
                "bet_label": bet.label,
                "bet_odds": bet.decimal_odds,
                "bet_model_probability": bet.model_probability,
                "bet_implied_probability": bet.implied_probability,
                "bet_edge": bet.edge,
                "bet_expected_return": bet.expected_return,
                "bet_stake_fraction": bet.recommended_fraction,
                "bet_stake": bet.stake,
                "bet_reason": bet.reason,
                "ev_tip": recommendation.score,
                "tip": recommendation.score,
                "home_win": prediction.home_win,
                "draw": prediction.draw,
                "away_win": prediction.away_win,
                "raw_home_win": prediction.uncalibrated_home_win,
                "raw_draw": prediction.uncalibrated_draw,
                "raw_away_win": prediction.uncalibrated_away_win,
                "expected_points": recommendation.expected_points,
                "risk": recommendation.standard_deviation,
                "confidence": prediction.confidence,
                "confidence_label": prediction.confidence_label,
                "classification": recommendation.classification,
                "data_uncertainty": prediction.data_uncertainty,
                "home_form_points_5": feature_row.get("home_form_points_5"),
                "away_form_points_5": feature_row.get("away_form_points_5"),
                "home_form_trend_5": feature_row.get("home_form_trend_5"),
                "away_form_trend_5": feature_row.get("away_form_trend_5"),
                "status": match.get("status", "scheduled"),
                "actual_result": (
                    f"{int(match['actual_home_goals'])}:"
                    f"{int(match['actual_away_goals'])}"
                    if pd.notna(match.get("actual_home_goals"))
                    and pd.notna(match.get("actual_away_goals"))
                    else ""
                ),
            }
        )
    summary = allocate_match_portfolio(
        pd.DataFrame(rows),
        bankroll=bankroll,
        config=betting_rules,
        pending_bets=pending_bets,
        matches=bundle.matches,
    )
    summary = summary.sort_values(
        ["date", "official_match_number"],
        kind="stable",
    ).reset_index(drop=True)
    return features, predictions, recommendations, summary


def pct(value: float) -> str:
    return f"{value:.1%}"


def dashboard(
    summary: pd.DataFrame,
    strategy: str,
    bundle: DataBundle,
    scoring: ScoringRules,
    use_joker: bool,
    bankroll: float,
    use_calibration: bool,
):
    st.header("Dashboard")
    st.caption(
        f"Strategie: {strategy.title()} | Quelle: {bundle.source_label} | "
        f"{len(summary)} Spiele | Joker: {'aktiv' if use_joker else 'inaktiv'}"
    )
    st.info(
        f"WM-2026-Teilnehmer und Gruppenspielplan. {bundle.source_label}. "
        "Tipps, Expected Points und verbleibender Turnierausgang sind "
        "hypothetische Modellrechnungen. Gespielte, synchronisierte Partien "
        "werden in der Simulation fixiert."
    )
    st.caption(
        (
            "Die angezeigten 1/X/2-Wahrscheinlichkeiten sind mit einem zeitlich "
            "getrennten historischen Backtest kalibriert."
        )
        if use_calibration
        else "Die Backtest-Kalibrierung ist für diese Ansicht deaktiviert."
    )
    official_fifa_rows = bundle.ratings[
        bundle.ratings["source"].astype(str).str.contains(
            "Official FIFA", case=False, na=False
        )
    ]
    fifa_as_of = (
        official_fifa_rows["as_of"].max().strftime("%d.%m.%Y")
        if not official_fifa_rows.empty
        else "nicht geladen"
    )
    signal_columns = st.columns(5)
    signal_columns[0].metric(
        "Historische Spiele", f"{len(bundle.historical_results):,}".replace(",", "'")
    )
    signal_columns[1].metric(
        "Verfügbarkeitsmeldungen", len(bundle.availability)
    )
    signal_columns[2].metric("Aufstellungszeilen", len(bundle.lineups))
    signal_columns[3].metric("Kartenereignisse", len(bundle.match_events))
    signal_columns[4].metric("Quotensnapshots", len(bundle.odds))
    st.caption(f"FIFA-Rangliste: Stand {fifa_as_of}")
    if bundle.availability.empty or bundle.lineups.empty or bundle.odds.empty:
        st.caption(
            "Leere Echtzeitsignale werden neutral behandelt und erhöhen die "
            "Datenunsicherheit. Quelle und Zeitstempel sind bei jedem Import Pflicht."
        )

    display = summary.copy()
    for column in [
        "home_win",
        "draw",
        "away_win",
        "winner_probability",
        "hypothetical_score_probability",
        "confidence",
        "data_uncertainty",
    ]:
        display[column] = display[column].map(pct)
    display["expected_points"] = display["expected_points"].map(lambda x: f"{x:.2f}")
    display["risk"] = display["risk"].map(lambda x: f"{x:.2f}")
    for column in [
        "swisslos_home_odds",
        "swisslos_draw_odds",
        "swisslos_away_odds",
        "swisslos_quote",
    ]:
        display[column] = display[column].map(
            lambda value: f"{value:.2f}" if pd.notna(value) else "nicht geladen"
        )
    display["model_swisslos_edge"] = display["model_swisslos_edge"].map(
        lambda value: f"{value:+.1%}" if pd.notna(value) else "nicht geladen"
    )
    display["bet_expected_return"] = display["bet_expected_return"].map(
        lambda value: f"{value:+.1%}" if pd.notna(value) else "nicht geladen"
    )
    display["bet_stake"] = display["bet_stake"].map(
        lambda value: f"CHF {value:.2f}" if value > 0 else "-"
    )

    def form_indicator(points: float, trend: float) -> str:
        if pd.isna(points):
            return "keine Daten"
        arrow = "↑" if trend > 0.08 else "↓" if trend < -0.08 else "→"
        return f"{arrow} {points:.2f} Pkt."

    display["home_form"] = display.apply(
        lambda row: form_indicator(
            row["home_form_points_5"], row["home_form_trend_5"]
        ),
        axis=1,
    )
    display["away_form"] = display.apply(
        lambda row: form_indicator(
            row["away_form_points_5"], row["away_form_trend_5"]
        ),
        axis=1,
    )
    display["bet_recommendation"] = display.apply(
        lambda row: (
            f"{row['bet_outcome']} - {row['bet_label']}"
            if row["bet_decision"] == "WETTEN"
            else "Keine Wette"
        ),
        axis=1,
    )
    st.dataframe(
        display[
            [
                "official_match_number",
                "date",
                "home_team",
                "away_team",
                "home_form",
                "away_form",
                "actual_result",
                "predicted_winner",
                "winner_probability",
                "home_win",
                "draw",
                "away_win",
                "swisslos_home_odds",
                "swisslos_draw_odds",
                "swisslos_away_odds",
                "model_swisslos_edge",
                "bet_recommendation",
                "bet_expected_return",
                "bet_stake",
                "hypothetical_score",
                "hypothetical_score_probability",
                "ev_tip",
                "expected_points",
                "confidence_label",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "official_match_number": "FIFA-Spiel",
            "home_win": "1",
            "draw": "X",
            "away_win": "2",
            "swisslos_home_odds": "Swisslos 1",
            "swisslos_draw_odds": "Swisslos X",
            "swisslos_away_odds": "Swisslos 2",
            "model_swisslos_edge": "Modell vs. Swisslos",
            "bet_recommendation": "Wettentscheidung",
            "bet_expected_return": "Erwarteter Ertrag",
            "bet_stake": f"Einsatz bei CHF {bankroll:.0f}",
            "actual_result": "Endstand",
            "home_form": "Form Heim",
            "away_form": "Form Auswärts",
            "predicted_winner": "Wahrscheinlichster Sieger",
            "winner_probability": "Sieger-/Tendenzchance",
            "hypothetical_score": "Hypothetischer Endstand",
            "hypothetical_score_probability": "Resultatchance",
            "ev_tip": "SRF-EV-Tipp",
            "expected_points": "Expected Points",
            "confidence_label": "Confidence",
        },
    )
    if st.button("Empfehlungen lokal speichern"):
        run_id = save_recommendations(
            DATABASE_PATH,
            summary,
            bundle.source_label,
            strategy,
            {**scoring.to_dict(), "joker_active": use_joker},
        )
        st.success(f"Analyse-Lauf {run_id} wurde in SQLite gespeichert.")


def match_analysis(
    summary: pd.DataFrame,
    features: pd.DataFrame,
    predictions: dict,
    recommendations: dict,
    bundle: DataBundle,
    scoring: ScoringRules,
    config: dict,
    strategy: str,
    betting_config: dict,
    bankroll: float,
):
    st.header("Spielanalyse")
    labels = {
        row["match_id"]: (
            f"FIFA-Spiel {int(row['official_match_number'])}: "
            f"{row['home_team']} - {row['away_team']} ({row['date']})"
        )
        for row in summary.to_dict("records")
    }
    match_id = st.selectbox(
        "Spiel",
        options=list(labels),
        format_func=lambda value: labels[value],
    )
    prediction = predictions[match_id]
    recommendation = recommendations[match_id]
    explanation = build_explanation(prediction, recommendation)
    feature_row = features.loc[features["match_id"].astype(str) == match_id].iloc[0]

    outcome_code, predicted_winner, winner_probability = (
        prediction.most_likely_outcome()
    )
    hypothetical_score, hypothetical_probability = (
        prediction.representative_score()
    )
    modal_score, modal_probability = prediction.modal_score()
    st.subheader("1. Wer gewinnt?")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modellprognose", predicted_winner)
    col2.metric("Wahrscheinlichkeit", f"{winner_probability:.1%}")
    col3.metric("Confidence", f"{prediction.confidence:.0%}")
    col4.metric("Datenunsicherheit", f"{prediction.data_uncertainty:.0%}")

    outcome_frame = pd.DataFrame(
        {
            "Ausgang": [
                f"1 - {prediction.home_team}",
                "X - Unentschieden",
                f"2 - {prediction.away_team}",
            ],
            "Wahrscheinlichkeit": [
                prediction.home_win,
                prediction.draw,
                prediction.away_win,
            ],
            "Rohmodell": [
                prediction.uncalibrated_home_win,
                prediction.uncalibrated_draw,
                prediction.uncalibrated_away_win,
            ],
        }
    )
    outcome_chart = px.bar(
        outcome_frame,
        x="Ausgang",
        y="Wahrscheinlichkeit",
        color="Ausgang",
        text_auto=".1%",
    )
    outcome_chart.update_yaxes(tickformat=".0%", range=[0, 1])
    outcome_chart.update_layout(showlegend=False)
    st.plotly_chart(outcome_chart, width="stretch")
    if prediction.uncalibrated_home_win is not None:
        shifts = [
            prediction.home_win - prediction.uncalibrated_home_win,
            prediction.draw - prediction.uncalibrated_draw,
            prediction.away_win - prediction.uncalibrated_away_win,
        ]
        if any(abs(value) > 1e-9 for value in shifts):
            st.caption(
                "Backtest-Kalibrierung gegenüber dem rohen Poisson-Modell: "
                f"1 {shifts[0]:+.1%}, X {shifts[1]:+.1%}, 2 {shifts[2]:+.1%}."
            )
        else:
            st.caption("Backtest-Kalibrierung ist für diese Analyse deaktiviert.")

    match = bundle.matches[
        bundle.matches["match_id"].astype(str) == str(match_id)
    ].iloc[0]
    team_ids = {str(match["home_team"]), str(match["away_team"])}
    kickoff = pd.Timestamp(
        match.get("kickoff_utc")
        if pd.notna(match.get("kickoff_utc"))
        else match["date"]
    )
    team_names = bundle.teams.set_index("team_id")["team_name"].to_dict()
    form_frames = []
    for team_id in (str(match["home_team"]), str(match["away_team"])):
        curve = team_form_curve(
            bundle.historical_results,
            team_id,
            kickoff,
            window=10,
        )
        if not curve.empty:
            curve["Team"] = team_names.get(team_id, team_id)
            form_frames.append(curve)
    st.subheader("Aktuelle Formkurve")
    form_metrics = st.columns(4)
    form_metrics[0].metric(
        f"{team_names.get(str(match['home_team']), match['home_team'])} Form 5",
        (
            f"{feature_row.get('home_form_points_5'):.2f} Pkt."
            if pd.notna(feature_row.get("home_form_points_5"))
            else "keine Daten"
        ),
    )
    form_metrics[1].metric(
        "Trend Heim",
        (
            f"{feature_row.get('home_form_trend_5'):+.2f}"
            if pd.notna(feature_row.get("home_form_trend_5"))
            else "keine Daten"
        ),
    )
    form_metrics[2].metric(
        f"{team_names.get(str(match['away_team']), match['away_team'])} Form 5",
        (
            f"{feature_row.get('away_form_points_5'):.2f} Pkt."
            if pd.notna(feature_row.get("away_form_points_5"))
            else "keine Daten"
        ),
    )
    form_metrics[3].metric(
        "Trend Auswärts",
        (
            f"{feature_row.get('away_form_trend_5'):+.2f}"
            if pd.notna(feature_row.get("away_form_trend_5"))
            else "keine Daten"
        ),
    )
    if form_frames:
        form_curve = pd.concat(form_frames, ignore_index=True)
        form_chart = px.line(
            form_curve,
            x="date",
            y="rolling_points_5",
            color="Team",
            markers=True,
            hover_data={
                "opponent": True,
                "score": True,
                "result": True,
                "points": True,
                "rolling_points_5": ":.2f",
            },
            labels={
                "date": "Spieldatum",
                "rolling_points_5": "Rollierender Punkteschnitt (max. 3)",
                "opponent": "Gegner",
                "score": "Resultat",
                "result": "S/U/N",
                "points": "Punkte",
            },
        )
        form_chart.update_yaxes(range=[0, 3])
        st.plotly_chart(form_chart, width="stretch")
    st.caption(
        "Der Formwert gewichtet die letzten fünf Spiele. Der Trend misst, "
        "ob die Resultate innerhalb dieser Serie steigen oder fallen und "
        "fliesst mit kleinem Gewicht in die Prognose ein."
    )

    scoped_suspensions = bundle.availability[
        (bundle.availability["status"].astype(str).str.lower() == "suspended")
        & (bundle.availability["team_id"].astype(str).isin(team_ids))
    ].copy()
    if "match_id" in scoped_suspensions:
        scope = scoped_suspensions["match_id"].fillna("").astype(str)
        scoped_suspensions = scoped_suspensions[
            (scope == "") | (scope == str(match_id))
        ]
    if not scoped_suspensions.empty:
        suspended = ", ".join(
            f"{row['player_name']} "
            f"({team_names.get(str(row['team_id']), row['team_id'])})"
            for row in scoped_suspensions.to_dict("records")
        )
        st.warning(
            "Im Modell berücksichtigte Sperren für dieses Spiel: " + suspended
        )

    with st.expander("Berücksichtigte Karten und Sperren"):
        cards = bundle.match_events[
            bundle.match_events["team_id"].astype(str).isin(team_ids)
        ].copy()
        if cards.empty:
            st.caption("Für diese Teams sind noch keine WM-Karten erfasst.")
        else:
            team_names = bundle.teams.set_index("team_id")["team_name"].to_dict()
            cards["Team"] = cards["team_id"].astype(str).map(team_names)
            cards["Karte"] = cards["event_type"].map(
                {"yellow_card": "Gelb", "red_card": "Rot"}
            )
            cards["Minute"] = cards["minute"].astype(int)
            st.dataframe(
                cards[["match_id", "Team", "player_name", "Karte", "Minute"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "match_id": "Spiel",
                    "player_name": "Spieler",
                },
            )
        st.caption(
            "Rot bedeutet mindestens eine Sperre für das nächste Spiel. "
            "Ohne bestätigte FIFA-Entscheidung wird keine längere Sperre angenommen. "
            "Gelbe Karten fliessen als kleiner Disziplintrend ein."
        )

    st.subheader("Swisslos Sporttip")
    swisslos_values = [
        feature_row.get("swisslos_home_odds"),
        feature_row.get("swisslos_draw_odds"),
        feature_row.get("swisslos_away_odds"),
    ]
    swisslos_cols = st.columns(4)
    for column, label, value in zip(
        swisslos_cols[:3],
        ["Quote 1", "Quote X", "Quote 2"],
        swisslos_values,
    ):
        column.metric(
            label,
            f"{value:.2f}" if pd.notna(value) else "nicht geladen",
        )
    swisslos_cols[3].metric(
        "Buchmachermarge",
        (
            f"{feature_row.get('swisslos_margin'):.1%}"
            if pd.notna(feature_row.get("swisslos_margin"))
            else "nicht geladen"
        ),
    )
    if pd.notna(summary.loc[summary["match_id"] == match_id, "swisslos_quote"].iloc[0]):
        selected_row = summary.loc[summary["match_id"] == match_id].iloc[0]
        st.info(
            f"Zur Modellprognose {predicted_winner} bietet Swisslos die Quote "
            f"{selected_row['swisslos_quote']:.2f}. Die bereinigte implizite "
            f"Wahrscheinlichkeit beträgt "
            f"{selected_row['swisslos_implied_probability']:.1%}; "
            f"Modellabweichung: {selected_row['model_swisslos_edge']:+.1%}."
        )
    else:
        st.warning(
            "Keine Swisslos-Quote geladen. Importiere auf der Seite Datenimport "
            "eine aktuelle odds.csv mit bookmaker=Swisslos."
        )
    st.link_button(
        "Swisslos Sporttip öffnen",
        "https://www.swisslos.ch/de/sporttip/sportwetten/fussball",
    )

    selected_row = summary.loc[summary["match_id"] == match_id].iloc[0]
    bet = assess_market(
        prediction,
        decimal_odds={
            "1": feature_row.get("swisslos_home_odds"),
            "X": feature_row.get("swisslos_draw_odds"),
            "2": feature_row.get("swisslos_away_odds"),
        },
        implied_probabilities={
            "1": feature_row.get("swisslos_home_probability"),
            "X": feature_row.get("swisslos_draw_probability"),
            "2": feature_row.get("swisslos_away_probability"),
        },
        bankroll=bankroll,
        config=betting_config,
        odds_age_hours=selected_row.get("swisslos_current_age_hours"),
        market_open=str(selected_row.get("status", "scheduled")).lower()
        not in {"completed", "finished", "final"},
    )
    portfolio_row = summary.loc[summary["match_id"] == match_id].iloc[0]
    st.subheader("Wettentscheidung")
    bet_cols = st.columns(4)
    bet_cols[0].metric("Portfolio-Entscheid", portfolio_row["bet_decision"])
    bet_cols[1].metric(
        "Bester Markt",
        f"{bet.outcome} - {bet.label}" if bet.outcome else "nicht verfügbar",
    )
    bet_cols[2].metric(
        "Expected Return",
        f"{bet.expected_return:+.1%}"
        if pd.notna(bet.expected_return)
        else "nicht verfügbar",
    )
    bet_cols[3].metric(
        "Portfolio-Einsatz",
        (
            f"CHF {portfolio_row['bet_stake']:.2f}"
            if portfolio_row["bet_stake"] > 0
            else "CHF 0.00"
        ),
    )
    if portfolio_row["bet_decision"] == "WETTEN":
        st.success(portfolio_row["bet_reason"])
    else:
        st.warning(portfolio_row["bet_reason"])
    st.caption(
        "Expected Return ist eine Modellschätzung pro eingesetztem Franken, "
        "keine Gewinnzusage. Auch positive Modellwerte können verlieren."
    )

    st.subheader("2. Rein hypothetischer Endstand")
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric(
        "Passend zur wahrscheinlichsten Tendenz",
        hypothetical_score,
    )
    result_col2.metric(
        "Wahrscheinlichkeit genau dieses Resultats",
        f"{hypothetical_probability:.1%}",
    )
    result_col3.metric(
        "Erwartete Tore",
        f"{prediction.expected_home_goals:.2f} : "
        f"{prediction.expected_away_goals:.2f}",
    )
    if modal_score != hypothetical_score:
        st.caption(
            f"Das häufigste einzelne Resultat über alle Ausgänge ist "
            f"{modal_score} ({modal_probability:.1%}). Es kann von der "
            "wahrscheinlichsten 1/X/2-Tendenz abweichen, weil sich die "
            "Siegwahrscheinlichkeit auf mehrere Resultate verteilt."
        )

    st.subheader("3. Punkte-optimierter SRF-Tipp")
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    tip_col1.metric("SRF-EV-Tipp", recommendation.score)
    tip_col2.metric("Expected Points", f"{recommendation.expected_points:.2f}")
    tip_col3.metric("Strategie", recommendation.classification)
    st.caption(
        "Der SRF-EV-Tipp maximiert erwartete Tippspielpunkte. Er muss weder "
        "das häufigste Einzelresultat noch der repräsentative Endstand sein."
    )
    st.write(explanation["why"])
    st.info(
        f"Nächste Alternative: {explanation['runner_up']}. "
        f"EV-Abstand: {explanation['ev_gap']:.3f} Punkte. "
        f"Einordnung: {explanation['classification']}."
    )
    st.caption(explanation["data_warning"])

    left, right = st.columns(2)
    with left:
        st.subheader("Top 10 hypothetische Resultate")
        probable = prediction.score_probabilities().head(10)
        chart = px.bar(
            probable,
            x="score",
            y="probability",
            labels={"score": "Resultat", "probability": "Wahrscheinlichkeit"},
        )
        chart.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart, width="stretch")
    with right:
        st.subheader("Top 10 SRF-Tipps nach Expected Value")
        all_tips = optimize_tip(
            prediction,
            scoring,
            strategy="safe",
            optimizer_config=config.get("optimizer", {}),
        ).alternatives
        ev_chart = px.bar(
            all_tips.sort_values("expected_points"),
            x="expected_points",
            y="tip",
            orientation="h",
            labels={"expected_points": "Expected Points", "tip": "Tipp"},
        )
        st.plotly_chart(ev_chart, width="stretch")

    st.subheader("Modelltreiber")
    drivers = explanation["drivers"]
    driver_chart = px.bar(
        drivers.sort_values("effect"),
        x="effect",
        y="factor",
        orientation="h",
        color="direction",
        labels={"effect": "Einfluss auf Heim- vs. Auswärtsteam", "factor": "Faktor"},
    )
    st.plotly_chart(driver_chart, width="stretch")
    with st.expander("Form- und Modellwerte"):
        selected = {
            key: feature_row.get(key)
            for key in [
                "rating_diff",
                "form_diff",
                "home_form_trend_5",
                "away_form_trend_5",
                "form_trend_diff",
                "home_goals_for_5",
                "away_goals_for_5",
                "home_goals_against_5",
                "away_goals_against_5",
                "h2h_goal_diff",
                "travel_diff_1000km",
                "home_availability_burden",
                "away_availability_burden",
                "home_yellow_cards_5",
                "away_yellow_cards_5",
                "home_red_cards_5",
                "away_red_cards_5",
                "discipline_edge",
                "lineup_strength_diff",
                "market_home_probability",
                "market_draw_probability",
                "market_away_probability",
                "odds_age_hours",
            ]
        }
        selected["expected_home_goals"] = prediction.expected_home_goals
        selected["expected_away_goals"] = prediction.expected_away_goals
        st.json(selected)


def scoring_configuration(config: dict):
    st.header("Scoring-Konfiguration")
    st.warning(
        "Die Voreinstellung ist ein Demo-Regelwerk. Bitte die tatsächlich geltenden "
        "SRF-Regeln prüfen und hier exakt abbilden."
    )
    current = get_scoring(config).to_dict()
    current_model = get_model_config(config)
    with st.form("scoring_form"):
        col1, col2 = st.columns(2)
        exact = col1.number_input(
            "Punkte exaktes Resultat", 0.0, 100.0, float(current["exact_points"]), 0.5
        )
        tendency = col2.number_input(
            "Punkte richtige Tendenz", 0.0, 100.0, float(current["tendency_points"]), 0.5
        )
        difference = col1.number_input(
            "Punkte richtige Tordifferenz",
            0.0,
            100.0,
            float(current["goal_difference_points"]),
            0.5,
        )
        team_goals = col2.number_input(
            "Punkte je richtige Team-Torzahl",
            0.0,
            100.0,
            float(current["team_goals_points"]),
            0.5,
        )
        bonus = col1.number_input(
            "Bonus bei mindestens einem Treffer",
            0.0,
            100.0,
            float(current["bonus_points"]),
            0.5,
        )
        multiplier = col2.number_input(
            "Allgemeiner Multiplikator",
            0.1,
            20.0,
            float(current["multiplier"]),
            0.1,
        )
        joker = col1.number_input(
            "Joker-Multiplikator",
            0.1,
            20.0,
            float(current["joker_multiplier"]),
            0.1,
        )
        mode = st.radio(
            "Regeln kombinieren",
            ["additive", "highest_only"],
            index=0 if current["combination_mode"] == "additive" else 1,
            format_func=lambda value: (
                "Additiv: alle erfüllten Regeln zählen"
                if value == "additive"
                else "Nur die höchste erfüllte Regel zählt"
            ),
        )
        st.subheader("Ensemble-Gewichte")
        st.caption(
            "Die Gewichte müssen nicht exakt 1 ergeben; sie skalieren den Einfluss "
            "des jeweiligen Faktors auf die erwarteten Tore."
        )
        model_weights = {}
        weight_labels = {
            "rating": "Rating/FIFA",
            "form": "Aktuelle Form",
            "attack_defense": "Angriff/Verteidigung",
            "home_context": "Austragungsort/Kontinent",
            "travel": "Reise",
            "head_to_head": "Direktduelle",
            "availability": "Verletzungen/Verfügbarkeit",
            "lineup": "Aufstellung",
            "market": "Marktquoten",
        }
        for key, label in weight_labels.items():
            model_weights[key] = st.slider(
                label,
                min_value=0.0,
                max_value=1.0,
                value=float(current_model["weights"].get(key, 0.0)),
                step=0.01,
            )
        submitted = st.form_submit_button("Scoring übernehmen")
    if submitted:
        st.session_state.scoring = {
            "exact_points": exact,
            "tendency_points": tendency,
            "goal_difference_points": difference,
            "team_goals_points": team_goals,
            "combination_mode": mode,
            "bonus_points": bonus,
            "multiplier": multiplier,
            "joker_multiplier": joker,
        }
        updated_model = copy.deepcopy(current_model)
        updated_model["weights"] = model_weights
        st.session_state.model_config = updated_model
        st.success("Scoring-Regeln und Modellgewichte wurden übernommen.")


def betting_page(
    summary: pd.DataFrame,
    features: pd.DataFrame,
    predictions: dict,
    betting_config: dict,
    bankroll: float,
):
    st.header("Wettanalyse")
    st.warning(
        "Diese Seite zeigt ausschließlich mathematische Modellwerte. Sie kann "
        "keinen Gewinn garantieren; Einsätze können vollständig verloren gehen. "
        "Die Anwendung platziert keine Wetten."
    )
    st.caption(
        "Eine Wette wird nur angezeigt, wenn Swisslos-Quote, Expected Return, "
        "Modellvorteil, Datenqualität und Aktualität alle die eingestellten "
        "Mindestwerte erfüllen."
    )

    available = int(summary["swisslos_home_odds"].notna().sum())
    recommendations = int((summary["bet_decision"] == "WETTEN").sum())
    portfolio_total_cap = float(summary["portfolio_total_cap"].iloc[0])
    portfolio_pending = float(summary["portfolio_pending_stake"].iloc[0])
    portfolio_new = float(summary["portfolio_new_stake"].iloc[0])
    portfolio_remaining = float(summary["portfolio_remaining_capacity"].iloc[0])
    cols = st.columns(6)
    cols[0].metric("Aktuelle Bankroll", f"CHF {bankroll:.2f}")
    cols[1].metric("Swisslos-Quoten", available)
    cols[2].metric("Portfolio-Wetten", recommendations)
    cols[3].metric("Bereits offen", f"CHF {portfolio_pending:.2f}")
    cols[4].metric("Neue Einsätze", f"CHF {portfolio_new:.2f}")
    cols[5].metric(
        "Freie Risikokapazität",
        f"CHF {portfolio_remaining:.2f}",
        help=f"Gesamtes Expositionslimit: CHF {portfolio_total_cap:.2f}",
    )

    display = summary.copy()
    display["Swisslos 1/X/2"] = display.apply(
        lambda row: (
            f"{row['swisslos_home_odds']:.2f} / "
            f"{row['swisslos_draw_odds']:.2f} / "
            f"{row['swisslos_away_odds']:.2f}"
            if pd.notna(row["swisslos_home_odds"])
            else "nicht geladen"
        ),
        axis=1,
    )
    display["Empfehlung"] = display.apply(
        lambda row: (
            f"{row['bet_outcome']} - {row['bet_label']}"
            if row["bet_decision"] == "WETTEN"
            else "Keine Wette"
        ),
        axis=1,
    )
    display["Modellchance"] = display["bet_model_probability"].map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "-"
    )
    display["Quote"] = display["bet_odds"].map(
        lambda value: f"{value:.2f}" if pd.notna(value) else "-"
    )
    display["Edge"] = display["bet_edge"].map(
        lambda value: f"{value:+.1%}" if pd.notna(value) else "-"
    )
    display["Expected Return"] = display["bet_expected_return"].map(
        lambda value: f"{value:+.1%}" if pd.notna(value) else "-"
    )
    display["Einsatz"] = display["bet_stake"].map(
        lambda value: f"CHF {value:.2f}" if value > 0 else "CHF 0.00"
    )
    st.dataframe(
        display[
            [
                "date",
                "home_team",
                "away_team",
                "home_win",
                "draw",
                "away_win",
                "Swisslos 1/X/2",
                "Empfehlung",
                "Modellchance",
                "Quote",
                "Edge",
                "Expected Return",
                "Einsatz",
                "bet_reason",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "date": "Datum",
            "home_team": "Heim",
            "away_team": "Auswärts",
            "home_win": st.column_config.NumberColumn("1", format="percent"),
            "draw": st.column_config.NumberColumn("X", format="percent"),
            "away_win": st.column_config.NumberColumn("2", format="percent"),
            "bet_reason": "Begründung",
        },
    )

    labels = {
        row["match_id"]: f"{row['home_team']} - {row['away_team']} ({row['date']})"
        for row in summary.to_dict("records")
    }
    match_id = st.selectbox(
        "Detailprüfung für ein Spiel",
        options=list(labels),
        format_func=lambda value: labels[value],
        key="betting_match",
    )
    feature_row = features.loc[features["match_id"].astype(str) == match_id].iloc[0]
    selected = summary.loc[summary["match_id"] == match_id].iloc[0]
    recommendation = assess_market(
        predictions[match_id],
        decimal_odds={
            "1": feature_row.get("swisslos_home_odds"),
            "X": feature_row.get("swisslos_draw_odds"),
            "2": feature_row.get("swisslos_away_odds"),
        },
        implied_probabilities={
            "1": feature_row.get("swisslos_home_probability"),
            "X": feature_row.get("swisslos_draw_probability"),
            "2": feature_row.get("swisslos_away_probability"),
        },
        bankroll=bankroll,
        config=betting_config,
        odds_age_hours=selected.get("swisslos_current_age_hours"),
        market_open=str(selected.get("status", "scheduled")).lower()
        not in {"completed", "finished", "final"},
    )
    detail = recommendation.as_frame()
    for column in [
        "model_probability",
        "implied_probability",
        "edge",
        "expected_return",
        "full_kelly_fraction",
        "recommended_fraction",
    ]:
        detail[column] = detail[column].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "-"
        )
    detail["decimal_odds"] = detail["decimal_odds"].map(
        lambda value: f"{value:.2f}" if pd.notna(value) else "-"
    )
    detail["stake"] = detail["stake"].map(lambda value: f"CHF {value:.2f}")
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        column_config={
            "outcome": "Markt",
            "label": "Ausgang",
            "model_probability": "Modellchance",
            "decimal_odds": "Swisslos-Quote",
            "implied_probability": "Swisslos-Wahrscheinlichkeit",
            "edge": "Modell-Edge",
            "expected_return": "Expected Return",
            "full_kelly_fraction": "Volles Kelly",
            "recommended_fraction": "Empfohlener Anteil",
            "stake": "Einsatz",
            "decision": "Entscheid",
            "reason": "Begründung",
        },
    )
    st.caption(
        "Die Detailtabelle zeigt den isolierten Kelly-Wert je Markt. Der "
        "tatsächliche Portfolio-Einsatz in der Übersicht berücksichtigt "
        "zusätzlich offene Wetten, Gesamtlimit, Tageslimit und Priorität."
    )


def _local_timestamp(selected_date: date, selected_time) -> pd.Timestamp:
    local = pd.Timestamp(
        datetime.combine(selected_date, selected_time),
        tz="Europe/Zurich",
    )
    return local.tz_convert("UTC").tz_localize(None)


def _match_kickoff_utc(match: pd.Series | dict) -> pd.Timestamp:
    kickoff_utc = match.get("kickoff_utc")
    return (
        pd.Timestamp(kickoff_utc)
        if pd.notna(kickoff_utc)
        else pd.Timestamp(match["date"])
    )


def swisslos_odds_page(bundle: DataBundle):
    st.header("Swisslos Sporttip")
    st.write(
        "Aktuelle 1/X/2- und Weltmeisterquoten können direkt aus dem "
        "offiziellen öffentlichen Sporttip-Widget gelesen werden. Jeder "
        "Abruf erzeugt einen lokalen, zeitgestempelten Snapshot."
    )
    st.caption(
        "Nur lesender Abruf: kein Login, keine Wettabgabe und keine "
        "Gewinnzusage. Die manuelle Erfassung bleibt als Rückfalloption."
    )
    database = DATABASE_PATH
    source = SWISSLOS_SOURCE
    names = bundle.teams.set_index("team_id")["team_name"].to_dict()
    notice = st.session_state.pop("swisslos_notice", None)
    if notice:
        st.success(notice)
    action_columns = st.columns([1, 1])
    if action_columns[0].button(
        "Offizielle Swisslos-Quoten aktualisieren",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner("Swisslos Sporttip wird gelesen ..."):
                snapshot = fetch_swisslos_odds(bundle.teams, bundle.matches)
                saved_matches = save_match_odds(database, snapshot.match_odds)
                saved_outrights = save_outright_odds(
                    database, snapshot.outright_odds
                )
                bundle.odds = _merge_snapshots(
                    bundle.odds,
                    snapshot.match_odds,
                    ["match_id", "bookmaker", "collected_at"],
                )
                bundle.outright_odds = _merge_snapshots(
                    bundle.outright_odds,
                    snapshot.outright_odds,
                    ["team_id", "bookmaker", "market", "collected_at"],
                )
            collected_local = (
                snapshot.collected_at.tz_localize("UTC")
                .tz_convert("Europe/Zurich")
                .strftime("%d.%m.%Y %H:%M:%S")
            )
            st.session_state.swisslos_notice = (
                f"{saved_matches} Spielquoten und {saved_outrights} "
                f"Weltmeisterquoten gespeichert. Stand: {collected_local} Uhr."
            )
            st.rerun()
        except SwisslosOddsError as exc:
            st.error(str(exc))
    action_columns[1].link_button(
        "Swisslos-Quelle öffnen",
        SWISSLOS_SOURCE,
        width="stretch",
    )
    match_labels = {
        str(row["match_id"]): (
            f"{names.get(str(row['home_team']), row['home_team'])} - "
            f"{names.get(str(row['away_team']), row['away_team'])} "
            f"({pd.Timestamp(row['date']).strftime('%d.%m.%Y %H:%M')})"
        )
        for row in bundle.matches.to_dict("records")
    }

    match_tab, champion_tab, history_tab = st.tabs(
        ["1/X/2-Spielquote", "Weltmeisterquoten", "Snapshot-Verlauf"]
    )
    with match_tab:
        with st.form("manual_match_odds"):
            match_id = st.selectbox(
                "Spiel",
                options=list(match_labels),
                format_func=lambda value: match_labels[value],
            )
            quote_columns = st.columns(3)
            home_odds = quote_columns[0].number_input(
                "Swisslos 1",
                min_value=1.01,
                value=None,
                step=0.01,
                placeholder="z. B. 2.10",
            )
            draw_odds = quote_columns[1].number_input(
                "Swisslos X",
                min_value=1.01,
                value=None,
                step=0.01,
                placeholder="z. B. 3.40",
            )
            away_odds = quote_columns[2].number_input(
                "Swisslos 2",
                min_value=1.01,
                value=None,
                step=0.01,
                placeholder="z. B. 3.10",
            )
            captured = st.columns(2)
            captured_date = captured[0].date_input(
                "Erfasst am",
                value=date.today(),
            )
            captured_time = captured[1].time_input(
                "Erfasst um (Schweizer Zeit)",
                value=datetime.now().time().replace(second=0, microsecond=0),
            )
            save_match = st.form_submit_button(
                "1/X/2-Snapshot speichern",
                type="primary",
            )
        if save_match:
            if any(value is None for value in (home_odds, draw_odds, away_odds)):
                st.error("Bitte alle drei Quoten 1, X und 2 eintragen.")
            else:
                frame = validate_dataframe(
                    pd.DataFrame(
                        [
                            {
                                "match_id": match_id,
                                "bookmaker": "Swisslos",
                                "collected_at": _local_timestamp(
                                    captured_date, captured_time
                                ),
                                "home_odds": home_odds,
                                "draw_odds": draw_odds,
                                "away_odds": away_odds,
                                "source": source,
                            }
                        ]
                    ),
                    "odds",
                )
                saved = save_match_odds(database, frame)
                bundle.odds = _merge_snapshots(
                    bundle.odds,
                    frame,
                    ["match_id", "bookmaker", "collected_at"],
                )
                st.success(
                    "Spielquote gespeichert."
                    if saved
                    else "Dieser Snapshot war bereits gespeichert."
                )
                st.rerun()

    with champion_tab:
        latest = pd.DataFrame(
            columns=["team_id", "decimal_odds", "collected_at"]
        )
        if not bundle.outright_odds.empty:
            latest = (
                bundle.outright_odds[
                    (bundle.outright_odds["bookmaker"].str.lower() == "swisslos")
                    & (bundle.outright_odds["market"].str.lower() == "champion")
                ]
                .sort_values("collected_at")
                .groupby("team_id", as_index=False)
                .tail(1)[["team_id", "decimal_odds", "collected_at"]]
            )
        editor = bundle.teams[["team_id", "team_name"]].merge(
            latest[["team_id", "decimal_odds"]],
            on="team_id",
            how="left",
        )
        with st.form("manual_outright_odds"):
            edited = st.data_editor(
                editor,
                width="stretch",
                hide_index=True,
                disabled=["team_id", "team_name"],
                column_config={
                    "team_id": "ID",
                    "team_name": "Team",
                    "decimal_odds": st.column_config.NumberColumn(
                        "Swisslos-Weltmeisterquote",
                        min_value=1.01,
                        step=0.1,
                        format="%.2f",
                    ),
                },
            )
            captured = st.columns(2)
            outright_date = captured[0].date_input(
                "Quoten erfasst am",
                value=date.today(),
                key="outright_date",
            )
            outright_time = captured[1].time_input(
                "Quoten erfasst um (Schweizer Zeit)",
                value=datetime.now().time().replace(second=0, microsecond=0),
                key="outright_time",
            )
            save_outrights = st.form_submit_button(
                "Weltmeisterquoten speichern",
                type="primary",
            )
        entered = edited[pd.to_numeric(edited["decimal_odds"], errors="coerce") > 1]
        st.caption(
            f"Aktuell ausgefüllt: {len(entered)} von {len(editor)} Teams "
            f"({len(entered) / len(editor):.0%} Marktabdeckung)."
        )
        if save_outrights:
            if entered.empty:
                st.error("Mindestens eine gültige Weltmeisterquote eintragen.")
            else:
                timestamp = _local_timestamp(outright_date, outright_time)
                frame = entered[["team_id", "decimal_odds"]].copy()
                frame["bookmaker"] = "Swisslos"
                frame["market"] = "champion"
                frame["collected_at"] = timestamp
                frame["source"] = source
                frame = validate_dataframe(
                    frame[
                        [
                            "team_id",
                            "bookmaker",
                            "market",
                            "collected_at",
                            "decimal_odds",
                            "source",
                        ]
                    ],
                    "outright_odds",
                )
                saved = save_outright_odds(database, frame)
                bundle.outright_odds = _merge_snapshots(
                    bundle.outright_odds,
                    frame,
                    ["team_id", "bookmaker", "market", "collected_at"],
                )
                st.success(f"{saved} Weltmeisterquote(n) gespeichert.")
                st.rerun()

    with history_tab:
        st.subheader("Letzte 1/X/2-Snapshots")
        match_history = load_match_odds(database)
        st.dataframe(
            match_history.sort_values("collected_at", ascending=False).head(100),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Letzte Weltmeisterquoten")
        outright_history = load_outright_odds(database)
        if not outright_history.empty:
            outright_history["team"] = (
                outright_history["team_id"].map(names).fillna(
                    outright_history["team_id"]
                )
            )
        st.dataframe(
            outright_history.sort_values("collected_at", ascending=False).head(100),
            width="stretch",
            hide_index=True,
        )


def matchday_page(bundle: DataBundle, summary: pd.DataFrame):
    st.header("Matchday-Center")
    st.write(
        "Erfasste WM-Endstände werden dauerhaft gespeichert. Danach werden "
        "Form, Rolling Elo, alle folgenden Matchprognosen, die "
        "Turniersimulation und offene Paper-Wetten aktualisiert."
    )
    completed_mask = (
        bundle.matches["actual_home_goals"].notna()
        & bundle.matches["actual_away_goals"].notna()
    )
    completed = int(completed_mask.sum())
    remaining = int(len(bundle.matches) - completed)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    open_matches = bundle.matches.loc[~completed_mask].copy()
    open_matches["_kickoff_utc"] = open_matches.apply(
        _match_kickoff_utc, axis=1
    )
    open_matches = open_matches.sort_values("_kickoff_utc")
    next_kickoff = (
        _match_kickoff_utc(open_matches.iloc[0])
        if not open_matches.empty
        else None
    )
    next_kickoff_local = (
        next_kickoff.tz_localize("UTC").tz_convert("Europe/Zurich")
        if next_kickoff is not None
        else None
    )
    metrics = st.columns(4)
    metrics[0].metric("Gespielte Matches", completed)
    metrics[1].metric("Offene Matches", remaining)
    metrics[2].metric(
        "Nächster Anpfiff",
        (
            next_kickoff_local.strftime("%d.%m. %H:%M")
            if next_kickoff_local is not None
            else "-"
        ),
    )
    metrics[3].metric(
        "Historische Spiele im Modell",
        f"{len(bundle.historical_results):,}".replace(",", "'"),
    )

    st.subheader("Nächste Spiele")
    upcoming = summary[
        ~summary["status"].astype(str).str.lower().isin(
            {"completed", "played", "finished", "final"}
        )
    ].head(12).copy()
    if upcoming.empty:
        st.success("Alle geladenen Spiele sind abgeschlossen.")
    else:
        upcoming["home_win"] = upcoming["home_win"].map(lambda value: f"{value:.1%}")
        upcoming["draw"] = upcoming["draw"].map(lambda value: f"{value:.1%}")
        upcoming["away_win"] = upcoming["away_win"].map(lambda value: f"{value:.1%}")
        upcoming["confidence"] = upcoming["confidence"].map(
            lambda value: f"{value:.0%}"
        )
        for column in [
            "swisslos_home_odds",
            "swisslos_draw_odds",
            "swisslos_away_odds",
        ]:
            upcoming[column] = upcoming[column].map(
                lambda value: f"{value:.2f}" if pd.notna(value) else "-"
            )
        st.dataframe(
            upcoming[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "home_win",
                    "draw",
                    "away_win",
                    "swisslos_home_odds",
                    "swisslos_draw_odds",
                    "swisslos_away_odds",
                    "hypothetical_score",
                    "confidence",
                    "bet_decision",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "date": "Datum",
                "home_team": "Heim",
                "away_team": "Auswärts",
                "home_win": "1",
                "draw": "X",
                "away_win": "2",
                "swisslos_home_odds": "Swisslos 1",
                "swisslos_draw_odds": "Swisslos X",
                "swisslos_away_odds": "Swisslos 2",
                "hypothetical_score": "Hypothetisches Resultat",
                "confidence": "Confidence",
                "bet_decision": "Wettentscheid",
            },
        )

    st.subheader("Endstand erfassen oder korrigieren")
    names = bundle.teams.set_index("team_id")["team_name"].to_dict()
    match_labels = {
        str(row["match_id"]): (
            f"{names.get(str(row['home_team']), row['home_team'])} - "
            f"{names.get(str(row['away_team']), row['away_team'])} "
            f"({pd.Timestamp(row['date']).strftime('%d.%m.%Y %H:%M')})"
        )
        for row in bundle.matches.sort_values("date").to_dict("records")
    }
    with st.form("matchday_result"):
        match_id = st.selectbox(
            "Spiel",
            options=list(match_labels),
            format_func=lambda value: match_labels[value],
            key="matchday_match",
        )
        selected_match = bundle.matches[
            bundle.matches["match_id"].astype(str) == match_id
        ].iloc[0]
        existing_home = selected_match.get("actual_home_goals")
        existing_away = selected_match.get("actual_away_goals")
        score_columns = st.columns(2)
        home_goals = score_columns[0].number_input(
            f"Tore {names.get(str(selected_match['home_team']), selected_match['home_team'])}",
            min_value=0,
            max_value=30,
            value=int(existing_home) if pd.notna(existing_home) else 0,
            step=1,
        )
        away_goals = score_columns[1].number_input(
            f"Tore {names.get(str(selected_match['away_team']), selected_match['away_team'])}",
            min_value=0,
            max_value=30,
            value=int(existing_away) if pd.notna(existing_away) else 0,
            step=1,
        )
        source = st.text_input(
            "Quelle",
            value="Manuell nach offiziellem Endstand bestätigt",
        )
        confirmed = st.checkbox(
            "Ich bestätige, dass dies der offizielle Endstand nach regulärer "
            "Spielzeit inklusive Nachspielzeit ist.",
        )
        submit_result = st.form_submit_button(
            "Endstand speichern und Modell aktualisieren",
            type="primary",
        )
    if submit_result:
        kickoff = _match_kickoff_utc(selected_match)
        if not confirmed:
            st.error("Bitte den offiziellen Endstand ausdrücklich bestätigen.")
        elif not source.strip():
            st.error("Eine nachvollziehbare Quelle ist Pflicht.")
        elif kickoff > now and pd.isna(existing_home):
            st.error(
                "Der gespeicherte Anpfiff liegt noch in der Zukunft. "
                "Bitte Datum und ausgewähltes Spiel prüfen."
            )
        else:
            save_match_result(
                DATABASE_PATH,
                match_id=match_id,
                home_goals=int(home_goals),
                away_goals=int(away_goals),
                source=source.strip(),
            )
            results = load_match_results(DATABASE_PATH)
            st.session_state.bundle = apply_match_results(bundle, results)
            settled = settle_paper_bets(
                DATABASE_PATH,
                st.session_state.bundle.matches,
            )
            st.session_state.matchday_notice = (
                f"Endstand gespeichert; {settled} Paper-Wette(n) abgerechnet."
            )
            st.rerun()

    notice = st.session_state.pop("matchday_notice", None)
    if notice:
        st.success(notice)
    if pd.notna(existing_home) and pd.notna(existing_away):
        st.warning(
            f"Für dieses Spiel ist bereits {int(existing_home)}:"
            f"{int(existing_away)} gespeichert. Eine neue Eingabe wird als "
            "Korrektur-Snapshot protokolliert."
        )

    with st.expander("Resultat-Snapshot-Verlauf"):
        result_history = load_match_results(DATABASE_PATH, latest_only=False)
        if not result_history.empty:
            result_history["Spiel"] = result_history["match_id"].map(match_labels)
            result_history["Endstand"] = (
                result_history["home_goals"].astype(str)
                + ":"
                + result_history["away_goals"].astype(str)
            )
        st.dataframe(
            result_history.sort_values("recorded_at", ascending=False),
            width="stretch",
            hide_index=True,
        )


def paper_betting_page(
    summary: pd.DataFrame,
    bundle: DataBundle,
    starting_bankroll: float,
):
    st.header("Paper-Wettjournal")
    st.warning(
        "Das Journal simuliert Wetten ausschließlich zur Modellkontrolle. "
        "Es übermittelt keine Wette an Swisslos und ist keine Gewinnzusage."
    )
    database_path = DATABASE_PATH
    actions = st.columns(2)
    if actions[0].button(
        "Aktuelle Value-Signale ins Journal übernehmen",
        type="primary",
    ):
        saved = save_paper_bets(database_path, summary)
        if saved:
            st.success(f"{saved} neue Paper-Wette(n) gespeichert.")
        else:
            st.info(
                "Keine neuen zulässigen Value-Signale vorhanden. Fehlende "
                "Swisslos-Quoten oder No-Bet-Regeln werden respektiert."
            )
    if actions[1].button("Offene Paper-Wetten abrechnen"):
        settled = settle_paper_bets(database_path, bundle.matches)
        if settled:
            st.success(f"{settled} Paper-Wette(n) abgerechnet.")
        else:
            st.info("Keine neuen abgeschlossenen Spiele für die Abrechnung gefunden.")

    paper_bets = load_paper_bets(database_path)
    performance = calculate_betting_performance(
        paper_bets,
        starting_bankroll=starting_bankroll,
        odds=bundle.odds,
        matches=bundle.matches,
    )
    metrics = st.columns(6)
    metrics[0].metric("Paper-Wetten", performance.total_bets)
    metrics[1].metric("Offen", performance.pending_bets)
    metrics[2].metric("Trefferquote", f"{performance.hit_rate:.1%}")
    metrics[3].metric(
        "Realisierter ROI",
        f"{performance.roi:+.1%}",
        help="Gewinn oder Verlust geteilt durch die Summe abgerechneter Einsätze.",
    )
    metrics[4].metric(
        "Offene Einsätze",
        f"CHF {performance.pending_stake:.2f}",
    )
    metrics[5].metric(
        "Simulierte Bankroll",
        f"CHF {performance.current_bankroll:.2f}",
        delta=f"CHF {performance.total_profit:+.2f}",
    )
    st.caption(
        f"Nach Abzug offener Einsätze frei verfügbar: "
        f"CHF {performance.available_bankroll:.2f}."
    )
    clv = performance.average_closing_line_value
    if pd.notna(clv):
        st.metric(
            "Durchschnittlicher Closing-Line-Value",
            f"{clv:+.1%}",
            help=(
                "Positive Werte bedeuten, dass die gespeicherte Quote besser "
                "war als die letzte importierte Swisslos-Quote vor Anpfiff."
            ),
        )
    else:
        st.caption(
            "Closing-Line-Value wird sichtbar, sobald mehrere Swisslos-"
            "Quotensnapshots vor dem Anpfiff importiert wurden."
        )

    if performance.ledger.empty:
        st.info(
            "Das Journal ist leer. Erst aktuelle Swisslos-Quoten laden; danach "
            "können qualifizierte Value-Signale als Paper-Wetten gespeichert werden."
        )
        return

    ledger = performance.ledger.copy()
    ledger["Spiel"] = ledger["home_team"] + " - " + ledger["away_team"]
    ledger["Tipp"] = ledger["outcome"]
    ledger["Quote"] = ledger["odds"].map(lambda value: f"{value:.2f}")
    ledger["Modellchance"] = ledger["model_probability"].map(
        lambda value: f"{value:.1%}"
    )
    ledger["Edge"] = ledger["edge"].map(lambda value: f"{value:+.1%}")
    ledger["Einsatz"] = ledger["stake"].map(lambda value: f"CHF {value:.2f}")
    ledger["Gewinn/Verlust"] = ledger["profit"].map(
        lambda value: f"CHF {value:+.2f}" if pd.notna(value) else "-"
    )
    ledger["CLV"] = ledger["closing_line_value"].map(
        lambda value: f"{value:+.1%}" if pd.notna(value) else "-"
    )
    ledger["Endstand"] = ledger.apply(
        lambda row: (
            f"{int(row['actual_home_goals'])}:{int(row['actual_away_goals'])}"
            if pd.notna(row["actual_home_goals"])
            and pd.notna(row["actual_away_goals"])
            else "-"
        ),
        axis=1,
    )
    st.dataframe(
        ledger[
            [
                "created_at",
                "Spiel",
                "Tipp",
                "Quote",
                "Modellchance",
                "Edge",
                "Einsatz",
                "status",
                "Endstand",
                "Gewinn/Verlust",
                "CLV",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "created_at": "Gespeichert",
            "status": "Status",
        },
    )

    settled = performance.ledger[
        performance.ledger["status"].isin(["won", "lost"])
    ].sort_values("settled_at")
    if not settled.empty:
        curve = settled[["settled_at", "profit"]].copy()
        curve["bankroll"] = starting_bankroll + curve["profit"].cumsum()
        chart = px.line(
            curve,
            x="settled_at",
            y="bankroll",
            markers=True,
            labels={
                "settled_at": "Abrechnung",
                "bankroll": "Simulierte Bankroll (CHF)",
            },
        )
        st.plotly_chart(chart, width="stretch")
    st.download_button(
        "Paper-Journal als CSV herunterladen",
        performance.ledger.to_csv(index=False).encode("utf-8-sig"),
        "wm2026_paper_wetten.csv",
        "text/csv",
    )


def backtest_page(bundle: DataBundle, model_config: dict):
    st.header("Modell-Backtest")
    st.write(
        "Der Backtest stellt für historische Spiele jeweils nur Informationen "
        "bereit, die vor dem Anpfiff bekannt waren. Aktuelle FIFA-Ratings, "
        "Verletzungen, Aufstellungen und Quoten werden nicht rückwirkend verwendet."
    )
    st.caption(
        "Damit werden der Poisson-, Form-, Angriffs-/Verteidigungs- und "
        "H2H-Kern geprüft. Für eine historische Wett-Rendite wären zusätzlich "
        "damalige Swisslos-Quoten erforderlich."
    )
    min_date = bundle.historical_results["date"].min().date()
    max_date = bundle.historical_results["date"].max().date()
    default_start = max(min_date, date(2024, 1, 1))
    controls = st.columns(3)
    start_date = controls[0].date_input(
        "Testzeitraum ab",
        value=default_start,
        min_value=min_date,
        max_value=max_date,
    )
    min_prior = controls[1].number_input(
        "Mindestens frühere Spiele je Team",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
    )
    max_matches = controls[2].selectbox(
        "Maximale Testspiele",
        options=[250, 500, 1000, 2000],
        index=1,
    )
    if st.button("Backtest berechnen", type="primary"):
        try:
            with st.spinner("Historische Prognosen werden zeitlich sauber berechnet..."):
                report = run_backtest(
                    bundle.historical_results,
                    model_config,
                    known_teams=bundle.teams,
                    start_date=start_date.isoformat(),
                    min_prior_matches=int(min_prior),
                    max_matches=int(max_matches),
                )
            st.session_state.backtest_report = report
            st.session_state.backtest_parameters = (
                start_date.isoformat(),
                int(min_prior),
                int(max_matches),
            )
        except ValueError as error:
            st.error(str(error))

    report = st.session_state.get("backtest_report")
    if report is None:
        st.info(
            "Starte den Backtest, um Prognosegüte und Kalibrierung zu prüfen."
        )
        return

    params = st.session_state.get("backtest_parameters")
    if params:
        st.caption(
            f"Auswertung ab {params[0]}, mindestens {params[1]} frühere Spiele, "
            f"maximal {params[2]} Testspiele."
        )
    metrics = st.columns(4)
    metrics[0].metric("Testspiele", report.matches)
    metrics[1].metric(
        "1/X/2-Trefferquote",
        f"{report.accuracy:.1%}",
    )
    metrics[2].metric(
        "Brier Score",
        f"{report.brier_score:.3f}",
        delta=f"{report.baseline_brier_score - report.brier_score:+.3f} besser",
        delta_color="normal",
        help="Kleiner ist besser. Die Vergleichsprognose nutzt nur historische Grundhäufigkeiten.",
    )
    metrics[3].metric(
        "Log Loss",
        f"{report.log_loss:.3f}",
        delta=f"{report.baseline_log_loss - report.log_loss:+.3f} besser",
        delta_color="normal",
        help="Kleiner ist besser; selbstsichere Fehlprognosen werden stark bestraft.",
    )
    if (
        report.brier_score < report.baseline_brier_score
        and report.log_loss < report.baseline_log_loss
    ):
        st.success(
            "Der Modellkern schlägt im gewählten Zeitraum die konstante "
            "Basisprognose bei Brier Score und Log Loss."
        )
    else:
        st.warning(
            "Der Modellkern schlägt die Basisprognose nicht in allen Gütemaßen. "
            "Wett-Edges sollten deshalb besonders vorsichtig interpretiert werden."
        )

    chronological = report.predictions.sort_values("date").reset_index(drop=True)
    split_index = int(len(chronological) * 0.70)
    if split_index >= 30 and len(chronological) - split_index >= 30:
        validation_start = chronological.iloc[split_index]["date"]
        validation = fit_and_validate_calibrator(
            chronological,
            validation_start=validation_start,
        )
        st.subheader("Zeitlich getrennte Kalibrierungsprüfung")
        st.caption(
            f"{validation.training_matches} ältere Spiele trainieren die "
            f"Kalibrierung; {validation.validation_matches} spätere Spiele "
            "prüfen sie außerhalb des Trainingszeitraums."
        )
        comparison = pd.DataFrame(
            [
                {
                    "Kennzahl": "Brier Score",
                    "Rohmodell": validation.raw_brier_score,
                    "Kalibriert": validation.calibrated_brier_score,
                    "Verbesserung": (
                        validation.raw_brier_score
                        - validation.calibrated_brier_score
                    ),
                },
                {
                    "Kennzahl": "Log Loss",
                    "Rohmodell": validation.raw_log_loss,
                    "Kalibriert": validation.calibrated_log_loss,
                    "Verbesserung": (
                        validation.raw_log_loss
                        - validation.calibrated_log_loss
                    ),
                },
            ]
        )
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
            column_config={
                "Rohmodell": st.column_config.NumberColumn(format="%.4f"),
                "Kalibriert": st.column_config.NumberColumn(format="%.4f"),
                "Verbesserung": st.column_config.NumberColumn(format="%+.4f"),
            },
        )
        if (
            validation.calibrated_brier_score <= validation.raw_brier_score
            and validation.calibrated_log_loss <= validation.raw_log_loss
        ):
            st.success(
                "Die Kalibrierung verbessert beide Gütemaße auf den späteren "
                "Validierungsspielen."
            )
        else:
            st.warning(
                "Die Kalibrierung verbessert nicht beide Gütemaße. Für den "
                "produktiven Einsatz sollte sie deaktiviert oder neu trainiert werden."
            )

    st.subheader("Kalibrierung")
    calibration = report.calibration.copy()
    chart = px.scatter(
        calibration,
        x="predicted_probability",
        y="observed_frequency",
        size="observations",
        hover_name="bucket_label",
        labels={
            "predicted_probability": "Prognostizierte Wahrscheinlichkeit",
            "observed_frequency": "Tatsächlich eingetreten",
            "observations": "Beobachtungen",
        },
    )
    chart.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfekte Kalibrierung",
        line={"dash": "dash", "color": "#70efb3"},
    )
    chart.update_xaxes(tickformat=".0%", range=[0, 1])
    chart.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(chart, width="stretch")

    outcome_display = report.outcome_metrics.copy()
    st.subheader("Kalibrierung nach Ausgang")
    st.dataframe(
        outcome_display,
        width="stretch",
        hide_index=True,
        column_config={
            "outcome": "Ausgang",
            "mean_predicted_probability": st.column_config.NumberColumn(
                "Mittlere Modellchance", format="percent"
            ),
            "observed_frequency": st.column_config.NumberColumn(
                "Tatsächliche Häufigkeit", format="percent"
            ),
            "calibration_gap": st.column_config.NumberColumn(
                "Abweichung", format="percent"
            ),
        },
    )
    with st.expander("Historische Einzelprognosen"):
        predictions = report.predictions.sort_values("date", ascending=False).copy()
        st.dataframe(
            predictions[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "actual_result",
                    "actual_outcome",
                    "predicted_outcome",
                    "predicted_probability",
                    "home_win",
                    "draw",
                    "away_win",
                    "correct",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "predicted_probability": st.column_config.NumberColumn(
                    "Prognosechance", format="percent"
                ),
                "home_win": st.column_config.NumberColumn("1", format="percent"),
                "draw": st.column_config.NumberColumn("X", format="percent"),
                "away_win": st.column_config.NumberColumn("2", format="percent"),
                "correct": st.column_config.CheckboxColumn("Richtig"),
            },
        )


def data_import(bundle: DataBundle):
    st.header("Datenimport")
    st.write(
        "Je Tabelle kann eine CSV- oder JSON-Datei geladen werden. Nicht geladene "
        "Tabellen bleiben unverändert. API-Schlüssel gehören später in Umgebungsvariablen, "
        "nicht in den Quellcode."
    )
    st.caption(
        "Zeitkritische Tabellen benötigen immer Quelle und Erfassungszeit. "
        "Veraltete Daten werden in der Confidence abgewertet."
    )
    swisslos_template = bundle.matches[
        ["match_id", "date", "home_team", "away_team"]
    ].copy()
    swisslos_template.insert(1, "bookmaker", "Swisslos")
    swisslos_template.insert(2, "collected_at", "")
    swisslos_template["home_odds"] = ""
    swisslos_template["draw_odds"] = ""
    swisslos_template["away_odds"] = ""
    swisslos_template["source"] = (
        "https://www.swisslos.ch/de/sporttip/sportwetten/fussball"
    )
    st.download_button(
        "Swisslos-Quotenvorlage herunterladen",
        swisslos_template.to_csv(index=False).encode("utf-8-sig"),
        "swisslos_odds_template.csv",
        "text/csv",
    )
    st.caption(
        "Trage die aktuellen Dezimalquoten 1/X/2 und den Erfassungszeitpunkt ein "
        "und lade die Datei anschließend als Tabelle `odds` hoch."
    )
    outright_template = bundle.teams[["team_id", "team_name"]].copy()
    outright_template.insert(1, "bookmaker", "Swisslos")
    outright_template.insert(2, "market", "champion")
    outright_template.insert(3, "collected_at", "")
    outright_template["decimal_odds"] = ""
    outright_template["source"] = (
        "https://www.swisslos.ch/de/sporttip/sportwetten/fussball"
    )
    st.download_button(
        "Swisslos-Weltmeisterquoten-Vorlage herunterladen",
        outright_template.drop(columns=["team_name"])
        .to_csv(index=False)
        .encode("utf-8-sig"),
        "swisslos_weltmeisterquoten_template.csv",
        "text/csv",
    )
    st.caption(
        "Für die Langzeitwette auf den Weltmeister alle angebotenen Teams mit "
        "Dezimalquote eintragen und als Tabelle `outright_odds` hochladen."
    )
    uploads = {}
    columns = st.columns(2)
    table_names = [
        "teams",
        "matches",
        "historical_results",
        "ratings",
        "availability",
        "lineups",
        "match_events",
        "odds",
        "outright_odds",
        "tips",
    ]
    for index, name in enumerate(table_names):
        uploads[name] = columns[index % 2].file_uploader(
            name, type=["csv", "json"], key=f"upload_{name}"
        )
    if st.button("Dateien validieren und übernehmen", type="primary"):
        values = {
            "teams": bundle.teams,
            "matches": bundle.matches,
            "historical_results": bundle.historical_results,
            "ratings": bundle.ratings,
            "availability": bundle.availability,
            "lineups": bundle.lineups,
            "match_events": bundle.match_events,
            "odds": bundle.odds,
            "outright_odds": bundle.outright_odds,
            "tips": bundle.tips,
        }
        try:
            changed = []
            for name, uploaded in uploads.items():
                if uploaded is not None:
                    values[name] = load_uploaded_bytes(
                        uploaded.getvalue(), uploaded.name, name
                    )
                    changed.append(name)
            if not changed:
                st.warning("Keine Datei ausgewählt.")
                return
            new_bundle = DataBundle(
                **values,
                source_label="Benutzerimport",
                is_demo=False,
            )
            warnings = validate_references(new_bundle)
            database = DATABASE_PATH
            if "odds" in changed:
                save_match_odds(database, new_bundle.odds)
            if "outright_odds" in changed:
                save_outright_odds(database, new_bundle.outright_odds)
            st.session_state.bundle = new_bundle
            st.success("Übernommen: " + ", ".join(changed))
            for warning in warnings:
                st.warning(warning)
        except DataValidationError as error:
            st.error(str(error))

    if st.button("WM-2026-Startdaten wiederherstellen"):
        st.session_state.bundle = merge_persisted_market_data(
            copy.deepcopy(load_world_cup_bundle())
        )
        st.success("Offizieller WM-2026-Gruppenplan wiederhergestellt.")
    if st.button("Historische Resultate jetzt online aktualisieren"):
        try:
            with st.spinner(
                "Länderspiele, WM-Endstände und Karten werden geladen..."
            ):
                history, completed = update_history(
                    ROOT / "data" / "world_cup_2026" / "historical_results.csv",
                    since="2018-01-01",
                    as_of=date.today().isoformat(),
                    matches_path=ROOT / "data" / "world_cup_2026" / "matches.csv",
                )
                tournament = update_world_cup_files(
                    ROOT / "data" / "world_cup_2026",
                    as_of=date.today().isoformat(),
                )
                load_world_cup_bundle.clear()
                st.session_state.bundle = merge_persisted_market_data(
                    copy.deepcopy(load_world_cup_bundle())
                )
            st.success(
                f"{len(history)} historische Spiele geladen; "
                f"{max(completed, tournament['completed_matches'])} "
                "WM-Endstände und "
                f"{tournament['card_events']} Karten synchronisiert."
            )
            st.rerun()
        except Exception as error:
            st.error(f"Aktualisierung fehlgeschlagen: {error}")
    if st.button("Offizielle FIFA-Rangliste jetzt aktualisieren"):
        try:
            with st.spinner("Die neueste offizielle FIFA-Rangliste wird geladen..."):
                rankings, schedule_id = update_local_fifa_rankings(
                    ROOT / "data" / "world_cup_2026" / "teams.csv",
                    ROOT / "data" / "world_cup_2026" / "ratings.csv",
                )
                load_world_cup_bundle.clear()
                st.session_state.bundle = merge_persisted_market_data(
                    copy.deepcopy(load_world_cup_bundle())
                )
            st.success(
                f"{len(rankings)} offizielle FIFA-Werte vom "
                f"{rankings.iloc[0]['as_of']} geladen ({schedule_id})."
            )
            st.rerun()
        except Exception as error:
            st.error(f"FIFA-Ranglistenaktualisierung fehlgeschlagen: {error}")
    st.subheader("Aktueller Datenbestand")
    st.dataframe(
        pd.DataFrame(
            {
                "Tabelle": table_names,
                "Zeilen": [
                    len(bundle.teams),
                    len(bundle.matches),
                    len(bundle.historical_results),
                    len(bundle.ratings),
                    len(bundle.availability),
                    len(bundle.lineups),
                    len(bundle.match_events),
                    len(bundle.odds),
                    len(bundle.outright_odds),
                    len(bundle.tips),
                ],
            }
        ),
        hide_index=True,
    )
    for warning in validate_references(bundle):
        st.warning(warning)


def simulation_page(
    bundle: DataBundle,
    predictions: dict,
    config: dict,
    betting_config: dict,
    bankroll: float,
):
    st.header("Turniersimulation")
    st.write(
        "Monte Carlo simuliert die 72 Gruppenspiele und danach die K.-o.-Phase "
        "bis zum Finale. Bei Punktegleichheit gelten Tordifferenz, erzielte Tore "
        "und anschließend ein Zufallsentscheid."
    )
    st.caption(
        "Die zwölf Gruppensieger und -zweiten sowie die acht besten Dritten "
        "kommen weiter. Die Dritten werden auf zulässige FIFA-Slots verteilt; "
        "das ist eine transparente Näherung der Kombinationstabelle."
    )
    runs = st.slider(
        "Simulationsläufe",
        min_value=500,
        max_value=30000,
        value=int(config["simulation"]["default_runs"]),
        step=500,
    )
    seed = st.number_input(
        "Zufalls-Seed", value=int(config["simulation"]["random_seed"]), step=1
    )
    if st.button("Simulation starten", type="primary"):
        try:
            group_result = simulate_tournament(
                bundle.matches, predictions, runs=runs, seed=int(seed)
            )
            names = bundle.teams.set_index("team_id")["team_name"].to_dict()
            group_result["team"] = (
                group_result["team_id"].map(names).fillna(group_result["team_id"])
            )
            display = group_result[
                [
                    "group",
                    "team",
                    "group_win_probability",
                    "top_two_probability",
                    "expected_group_points",
                ]
            ]
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    "group_win_probability": st.column_config.ProgressColumn(
                        "Gruppensieg", min_value=0, max_value=1, format="percent"
                    ),
                    "top_two_probability": st.column_config.ProgressColumn(
                        "Top 2", min_value=0, max_value=1, format="percent"
                    ),
                    "expected_group_points": st.column_config.NumberColumn(
                        "Erwartete Punkte", format="%.2f"
                    ),
                },
            )
            group_chart = px.bar(
                display,
                x="team",
                y="top_two_probability",
                color="group",
                barmode="group",
                labels={"team": "Team", "top_two_probability": "Top-2-Chance"},
            )
            group_chart.update_yaxes(tickformat=".0%")
            st.plotly_chart(group_chart, width="stretch")

            with st.spinner("K.-o.-Turnier wird simuliert..."):
                simulation_teams = teams_with_current_elo(
                    bundle.teams,
                    bundle.historical_results,
                )
                world_cup = simulate_world_cup(
                    bundle.matches,
                    predictions,
                    simulation_teams,
                    runs=runs,
                    seed=int(seed),
                )
            probabilities = world_cup.team_probabilities.copy()
            probabilities["team"] = (
                probabilities["team_id"].map(names).fillna(probabilities["team_id"])
            )
            champion_name = names.get(world_cup.champion, world_cup.champion)
            finalist_names = [
                names.get(team, team) for team in world_cup.likely_finalists
            ]
            st.subheader("Rein hypothetischer Turnierausgang")
            col1, col2 = st.columns(2)
            col1.metric(
                "Wahrscheinlichster Weltmeister",
                champion_name,
                f"{world_cup.champion_probability:.1%} Titelchance",
            )
            col2.metric(
                "Häufigstes simuliertes Finale",
                " / ".join(finalist_names),
            )
            knockout_display = probabilities[
                [
                    "team",
                    "round_of_32_probability",
                    "quarterfinal_probability",
                    "semifinal_probability",
                    "final_probability",
                    "champion_probability",
                ]
            ].head(20)
            st.dataframe(
                knockout_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "round_of_32_probability": st.column_config.NumberColumn(
                        "Sechzehntelfinale", format="percent"
                    ),
                    "quarterfinal_probability": st.column_config.NumberColumn(
                        "Viertelfinale", format="percent"
                    ),
                    "semifinal_probability": st.column_config.NumberColumn(
                        "Halbfinale", format="percent"
                    ),
                    "final_probability": st.column_config.NumberColumn(
                        "Finale", format="percent"
                    ),
                    "champion_probability": st.column_config.NumberColumn(
                        "Weltmeister", format="percent"
                    ),
                },
            )
            title_chart = px.bar(
                probabilities.head(16).sort_values("champion_probability"),
                x="champion_probability",
                y="team",
                orientation="h",
                labels={"champion_probability": "Titelchance", "team": "Team"},
            )
            title_chart.update_xaxes(tickformat=".0%")
            st.plotly_chart(title_chart, width="stretch")

            st.subheader("Swisslos-Weltmeistermarkt")
            probabilities["simulation_uncertainty"] = 0.25
            outright = evaluate_outright_market(
                probabilities,
                bundle.outright_odds,
                bankroll=bankroll,
                config=betting_config,
            )
            coverage = (
                float(outright["market_coverage"].max())
                if not outright.empty
                else 0.0
            )
            st.caption(
                f"Quotenabdeckung: {coverage:.0%} der simulierten Teams. "
                "Unter 80 % werden keine Langzeitwetten empfohlen."
            )
            outright_display = outright.copy()
            outright_display["champion_probability"] = outright_display[
                "champion_probability"
            ].map(lambda value: f"{value:.1%}")
            outright_display["decimal_odds"] = outright_display[
                "decimal_odds"
            ].map(lambda value: f"{value:.2f}" if pd.notna(value) else "-")
            for column in ["market_probability", "edge", "expected_return"]:
                outright_display[column] = outright_display[column].map(
                    lambda value: f"{value:+.1%}" if pd.notna(value) else "-"
                )
            outright_display["stake"] = outright_display["stake"].map(
                lambda value: f"CHF {value:.2f}" if pd.notna(value) and value > 0 else "-"
            )
            st.dataframe(
                outright_display[
                    [
                        "team",
                        "champion_probability",
                        "decimal_odds",
                        "market_probability",
                        "edge",
                        "expected_return",
                        "decision",
                        "stake",
                        "reason",
                    ]
                ].head(20),
                width="stretch",
                hide_index=True,
                column_config={
                    "team": "Team",
                    "champion_probability": "Modell-Titelchance",
                    "decimal_odds": "Swisslos-Quote",
                    "market_probability": "Swisslos-Wahrscheinlichkeit",
                    "edge": "Modell-Edge",
                    "expected_return": "Expected Return",
                    "decision": "Entscheid",
                    "stake": f"Einsatz bei CHF {bankroll:.0f}",
                    "reason": "Begründung",
                },
            )
            if bundle.outright_odds.empty:
                st.warning(
                    "Keine Swisslos-Weltmeisterquoten geladen. Die Vorlage "
                    "steht auf der Seite Datenimport bereit."
                )
            st.download_button(
                "Titelchancen und Langzeitquoten als CSV",
                outright.to_csv(index=False).encode("utf-8-sig"),
                "wm2026_weltmeister_value.csv",
                "text/csv",
            )
        except ValueError as error:
            st.error(str(error))


def export_page(summary: pd.DataFrame):
    st.header("Export")
    export_columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "predicted_winner",
        "winner_probability",
        "swisslos_home_odds",
        "swisslos_draw_odds",
        "swisslos_away_odds",
        "model_swisslos_edge",
        "raw_home_win",
        "raw_draw",
        "raw_away_win",
        "bet_decision",
        "bet_outcome",
        "bet_odds",
        "bet_edge",
        "bet_expected_return",
        "bet_stake",
        "bet_reason",
        "hypothetical_score",
        "hypothetical_score_probability",
        "ev_tip",
        "expected_points",
        "confidence",
        "classification",
    ]
    export_frame = summary[export_columns].copy()
    st.dataframe(export_frame, width="stretch", hide_index=True)
    csv = export_frame.to_csv(index=False).encode("utf-8-sig")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "CSV herunterladen", csv, "wm2026_tipps.csv", "text/csv"
    )
    try:
        excel = to_excel_bytes(export_frame)
        col2.download_button(
            "Excel herunterladen",
            excel,
            "wm2026_tipps.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        col2.warning("Excel-Export benötigt openpyxl.")
    markdown = to_markdown(export_frame)
    col3.download_button(
        "Markdown herunterladen",
        markdown.encode("utf-8"),
        "wm2026_tipps.md",
        "text/markdown",
    )
    st.subheader("Copy-Paste-Format")
    st.code(to_srf_text(export_frame), language=None)


def main():
    config = load_config()
    st.set_page_config(
        page_title=config["app"]["title"],
        layout="wide",
    )
    apply_website_style()
    render_website_header()
    bundle = get_bundle()
    scoring = get_scoring(config)
    model_config = get_model_config(config)

    page = st.sidebar.radio(
        "Bereich",
        [
            "Dashboard",
            "Matchday-Center",
            "Spielanalyse",
            "Swisslos-Quoten",
            "Wettanalyse",
            "Paper-Wettjournal",
            "Modell-Backtest",
            "Scoring-Konfiguration",
            "Datenimport",
            "Turniersimulation",
            "Export",
        ],
    )
    strategy = st.sidebar.selectbox(
        "Strategie",
        ["safe", "value", "risk"],
        format_func=lambda value: {
            "safe": "Sicher",
            "value": "Value",
            "risk": "Risiko",
        }[value],
    )
    use_joker = st.sidebar.checkbox(
        "Joker-Multiplikator anwenden",
        help="Aktiviert den konfigurierten Joker für die aktuelle Was-wäre-wenn-Analyse.",
    )
    calibration_default = bool(
        model_config.get("calibration", {}).get("enabled", False)
    )
    use_calibration = st.sidebar.checkbox(
        "Backtest-Kalibrierung verwenden",
        value=calibration_default,
        help=(
            "Korrigiert systematische Über- oder Unterschätzung der "
            "1/X/2-Wahrscheinlichkeiten anhand zeitlich getrennter historischer Spiele."
        ),
    )
    st.sidebar.caption(
        "Sicher priorisiert stabilen EV, Value erlaubt kleine EV-Abstriche für "
        "weniger offensichtliche Tipps, Risiko gewichtet Streuung stärker."
    )
    betting_config = copy.deepcopy(config.get("betting", {}))
    analysis_model_config = copy.deepcopy(model_config)
    analysis_model_config.setdefault("calibration", {})["enabled"] = use_calibration
    with st.sidebar.expander("Wett-Risikoregeln"):
        bankroll = st.number_input(
            "Bankroll in CHF",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(betting_config.get("bankroll", 100.0)),
            step=10.0,
        )
        betting_config["fractional_kelly"] = st.slider(
            "Kelly-Anteil",
            min_value=0.0,
            max_value=0.5,
            value=float(betting_config.get("fractional_kelly", 0.25)),
            step=0.05,
            help="Je kleiner der Wert, desto konservativer die Einsatzberechnung.",
        )
        betting_config["max_stake_fraction"] = (
            st.slider(
                "Maximaler Einsatz pro Spiel",
                min_value=0.0,
                max_value=10.0,
                value=float(betting_config.get("max_stake_fraction", 0.02)) * 100,
                step=0.5,
                format="%.1f%%",
            )
            / 100
        )
        betting_config["min_edge"] = (
            st.slider(
                "Minimaler Modellvorteil",
                min_value=0.0,
                max_value=20.0,
                value=float(betting_config.get("min_edge", 0.04)) * 100,
                step=0.5,
                format="%.1f%%",
            )
            / 100
        )
        betting_config["min_expected_return"] = (
            st.slider(
                "Minimaler Expected Return",
                min_value=0.0,
                max_value=20.0,
                value=float(betting_config.get("min_expected_return", 0.03)) * 100,
                step=0.5,
                format="%.1f%%",
            )
            / 100
        )
        betting_config["max_odds_age_hours"] = st.number_input(
            "Maximales Quotenalter in Stunden",
            min_value=1.0,
            max_value=168.0,
            value=float(betting_config.get("max_odds_age_hours", 24.0)),
            step=1.0,
        )
        betting_config["max_total_exposure_fraction"] = (
            st.slider(
                "Maximale gesamte offene Exposition",
                min_value=1.0,
                max_value=30.0,
                value=float(
                    betting_config.get("max_total_exposure_fraction", 0.10)
                )
                * 100,
                step=1.0,
                format="%.0f%%",
            )
            / 100
        )
        betting_config["max_daily_exposure_fraction"] = (
            st.slider(
                "Maximale Exposition pro Spieltag",
                min_value=1.0,
                max_value=20.0,
                value=float(
                    betting_config.get("max_daily_exposure_fraction", 0.05)
                )
                * 100,
                step=1.0,
                format="%.0f%%",
            )
            / 100
        )
        betting_config["minimum_portfolio_stake"] = st.number_input(
            "Mindesteinsatz im Portfolio",
            min_value=0.10,
            max_value=1000.0,
            value=float(betting_config.get("minimum_portfolio_stake", 0.50)),
            step=0.10,
        )
        betting_config["max_bets_per_day"] = st.number_input(
            "Maximale Wetten pro Spieltag",
            min_value=1,
            max_value=20,
            value=int(betting_config.get("max_bets_per_day", 3)),
            step=1,
        )

    paper_bets = load_paper_bets(DATABASE_PATH)
    bankroll_performance = calculate_betting_performance(
        paper_bets,
        starting_bankroll=bankroll,
        odds=bundle.odds,
        matches=bundle.matches,
    )
    portfolio_bankroll = bankroll_performance.current_bankroll

    try:
        features, predictions, recommendations, summary = build_analysis(
            bundle,
            config,
            analysis_model_config,
            scoring,
            strategy,
            use_joker,
            betting_config,
            portfolio_bankroll,
            paper_bets,
        )
    except (DataValidationError, KeyError, ValueError) as error:
        st.error(f"Analyse nicht möglich: {error}")
        st.stop()

    if page == "Dashboard":
        dashboard(
            summary,
            strategy,
            bundle,
            scoring,
            use_joker,
            portfolio_bankroll,
            use_calibration,
        )
    elif page == "Matchday-Center":
        matchday_page(bundle, summary)
    elif page == "Spielanalyse":
        match_analysis(
            summary,
            features,
            predictions,
            recommendations,
            bundle,
            scoring,
            config,
            strategy,
            betting_config,
            portfolio_bankroll,
        )
    elif page == "Swisslos-Quoten":
        swisslos_odds_page(bundle)
    elif page == "Wettanalyse":
        betting_page(
            summary,
            features,
            predictions,
            betting_config,
            portfolio_bankroll,
        )
    elif page == "Paper-Wettjournal":
        paper_betting_page(summary, bundle, bankroll)
    elif page == "Modell-Backtest":
        backtest_page(bundle, analysis_model_config)
    elif page == "Scoring-Konfiguration":
        scoring_configuration(config)
    elif page == "Datenimport":
        data_import(bundle)
    elif page == "Turniersimulation":
        simulation_page(
            bundle,
            predictions,
            config,
            betting_config,
            portfolio_bankroll,
        )
    else:
        export_page(summary)
    render_footer()


if __name__ == "__main__":
    main()
