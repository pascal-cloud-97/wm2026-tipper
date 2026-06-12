from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendation_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_label TEXT NOT NULL,
    strategy TEXT NOT NULL,
    scoring_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
    run_id INTEGER NOT NULL,
    match_id TEXT NOT NULL,
    tip_home INTEGER NOT NULL,
    tip_away INTEGER NOT NULL,
    expected_points REAL NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (run_id, match_id),
    FOREIGN KEY (run_id) REFERENCES recommendation_runs(run_id)
);
CREATE TABLE IF NOT EXISTS paper_bets (
    paper_bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    match_id TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    outcome TEXT NOT NULL,
    odds REAL NOT NULL,
    odds_collected_at TEXT,
    model_probability REAL NOT NULL,
    implied_probability REAL NOT NULL,
    edge REAL NOT NULL,
    expected_return REAL NOT NULL,
    stake REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    payout REAL,
    profit REAL,
    settled_at TEXT,
    UNIQUE(match_id, outcome)
);
CREATE TABLE IF NOT EXISTS match_odds_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    home_odds REAL NOT NULL,
    draw_odds REAL NOT NULL,
    away_odds REAL NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, bookmaker, collected_at)
);
CREATE TABLE IF NOT EXISTS outright_odds_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    decimal_odds REAL NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, bookmaker, market, collected_at)
);
CREATE TABLE IF NOT EXISTS match_result_snapshots (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, recorded_at)
);
"""


@contextmanager
def _database_connection(path: str | Path):
    connection = sqlite3.connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(path: str | Path) -> None:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with _database_connection(database) as connection:
        connection.executescript(SCHEMA)


def save_recommendations(
    path: str | Path,
    frame: pd.DataFrame,
    source_label: str,
    strategy: str,
    scoring: dict,
) -> int:
    initialize_database(path)
    with _database_connection(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO recommendation_runs (source_label, strategy, scoring_json)
            VALUES (?, ?, ?)
            """,
            (source_label, strategy, json.dumps(scoring, sort_keys=True)),
        )
        run_id = int(cursor.lastrowid)
        rows = [
            (
                run_id,
                str(row["match_id"]),
                int(str(row["tip"]).split(":")[0]),
                int(str(row["tip"]).split(":")[1]),
                float(row["expected_points"]),
                float(row["confidence"]),
            )
            for row in frame.to_dict("records")
        ]
        connection.executemany(
            """
            INSERT INTO recommendations
            (run_id, match_id, tip_home, tip_away, expected_points, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return run_id


def save_paper_bets(path: str | Path, frame: pd.DataFrame) -> int:
    initialize_database(path)
    candidates = frame[
        (frame["bet_decision"] == "WETTEN")
        & (pd.to_numeric(frame["bet_stake"], errors="coerce") > 0)
    ]
    if candidates.empty:
        return 0
    rows = []
    for row in candidates.to_dict("records"):
        collected_at = row.get("swisslos_collected_at")
        rows.append(
            (
                str(row["match_id"]),
                str(row["home_team"]),
                str(row["away_team"]),
                str(row["bet_outcome"]),
                float(row["bet_odds"]),
                (
                    pd.Timestamp(collected_at).isoformat()
                    if pd.notna(collected_at)
                    else None
                ),
                float(row["bet_model_probability"]),
                float(row["bet_implied_probability"]),
                float(row["bet_edge"]),
                float(row["bet_expected_return"]),
                float(row["bet_stake"]),
            )
        )
    with _database_connection(path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO paper_bets (
                match_id, home_team, away_team, outcome, odds,
                odds_collected_at, model_probability, implied_probability,
                edge, expected_return, stake
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return int(connection.total_changes - before)


def settle_paper_bets(path: str | Path, matches: pd.DataFrame) -> int:
    initialize_database(path)
    completed = matches[
        matches["actual_home_goals"].notna()
        & matches["actual_away_goals"].notna()
    ].copy()
    if completed.empty:
        return 0
    results = completed.set_index(completed["match_id"].astype(str)).to_dict("index")
    settled = 0
    with _database_connection(path) as connection:
        pending = connection.execute(
            """
            SELECT paper_bet_id, match_id, outcome, odds, stake
            FROM paper_bets
            WHERE status = 'pending'
            """
        ).fetchall()
        for paper_bet_id, match_id, outcome, odds, stake in pending:
            result = results.get(str(match_id))
            if result is None:
                continue
            home_goals = int(result["actual_home_goals"])
            away_goals = int(result["actual_away_goals"])
            actual_outcome = (
                "1"
                if home_goals > away_goals
                else "2"
                if home_goals < away_goals
                else "X"
            )
            won = str(outcome) == actual_outcome
            payout = float(stake) * float(odds) if won else 0.0
            profit = payout - float(stake)
            connection.execute(
                """
                UPDATE paper_bets
                SET status = ?, actual_home_goals = ?, actual_away_goals = ?,
                    payout = ?, profit = ?, settled_at = CURRENT_TIMESTAMP
                WHERE paper_bet_id = ?
                """,
                (
                    "won" if won else "lost",
                    home_goals,
                    away_goals,
                    payout,
                    profit,
                    int(paper_bet_id),
                ),
            )
            settled += 1
    return settled


def load_paper_bets(path: str | Path) -> pd.DataFrame:
    initialize_database(path)
    with _database_connection(path) as connection:
        return pd.read_sql_query(
            "SELECT * FROM paper_bets ORDER BY created_at DESC, paper_bet_id DESC",
            connection,
            parse_dates=["created_at", "odds_collected_at", "settled_at"],
        )


def save_match_odds(path: str | Path, frame: pd.DataFrame) -> int:
    initialize_database(path)
    if frame.empty:
        return 0
    rows = [
        (
            str(row["match_id"]),
            str(row["bookmaker"]),
            pd.Timestamp(row["collected_at"]).isoformat(),
            float(row["home_odds"]),
            float(row["draw_odds"]),
            float(row["away_odds"]),
            str(row["source"]),
        )
        for row in frame.to_dict("records")
    ]
    with _database_connection(path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO match_odds_snapshots (
                match_id, bookmaker, collected_at, home_odds,
                draw_odds, away_odds, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return int(connection.total_changes - before)


def load_match_odds(path: str | Path) -> pd.DataFrame:
    initialize_database(path)
    with _database_connection(path) as connection:
        return pd.read_sql_query(
            """
            SELECT match_id, bookmaker, collected_at, home_odds,
                   draw_odds, away_odds, source
            FROM match_odds_snapshots
            ORDER BY collected_at
            """,
            connection,
            parse_dates=["collected_at"],
        )


def save_outright_odds(path: str | Path, frame: pd.DataFrame) -> int:
    initialize_database(path)
    if frame.empty:
        return 0
    rows = [
        (
            str(row["team_id"]),
            str(row["bookmaker"]),
            str(row["market"]),
            pd.Timestamp(row["collected_at"]).isoformat(),
            float(row["decimal_odds"]),
            str(row["source"]),
        )
        for row in frame.to_dict("records")
    ]
    with _database_connection(path) as connection:
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO outright_odds_snapshots (
                team_id, bookmaker, market, collected_at,
                decimal_odds, source
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return int(connection.total_changes - before)


def load_outright_odds(path: str | Path) -> pd.DataFrame:
    initialize_database(path)
    with _database_connection(path) as connection:
        return pd.read_sql_query(
            """
            SELECT team_id, bookmaker, market, collected_at,
                   decimal_odds, source
            FROM outright_odds_snapshots
            ORDER BY collected_at
            """,
            connection,
            parse_dates=["collected_at"],
        )


def save_match_result(
    path: str | Path,
    match_id: str,
    home_goals: int,
    away_goals: int,
    source: str,
    recorded_at: pd.Timestamp | None = None,
) -> int:
    initialize_database(path)
    if int(home_goals) < 0 or int(away_goals) < 0:
        raise ValueError("Tore dürfen nicht negativ sein.")
    timestamp = (
        pd.Timestamp(recorded_at)
        if recorded_at is not None
        else pd.Timestamp.now(tz="UTC").tz_localize(None)
    )
    with _database_connection(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO match_result_snapshots (
                match_id, recorded_at, home_goals, away_goals, source
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(match_id),
                timestamp.isoformat(),
                int(home_goals),
                int(away_goals),
                str(source),
            ),
        )
        return int(cursor.lastrowid)


def load_match_results(
    path: str | Path,
    latest_only: bool = True,
) -> pd.DataFrame:
    initialize_database(path)
    with _database_connection(path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT result_id, match_id, recorded_at, home_goals,
                   away_goals, source, created_at
            FROM match_result_snapshots
            ORDER BY recorded_at, result_id
            """,
            connection,
            parse_dates=["recorded_at", "created_at"],
        )
    if latest_only and not frame.empty:
        frame = (
            frame.sort_values(["recorded_at", "result_id"])
            .groupby("match_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
    return frame
