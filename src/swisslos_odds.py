from __future__ import annotations

import json
import time
import unicodedata
import uuid
import zlib
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from websockets.sync.client import connect

from src.data_loader import validate_dataframe


SWISSLOS_SOURCE = (
    "https://www.swisslos.ch/de/sporttip/sportwetten/fussball/wm-2026"
)
SPORTSBOOK_WEBSOCKET = "wss://ws.ch.admiral.at/"
CONFIGURATION_NODE = (
    "asw:node:admiral:device:1886b7bc-2f48-486a-ac40-4fe5828886a8"
)
MATCH_ROUTE = "/fussball/wm-2026"
OUTRIGHT_ROUTE = "/fussball/wm-2026/wm-2026-turnierwetten"
MAIN_MARKET_TYPE = "asw:markettype:1"
WINNER_MARKET_TYPE = "asw:markettype:534:pre:markettext:176991"
TEAM_ID_ALIASES = {"CUR": "CUW", "IRI": "IRN"}


class SwisslosOddsError(RuntimeError):
    """Raised when the official Sporttip widget cannot be read safely."""


@dataclass(frozen=True)
class SwisslosOddsSnapshot:
    match_odds: pd.DataFrame
    outright_odds: pd.DataFrame
    collected_at: pd.Timestamp


EntityStore = dict[str, dict[str, dict]]


def _encode_message(payload: dict) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return compressor.compress(raw) + compressor.flush()


def _decode_message(frame: bytes | str) -> dict:
    if isinstance(frame, str):
        return json.loads(frame)
    return json.loads(zlib.decompress(frame, wbits=-15))


def _payload_messages(message: dict) -> list[dict]:
    payload = message.get("payload", [])
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        payload = [payload]
    return [item for item in payload if isinstance(item, dict)]


def _apply_snapshot(store: EntityStore, message: dict) -> None:
    if message.get("type") != "SportsbookSnapshotUpdated":
        return
    updates = (
        message.get("body", {})
        .get("snapshotUpdate", {})
        .get("snapshotUpdateItems", [])
    )
    for update in updates:
        entity_type = str(update.get("type", ""))
        entity = update.get("entity") or {}
        urn = entity.get("urn")
        if not entity_type or not urn:
            continue
        if int(update.get("kind", 0)) == 0:
            store.setdefault(entity_type, {})[str(urn)] = entity
        else:
            store.get(entity_type, {}).pop(str(urn), None)


def _timestamp_from_messages(
    current: pd.Timestamp | None,
    messages: list[dict],
) -> pd.Timestamp | None:
    timestamps = [
        pd.Timestamp(message["timestamp"])
        for message in messages
        if message.get("timestamp")
    ]
    if not timestamps:
        return current
    latest = max(timestamps)
    if latest.tzinfo is not None:
        latest = latest.tz_convert("UTC").tz_localize(None)
    return max(current, latest) if current is not None else latest


def _send(websocket, payload: list[dict] | dict) -> None:
    commands = payload if isinstance(payload, list) else [payload]
    websocket.send(
        _encode_message(
            {
                "payload": commands,
                "properties": {"type": "binary", "service": "sportsbook"},
            }
        )
    )


def _receive_until(
    websocket,
    store: EntityStore,
    predicate: Callable[[list[dict], EntityStore], bool],
    timeout: float,
) -> tuple[list[dict], pd.Timestamp | None]:
    deadline = time.monotonic() + timeout
    latest_timestamp: pd.Timestamp | None = None
    while time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        try:
            outer = _decode_message(websocket.recv(timeout=remaining))
        except TimeoutError:
            break
        messages = _payload_messages(outer)
        latest_timestamp = _timestamp_from_messages(latest_timestamp, messages)
        for message in messages:
            _apply_snapshot(store, message)
            if message.get("type") in {"CommandFailed", "Error"}:
                raise SwisslosOddsError(
                    f"Swisslos meldet einen Fehler: {message.get('body', message)}"
                )
        if predicate(messages, store):
            return messages, latest_timestamp
    raise SwisslosOddsError("Zeitüberschreitung beim Laden der Swisslos-Quoten.")


def _open_sportsbook(timeout: float):
    websocket = connect(
        SPORTSBOOK_WEBSOCKET,
        origin="https://www.swisslos.ch",
        user_agent_header="Mozilla/5.0",
        max_size=None,
        open_timeout=timeout,
        close_timeout=2,
    )
    websocket.send(
        _encode_message(
            {
                "connect": {"headers": {}},
                "properties": {"type": "binary", "service": "sportsbook"},
            }
        )
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = _decode_message(
                websocket.recv(timeout=max(0.05, deadline - time.monotonic()))
            )
        except TimeoutError as exc:
            websocket.close()
            raise SwisslosOddsError(
                "Keine Verbindung zum Swisslos-Sporttip-Widget."
            ) from exc
        if "connected" in response:
            break
    else:
        websocket.close()
        raise SwisslosOddsError("Swisslos-Sporttip-Verbindung nicht bestätigt.")

    _send(
        websocket,
        {
            "type": "ConfigureEnvironment",
            "name": "ConfigureEnvironment",
            "body": {
                "timeZone": "Europe/Zurich",
                "languageCode": "de",
                "configurationNodeUrn": CONFIGURATION_NODE,
            },
        },
    )
    _receive_until(
        websocket,
        {},
        lambda messages, _: any(
            message.get("type") == "EnvironmentConfigured"
            for message in messages
        ),
        timeout,
    )
    return websocket


def _find_view_node(node: object, node_type: str) -> dict | None:
    if isinstance(node, dict):
        if node.get("type") == node_type:
            return node
        for value in node.values():
            found = _find_view_node(value, node_type)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_view_node(value, node_type)
            if found is not None:
                return found
    return None


def _event_urns(node: object) -> list[str]:
    urns: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "EventViewNode" and node.get("eventUrn"):
            urns.append(str(node["eventUrn"]))
        for value in node.values():
            urns.extend(_event_urns(value))
    elif isinstance(node, list):
        for value in node:
            urns.extend(_event_urns(value))
    return list(dict.fromkeys(urns))


def _main_market(event: dict, store: EntityStore) -> dict | None:
    markets = store.get("Market", {})
    for market_urn in event.get("markets", []):
        market = markets.get(str(market_urn))
        if (
            market
            and market.get("type") == MAIN_MARKET_TYPE
            and int(market.get("state", 0)) == 1
        ):
            return market
    return None


def _main_events_complete(
    event_urns: list[str],
    store: EntityStore,
) -> bool:
    events = store.get("Event", {})
    selections = store.get("Selection", {})
    competitors = store.get("Competitor", {})
    for event_urn in event_urns:
        event = events.get(event_urn)
        if not event:
            return False
        if int(event.get("type", -1)) != 0:
            continue
        if any(
            str(item.get("competitor")) not in competitors
            for item in event.get("eventCompetitors", [])
        ):
            return False
        market = _main_market(event, store)
        if not market or len(market.get("selections", [])) != 3:
            return False
        if any(str(urn) not in selections for urn in market["selections"]):
            return False
    return True


def _normalize(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in ascii_value.lower() if character.isalnum())


def _competitor_candidates(competitor: dict) -> set[str]:
    values = {
        competitor.get("name"),
        competitor.get("abbreviation"),
        competitor.get("countryCode"),
    }
    values.update((competitor.get("translations") or {}).values())
    values.update((competitor.get("properties") or {}).get("AliasNames", []))
    return {_normalize(value) for value in values if value}


def _team_aliases(teams: pd.DataFrame, store: EntityStore) -> dict[str, str]:
    aliases: dict[str, str] = {}
    team_ids = set(teams["team_id"].astype(str))
    for row in teams.to_dict("records"):
        team_id = str(row["team_id"])
        for value in (team_id, row.get("team_name"), row.get("country")):
            if value:
                aliases[_normalize(value)] = team_id

    for competitor in store.get("Competitor", {}).values():
        abbreviation = str(competitor.get("abbreviation", ""))
        candidate_id = TEAM_ID_ALIASES.get(abbreviation, abbreviation)
        if candidate_id not in team_ids:
            candidate_id = next(
                (
                    aliases[candidate]
                    for candidate in _competitor_candidates(competitor)
                    if candidate in aliases
                ),
                "",
            )
        if candidate_id in team_ids:
            for candidate in _competitor_candidates(competitor):
                aliases[candidate] = candidate_id
    return aliases


def _competitor_team_id(
    competitor_urn: object,
    store: EntityStore,
    aliases: dict[str, str],
) -> str | None:
    competitor = store.get("Competitor", {}).get(str(competitor_urn))
    if not competitor:
        return None
    for candidate in _competitor_candidates(competitor):
        if candidate in aliases:
            return aliases[candidate]
    return None


def _extract_match_odds(
    store: EntityStore,
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    event_urns: list[str],
    collected_at: pd.Timestamp,
) -> pd.DataFrame:
    aliases = _team_aliases(teams, store)
    match_lookup = {
        (str(row["home_team"]), str(row["away_team"])): str(row["match_id"])
        for row in matches.to_dict("records")
    }
    selections = store.get("Selection", {})
    rows: list[dict] = []
    unresolved: list[str] = []

    for event_urn in event_urns:
        event = store.get("Event", {}).get(event_urn)
        if not event or int(event.get("type", -1)) != 0:
            continue
        competitors = {
            str(item.get("qualifier")): _competitor_team_id(
                item.get("competitor"), store, aliases
            )
            for item in event.get("eventCompetitors", [])
        }
        home_id = competitors.get("home")
        away_id = competitors.get("away")
        match_id = match_lookup.get((str(home_id), str(away_id)))
        market = _main_market(event, store)
        if not home_id or not away_id or not match_id or not market:
            unresolved.append(str(event.get("name", event_urn)))
            continue
        prices: dict[str, float] = {}
        for selection_urn in market.get("selections", []):
            selection = selections.get(str(selection_urn), {})
            if int(selection.get("state", 0)) != 1:
                continue
            prices[str(selection.get("type"))] = float(selection["odds"])
        required = {
            "asw:selectiontype:1",
            "asw:selectiontype:2",
            "asw:selectiontype:3",
        }
        if not required.issubset(prices):
            unresolved.append(str(event.get("name", event_urn)))
            continue
        rows.append(
            {
                "match_id": match_id,
                "bookmaker": "Swisslos",
                "collected_at": collected_at,
                "home_odds": prices["asw:selectiontype:1"],
                "draw_odds": prices["asw:selectiontype:2"],
                "away_odds": prices["asw:selectiontype:3"],
                "source": SWISSLOS_SOURCE,
            }
        )
    if unresolved:
        raise SwisslosOddsError(
            "Swisslos-Spiele konnten nicht eindeutig zugeordnet werden: "
            + ", ".join(unresolved[:5])
        )
    if not rows:
        raise SwisslosOddsError("Swisslos liefert derzeit keine offenen 1/X/2-Quoten.")
    return validate_dataframe(pd.DataFrame(rows), "odds").sort_values("match_id")


def _winner_market(store: EntityStore) -> dict | None:
    return next(
        (
            market
            for market in store.get("Market", {}).values()
            if market.get("type") == WINNER_MARKET_TYPE
            and int(market.get("state", 0)) == 1
        ),
        None,
    )


def _winner_market_complete(store: EntityStore) -> bool:
    market = _winner_market(store)
    if not market or not market.get("selections"):
        return False
    selections = store.get("Selection", {})
    selection_types = store.get("SelectionType", {})
    return all(
        str(urn) in selections
        and str(selections[str(urn)].get("type")) in selection_types
        for urn in market["selections"]
    )


def _extract_outright_odds(
    store: EntityStore,
    teams: pd.DataFrame,
    collected_at: pd.Timestamp,
) -> pd.DataFrame:
    market = _winner_market(store)
    if not market:
        raise SwisslosOddsError("Swisslos-Weltmeistermarkt wurde nicht gefunden.")
    aliases = _team_aliases(teams, store)
    selection_entities = store.get("Selection", {})
    selection_types = store.get("SelectionType", {})
    rows: list[dict] = []
    unresolved: list[str] = []
    for selection_urn in market.get("selections", []):
        selection = selection_entities.get(str(selection_urn), {})
        if int(selection.get("state", 0)) != 1:
            continue
        selection_type = selection_types.get(str(selection.get("type")), {})
        candidates = {
            _normalize(selection_type.get("name", "")),
            *{
                _normalize(value)
                for value in (selection_type.get("translations") or {}).values()
            },
        }
        team_id = next(
            (aliases[candidate] for candidate in candidates if candidate in aliases),
            None,
        )
        if not team_id:
            unresolved.append(str(selection_type.get("name", selection_urn)))
            continue
        rows.append(
            {
                "team_id": team_id,
                "bookmaker": "Swisslos",
                "market": "champion",
                "collected_at": collected_at,
                "decimal_odds": float(selection["odds"]),
                "source": SWISSLOS_SOURCE,
            }
        )
    if unresolved:
        raise SwisslosOddsError(
            "Swisslos-Weltmeisterquoten konnten nicht zugeordnet werden: "
            + ", ".join(unresolved[:5])
        )
    if not rows:
        raise SwisslosOddsError(
            "Swisslos liefert derzeit keine aktiven Weltmeisterquoten."
        )
    return validate_dataframe(
        pd.DataFrame(rows), "outright_odds"
    ).sort_values(["decimal_odds", "team_id"])


def _fetch_match_entities(
    timeout: float,
) -> tuple[EntityStore, list[str], pd.Timestamp]:
    store: EntityStore = {}
    latest: pd.Timestamp | None = None
    try:
        with _open_sportsbook(timeout) as websocket:
            request_id = str(uuid.uuid4())
            _send(
                websocket,
                {
                    "type": "RetrieveSportsbookPage",
                    "name": "RetrieveSportsbookPage/RetrieveSportsbookPage",
                    "body": {"urlPart": MATCH_ROUTE, "pageScope": 1},
                    "requestId": request_id,
                },
            )
            response_holder: list[dict] = []

            def page_received(messages: list[dict], _: EntityStore) -> bool:
                response_holder.extend(
                    message
                    for message in messages
                    if message.get("type") == "SportsbookPageRetrieved"
                    and message.get("requestId") == request_id
                )
                return bool(response_holder)

            _, timestamp = _receive_until(
                websocket, store, page_received, timeout
            )
            latest = _timestamp_from_messages(latest, response_holder)
            latest = max(latest, timestamp) if latest is not None and timestamp else (
                latest or timestamp
            )
            page = response_holder[-1]["body"]
            grid = _find_view_node(page.get("viewNode"), "SportsGridViewNode")
            if not grid or not grid.get("id"):
                raise SwisslosOddsError(
                    "Swisslos-Spielraster wurde nicht gefunden."
                )
            event_urns = _event_urns(grid)
            if not event_urns:
                raise SwisslosOddsError(
                    "Swisslos liefert derzeit keine offenen WM-Spiele."
                )
            _send(
                websocket,
                {
                    "type": "UpdateSportsbookPage",
                    "name": "wm2026-tipper/UpdateSportsbookPage",
                    "body": {
                        "sportsGridViewNodeId": grid["id"],
                        "pageSubscriptionName": (
                            "RetrieveSportsbookPage/RetrieveSportsbookPage"
                        ),
                        "commandProperties": [
                            {"type": 0},
                            {"type": 1, "value": '["asw:marketconfig:1"]'},
                        ],
                    },
                    "requestId": request_id,
                },
            )
            _, timestamp = _receive_until(
                websocket,
                store,
                lambda _, entities: _main_events_complete(
                    event_urns, entities
                ),
                timeout,
            )
            latest = max(latest, timestamp) if latest is not None and timestamp else (
                latest or timestamp
            )
    except SwisslosOddsError:
        raise
    except Exception as exc:
        raise SwisslosOddsError(
            f"Swisslos-Spielquoten konnten nicht geladen werden: {exc}"
        ) from exc
    return (
        store,
        event_urns,
        latest or pd.Timestamp.now(tz="UTC").tz_localize(None),
    )


def _fetch_outright_entities(
    timeout: float,
) -> tuple[EntityStore, pd.Timestamp]:
    store: EntityStore = {}
    latest: pd.Timestamp | None = None
    try:
        with _open_sportsbook(timeout) as websocket:
            request_id = str(uuid.uuid4())
            _send(
                websocket,
                {
                    "type": "RetrieveSportsbookPage",
                    "name": "RetrieveSportsbookPage/RetrieveSportsbookPage",
                    "body": {"urlPart": OUTRIGHT_ROUTE, "pageScope": 1},
                    "requestId": request_id,
                },
            )
            response_holder: list[dict] = []

            def page_received(messages: list[dict], _: EntityStore) -> bool:
                response_holder.extend(
                    message
                    for message in messages
                    if message.get("type") == "SportsbookPageRetrieved"
                    and message.get("requestId") == request_id
                )
                return bool(response_holder)

            _, timestamp = _receive_until(
                websocket, store, page_received, timeout
            )
            latest = timestamp
            path = response_holder[-1]["body"].get("categoryPath", {})
            event_urn = path.get("event")
            categories = path.get("categories", [])
            category_urn = categories[-1] if categories else None
            if not event_urn or not category_urn:
                raise SwisslosOddsError(
                    "Swisslos-Weltmeistermarkt wurde nicht geöffnet."
                )
            _send(
                websocket,
                [
                    {
                        "type": "RetrieveEvents",
                        "name": "wm2026-tipper/RetrieveEvents",
                        "body": {"events": [event_urn]},
                        "requestId": str(uuid.uuid4()),
                    },
                    {
                        "type": "RetrieveDetailView",
                        "name": "wm2026-tipper/RetrieveDetailView",
                        "body": {
                            "event": event_urn,
                            "category": category_urn,
                        },
                        "requestId": str(uuid.uuid4()),
                    },
                ],
            )
            _, timestamp = _receive_until(
                websocket,
                store,
                lambda _, entities: _winner_market_complete(entities),
                timeout,
            )
            latest = max(latest, timestamp) if latest is not None and timestamp else (
                latest or timestamp
            )
    except SwisslosOddsError:
        raise
    except Exception as exc:
        raise SwisslosOddsError(
            f"Swisslos-Weltmeisterquoten konnten nicht geladen werden: {exc}"
        ) from exc
    return store, latest or pd.Timestamp.now(tz="UTC").tz_localize(None)


def fetch_swisslos_odds(
    teams: pd.DataFrame,
    matches: pd.DataFrame,
    timeout: float = 15.0,
) -> SwisslosOddsSnapshot:
    """Read current public Swisslos 1/X/2 and champion prices, without betting."""
    match_store, event_urns, match_timestamp = _fetch_match_entities(timeout)
    outright_store, outright_timestamp = _fetch_outright_entities(timeout)
    match_timestamp = pd.Timestamp(match_timestamp).floor("s")
    outright_timestamp = pd.Timestamp(outright_timestamp).floor("s")
    collected_at = max(match_timestamp, outright_timestamp)
    match_odds = _extract_match_odds(
        match_store, teams, matches, event_urns, match_timestamp
    )
    outright_odds = _extract_outright_odds(
        outright_store, teams, outright_timestamp
    )
    return SwisslosOddsSnapshot(
        match_odds=match_odds,
        outright_odds=outright_odds,
        collected_at=collected_at,
    )
