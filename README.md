# WM 2026 Tipper - USA, Kanada und Mexiko

Lokale Streamlit-App zur datenbasierten Tippoptimierung. Das Ziel ist nicht
einfach das wahrscheinlichste Resultat, sondern der Tipp mit dem höchsten
erwarteten Punktwert unter einem frei konfigurierbaren Scoring-Regelwerk.

> Stand 10. Juni 2026: Teilnehmer, Gruppen, Paarungen, Austragungsorte und
> Anspielzeiten wurden gegen den offiziellen FIFA-Spielplan vom
> 10. April 2026 geprüft. Resultate, Stärkeindizes,
> Wahrscheinlichkeiten und Turnierausgang sind rein hypothetische
> Modellrechnungen. Die Scoring-Voreinstellung ist keine bestätigte
> SRF-Regelquelle.

## Funktionen

- CSV- und JSON-Import mit Schema-, Typ- und Referenzprüfung
- Form der letzten 5 und 10 Spiele ohne Future Leakage
- Rating-, Angriffs-, Abwehr-, Reise-, Kontinental- und H2H-Faktoren
- Rolling Elo kombiniert mit offiziellen FIFA-Ranglistenpunkten
- zeitgestempelte Verletzungs-/Verfügbarkeitsmeldungen
- erwartete oder bestätigte Aufstellungen mit Spielerwerten
- de-viggte 1/X/2-Konsenswahrscheinlichkeiten aus Dezimalquoten
- transparentes Poisson-Ensemble für Resultate von `0:0` bis `6:6`
- vollständige Expected-Points-Auswertung aller möglichen Tipps
- klare Trennung zwischen 1/X/2-Prognose, hypothetischem Endstand und
  punkte-optimiertem SRF-Tipp
- Strategien Sicher, Value und Risiko
- Erklärungen mit EV-Abstand, Alternativen und Modelltreibern
- Monte-Carlo-Simulation aller 72 Gruppenspiele und der K.-o.-Phase
- Export als CSV, Excel, Markdown und Copy-Paste-Text
- lokale Speicherung von Analyse-Läufen in SQLite

## Installation

Voraussetzung ist Python 3.11 oder 3.12.

```powershell
cd "C:\Users\pasca\OneDrive\Dokumente\WM - Tippspiel\wm2026-tipper"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Unter macOS oder Linux wird die Umgebung mit
`source .venv/bin/activate` aktiviert.

## Start

```powershell
.\start_app.cmd
```

Anschließend öffnet Streamlit normalerweise `http://localhost:8501`.

Vom übergeordneten Ordner `WM - Tippspiel`:

```powershell
.\start_wm_tipper.cmd
```

Die Starter verwenden bewusst `.venv312`, weil die globale Python-3.14-
Installation und Pip-Launcher in OneDrive auf diesem Rechner blockiert werden.

## Als öffentliche Website bereitstellen

Für eine dauerhafte kostenlose `streamlit.app`-Adresse:

1. Projekt in ein öffentliches GitHub-Repository übertragen.
2. Auf [share.streamlit.io](https://share.streamlit.io/) mit GitHub anmelden.
3. Repository, Branch `main` und Startdatei `app/main.py` auswählen.
4. App veröffentlichen und die erzeugte HTTPS-Adresse teilen.

`runtime.txt`, `requirements.txt` und `.streamlit/config.toml` enthalten die
benötigte Laufzeitkonfiguration. Die SQLite-Datei ist absichtlich nicht Teil
des Repositorys. Die im Datensatz archivierten Quoten werden aus CSV geladen;
Änderungen während einer kostenlosen Cloud-Sitzung können nach einem Neustart
verloren gehen.

Für eine sofortige, temporäre Freigabe kann die lokal laufende App über einen
HTTPS-Tunnel veröffentlicht werden. Diese Adresse funktioniert nur, solange
der lokale Rechner und beide Hintergrunddienste laufen.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Alternativ:

```powershell
pytest
```

## Datenformat

Jede Tabelle kann als CSV oder als JSON-Array importiert werden. JSON darf
alternativ die Datensätze unter `data` oder `records` enthalten. Die IDs in
`matches.home_team` und `matches.away_team` referenzieren `teams.team_id`.

### `teams`

Pflicht: `team_id`, `team_name`

Optional: `country`, `continent`, `rating`, `fifa_rank`, `fifa_points`,
`is_host`, `latitude`, `longitude`

### `matches`

Pflicht: `match_id`, `date`, `home_team`, `away_team`, `stage`

Optional: `official_match_number`, `group`, `venue`, `venue_country`,
`venue_continent`,
`home_travel_km`, `away_travel_km`, `status`

### `historical_results`

Pflicht: `date`, `home_team`, `away_team`, `home_goals`, `away_goals`

Optional: `neutral`, `competition`

### `ratings`

Pflicht: `team_id`, `as_of`, `rating`

Optional: `fifa_rank`, `fifa_points`, `source`. Pro Team wird der jüngste
Eintrag verwendet.

### `tips`

Pflicht: `match_id`, `tip_home`, `tip_away`

Optional: `strategy`, `created_at`

Fehlende optionale Daten werden nicht erfunden. Der mitgelieferte
WM-2026-Datensatz enthält 3.725 reale Männer-A-Länderspiele seit 2018 mit
mindestens einem WM-Team, begrenzt auf den Stichtag 10. Juni 2026. Quelle ist
`martj42/international_results` unter CC0 1.0.

Verletzungen, erwartete Aufstellungen und Quoten ändern sich kurzfristig.
Ihre Tabellen sind vorbereitet, aber initial leer. Ohne datierte Quelle
behandelt das Modell diese Signale neutral und erhöht die Unsicherheit.

Die Datei `official_schedule_manifest.csv` enthält den strukturierten
Vergleichsabzug der 72 Gruppenspiele. `match_id` bleibt eine stabile interne
Kennung; `official_match_number` ist die offizielle FIFA-Spielnummer.

### `availability`

Pflicht: `team_id`, `player_name`, `status`, `impact`, `as_of`, `source`

`status`: `out`, `suspended`, `doubtful`, `questionable` oder `available`.
`impact` liegt zwischen 0 und 1 und beschreibt die sportliche Bedeutung des
Spielers.

### `lineups`

Pflicht: `match_id`, `team_id`, `player_name`, `is_starting`,
`player_rating`, `as_of`, `source`

Optional: `expected_minutes`. `player_rating` verwendet eine Skala von 0 bis
100. Eine Aufstellung wird erst ab sieben Startern als belastbares Signal
gewertet.

### `odds`

Pflicht: `match_id`, `bookmaker`, `collected_at`, `home_odds`, `draw_odds`,
`away_odds`, `source`

Die Quoten sind Dezimalquoten. Die App entfernt die Buchmachermarge pro
Anbieter und mittelt anschließend die normalisierten 1/X/2-Wahrscheinlichkeiten.

### Swisslos Sporttip

Swisslos-Quoten werden im Dashboard immer als eigene Spalten `Swisslos 1`,
`Swisslos X` und `Swisslos 2` angezeigt. In der Spielanalyse erscheinen
zusätzlich die bereinigte implizite Wahrscheinlichkeit, Buchmachermarge und
Abweichung zur Modellwahrscheinlichkeit.

Auf der Seite **Swisslos-Quoten** lädt der Knopf
`Offizielle Swisslos-Quoten aktualisieren` die aktuellen 1/X/2-Quoten und
Weltmeisterquoten aus dem öffentlichen Sporttip-Widget der
[offiziellen WM-2026-Seite](https://www.swisslos.ch/de/sporttip/sportwetten/fussball/wm-2026).
Der Zugriff ist rein lesend: Die App meldet sich nicht an und gibt keine
Wette ab.

Jeder Snapshot wird mit Zeitstempel in `data/wm2026_tipper.sqlite` gespeichert
und nach einem Neustart automatisch wieder geladen. Frühere Werte bleiben
erhalten, damit Quotenbewegung und Closing-Line-Value nachvollziehbar sind.
Alternativ kann die Aktualisierung im Projektordner gestartet werden:

```powershell
.\.venv312\Scripts\python.exe scripts\update_swisslos_odds.py
```

Die manuelle Erfassung und die CSV-Vorlagen bleiben als Rückfalloption
erhalten, falls Swisslos die technische Struktur seines Widgets ändert.

Für den Weltmeistermarkt gibt es eine zweite Vorlage
`Swisslos-Weltmeisterquoten-Vorlage`. Sie wird als `outright_odds` importiert
und enthält `team_id`, `bookmaker`, `market=champion`, `collected_at`,
`decimal_odds` und `source`. Die Turniersimulation vergleicht danach jede
Titelchance direkt mit der Swisslos-Langzeitquote. Bei weniger als 80 Prozent
Marktabdeckung wird bewusst keine Wette empfohlen.

### Wettanalyse

Die Seite **Wettanalyse** prüft für jedes Spiel alle drei Swisslos-Märkte
`1`, `X` und `2`. Der wahrscheinlichste Sieger ist dabei nicht automatisch
die beste Wette. Entscheidend sind:

```text
Expected Return = Modellwahrscheinlichkeit * Dezimalquote - 1
Edge = Modellwahrscheinlichkeit - bereinigte Swisslos-Wahrscheinlichkeit
```

Nur wenn Quote, Edge, Expected Return, Confidence, Datenunsicherheit und
Quotenalter alle die konfigurierten Grenzen erfüllen, erscheint `WETTEN`.
Andernfalls zeigt die App ausdrücklich `KEINE WETTE` und den Grund. Der
Einsatz basiert auf einem reduzierten Kelly-Wert, wird durch Datenqualität
weiter verkleinert und standardmäßig auf 2 Prozent der Bankroll begrenzt.

Die Regeln stehen in `config.yaml` unter `betting` und können in der
Seitenleiste angepasst werden. Positive Modellwerte sind keine Garantie;
auch eine mathematisch positive Wette kann den gesamten Einsatz verlieren.

Die endgültigen Einsätze werden nicht mehr isoliert je Match freigegeben,
sondern als gemeinsames Portfolio verteilt. Standardmäßig gelten:

- höchstens 2 Prozent pro Einzelwette,
- höchstens 5 Prozent neue/offene Exposition pro Spieltag,
- höchstens 10 Prozent gesamte offene Exposition,
- maximal drei Wetten pro Spieltag,
- offene Paper-Wetten werden vom verfügbaren Risikobudget abgezogen.

Wenn mehrere Signale konkurrieren, priorisiert die App Expected Return,
Confidence und Datenqualität. Ein rechnerisch positives Signal kann deshalb
als `KEINE WETTE` erscheinen, wenn das Portfolio-Limit bereits ausgeschöpft
ist.

### Paper-Wettjournal

Qualifizierte Value-Signale können auf der Seite **Paper-Wettjournal** lokal
in SQLite gespeichert werden. Es werden keine echten Wetten platziert. Sobald
Endstände in `matches.csv` vorliegen, rechnet die App die simulierten Wetten
ab und zeigt:

- offene und abgerechnete Paper-Wetten
- Trefferquote, realisierten Gewinn/Verlust und ROI
- simulierten Bankroll-Verlauf
- Closing-Line-Value gegenüber der letzten importierten Swisslos-Quote vor
  dem Anpfiff

Mehrere zeitgestempelte Swisslos-Snapshots sind besonders wichtig: Ein
positiver Closing-Line-Value ist langfristig meist aussagekräftiger als eine
kurze Glücksserie, beweist aber ebenfalls keinen zukünftigen Gewinn.

### Matchday-Center

Im **Matchday-Center** kann ein offizieller WM-Endstand erfasst oder korrigiert
werden. Jeder Resultatssnapshot bleibt in SQLite protokolliert. Der jeweils
neueste Stand:

- markiert das Match als abgeschlossen,
- wird als echtes FIFA-WM-Spiel in die Formhistorie übernommen,
- aktualisiert Rolling Elo und Form 5/10 der folgenden Matches,
- fixiert das Resultat in der Monte-Carlo-Turniersimulation,
- rechnet passende offene Paper-Wetten automatisch ab.

Die Quelle und eine ausdrückliche Bestätigung des offiziellen Endstands sind
Pflicht. Korrekturen ersetzen den Matchwert im Modell, löschen aber nicht den
Snapshot-Verlauf.

### Modell-Backtest

Die Seite **Modell-Backtest** prüft historische 1/X/2-Prognosen ohne
Zukunftsinformationen. Für jedes Testspiel werden Form, Angriff/Verteidigung
und H2H ausschließlich aus früheren Spielen berechnet. Heutige FIFA-Ratings,
Quoten, Verletzungen und Aufstellungen werden nicht rückwirkend eingesetzt.

Ausgegeben werden:

- 1/X/2-Trefferquote
- multiclass Brier Score
- Log Loss
- Vergleich mit einer konstanten historischen Basisprognose
- Kalibrierung nach Wahrscheinlichkeitsklassen und Spielausgang

Ein guter Backtest beweist keinen zukünftigen Gewinn. Für einen belastbaren
historischen Wett-Renditetest werden zusätzlich archivierte Swisslos-Quoten
mit damaligem Erfassungszeitpunkt benötigt.

### Wahrscheinlichkeitskalibrierung

Die produktiven 1/X/2-Wahrscheinlichkeiten werden standardmäßig mit einer
Multiclass-Temperaturkalibrierung korrigiert. Trainiert wurde sie auf 1.353
historischen Spielen bis Ende 2024 und ausschließlich auf 594 späteren Spielen
ab 2025 validiert. Auf diesem getrennten Validierungszeitraum verbesserten
sich die Kennzahlen:

```text
Brier Score: 0.5228 -> 0.5192
Log Loss:    0.8913 -> 0.8810
```

Für den Rating-Faktor verwendet der historische Test ein Rolling Elo, das vor
jedem Anpfiff nur aus bereits abgeschlossenen Länderspielen berechnet wird.
Der aktuelle Elo-Stand wird aus derselben Historie erzeugt und ersetzt im
Modell die bloß aus einem FIFA-Rang abgeleitete Hilfszahl.

Die Kalibrierung skaliert die Heim-, Remis- und Auswärtssieg-Blöcke der
vollständigen Resultatmatrix. Dadurch bleiben Dashboard, SRF-EV-Tipp,
Wettanalyse und Turniersimulation konsistent. In der Seitenleiste kann sie
für einen direkten Vergleich deaktiviert werden.

Historie aktualisieren:

```powershell
python scripts/update_historical_results.py --as-of 2026-06-10
```

Der gleiche Vorgang ist auf der Seite **Datenimport** über
„Historische Resultate jetzt online aktualisieren“ verfügbar. Dabei werden
abgeschlossene WM-Spiele in `matches.csv` als `completed` markiert und mit
ihrem Endstand gespeichert. Die nächsten Spiele verwenden diese Resultate in
Form 5/10 und H2H; die Turniersimulation hält abgeschlossene Partien fest und
simuliert nur offene Begegnungen.

`rating` im Startdatensatz entspricht der offiziellen FIFA-Punktzahl. Für die
Matchprognose wird sie mit dem aus historischen A-Länderspielen berechneten
Rolling Elo kombiniert. Der FIFA-Anteil ist in `config.yaml` konfigurierbar:

```text
model.fifa_blend_weight = 0.25
```

Aktualisierung über die offizielle FIFA-Ranglistenseite:

```powershell
python scripts/update_fifa_rankings.py
```

## Expected-Points-Logik

Für jeden Tipp im konfigurierten Torraum wird über alle möglichen echten
Resultate summiert:

```text
EV(Tipp) = Summe[ P(echtes Resultat) * Punkte(Tipp, echtes Resultat) ]
```

Die Scoring-Seite unterstützt exaktes Resultat, Tendenz, Tordifferenz,
richtige Team-Torzahl, Bonus, allgemeinen Multiplikator und Joker. Regeln
können additiv wirken oder nur den höchsten erfüllten Punktwert vergeben.

- **Sicher**: maximiert strikt den EV; geringere Varianz entscheidet nur bei
  Gleichstand.
- **Value**: akzeptiert nur Tipps nahe am maximalen EV und bevorzugt weniger
  offensichtliche Resultate.
- **Risiko**: erlaubt einen größeren, begrenzten EV-Abstand und gewichtet
  Punkte-Streuung als Upside.

Die Grenzwerte und Modellgewichte liegen in `config.yaml`.

## Drei unterschiedliche Aussagen pro Spiel

1. **Wahrscheinlichster Ausgang:** Vergleich der gesamten Wahrscheinlichkeiten
   für Heimsieg, Remis und Auswärtssieg.
2. **Hypothetischer Endstand:** Das wahrscheinlichste exakte Resultat, das zur
   wahrscheinlichsten 1/X/2-Tendenz passt.
3. **SRF-EV-Tipp:** Der Tipp mit dem höchsten erwarteten Punktwert unter den
   eingestellten Scoring-Regeln.

Diese Werte können voneinander abweichen. Beispielsweise können alle
Heimsiegresultate zusammen wahrscheinlicher sein als ein Remis, obwohl `1:1`
das häufigste einzelne exakte Resultat ist. Ebenso bevorzugt das Scoring
häufig `1:0`, weil dieser Tipp über mehrere mögliche Endstände Punkte für
Tendenz, Differenz oder Team-Tore sammelt.

## Projektstruktur

```text
wm2026-tipper/
  app/main.py
  data/world_cup_2026/
  src/
    data_loader.py
    feature_engineering.py
    prediction.py
    scoring.py
    optimizer.py
    simulation.py
    explainability.py
    exporting.py
    storage.py
  tests/
  config.yaml
  requirements.txt
  Dockerfile
```

## Spätere Datenquellen

Neue APIs sollten in einem eigenen Adapter in `src/` auf dieselben
DataFrame-Verträge abgebildet werden. API-Keys gehören in Umgebungsvariablen
oder lokale Streamlit-Secrets und dürfen nicht in `config.yaml` oder Git
gespeichert werden.
