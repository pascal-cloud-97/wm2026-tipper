# FIFA Fussball-Weltmeisterschaft 2026

Stand der Turnierdaten: 14. Juni 2026.

- Teilnehmer, Gruppeneinteilung, Paarungen, Austragungsorte und Anspielzeiten
  wurden gegen den offiziellen FIFA-Spielplan vom 10. April 2026 geprüft.
- `official_schedule_manifest.csv` ist der strukturierte Vergleichsabzug.
  Die FIFA-PDF nennt alle Zeiten in Eastern Time; `matches.csv` verwendet
  weiterhin die jeweilige lokale Stadionzeit.
- FIFA-Rang und FIFA-Punkte stammen aus der letzten offiziellen
  FIFA/Coca-Cola-Männerweltrangliste vom 1. April 2026.
- `rating` in `teams.csv` entspricht der offiziellen FIFA-Punktzahl. Für die
  Prognose wird sie mit dem Rolling Elo aus den historischen Resultaten
  kombiniert.
- `historical_results.csv` enthält reale Männer-A-Länderspiele seit 2018,
  soweit mindestens eines der 48 WM-Teams beteiligt war. Die Daten stammen
  aus `martj42/international_results` (CC0) und werden auf den Stichtag
  10. Juni 2026 begrenzt; bestätigte WM-Endstände werden danach fortlaufend
  ergänzt.
- Alle acht bis zum Morgen des 14. Juni 2026 beendeten Partien sowie
  28 Kartenereignisse wurden aus den ESPN-Matchcentern übernommen.
- `match_events.csv` enthält Gelb-/Rot-Ereignisse mit Spieler, Minute und
  Starterstatus. Rote Karten werden mindestens für das nächste Spiel als
  Sperre behandelt; längere Sperren erst nach bestätigter FIFA-Entscheidung.
- Der Sperreneinfluss verwendet eine offengelegte Modellheuristik:
  0.65 Belastung für einen gesperrten Starter, 0.35 für einen eingesetzten
  Ersatzspieler. Diese Werte sind Modellannahmen, keine offiziellen Ratings.
- Verletzungen, künftige Aufstellungen und Quoten sind zeitkritische Daten.
  Fehlende Informationen werden nicht erfunden und erhöhen die Unsicherheit.
- Datums- und Zeitangaben in `matches.csv` sind lokale Stadionzeiten ohne
  Zeitzonen-Konvertierung.

Quellen:

- FIFA World Cup 26 Match Schedule, 10. April 2026:
  https://digitalhub.fifa.com/m/1be9ce37eb98fcc5/original/FWC26-Match-Schedule_English.pdf
- FIFA Final Draw results:
  https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/final-draw-results
- FIFA/Coca-Cola Men's World Ranking, 1. April 2026:
  https://inside.fifa.com/fifa-world-ranking/men
- martj42/international_results, CC0 1.0
- ESPN Matchcenter, Mexiko–Südafrika:
  https://www.espn.com/soccer/match/_/gameId/760415/south-africa-mexico
- ESPN Matchcenter, Südkorea–Tschechien:
  https://www.espn.com/soccer/match/_/gameId/760414/czechia-south-korea

Historische Resultate aktualisieren:

```powershell
python scripts/update_historical_results.py --as-of 2026-06-10
```

Laufende WM-Endstände, Karten und daraus folgende Sperren aktualisieren:

```powershell
python scripts/update_world_cup_results.py --as-of 2026-06-14
```

Offizielle FIFA-Rangliste aktualisieren:

```powershell
python scripts/update_fifa_rankings.py
```
