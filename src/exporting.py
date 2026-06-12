from __future__ import annotations

import io

import pandas as pd


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Tipps")
    return output.getvalue()


def to_markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def to_srf_text(frame: pd.DataFrame) -> str:
    lines = []
    for row in frame.to_dict("records"):
        lines.append(
            f"{row['date']} | {row['home_team']} - {row['away_team']} | "
            f"{row.get('ev_tip', row.get('tip', ''))}"
        )
    return "\n".join(lines)
