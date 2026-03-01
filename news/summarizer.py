from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dateutil.tz import gettz


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DigestOptions:
    timezone: str = "Asia/Colombo"
    top_n: int = 10


def _today_str(tz_name: str) -> str:
    tz = gettz(tz_name)
    return datetime.now(tz).strftime("%Y-%m-%d")


class Summarizer:
    def build_digest(self, df: pd.DataFrame, opts: DigestOptions) -> Dict[str, str]:
        """
        Returns {"txt": ..., "md": ..., "html": ...}
        """
        date_str = _today_str(opts.timezone)

        if df.empty:
            base = f"📅 Daily News Digest – {date_str}\n\nNo headlines matched your filters today.\n"
            return {"txt": base, "md": base, "html": f"<h2>Daily News Digest – {date_str}</h2><p>No headlines matched today.</p>"}

        top = df.head(opts.top_n).copy()

        grouped = []
        for cat, g in top.groupby("category", sort=False):
            lines = []
            for _, r in g.iterrows():
                lines.append(f"- {r['title']} ({r['source']})")
            grouped.append((cat, lines))

        txt = [f"📅 Daily News Digest – {date_str}", ""]
        md = [f"📅 Daily News Digest – {date_str}", ""]
        html = [f"<h2>📅 Daily News Digest – {date_str}</h2>"]

        for cat, lines in grouped:
            txt.append(f"🔹 {cat}")
            md.append(f"## 🔹 {cat}")
            html.append(f"<h3>🔹 {cat}</h3><ul>")

            for line in lines:
                txt.append(line)
                md.append(line)
            for _, r in top[top["category"] == cat].iterrows():
                html.append(f"<li><a href=\"{r['link']}\">{r['title']}</a> <em>({r['source']})</em></li>")

            txt.append("")
            md.append("")
            html.append("</ul>")

        return {"txt": "\n".join(txt).strip() + "\n", "md": "\n".join(md).strip() + "\n", "html": "\n".join(html).strip() + "\n"}

    def save_outputs(self, digest: Dict[str, str], out_dir: str, tz_name: str) -> Dict[str, str]:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        date_str = _today_str(tz_name)

        paths = {
            "txt": str(Path(out_dir) / f"digest_{date_str}.txt"),
            "md": str(Path(out_dir) / f"digest_{date_str}.md"),
            "html": str(Path(out_dir) / f"digest_{date_str}.html"),
        }

        for k, p in paths.items():
            Path(p).write_text(digest[k], encoding="utf-8")

        logger.info("Saved digest outputs: %s", paths)
        return paths