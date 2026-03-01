from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import feedparser
import requests
import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawItem:
    title: str
    source: str
    published: Optional[str]
    link: str
    summary: str
    raw: Dict[str, Any]


def _resolve_env_placeholders(value: Any) -> Any:
    """
    Supports "${ENV:VAR_NAME}" in YAML config.
    """
    if isinstance(value, str) and value.startswith("${ENV:") and value.endswith("}"):
        env_name = value[len("${ENV:") : -1]
        return os.getenv(env_name, "")
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_env_placeholders(cfg)


class NewsFetcher:
    def __init__(self, timeout_s: int = 12, max_retries: int = 2, backoff_s: float = 0.8):
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def fetch_all(self, cfg: Dict[str, Any]) -> List[RawItem]:
        items: List[RawItem] = []

        for src in cfg.get("rss_sources", []):
            name = src["name"]
            url = src["url"]
            items.extend(self.fetch_rss(name=name, url=url))

        for src in cfg.get("api_sources", []):
            if not src.get("enabled", False):
                continue
            items.extend(self.fetch_api(src))

        return items

    def fetch_rss(self, name: str, url: str) -> List[RawItem]:
        logger.info("Fetching RSS: %s (%s)", name, url)

        # feedparser can fetch by itself; still handle timeouts via requests for better control.
        content = self._get(url)
        if content is None:
            return []

        feed = feedparser.parse(content)
        out: List[RawItem] = []

        for e in feed.entries:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            summary = (e.get("summary") or e.get("description") or "").strip()
            published = e.get("published") or e.get("updated") or None

            if not title or not link:
                continue

            out.append(
                RawItem(
                    title=title,
                    source=name,
                    published=published,
                    link=link,
                    summary=summary,
                    raw=dict(e),
                )
            )

        logger.info("RSS fetched %d items from %s", len(out), name)
        return out

    def fetch_api(self, src_cfg: Dict[str, Any]) -> List[RawItem]:
        """
        Generic JSON API fetch. Includes a ready-made implementation for NewsAPI.org style responses.
        """
        name = src_cfg["name"]
        url = src_cfg["url"]
        api_type = (src_cfg.get("type") or "generic").lower()
        params = src_cfg.get("params") or {}
        headers = src_cfg.get("headers") or {}

        logger.info("Fetching API: %s (%s, type=%s)", name, url, api_type)

        json_data = self._get_json(url, params=params, headers=headers)
        if json_data is None:
            return []

        out: List[RawItem] = []

        if api_type == "newsapi":
            # Expected: {"articles":[{"title","url","publishedAt","description","source":{"name":...}}]}
            for a in (json_data.get("articles") or []):
                title = (a.get("title") or "").strip()
                link = (a.get("url") or "").strip()
                published = a.get("publishedAt")
                summary = (a.get("description") or "").strip()
                src_name = (a.get("source") or {}).get("name") or name

                if not title or not link:
                    continue

                out.append(
                    RawItem(
                        title=title,
                        source=str(src_name),
                        published=published,
                        link=link,
                        summary=summary,
                        raw=dict(a),
                    )
                )
        else:
            # Generic: try common keys
            records = json_data.get("items") or json_data.get("data") or []
            if isinstance(records, dict):
                records = [records]

            for r in records:
                title = str(r.get("title") or "").strip()
                link = str(r.get("link") or r.get("url") or "").strip()
                published = r.get("published") or r.get("publishedAt") or r.get("date")
                summary = str(r.get("summary") or r.get("description") or "").strip()

                if not title or not link:
                    continue

                out.append(
                    RawItem(
                        title=title,
                        source=name,
                        published=str(published) if published else None,
                        link=link,
                        summary=summary,
                        raw=dict(r),
                    )
                )

        logger.info("API fetched %d items from %s", len(out), name)
        return out

    def _get(self, url: str) -> Optional[bytes]:
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(url, timeout=self.timeout_s, headers={"User-Agent": "SmartNewsAggregator/1.0"})
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                logger.warning("GET failed (%s) attempt %d/%d: %s", url, attempt + 1, self.max_retries + 1, e)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_s * (2**attempt))
        return None

    def _get_json(self, url: str, params: Dict[str, Any], headers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": "SmartNewsAggregator/1.0", **headers},
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning("GET JSON failed (%s) attempt %d/%d: %s", url, attempt + 1, self.max_retries + 1, e)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_s * (2**attempt))
        return None