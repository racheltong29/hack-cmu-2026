"""Client for the free SEC EDGAR APIs (no API key required).

SEC fair-access rules: identify yourself via User-Agent and stay under
10 requests/second. https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import re
import time

import httpx

from ..config import get_settings

_TICKER_CACHE: dict[str, str] = {}


class EdgarClient:
    def __init__(self, client: httpx.Client | None = None):
        s = get_settings()
        self._settings = s
        self._client = client or httpx.Client(
            headers={"User-Agent": s.sec_user_agent, "Accept-Encoding": "gzip"},
            timeout=30,
        )
        self._last_request = 0.0

    def _throttle(self) -> None:
        # SEC allows 10 req/s; keep a safe 8/s ceiling.
        elapsed = time.monotonic() - self._last_request
        if elapsed < 0.125:
            time.sleep(0.125 - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    def cik_for_ticker(self, ticker: str) -> str:
        """Resolve a ticker to a zero-padded 10-digit CIK."""
        ticker = ticker.upper()
        if not _TICKER_CACHE:
            data = self._get(f"{self._settings.sec_base_url}/files/company_tickers.json").json()
            for row in data.values():
                _TICKER_CACHE[row["ticker"].upper()] = f"{row['cik_str']:010d}"
        cik = _TICKER_CACHE.get(ticker)
        if cik is None:
            raise KeyError(f"Unknown ticker: {ticker}")
        return cik

    def recent_filings(self, ticker: str, form_types: tuple[str, ...] = ("10-K", "10-Q")) -> list[dict]:
        """List recent filings metadata for a ticker from the submissions API."""
        cik = self.cik_for_ticker(ticker)
        data = self._get(f"{self._settings.sec_data_url}/submissions/CIK{cik}.json").json()
        recent = data["filings"]["recent"]
        out = []
        for i, form in enumerate(recent["form"]):
            if form in form_types:
                accession = recent["accessionNumber"][i].replace("-", "")
                doc = recent["primaryDocument"][i]
                out.append(
                    {
                        "form_type": form,
                        "filing_date": recent["filingDate"][i],
                        "report_date": recent["reportDate"][i],
                        "url": (
                            f"{self._settings.sec_base_url}/Archives/edgar/data/"
                            f"{int(cik)}/{accession}/{doc}"
                        ),
                    }
                )
        return out

    def fetch_filing_text(self, url: str) -> str:
        """Download a filing document and strip it to plain text."""
        html = self._get(url).text
        return strip_html(html)


def strip_html(html: str) -> str:
    """Minimal, dependency-free HTML→text good enough for filing bodies."""
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|tr|table|h[1-6]|li)>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()
