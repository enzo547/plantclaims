"""Python client for PlantClaims: source-grounded claims extracted from plant science papers.

    import plantclaims
    hits = plantclaims.search_claims("wheat stripe rust resistance", page_size=5)
    for hit in hits["results"]:
        print(hit["claim"]["statement"], hit["paper"]["doi"])

Every function wraps one endpoint of the read-only JSON API (see /docs on the site) and returns its JSON as a dict, unchanged. The claims
are candidates extracted by a language model; each response carries `human_reviewed` (how many claims a person has reviewed) and the
`data_sha256` of the snapshot it was read from. No dependencies beyond the standard library.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator, Optional

__all__ = ["PlantClaims", "APIError", "DEFAULT_BASE_URL", "search_claims", "iter_claims", "get_claim", "search_papers", "iter_papers",
           "get_paper", "facets", "stats"]
__version__ = "0.1.1"

DEFAULT_BASE_URL = "https://plantclaims.org"
MAX_PAGE_SIZE = 100


class APIError(Exception):
    """The API answered with an error: `status` is the HTTP status, `message` the API's own explanation."""

    def __init__(self, status: int, message: str, url: str):
        super().__init__(f"{status}: {message} ({url})")
        self.status, self.message, self.url = status, message, url


def _params(query: str, page: Optional[int], page_size: Optional[int], sort: Optional[str], filters: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    if query:
        out["q"] = query
    if page is not None:
        out["page"] = str(page)
    if page_size is not None:
        out["pageSize"] = str(page_size)
    if sort:
        out["sort"] = sort
    for key, value in filters.items():
        if value is None or value is False:
            continue
        out[key] = "1" if value is True else str(value)
    return out


class PlantClaims:
    """A client bound to one site. `base_url` is the site's address; the API lives under /api on it."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0, user_agent: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent or f"plantclaims-python/{__version__}"

    def _get(self, path: str, params: Optional[dict[str, str]] = None) -> dict[str, Any]:
        url = f"{self.base_url}/api{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            try:
                message = json.loads(body).get("error") or body
            except ValueError:
                message = body[:200] or error.reason
            raise APIError(error.code, str(message), url) from None

    # claims
    def search_claims(self, query: str = "", *, page: int = 1, page_size: int = 10, sort: Optional[str] = None, **filters: Any) -> dict[str, Any]:
        """One page of claims, ranked as on the site's Search page.

        Filters are the API's search parameters: kind, bucket, pol, stat, es, origin, role, taxon, gene, interactor, decade, paper,
        oa=True, doi=True, group. Their values are listed by facets["vocab"]. sort is relevance, newest, oldest or cited.
        Returns total, pages, match and results: a list of {"claim", "paper", "score"}."""
        return self._get("/claims", _params(query, page, page_size, sort, filters))

    def iter_claims(self, query: str = "", *, limit: Optional[int] = None, page_size: int = MAX_PAGE_SIZE, sort: Optional[str] = None,
                    **filters: Any) -> Iterator[dict[str, Any]]:
        """Every matching claim hit ({"claim", "paper", "score"}), page after page, up to `limit` hits."""
        yield from self._iterate("/claims", query, limit, page_size, sort, filters)

    def get_claim(self, claim_id: str) -> dict[str, Any]:
        """One claim with its paper record: {"claim", "paper", …}."""
        return self._get(f"/claims/{urllib.parse.quote(claim_id, safe='')}")

    # papers
    def search_papers(self, query: str = "", *, page: int = 1, page_size: int = 25, sort: Optional[str] = None, **filters: Any) -> dict[str, Any]:
        """One page of papers. Filters: label (core, adjacent, off_topic, insufficient_evidence), reason, status, decade, oa=True, doi=True.
        Results are {"paper", "score"}; a DOI or an OpenAlex id (W…) as the query finds that paper (match == "identifier")."""
        return self._get("/papers", _params(query, page, page_size, sort, filters))

    def iter_papers(self, query: str = "", *, limit: Optional[int] = None, page_size: int = MAX_PAGE_SIZE, sort: Optional[str] = None,
                    **filters: Any) -> Iterator[dict[str, Any]]:
        """Every matching paper hit ({"paper", "score"}), page after page, up to `limit` hits."""
        yield from self._iterate("/papers", query, limit, page_size, sort, filters)

    def get_paper(self, paper_id: str) -> dict[str, Any]:
        """One paper by OpenAlex id (W…): {"paper", "text", "claims"}. text holds the title and abstract exactly as the model read them;
        evidence start/end offsets are Unicode code points into these blocks. A DOI is looked up through search_papers first."""
        if not paper_id.upper().startswith("W"):
            found = self.search_papers(paper_id, page_size=1)
            if found.get("match") != "identifier" or not found.get("results"):
                raise APIError(404, "Paper not found", f"{self.base_url}/api/papers?q={urllib.parse.quote(paper_id)}")
            paper_id = found["results"][0]["paper"]["id"]
        return self._get(f"/papers/{urllib.parse.quote(paper_id, safe='')}")

    # the dataset
    def facets(self) -> dict[str, Any]:
        """Browse trees and the vocabulary of every filter (facets["vocab"])."""
        return self._get("/facets")

    def stats(self) -> dict[str, Any]:
        """Counters, distributions, fill rates and first-pass checks of the snapshot."""
        return self._get("/stats")

    def _iterate(self, path: str, query: str, limit: Optional[int], page_size: int, sort: Optional[str], filters: dict[str, Any]):
        page_size = max(1, min(MAX_PAGE_SIZE, page_size))
        seen, page = 0, 1
        while True:
            data = self._get(path, _params(query, page, page_size, sort, filters))
            for hit in data.get("results", []):
                if limit is not None and seen >= limit:
                    return
                seen += 1
                yield hit
            if page >= data.get("pages", 0) or not data.get("results"):
                return
            page += 1


_default: Optional[PlantClaims] = None


def _client() -> PlantClaims:
    global _default
    if _default is None:
        _default = PlantClaims()
    return _default


def search_claims(query: str = "", **kwargs: Any) -> dict[str, Any]:
    return _client().search_claims(query, **kwargs)


def iter_claims(query: str = "", **kwargs: Any) -> Iterator[dict[str, Any]]:
    return _client().iter_claims(query, **kwargs)


def get_claim(claim_id: str) -> dict[str, Any]:
    return _client().get_claim(claim_id)


def search_papers(query: str = "", **kwargs: Any) -> dict[str, Any]:
    return _client().search_papers(query, **kwargs)


def iter_papers(query: str = "", **kwargs: Any) -> Iterator[dict[str, Any]]:
    return _client().iter_papers(query, **kwargs)


def get_paper(paper_id: str) -> dict[str, Any]:
    return _client().get_paper(paper_id)


def facets() -> dict[str, Any]:
    return _client().facets()


def stats() -> dict[str, Any]:
    return _client().stats()
