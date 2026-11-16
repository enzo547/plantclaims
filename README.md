# PlantClaims

**Search plant science by finding, not by paper.**

PlantClaims turns the plant science literature into individual, source-grounded claims. Each claim is a single finding, such as a gene
that confers resistance, a treatment that raises yield, or a pathogen that infects a host, and it carries the verbatim sentences from
its paper that support it. You search the findings themselves and filter them by organism, gene, compound, kind of result and
direction of effect. Each finding links back to the paper it came from.

**Live at [plantclaims.org](https://plantclaims.org)** · [API docs](https://plantclaims.org/docs) · [MCP server](#ai-agents-mcp) · [Python SDK](#python-sdk)

## What's inside

- **11.76 million claims** from **1.73 million plant science papers**, drawn from an index of 3.55 million papers.
- **22.0 million verbatim evidence quotes.** Every claim can be checked against the words of its source in one click.
- **Structured fields on every claim:**
  - organism, material, gene, compound, organ or stage, treatment, and an interacting organism with its role (pathogen, pest,
    symbiont, host, parasite, weed);
  - the kind of claim (observation, intervention effect, association, mechanism, method or resource, recommendation);
  - direction and statistical support of the effect;
  - whether the claim reports the paper's own result, summarises a review, or cites background.
- **Normalised names:**
  - organisms mapped to NCBI Taxonomy (107,752 taxa);
  - genes to official symbols;
  - compounds to ChEBI.

  Browse by organism or gene and you get every claim about it, whatever name the paper used.
- **Research questions across papers:** claims that answer the same question are grouped, so you can see which papers report the same
  result and which report different ones.
- **Four ways in:** a web interface, a REST API, a Python SDK and an MCP server for AI agents.

## Quick start

### Web

Open [plantclaims.org](https://plantclaims.org) and search, for example, `wheat stripe rust resistance`.

### REST API

```bash
# Search claims (same ranking as the website)
curl 'https://plantclaims.org/api/claims?q=drought+tolerance&kind=observation&sort=newest&pageSize=5'

# One claim with its paper and evidence quotes
curl 'https://plantclaims.org/api/claims/{claim_id}'

# One paper (OpenAlex id or DOI) with all its claims
curl 'https://plantclaims.org/api/papers/W2945897111'

# The values every filter accepts
curl 'https://plantclaims.org/api/facets'
```

Full parameter list: [plantclaims.org/docs](https://plantclaims.org/docs).

### Python SDK

```bash
pip install https://plantclaims.org/sdk/plantclaims-0.1.1-py3-none-any.whl
```

```python
import plantclaims

hits = plantclaims.search_claims("wheat stripe rust resistance", page_size=5)
for hit in hits["results"]:
    claim, paper = hit["claim"], hit["paper"]
    print(claim["statement"], "|", claim["ev"][0]["quote"], "|", paper["doi"])

# every claim on Fusarium head blight that reports a significant result, newest first
for hit in plantclaims.iter_claims("Fusarium head blight", stat="significant", sort="newest", limit=200):
    ...

paper = plantclaims.get_paper("10.1186/s12870-019-1764-4")   # OpenAlex id (W…) or DOI
```

The SDK depends on the Python standard library only and runs on Python 3.8 or later. Source code is in [`sdk/`](sdk/).

### AI agents (MCP)

PlantClaims runs a [Model Context Protocol](https://modelcontextprotocol.io) server at `https://plantclaims.org/mcp`. With it, an AI
agent can answer a plant science question from cited findings instead of from memory.

```bash
# Claude Code
claude mcp add --transport http plantclaims https://plantclaims.org/mcp
```

```json
{ "mcpServers": { "plantclaims": { "url": "https://plantclaims.org/mcp" } } }
```

| Tool | What it does |
|---|---|
| `search_claims` | Search claims with the website's ranking and filters |
| `get_claim` | One claim: statement, fields, evidence quotes, source paper |
| `search_papers` | Search papers by title, authors, venue, DOI or topic |
| `get_paper` | One paper with the text the model read and its claims |
| `list_filters` | The values each filter accepts |
| `stats` | Size of the dataset and the snapshot it comes from |

## License

Code in this repository is released under the [MIT License](LICENSE).
