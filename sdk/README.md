# plantclaims

Python client for [PlantClaims](https://plantclaims.org), claims extracted from the titles and abstracts of plant
science papers. Each claim carries verbatim quotes from its paper as evidence. No claim has been reviewed by a person.

```
pip install https://plantclaims.org/sdk/plantclaims-0.1.1-py3-none-any.whl
```

```python
import plantclaims

hits = plantclaims.search_claims("wheat stripe rust resistance", page_size=5)
print(hits["total"])
for hit in hits["results"]:
    claim, paper = hit["claim"], hit["paper"]
    print(claim["statement"], "|", claim["ev"][0]["quote"], "|", paper["doi"])

# every claim on Fusarium head blight that reports a significant result, newest first
for hit in plantclaims.iter_claims("Fusarium head blight", stat="significant", sort="newest", limit=200):
    ...

paper = plantclaims.get_paper("10.1186/s12870-019-1764-4")   # an OpenAlex id (W…) or a DOI
vocab = plantclaims.facets()["vocab"]                         # the values every filter accepts
```

`PlantClaims(base_url=…)` binds a client to another copy of the site. Errors raise `plantclaims.APIError` with the API's message.
No dependencies beyond the standard library; Python 3.8 or later.
