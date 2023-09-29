import logging
import os
import re
from typing import Any, Callable, Iterator

from Bio import Entrez
from diskcache import Cache

from topical.common import get_cache_dir

# Set up logging
logger = logging.getLogger(__name__)

# We give these their own cache directories so that we can clear them separately if things get corrupted
ENTREZ_CACHE_DIR = get_cache_dir() / "entrez"
EFETCH_CACHE_DIR = ENTREZ_CACHE_DIR / "efetch"
ESEARCH_CACHE_DIR = ENTREZ_CACHE_DIR / "esearch"


# Complile any regular expressions
date_regex = re.compile(r"\b\d{4}\b")  # match exactly four digits

# Set up Entrez
Entrez.email = os.environ.get("ENTREZ_EMAIL")
Entrez.api_key = os.environ.get("ENTREZ_API_KEY")
if Entrez.email:
    logger.info(f"Found ENTREZ_EMAIL environment variable: {Entrez.email}")
if Entrez.api_key:
    logger.info(f"Found ENTREZ_API_KEY environment variable: {Entrez.api_key}")


def get_year_from_medline_date(medline_date: str) -> str | None:
    """Attempts to extract a year from a MEDLINE date string. Returns None if no year is found.

    See https://www.nlm.nih.gov/bsd/licensee/elements_descriptions.html#pubdate for example strings.
    """
    candidates = date_regex.findall(medline_date)
    # Take the maximum year, if any
    candidate = str(max(int(candidate) for candidate in candidates)) if candidates else None
    return candidate


def efetch(queries: str | list[str], *, use_cache: bool = False, **kwargs) -> str:
    """Query the Entrez EFetch API and return the results.

    If `use_cache`, identical queries will be cached (and retrieved) from disk. `**kwargs` will be passed to the
    Entrez EFetch API. See here for details: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch. Note,
    It is highly recommended to provide an email address and API key via the `email` and `api_key` keyword
    arguments. Alternatively, you can set the `ENTREZ_EMAIL` and `ENTREZ_API_KEY` and environment variables.

    Examples:
    ```python
    >>> # Basic query for a single abstract from pubmed
    >>> next(nlm.efetch("26209480", db="pubmed", rettype="abstract", retmode="text"))
    >>> "1. Ann Thorac Surg. 2015 Oct;100(4):1298-304; discussion 1304. doi: ..."
    >>> # Search for multiple abstracts
    >>> list(nlm.efetch(["26209480", "36211543"], db="pubmed", rettype="abstract", retmode="text"))[0]
    >>> "1. Ann Thorac Surg. 2015 Oct;100(4):1298-304; discussion 1304. doi: ... 2. Front Cardiovasc Med. 2022 ..."
    ```

    Args:
        queries (str or list[str]): The query or queries to pass to the Entrez EFetch API.
        use_cache (bool): Whether to use a disk cache to store and retrieve the results of identical queries.
        **kwargs: Additional arguments to pass to the Entrez API.

    Returns:
        str: The results of the Entrez EFetch API.
    """
    # Pop any kwargs we would like to have defaults for. We set more patient defaults than the Entrez API.
    max_tries = kwargs.pop("max_tries", None) or 5
    sleep_between_tries = kwargs.pop("sleep_between_tries", None) or 30

    def _efetch(queries: str | list[str]) -> str:
        handle = Entrez.efetch(id=queries, max_tries=max_tries, sleep_between_tries=sleep_between_tries, **kwargs)
        results = Entrez.read(handle)
        handle.close()
        return results

    if not use_cache:
        return _efetch(queries)
    else:
        with Cache(EFETCH_CACHE_DIR) as reference:
            # Create a key from the term and all keyword arguments that is deterministic and invariant to order
            key = repr((queries, tuple(sorted(kwargs.items()))))
            if key in reference:
                return reference[key]
            else:
                results = _efetch(queries)
                reference.add(key, results, retry=True)
                return results


def esearch(
    queries: str | list[str], *, preprocessor: Callable[[str], str] | None = None, use_cache: bool = False, **kwargs
) -> Iterator[dict[str, Any]]:
    """Query the Entrez ESearch API and yield the results one at a time.

    If `use_cache`, identical queries will be cached (and retrieved) from disk. `**kwargs` will be passed to the
    Entrez ESearch API. See here for details: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch. Note,
    It is highly recommended to provide an email address and API key via the `email` and `api_key` keyword
    arguments. Alternatively, you can set the `ENTREZ_EMAIL` and `ENTREZ_API_KEY` and environment variables.

    Examples:
    ```python
    >>> # Query for the term "covid-19" in the PubMed database, returning the most relevant item
    >>> next(esearch("covid-19", db="pubmed", retmax=1))
    >>> {'Count': '370497', 'RetMax': '1', 'RetStart': '0', 'IdList': 'IdList': ['37368487'], ...}
    >>> # Get the total number of items matching the query
    >>> next(esearch("covid-19", db="pubmed", rettype="count"))
    >>> {'Count': '370497'}
    >>> # Retrict articles to those published in 2019 or before
    >>> next(esearch("covid-19", db="pubmed", datetype="pdat", mindate="0000", maxdate="2019"))
    >>> {'Count': '358', 'RetMax': '20', 'RetStart': '0', 'IdList': ['20301464', ...}
    ```

    Args:
        queries (str or list[str]): The query or queries to pass to the Entrez ESearch API.
        preprocessor (Callable[[str], str], optional): Apply this function to each query before passing it to the
        Entrez ESearch API
        use_cache (bool): Whether to use a disk cache to store and retrieve the results of identical queries.
        **kwargs: Additional arguments to pass to the Entrez ESearch API.

    Returns:
        Iterator[dict]: An iterator over the results of the Entrez ESearch API.
    """
    # Pop any kwargs we would like to have defaults for. We set more patient defaults than the Entrez API.
    max_tries = kwargs.pop("max_tries", None) or 5
    sleep_between_tries = kwargs.pop("sleep_between_tries", None) or 30

    if isinstance(queries, str):
        queries = [queries]

    def _esearch(term: str) -> dict:
        handle = Entrez.esearch(term=term, max_tries=max_tries, sleep_between_tries=sleep_between_tries, **kwargs)
        results = Entrez.read(handle)
        handle.close()
        return results

    # Open the cache ONCE! Opening Cache objects is slow, and since all operations are atomic, may be safely left open.
    cache = Cache(ESEARCH_CACHE_DIR) if use_cache else None
    for query in queries:
        if preprocessor is not None:
            query = preprocessor(query)

        if cache is None:
            yield _esearch(query)
        else:
            # Create a key from the term and all keyword arguments that is deterministic and invariant to order
            key = repr((query, tuple(sorted(kwargs.items()))))
            if key not in cache:
                results = _esearch(query)
                cache.add(key, results, retry=True)
            yield cache[key]

    if cache is not None:
        cache.close()
