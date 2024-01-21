import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Callable, Iterator

import requests
from Bio import Entrez
from cached_path import cached_path
from diskcache import Cache
from pydantic import BaseModel

from topical import util
from topical.common import get_cache_dir

# Set up logging
logger = logging.getLogger(__name__)

# Set up constants
MESH_LATEST_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{}.xml"
# We give these their own cache directories so that we can clear them separately if things get corrupted
ENTREZ_CACHE_DIR = get_cache_dir() / "entrez"
EFETCH_CACHE_DIR = ENTREZ_CACHE_DIR / "efetch"
ESEARCH_CACHE_DIR = ENTREZ_CACHE_DIR / "esearch"
MESH_CACHE_DIR = get_cache_dir() / "mesh"


# Complile any regular expressions
date_regex = re.compile(r"\b\d{4}\b")  # match exactly four digits

# Set up Entrez
Entrez.email = os.environ.get("ENTREZ_EMAIL")
Entrez.api_key = os.environ.get("ENTREZ_API_KEY")
if Entrez.email:
    logger.info(f"Found ENTREZ_EMAIL environment variable: {Entrez.email}")
if Entrez.api_key:
    logger.info(f"Found ENTREZ_API_KEY environment variable: {Entrez.api_key}")


class MeSHDescriptor(BaseModel):
    ui: str
    class_: str
    name: str
    date_created: util.Date
    date_revised: util.Date | None = None
    date_established: util.Date
    tree_numbers: list[str]

    def max_tree_depth(self) -> int | None:
        """Returns the maximum depth of this descriptor in the tree"""

        # For whatever reason, both "Male" and "Female" have no tree numbers, but are pretty clearly a depth of 2
        # So we handle these cases manually
        if self.ui == "D005260" or self.ui == "D008297":
            return 2
        return max(len(tree_num.split(".")) for tree_num in self.tree_numbers)


def _fetch_latest_year() -> int:
    """Fetches the latest year of MeSH from the NLM website. Note, this requires an active internet connection."""
    year = datetime.today().year
    while True:
        url = MESH_LATEST_URL.format(year)
        request = requests.head(url)
        request.raise_for_status()
        if request.status_code == 200:
            return year
        year -= 1


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


def fetch_mesh(year: int | None = None, **kwargs) -> Iterator[MeSHDescriptor]:
    """Fetch and parse the National Library of Medicine (NLM) MeSH Descriptors for a given year.

    Note that descriptors are cumulative. To get all current descriptors, specify the latest year. If no year is
    specified, the current year will be used. Downloading the descriptors can take upwards of 10 minutes (depending
    on your internet connection and speed). Subsequent calls for the same year will be faster, as downloaded
    descriptors are cached. See https://www.nlm.nih.gov/mesh/xml_data_elements.html for a description of the fields.

    Args:
        year (int, optional): Year to fetch descriptors for (defaults to current year). Note: descriptors are cumulative.
        kwargs: Keyword arguments to pass to `cached_path`.
    Returns:
        Iterator[MeSHDescriptor]: A generator of `MeSHDescriptor`'s, each of which represents a single descriptor.
    """
    year = year or _fetch_latest_year()
    # Download and cache the latest version of the MeSH XML file containing SCPs
    fp = cached_path(MESH_LATEST_URL.format(year), cache_dir=MESH_CACHE_DIR, **kwargs)
    # Parse the resulting XML file into a dictionary and yield it
    # NOTE: This does not include all fields, and may need to be extended in the future
    context = ET.iterparse(fp, events=("end",))
    for event, elem in context:
        if elem.tag == "DescriptorRecord" and event == "end":
            record_dict = {}
            record_dict["ui"] = elem.find("DescriptorUI").text
            record_dict["class_"] = elem.get("DescriptorClass")
            record_dict["name"] = elem.find("DescriptorName/String").text
            record_dict["date_created"] = {
                "year": elem.find("DateCreated/Year").text,
                "month": elem.find("DateCreated/Month").text,
                "day": elem.find("DateCreated/Day").text,
            }
            if elem.find("date_revised") is not None:
                record_dict["DateRevised"] = {
                    "year": elem.find("DateRevised/Year").text,
                    "month": elem.find("DateRevised/Month").text,
                    "day": elem.find("DateRevised/Day").text,
                }
            if elem.find("DateEstablished") is not None:
                record_dict["date_established"] = {
                    "year": elem.find("DateEstablished/Year").text,
                    "month": elem.find("DateEstablished/Month").text,
                    "day": elem.find("DateEstablished/Day").text,
                }

            record_dict["tree_numbers"] = [tree.text for tree in elem.findall("TreeNumberList/TreeNumber")]

            yield MeSHDescriptor(**record_dict)

            # This is important! It allows the memory occupied by the element to be freed.
            elem.clear()
