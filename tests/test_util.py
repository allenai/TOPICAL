import re

import pytest
from hypothesis import given
from hypothesis.strategies import booleans, text

from topical import util


def test_year_is_valid() -> None:
    # Valid case
    assert util.Date.year_is_valid("2021") == "2021"

    # Invalid case (length 1)
    with pytest.raises(ValueError):
        util.Date.year_is_valid("21")

    # Invalid case (length 2)
    with pytest.raises(ValueError):
        util.Date.year_is_valid("21")

    # Invalid case (length 3)
    with pytest.raises(ValueError):
        util.Date.year_is_valid("202")

    # Invalid case (length 5)
    with pytest.raises(ValueError):
        util.Date.year_is_valid("20211")


@given(text=text(), lowercase=booleans())
def test_sanitize_text(text: str, lowercase: bool) -> None:
    sanitized_text = util.sanitize_text(text, lowercase=lowercase)

    # There should be no cases of multiple spaces or tabs
    assert re.search(r"[ ]{2,}", sanitized_text) is None
    assert "\t" not in sanitized_text
    # The beginning and end of the string should be stripped of whitespace
    assert not sanitized_text.startswith(("\n", " "))
    assert not sanitized_text.endswith(("\n", " "))
    # Sometimes, hypothesis generates text that cannot be lowercased (like latin characters).
    # We don't particularly care about this, and it breaks this check.
    # Only run if the generated text can be lowercased.
    if lowercase and text.lower().islower():
        assert all(not char.isupper() for char in sanitized_text)


def test_get_pmids_from_text() -> None:
    # Example with no digits
    assert util.get_pmids_from_text("This is an example that contains no digits at all.") == []

    # Example with digits, but not in square brackets
    assert (
        util.get_pmids_from_text(
            "This is an example that contains digits, like 1 or 123, but not within square brackets."
        )
        == []
    )

    # Exampe with digits in square brackets
    assert util.get_pmids_from_text(
        "This is an example that contains digits, like [1] or [123], within square brackets."
    ) == ["1", "123"]


def test_replace_pmids_with_markdown_link() -> None:
    # Example with no digits
    assert (
        util.replace_pmids_with_markdown_link("This is an example that contains no digits at all.")
        == "This is an example that contains no digits at all."
    )

    # Example with digits, but not in square brackets
    assert (
        util.replace_pmids_with_markdown_link(
            "This is an example that contains digits, like 1 or 123, but not within square brackets."
        )
        == "This is an example that contains digits, like 1 or 123, but not within square brackets."
    )

    # Exampe with digits in square brackets
    assert util.replace_pmids_with_markdown_link(
        "This is an example that contains digits, like [1] or [123], within square brackets."
    ) == (
        "This is an example that contains digits, like [1](https://pubmed.ncbi.nlm.nih.gov/1) or"
        " [123](https://pubmed.ncbi.nlm.nih.gov/123), within square brackets."
    )
