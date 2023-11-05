from topical import nlm


def test_get_year_from_medline_date():
    # Examples taken from: https://www.nlm.nih.gov/bsd/licensee/elements_descriptions.html#pubdate
    expected = "2000"

    medline_date = "1999 Dec-2000 Jan"
    actual = nlm.get_year_from_medline_date(medline_date)
    assert actual == expected

    medline_date = "2000 Spring"
    expected = "2000"
    actual = nlm.get_year_from_medline_date(medline_date)
    assert actual == expected

    medline_date = "2000 Spring-Summer"
    expected = "2000"
    actual = nlm.get_year_from_medline_date(medline_date)
    assert actual == expected

    medline_date = "2000 Nov-Dec"
    expected = "2000"
    actual = nlm.get_year_from_medline_date(medline_date)
    assert actual == expected

    medline_date = "2000 Dec 23- 30"
    expected = "2000"
    actual = nlm.get_year_from_medline_date(medline_date)
    assert actual == expected


def test_efetch():
    # Sanity check that we can call the esearch API and get the expected count
    pmid = "26209480"
    result = nlm.efetch(pmid, db="pubmed", rettype="abstract")
    assert result["PubmedArticle"][0]["MedlineCitation"]["PMID"] == pmid


def test_esearch():
    # Sanity check that we can call the esearch API and get the expected count
    result = next(nlm.esearch("covid-19", db="pubmed", retmax=1))
    assert int(result["Count"]) > 36_000
