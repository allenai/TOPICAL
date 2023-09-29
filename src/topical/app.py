import os
import random
from datetime import datetime
from math import sqrt
from pathlib import Path

import guidance
import nltk
import streamlit as st
import tiktoken
import torch
from Bio import Entrez
from Bio.Entrez.Parser import DictionaryElement
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import util as sbert_util

from topical import nlm, util

MIN_SENT_LEN = 10  # minumum number of whitepsace tokens needed to retain an abstract sentence
MIN_SENTS = 5  # minimum number of sentences needed to retain an abstract

# TODO: This is artifically lowered because some tokens in the prompt are not accounted for in the total length
MODEL_MAX_LEN = 8000
TOPIC_PAGE_MAX_LEN = 768

# Debug settings
MAX_CLUSTER_SIZE = 5
MAX_CLUSTERS = 5

PROMPT_DIR = Path(__file__).parents[1] / "prompts"


# TODO: IMPORTANT TO REMOVE, JUST HERE FOR TESTING
Entrez.email = "johnmgiorgi@gmail.com"
Entrez.api_key = "bcc7945770d76bca7aa1742fd723b320dc08"
os.environ["OPENAI_API_KEY"] = "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"


random.seed(42)


@st.cache_data(show_spinner=False)
def plot_publications_per_year(records: DictionaryElement, end_year: str) -> dict[str, int]:
    pub_years = [int(nlm.get_year_from_medline_date(record["PubDate"])) for record in records]
    year_counts = {year: pub_years.count(year) for year in sorted(set(pub_years)) if int(year) <= int(end_year)}
    st.bar_chart(
        {"Year": list(year_counts.keys()), "Number of Publications": list(year_counts.values())},
        x="Year",
        y="Number of Publications",
        color="#ffbb00",
        use_container_width=True,
    )
    return year_counts


@st.cache_data(show_spinner=False)
def preprocess_pubmed_articles(records: DictionaryElement) -> list[dict[str, str | dict[str, str]]]:
    articles = []
    for article in records["PubmedArticle"]:
        medline_citation = article["MedlineCitation"]

        pmid = medline_citation["PMID"]
        title = util.sanitize_text(medline_citation["Article"]["ArticleTitle"])

        # Format the abstract
        abstract = medline_citation["Article"].get("Abstract", {})
        abstract = [section.strip() for section in abstract.get("AbstractText", "") if section]
        # Don't include the supplementary material section. Note that this check is not perfect, as
        # sometimes the supplementary material section is not the last section.
        if abstract and abstract[-1].startswith("Supplementary"):
            abstract = abstract[:-1]
        abstract = util.sanitize_text(" ".join(abstract))

        # Get the pubdate
        pubdate = medline_citation["Article"]["Journal"]["JournalIssue"]["PubDate"]
        if pubdate.get("MedlineDate") is not None:
            year = nlm.get_year_from_medline_date(pubdate["MedlineDate"].strip())
            month, day = None, None
        else:
            year = pubdate.get("Year")
            month = pubdate.get("Month")
            day = pubdate.get("Day")

        articles.append(
            {
                "id": pmid,
                "pub_date": {"year": year, "month": month, "day": day},
                "title": title,
                "abstract": abstract,
            }
        )
    return articles


@st.cache_resource(show_spinner="Loading encoder...")
def load_encoder(model: str = "hkunlp/instructor-base", quantize: bool = True):
    model = INSTRUCTOR(model)
    if not torch.cuda.is_available() and quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model


@st.cache_data(show_spinner="Embedding titles and abstracts (this could take up to a few minutes)...")
def embed_evidence(articles: list[str], _encoder, _batch_size: int = 64):
    instruction = "Represent the Biomedical abstract for clustering"
    return _encoder.encode(
        [[instruction, f"Title: {article['title']} Abstract: {article['abstract']}"] for article in articles],
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        batch_size=_batch_size,
    )


@st.cache_resource(show_spinner=False)
def load_tiktokenizer(model_choice: str):
    return tiktoken.encoding_for_model(model_choice)


@st.cache_data
def format_evidence(articles: list[dict[str, str]], clusters: list[list[int]], _tokenizer, prompt_len: int) -> str:
    curr_evidence_len = 0
    evidence = [[] for _ in clusters]
    max_length_reached = False

    def _format_article(article: dict[str, str]) -> str | None:
        id_, pub_date, title, abstract = article["id"], article["pub_date"], article["title"], article["abstract"]
        sents = nltk.tokenize.sent_tokenize(abstract)
        # Drop any sentences under the minimum length, and any abstract with less than the minimum number of sents
        sents = [sent for sent in sents if len(sent.split()) > MIN_SENT_LEN]
        if len(sents) < MIN_SENTS:
            return None
        # Take the first 3 and last 2 sentences of the abstract, which we assume are the most informative
        if len(sents) > 5:
            formatted_article = f"{' '.join(sents[:3])} [TRUNCATED] {' '.join(sents[-2:])}"
        else:
            formatted_article = " ".join(sents)
        date = pub_date["month"].strip() if pub_date["month"] else ""
        date += f" {pub_date['day'].strip()}" if pub_date["day"] else ""
        date += f" {pub_date['year'].strip()}" if pub_date["year"] else ""
        formatted_article = f"PMID: {id_} Publication Date: {date} Title: {title} Abstract: {formatted_article}"
        return util.sanitize_text(formatted_article)

    # Compute a weight for each cluster based on the sqrt of its size. If there is remaining space in the prompt
    # after including all centroids, we will sample from the clusters with probability proportional to these weights
    weights = [sqrt(len(cluster)) for cluster in clusters]

    # TODO: Code in this section is heavily duplicated, refactor

    # First, try adding as many centroids as possible
    for cluster_idx in range(len(clusters)):
        article_idx = clusters[cluster_idx].pop(0)
        formatted_article = _format_article(articles[article_idx])
        if formatted_article is None:
            continue
        formatted_article_len = len(_tokenizer.encode(formatted_article))
        if (curr_evidence_len + formatted_article_len) < (MODEL_MAX_LEN - prompt_len - TOPIC_PAGE_MAX_LEN):
            curr_evidence_len += formatted_article_len
            evidence[cluster_idx].append(formatted_article)
        else:
            max_length_reached = True
            break

    # If there is remaining space in the prompt, sample from the clusters with probability proportional to their size
    while not max_length_reached and any(clusters):
        cluster_idx = random.choices(range(len(clusters)), weights=weights, k=1)[0]
        if not clusters[cluster_idx]:
            continue
        article_idx = clusters[cluster_idx].pop(0)
        formatted_article = _format_article(articles[article_idx])
        if formatted_article is None:
            continue
        formatted_article_len = len(_tokenizer.encode(formatted_article))
        if (curr_evidence_len + formatted_article_len) < (MODEL_MAX_LEN - prompt_len - TOPIC_PAGE_MAX_LEN):
            curr_evidence_len += formatted_article_len
            evidence[cluster_idx].append(formatted_article)
        else:
            max_length_reached = True

    evidence = [cluster for cluster in evidence if cluster]
    # TODO: This extra text is not accounted for in the total length, so it's possible to go over the max length
    return "\n\n".join(f"Cluster {i + 1}:\n" + "\n".join(cluster) for i, cluster in enumerate(evidence))


@st.cache_resource(show_spinner=False)
def load_llm(model_choice: str):
    return guidance.llms.OpenAI(model_choice)


def main():
    with st.sidebar:
        st.image("https://prior.allenai.org/assets/logos/ai2-logo-header.png", use_column_width=True)

        st.header("Settings")
        st.write(
            "Please provide an __OpenAI API key__. Modifying additional settings is optional, reasonable defaults"
            " are provided."
        )

        st.subheader("OpenAI API")
        model_choice = st.text_input(
            "Choose a model:",
            value="gpt-4-0613",
            help="Any valid model name for the OpenAI API. It is strongly recommended to use GPT-4.",
        )
        openai_api_key = st.text_input("Enter your API Key:", type="password", help="Your key for the OpenAI API.")

        "---"

        st.subheader("Search")
        current_year = datetime.now().year
        end_year = str(
            st.number_input(
                "End year",
                min_value=0000,
                max_value=current_year,
                value=current_year,
                help="Include papers published up to and including this year in the search results.",
            )
        )
        retmax = st.number_input(
            "Maximum papers to consider",
            min_value=1,
            max_value=10_000,
            value=10_000,
            step=100,
            help="Determines the maximium number of papers to consider, starting from most to least relevant.",
        )

        st.subheader("Evidence Clustering")
        threshold = st.slider(
            "Titles and abstracts with a cosine similarity _greater than or equal to_ this value will be clustered",
            min_value=0.9,
            max_value=1.0,
            value=0.96,
        )
        min_community_size = st.slider(
            "Clusters smaller than this value will be discarded",
            min_value=1,
            max_value=10,
            value=5,
        )

        st.subheader("Debug")
        debug = st.toggle("Toggle for debug mode, which displays additional info", value=False)

    st.write("# 🪄📄 TOPICAL: TOPIC pages AutomagicaLly")
    st.write(
        "This demo generates the ability to produce topic pages for biomedical entities and concepts automatically."
        " Enter a __entity__ or __concept__ of interest below, and a topic page will be generated."
    )
    st.caption('An example is provided for you (just hit "__Generate Topic Page__"!)')
    with st.expander("Search tips 💡"):
        st.write(
            "For concrete entities that are likely to be mentioned precisely in the the literature"
            ' (e.g. _"beta-D-Galactoside alpha 2-6-Sialyltransferase"_), a search strategy that works well is to'
            " look for these mentions in paper titles: `<entity>[title] OR <synonym>[title]`. If the entity exists"
            " in the [MeSH ontology](https://meshb.nlm.nih.gov/), try adding: `OR <entity>[MeSH Major Topic]`."
        )

        st.write(
            "For more generic entities that are unlikely to be mentioned precisely"
            ' (e.g. _"Single-Cell Gene Expression Analysis"_), try providing the entity by itself.'
            " [PubMed](https://pubmed.ncbi.nlm.nih.gov/) will automatically expand your query to search for"
            " synonyms and related MeSH terms."
        )

    query = st.text_input(
        "Search query",
        value="microplastic[title] OR microplastics[title] OR Microplastics[MeSH Major Topic]",
        help=(
            "Enter a search query for your entity of interest. This supports the full syntax of the [PubMed"
            ' Advanced Search Builder](https://pubmed.ncbi.nlm.nih.gov/advanced/). See "_Search tips_" for more help.'
        ),
    )
    entity = st.text_input(
        "Canonicalized name",
        value="Microplastics",
        help=(
            "Enter a canonicalized name for the entity. This will not be used to query PubMed, but it can help keep"
            " the model on track when generating topic pages, especially in cases where the entity has multiple,"
            " potentially ambiguous names."
        ),
    )

    if st.button("Generate Topic Page", type="primary"):
        with st.status(f"Generating topic page for '_{entity}_'...", expanded=debug) as status:
            if debug:
                st.info("Debug mode is enabled, additional information will be displayed in blue.", icon="🐛")

            st.write("Querying PubMed...")
            esearch_results = next(
                nlm.esearch(
                    query,
                    db="pubmed",
                    retmax=10_0000,
                    sort="relevance",
                    mindate="0000",
                    maxdate=end_year,
                    use_cache=True,
                )
            )
            pmids = esearch_results["IdList"]
            total_publications = int(esearch_results["Count"])
            st.success(f"Found {len(pmids)} papers using query: `{query}`", icon="🔎")
            if debug:
                st.info(f'__Query translation__: `{esearch_results["QueryTranslation"]}`')

            st.write("Fetching publication years...")
            records = nlm.efetch(",".join(pmids), db="pubmed", rettype="docsum", use_cache=True)
            st.success("Done, publications by year plotted below", icon="📊")
            year_counts = plot_publications_per_year(records, end_year=end_year)

            if len(pmids) > retmax:
                st.warning(
                    f"Maximum papers to consider set to {retmax} but {len(pmids)} papers were found for your query."
                    f" Will only use the top {retmax} papers as input",
                    icon="⚠️",
                )
                pmids = pmids[:retmax]

            st.write("Downloading titles and abstracts...")
            records = nlm.efetch(",".join(pmids), db="pubmed", rettype="abstract", retmax=retmax, use_cache=True)
            st.success(f"Downloaded titles and abstracts for {len(records['PubmedArticle'])} papers", icon="⬇️")

            st.write("Preprocessing titles and abstracts...")
            articles = preprocess_pubmed_articles(records)
            st.success(f"Done {len(articles)} abstracts preprocessed", icon="⚙️")

            # This is a large object, so we delete it to free up memory
            del records

            if not articles:
                st.warning(
                    "No abstracts found for your query (it may not be a valid biomedical entity). Try a different"
                    " query?",
                    icon="😔",
                )
                st.stop()

            if len(articles) >= 100:
                st.write("Clustering evidence...")
                encoder = load_encoder()
                embeddings = embed_evidence(articles, encoder)
                clusters = sbert_util.community_detection(
                    embeddings, min_community_size=min_community_size, threshold=threshold
                )
                max_cluster_size = len(max(clusters, key=len))
                min_cluster_size = len(min(clusters, key=len))
                avg_cluster_size = sum(len(cluster) for cluster in clusters) / len(clusters)
                st.success(
                    f"Found {len(clusters)} clusters (max size: {max_cluster_size}, min size:"
                    f" {min_cluster_size}, mean size: {avg_cluster_size:.1f}) matching your query",
                    icon="✅",
                )

                if debug:
                    st.warning(
                        f"The first {MAX_CLUSTER_SIZE} titles of the first {MAX_CLUSTERS} clusters, useful for"
                        " spot checking the clustering. Clusters are sorted by decreasing size. The first"
                        " element of each cluster is its centroid.\n"
                        + "\n".join(
                            f"\n__Cluster {i + 1}__ (size: {len(cluster)}):\n"
                            + "\n".join(f"- {articles[idx]['title']}" for idx in cluster[:MAX_CLUSTER_SIZE])
                            for i, cluster in enumerate(clusters[:MAX_CLUSTERS])
                        )
                    )
            else:
                st.warning("Less than 100 total publications found, skipping clustering...", icon="⚠️")
                clusters = [[i] for i in range(len(articles))]

            if not clusters:
                st.warning(
                    "No clusters found for your query. Try relaxing the clustering criteria, or adding terms"
                    " to the query.",
                    icon="😔",
                )
                st.stop()

            # Design a prompt to get GPT to generate topic pages
            system_prompt = (PROMPT_DIR / "system.txt").read_text().format(domain="biomedical")
            instructions_prompt = (
                (PROMPT_DIR / "instructions.txt")
                .read_text()
                .format(
                    end_year=end_year,
                    id_type="PMID",
                    id_database="PubMed",
                    id_url="https://pubmed.ncbi.nlm.nih.gov",
                )
            )
            topic_page_prompt = (PROMPT_DIR / "topic_page.txt").read_text()

            llm = load_llm(model_choice)
            prompt = guidance(
                """
{{#system~}}
{{system_prompt}}
{{~/system}}

{{#user~}}
INSTRUCTIONS

{{instructions_prompt}}

ENTITY OR CONCEPT

Canonicalized entity name: {{canonicalized_entity_name}}
Publications per year: {{publications_per_year}}
Total number of publications: {{total_publications}}
Supporting literature:
{{evidence}}

TOPIC PAGE

{{topic_page_prompt}}
{{~/user}}

{{#assistant~}}
{{gen 'topic_page' temperature=0.0 max_tokens=768}}
{{~/assistant}}
        """,
                llm=llm,
                silent=True,
                stream=False,
                caching=True,
            )

            # Format evidence
            tokenizer = load_tiktokenizer(model_choice)
            publications_per_year = ", ".join(
                f"{year}: {count}" for year, count in year_counts.items() if int(year) <= int(end_year)
            )
            prompt_len = (
                len(tokenizer.encode(prompt.text))
                + len(tokenizer.encode(publications_per_year))
                + len(tokenizer.encode(str(total_publications)))
                + len(tokenizer.encode(system_prompt))
                + len(tokenizer.encode(instructions_prompt))
                + len(tokenizer.encode(topic_page_prompt))
                + len(tokenizer.encode(entity))
            )
            evidence = format_evidence(articles, clusters, tokenizer, prompt_len)

            if debug:
                st.warning(f"__Evidence__:\n\n{evidence}")
                st.warning(f"__Prompt__:\n\n{prompt.text}")

            if not evidence:
                st.warning(
                    f"No valid abstracts after filtering for a minimum sentence length of {MIN_SENT_LEN} whitespace"
                    f" tokens and minimum number of sentences of {MIN_SENTS}.",
                    icon="😔",
                )
                st.stop()

            with st.spinner("Prompting model..."):
                response = prompt(
                    system_prompt=system_prompt,
                    instructions_prompt=instructions_prompt,
                    topic_page_prompt=topic_page_prompt,
                    canonicalized_entity_name=entity,
                    publications_per_year=publications_per_year,
                    total_publications=total_publications,
                    evidence=evidence,
                    llm=llm,
                )

            status.update(label=f"Generated topic page for '_{entity}_'")

        if response:
            import re

            def strip_markdown(text):
                # Remove inline links [text](url)
                text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

                # Remove bold and italics
                text = re.sub(r"__([^\__]+)__", r"\1", text)  # double underscores
                text = re.sub(r"\*\*([^\*\*]+)\*\*", r"\1", text)  # double asterisks
                text = re.sub(r"_([^_]+)_", r"\1", text)  # single underscores
                text = re.sub(r"\*([^\*]+)\*", r"\1", text)  # single asterisks

                # Remove other potential markdown elements as needed
                # For example, headers, images, blockquotes, etc.
                # Add more regex patterns here if needed

                return text

            topic_page = f'### {entity.strip()}\n{response["topic_page"]}'
            st.write(topic_page)
            st.download_button("Download topic page", data=strip_markdown(topic_page), file_name="topic_page.md")


if __name__ == "__main__":
    main()
