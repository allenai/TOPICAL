import os
import random
import re
import urllib.parse
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
from lxml import html
from sentence_transformers import util as sbert_util

from topical import nlm, util

# Clustering settings
EMBEDDING_MODEL = "hkunlp/instructor-{model_size}"  # The model used to embed evidence for clustering
MIN_ARTICLES_TO_CLUSTER = 100  # If there are less than this number of articles, skip clustering
MIN_ARTICLES_TO_AVOID_BACKOFF = 2  # If there are less than this number of clusters, perform backoff
BACKOFF_THRESHOLD = 0.02  # The amount to reduce the cosine similarity by during backoff

# Prompt settings
MODEL_MAX_LEN = 8000  # TODO: Artifically low because some prompt tokens are not accounted for in the total length
TOPIC_PAGE_MAX_LEN = 512

# Debug settings
DEBUG_NUM_CLUSTERS = 5  # Number of clusters to display if debug is True
DEBUG_CLUSTER_SIZE = 5  # Size of each cluster to display if debug is True

# Path to the prompts
PROMPT_DIR = Path(__file__).parents[1] / "prompts"

# RegEx patterns
cited_pmid = re.compile(r"\[(\d+)\]")  # identify citations in model generated topic page

# Setup Entrez API
Entrez.email = os.environ.get("ENTREZ_EMAIL")
Entrez.api_key = os.environ.get("ENTREZ_API_KEY")

random.seed(42)


@st.cache_data(show_spinner=False)
def plot_publications_per_year(records: DictionaryElement, end_year: str) -> dict[str, int]:
    """Plots a histogram of publications per year (up to an including end_year) in records and returns the counts."""
    pub_years = [int(nlm.get_year_from_medline_date(record["PubDate"])) for record in records]
    year_counts = {year: pub_years.count(year) for year in sorted(set(pub_years)) if int(year) <= int(end_year)}
    st.bar_chart(
        {"Year": list(year_counts.keys()), "Number of Publications": list(year_counts.values())},
        x="Year",
        y="Number of Publications",
        color="#255ed3",
        use_container_width=True,
    )
    return year_counts


@st.cache_data(show_spinner=False)
def preprocess_pubmed_articles(records: DictionaryElement) -> list[dict[str, str | dict[str, str]]]:
    """Performs basic pre-processing of the articles in records and returns them as a list of dictionaries."""
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

        # remove any html markup, which sometimes exists in abstract text downloaded from PubMed
        if title:
            title = html.fromstring(title).text_content()
        if abstract:
            abstract = html.fromstring(abstract).text_content()

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
def load_encoder(model_size: str = "large", quantize: bool = True):
    """Load an Intructor-based text encoder for embedding titles and abstracts."""
    model = INSTRUCTOR(EMBEDDING_MODEL.format(model_size=model_size))
    if not torch.cuda.is_available() and quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model


@st.cache_data(show_spinner="Embedding titles and abstracts (this could take up to a few minutes)...")
def embed_evidence(articles: list[str], _encoder, _batch_size: int = 64):
    """Jointly embed the titles and abstracts in articles for the given encoder."""
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
    """Load an OpenAI tiktokenizer."""
    return tiktoken.encoding_for_model(model_choice)


@st.cache_data
def format_evidence(articles: list[dict[str, str]], clusters: list[list[int]], _tokenizer, prompt_len: int) -> str:
    """Format the supporting literature as a string for inclusion in the prompt."""
    curr_evidence_len = 0
    evidence = [[] for _ in clusters]
    max_length_reached = False

    def _format_article(article: dict[str, str]) -> str | None:
        id_, pub_date, title, abstract = article["id"], article["pub_date"], article["title"], article["abstract"]

        # Take the first 3 and last 2 sentences of the abstract, which we assume are the most informative
        sents = nltk.tokenize.sent_tokenize(abstract)
        if len(sents) > 5:
            abstract = f"{' '.join(sents[:3])} [TRUNCATED] {' '.join(sents[-2:])}"
        else:
            abstract = " ".join(sents)

        date = pub_date["month"].strip() if pub_date["month"] else ""
        date += f" {pub_date['day'].strip()}" if pub_date["day"] else ""
        date += f" {pub_date['year'].strip()}" if pub_date["year"] else ""

        formatted_article = f"PMID: {id_} Publication Date: {date} Title: {title} Abstract: {abstract}"
        formatted_article = util.sanitize_text(formatted_article)
        return formatted_article

    # Compute a weight for each cluster based on the sqrt of its size. If there is remaining space in the prompt
    # after including all centroids, we will sample from the clusters with probability proportional to these weights
    weights = [sqrt(len(cluster)) for cluster in clusters]

    # TODO: Code in this section is heavily duplicated, refactor

    # First, try adding as many centroids as possible
    for cluster_idx in range(len(clusters)):
        article_idx = clusters[cluster_idx].pop(0)
        formatted_article = _format_article(articles[article_idx])
        if not formatted_article:
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
        if not formatted_article:
            continue
        formatted_article_len = len(_tokenizer.encode(formatted_article))
        if (curr_evidence_len + formatted_article_len) < (MODEL_MAX_LEN - prompt_len - TOPIC_PAGE_MAX_LEN):
            curr_evidence_len += formatted_article_len
            evidence[cluster_idx].append(formatted_article)
        else:
            max_length_reached = True

    evidence = [cluster for cluster in evidence if cluster]
    num_abstracts = sum(len(cluster) for cluster in evidence)

    # TODO: This extra text is not accounted for in the total length, so it's possible to go over the max length
    return "\n\n".join(f"Cluster {i + 1}\n" + "\n".join(cluster) for i, cluster in enumerate(evidence))


@st.cache_resource(show_spinner=False)
def load_llm(model_choice: str):
    """Load an OpenAI large language model."""
    return guidance.llms.OpenAI(model_choice)


@st.cache_data(show_spinner=False)
def replace_pmid_with_markdown_link(text: str) -> str:
    """Replace any PubMed IDs cited by the model with a markdown formatted hyperlink to PubMed."""
    return cited_pmid.sub(r"[\1](https://pubmed.ncbi.nlm.nih.gov/\1)", text)


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
        openai_api_key = st.text_input(
            "Enter your API Key:",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Your key for the OpenAI API.",
        )

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
        embedder_size = st.selectbox(
            "Embedder model size",
            options=["base", "large", "xl"],
            index=1,
            format_func=lambda size: f"{size} (recommended)" if size == "large" else size,
            help=(
                "The size of the encoder used to produce embeddings for clustering. Larger models tend to perform"
                " better but are significantly slower."
            ),
        )
        threshold = st.slider(
            "Cosine similarity threshold",
            min_value=0.9,
            max_value=1.0,
            value=0.96,
            help="Titles and abstracts with a cosine similarity >= this value will be clustered",
        )
        min_community_size = st.slider(
            "Minimum cluster size",
            min_value=1,
            max_value=10,
            value=5,
            help="Clusters smaller than this value will be discarded",
        )

        "---"
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
    ).strip()

    if not openai_api_key:
        st.warning("Please provide an OpenAI API key in the sidebar.", icon="🔑")
        st.stop()

    # Any key provided in the sidebar will override the environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

    if st.button("Generate Topic Page", type="primary"):
        with st.status(f"Generating topic page for _{entity}_...", expanded=debug) as status:
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
                    # Only return papers with abstracts available
                    preprocessor=lambda x: x.strip() + " AND hasabstract[All Fields]",
                )
            )
            pmids = esearch_results["IdList"]
            total_publications = int(esearch_results["Count"])
            pubmed_query_linkout = (
                f'https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote_plus(esearch_results["QueryTranslation"])}'
            )
            st.success(
                f"Found {len(pmids)} papers using query:"
                f" [`{esearch_results['QueryTranslation']}`]({pubmed_query_linkout})",
                icon="🔎",
            )

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

            if len(articles) >= MIN_ARTICLES_TO_CLUSTER:
                st.write("Clustering evidence...")
                encoder = load_encoder(model_size=embedder_size)
                embeddings = embed_evidence(articles, encoder)
                clusters = sbert_util.community_detection(
                    embeddings, min_community_size=min_community_size, threshold=threshold
                )

                # Try lowering the threshold if no clusters are found
                backoff_threshold = threshold
                while len(clusters) < MIN_ARTICLES_TO_AVOID_BACKOFF and backoff_threshold > 0.9:
                    backoff_threshold -= BACKOFF_THRESHOLD
                    st.warning(
                        f"No clusters found with threshold {threshold},"
                        f" trying a lower threshold ({backoff_threshold:.2f})...",
                        icon="⚠️",
                    )
                    clusters = sbert_util.community_detection(
                        embeddings, min_community_size=min_community_size, threshold=backoff_threshold
                    )

                if not clusters:
                    st.warning(
                        "No clusters found for your query. Try relaxing the clustering criteria, or adding terms"
                        " to the query.",
                        icon="😔",
                    )
                    st.stop()

                DEBUG_NUM_CLUSTERS = len(max(clusters, key=len))
                min_cluster_size = len(min(clusters, key=len))
                avg_cluster_size = sum(len(cluster) for cluster in clusters) / len(clusters)
                st.success(
                    f"Found {len(clusters)} clusters (max size: {DEBUG_NUM_CLUSTERS}, min size:"
                    f" {min_cluster_size}, mean size: {avg_cluster_size:.1f}) matching your query",
                    icon="✅",
                )

                if debug:
                    st.warning(
                        f"The first {DEBUG_CLUSTER_SIZE} titles of the first {DEBUG_NUM_CLUSTERS} clusters, useful for"
                        " spot checking the clustering. Clusters are sorted by decreasing size. The first"
                        " element of each cluster is its centroid.\n"
                        + "\n".join(
                            f"\n__Cluster {i + 1}__ (size: {len(cluster)}):\n"
                            + "\n".join(f"- {articles[idx]['title']}" for idx in cluster[:DEBUG_CLUSTER_SIZE])
                            for i, cluster in enumerate(clusters[:DEBUG_NUM_CLUSTERS])
                        )
                    )
            else:
                st.warning(
                    f"Less than {MIN_ARTICLES_TO_CLUSTER} total publications found, skipping clustering...", icon="⚠️"
                )
                clusters = [[i] for i in range(len(articles))]

            # Design a prompt to get GPT to generate topic pages
            system_prompt = (PROMPT_DIR / "system.txt").read_text().format(domain="biomedical").strip()
            instructions_prompt = (
                (PROMPT_DIR / "instructions.txt")
                .read_text()
                .format(
                    domain="biomedical",
                    end_year=end_year,
                    min_articles_to_cluster=MIN_ARTICLES_TO_CLUSTER,
                )
            ).strip()
            how_to_cite_prompt = (
                (PROMPT_DIR / "how_to_cite.txt").read_text().format(id_database="PubMed", id_type="PMID")
            ).strip()
            topic_page_prompt = (PROMPT_DIR / "topic_page.txt").read_text().strip()

            llm = load_llm(model_choice)
            prompt = guidance(
                """
{{#system~}}
{{system_prompt}}
{{~/system}}

{{#user~}}
# INSTRUCTIONS

{{instructions_prompt}}

## HOW TO CITE YOUR CLAIMS

{{how_to_cite_prompt}}

# ENTITY OR CONCEPT

Canonicalized entity name: {{canonicalized_entity_name}}
Publications per year: {{publications_per_year}}
Total number of publications: {{total_publications}}
Supporting literature:

{{evidence}}

# TOPIC PAGE

{{topic_page_prompt}}
{{~/user}}

{{#assistant~}}
{{gen 'topic_page' temperature=0.0 max_tokens=512}}
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

            # TODO: Is there no way to format the entire prompt and get its length together?
            prompt_len = (
                len(tokenizer.encode(prompt.text))
                + len(tokenizer.encode(publications_per_year))
                + len(tokenizer.encode(str(total_publications)))
                + len(tokenizer.encode(system_prompt))
                + len(tokenizer.encode(instructions_prompt))
                + len(tokenizer.encode(how_to_cite_prompt))
                + len(tokenizer.encode(topic_page_prompt))
                + len(tokenizer.encode(entity))
            )
            evidence = format_evidence(articles, clusters, tokenizer, prompt_len)

            if debug:
                st.warning(f"__Evidence__:\n\n{evidence}")
                st.warning(f"__Prompt__:\n\n{prompt.text}")

            with st.spinner("Prompting model..."):
                response = prompt(
                    system_prompt=system_prompt,
                    instructions_prompt=instructions_prompt,
                    how_to_cite_prompt=how_to_cite_prompt,
                    topic_page_prompt=topic_page_prompt,
                    canonicalized_entity_name=entity,
                    publications_per_year=publications_per_year,
                    total_publications=total_publications,
                    evidence=evidence,
                    llm=llm,
                )

            status.update(label=f"Generated topic page for '_{entity}_'")

        if response:
            topic_page = f'### {entity}\n\n{response["topic_page"].strip()}'
            topic_page = replace_pmid_with_markdown_link(topic_page)
            st.write(topic_page)

            # Allow user to download raw markdown formatted topic page
            def prepare_topic_page_for_download(topic_page: str) -> str:
                pubmed_query = f"PubMed Query: [{esearch_results['QueryTranslation']}]({pubmed_query_linkout})"
                return f"{topic_page}\n\n---\n\n{pubmed_query}"

            st.download_button(
                "Download topic page",
                data=prepare_topic_page_for_download(topic_page),
                file_name=f'{util.sanitize_text(entity, lowercase=True).replace(" ", "_")}.md',
            )


if __name__ == "__main__":
    main()
