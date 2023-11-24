import os
import random
import urllib.parse
from pathlib import Path
from typing import Optional

import requests
import typer
import ujson as json
from Bio import Entrez
from lxml import html
from rich import print
from rich.progress import track
from typing_extensions import Annotated

from topical import nlm, util

# Setup Entrez API
Entrez.email = os.environ.get("ENTREZ_EMAIL")
Entrez.api_key = os.environ.get("ENTREZ_API_KEY")

random.seed(42)


def main(
    input_dir: Annotated[str, typer.Argument(help="Path to the directory containing the topic pages as JSON files")],
    output_dir: Annotated[
        Optional[str], typer.Option(help="Path to the directory to save the output. Defaults to --input-dir")
    ] = None,
) -> None:
    """Format the task 1 and task 2 inputs of the human evaluation for easy copy-pasting into the Google Sheet.

    Note, you will likely want to provide your Entrez email and API key via the ENTREZ_EMAIL and ENTREZ_API_KEY env vars.
    """
    if output_dir is None:
        output_dir = input_dir

    task_1_formatted_anns, task_2_formatted_anns = {}, {}

    input_dir = Path(input_dir)
    input_fps = list(Path(input_dir).glob("*.json"))
    for fp in track(input_fps, description="Formatting inputs"):
        topic_page = json.loads(fp.read_text().strip())

        # Parse the topic page
        entity = topic_page["entity"].strip()
        body = topic_page["topic_page"].strip()
        definition, main_content, future_directions = body.split("\n\n")
        pubmed_query = topic_page["pubmed_query"].strip()

        # Randomly select a PMID to evaluate for use in task 2
        definition_pmids = util.get_pmids_from_text(definition)
        main_content_pmids = util.get_pmids_from_text(main_content)
        future_directions_pmids = util.get_pmids_from_text(future_directions)

        pmid = random.choice(definition_pmids + main_content_pmids + future_directions_pmids)
        if pmid in definition_pmids:
            context = definition
        elif pmid in main_content_pmids:
            context = body
        else:
            context = future_directions
        context = util.sanitize_text(context)

        # Get the MeSH terms for the topic based on its title
        url_encoded_title = urllib.parse.quote(entity)
        r = requests.get(
            f"https://id.nlm.nih.gov/mesh/lookup/descriptor?label={url_encoded_title}&match=exact&limit=1"
        )
        r.raise_for_status()
        mesh_url = r.json()[0]["resource"]
        mesh_id = mesh_url.split("/")[-1]

        # Get the title and abstract of the article
        article = nlm.efetch(pmid, db="pubmed", rettype="abstract", retmax=1, use_cache=True)["PubmedArticle"][0][
            "MedlineCitation"
        ]
        title = util.sanitize_text(article["Article"]["ArticleTitle"])
        abstract = article["Article"].get("Abstract", {})
        abstract = [section.strip() for section in abstract.get("AbstractText", "") if section]
        # Don't include the supplementary material section. Note that this check is not perfect, as
        # sometimes the supplementary material section is not the last section.
        if abstract and abstract[-1].startswith("Supplementary"):
            abstract = abstract[:-1]
        abstract = util.sanitize_text(" ".join(abstract))
        # remove any html markup, which sometimes exists in abstract text downloaded from PubMed
        if title:
            title = html.fromstring(title).text_content()
        if abstract:
            abstract = html.fromstring(abstract).text_content()

        # Replace the PMID with a markdown formatted hyperlink to PubMed
        pmid = util.replace_pmids_with_markdown_link(f"[{pmid}]")

        # Format the input for easy copy-pasting into the Google Sheet
        task_1_formatted_anns[
            mesh_id
        ] = f"{entity}\n{definition}\n{main_content}\n{future_directions}\n{pubmed_query}\n{mesh_url}\n"
        task_2_formatted_anns[
            mesh_id
        ] = f"{entity}\n{pmid}\n{title}\n{abstract}\n{context}\n{pubmed_query}\n{mesh_url}\n"

    # Sort the inputs by MeSH ID so the output is deterministic
    task_1_formatted_anns = {mesh_id: task_1_formatted_anns[mesh_id] for mesh_id in sorted(task_1_formatted_anns)}
    task_2_formatted_anns = {mesh_id: task_2_formatted_anns[mesh_id] for mesh_id in sorted(task_2_formatted_anns)}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_1.txt").write_text("\n".join(task_1_formatted_anns.values()))
    (output_dir / "task_2.txt").write_text("\n".join(task_2_formatted_anns.values()))
    print(f"Formatted inputs written to '{output_dir}'")


if __name__ == "__main__":
    typer.run(main)
