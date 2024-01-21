import random
from pathlib import Path

import typer
import ujson as json
from typing_extensions import Annotated

from topical import nlm

random.seed(42)


def main(
    output_fp: Annotated[
        str, typer.Argument(help="Filepath to save the JSON lines file containing the sampled entities")
    ],
    n: Annotated[int, typer.Option(help="Number of entities to sample. Defaults to all possible entities")] = None,
    start_year: Annotated[
        int, typer.Option(help="Restrict the sampling to entities added to MeSH in this year or later (inclusive)")
    ] = 2013,
    end_year: Annotated[
        int, typer.Option(help="Restrict the sampling to entities added to MeSH in this year or earlier (inclusive)")
    ] = 2023,
    min_tree_depth: Annotated[
        int,
        typer.Option(
            help="Restrict the sampling to entities with a maximum tree depth equal to or greater than this value"
        ),
    ] = 7,
) -> None:
    """Sample n MeSH descriptors between start_year and end_year (inclusive) with a min_tree_depth"""
    descriptors = {
        descr.ui: descr
        for descr in nlm.fetch_mesh()
        if int(descr.date_created.year) >= start_year
        and int(descr.date_created.year) <= end_year
        and descr.max_tree_depth() >= min_tree_depth
    }
    print(
        f":white_check_mark: Loaded {len(descriptors)} MeSH descriptors for years {start_year}-{end_year}"
        f" (inclusive) with a min_tree_depth of {min_tree_depth}."
    )

    # Sample n entities, or, simply shuffle the keys if n is None
    if n is None:
        n = len(descriptors)
        print(f":warning: No number of entities specified. Sampling from all possible entities ({n}).")
    if n > len(descriptors):
        print(
            f":warning: Requested {n} entities, but only {len(descriptors)} entities are available."
            f" Sampling {len(descriptors)} entities."
        )
        n = len(descriptors)
    mesh_ids = sorted(random.sample(descriptors, k=n))
    sampled_entities = [{"mesh_id": mesh_id, "name": descriptors[mesh_id].name} for mesh_id in mesh_ids]

    # Write the sampled entities to disk
    output_fp = Path(output_fp)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text("\n".join(json.dumps(ent) for ent in sampled_entities))


if __name__ == "__main__":
    typer.run(main)
