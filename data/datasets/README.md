# Classification datasets

Each dataset is a binary question about one native class of a public dataset. `python -m scripts.build_datasets`
rebuilds the case files into this directory from the pinned public sources; `--check` compares each rebuilt file
against `checksums.json`. See the "Data and licences" section of the top-level README for the source, licence and
pinned revision of every dataset.

Shipped here:

- `arxiv_abstracts/cases.jsonl`: the arXiv abstracts used in the paper. They come from the live arXiv API, which
  returns the most recent submissions, so the builder cannot reproduce the exact set; arXiv metadata (titles and
  abstracts) is released under CC0, so the abstracts ship with the repository.
- `checksums.json`: the SHA-256 of `cases.jsonl` for every dataset, so a rebuild can be verified.

Not shipped, and rebuilt on demand by the builder:

- TREC-QC, AG News, DBpedia-14, Yahoo! Answers, patent abstracts, 20 Newsgroups, arXiv full text and IMDB: the
  builder recreates the exact files from their pinned sources.
- LexGLUE ECtHR: the builder recreates the case files, but no ECtHR court-decision text is shipped, because the
  decisions contain personal information.

The result files under `results/` reference documents only by case id, so the paper's tables and figures regenerate
without any dataset files.
