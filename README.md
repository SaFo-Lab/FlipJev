# FlipJev: Fast Black-Box Red Teaming Finds Universal Adversarial Suffixes for Typed Decision Models

[![Paper](https://img.shields.io/badge/Paper-arXiv%20%28coming%20soon%29-lightgrey)](#citation)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://flipjev.github.io)

This repository contains the code, the attack suffixes and all result files of the paper. It runs every experiment of
the paper and regenerates every table and figure from the released results without any API call.

## Authors

**[Xiaogeng Liu](https://xiaogeng-liu.com)**<sup>1*</sup>, **[Xinyan Wang](https://xinyan-wang-stat.github.io)**<sup>2</sup>, **[Chaowei Xiao](https://xiaocw11.github.io)**<sup>1</sup>

<sup>1</sup>Johns Hopkins University, <sup>2</sup>University of Wisconsin–Madison

*Corresponding author: xliu316@jhu.edu

## News

**[2026/10]** We release the code, the attack suffixes and all result files, together with the
[project page](https://flipjev.github.io).

## Overview

<p align="center">
  <img src="assets/overview.png" alt="A typed decision and its universal suffix" width="100%">
</p>

**A typed decision and its universal suffix.** The trusted question of the developer and the untrusted content travel
in separate fields (A). A black-box attacker appends a short suffix to the content that flips the decision without
naming the subject (B). One suffix, found on a single dataset, flips 90–100% of the documents in five other tasks
without any further search (C).

A *typed decision model* answers a fixed question about a piece of content and returns a typed decision, such as a
yes/no proposition, a choice among named options or a level on a scale, in a single fast query and without generating
text. Jev is the representative typed decision model; it is reached through the OpenRouter Decisions API.

The attack in this repository is a quick black-box query-based attack. It appends a short suffix to the untrusted
content and improves the suffix with an elitist evolutionary search that uses only the probabilities the API returns.
It calls no attacker language model, so it spends its whole budget on queries to Jev and makes each candidate with
cheap word-level edits. The suffix never contains a word that names the subject of the question, so the correct answer
does not change, and it is at most 32 tokens (64 for the trajectory judge). The same search produces a universal
suffix when it scores each candidate by its mean effect over a batch of training documents.

Headline results of the paper:

- Jev resists common prompt injections: no strategy changes a correct answer in 10,240 paired tests on emails, and
  none flips any of 1,620 pairs on the other datasets.
- The attack finds short suffixes that flip confident answers within minutes. Per-document search flips 58 of 180
  targets (32%) across nine public datasets; an attack of 2,000 queries takes about two minutes.
- The suffixes are universal: one suffix per dataset flips 43% of unseen documents (114 of 264), while a fixed
  natural-language baseline flips none. On agent trajectories, one reviewer note inside a tool output flips up to 30%
  of held-out unsafe trajectories.
- The suffixes transfer across tasks: one suffix generated on full arXiv papers flips 90–100% of the documents of
  five other datasets, which ask different questions, without any query on them.

## Installation

Python 3.9 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The attack and the evaluation need only `tiktoken`. Regenerating the figures also needs `matplotlib` and `numpy`;
rebuilding the datasets also needs `datasets` and `scikit-learn`.

## API configuration

Jev is served through the OpenRouter Decisions API at `https://openrouter.ai/api/alpha/decisions`. The requested model
is `typesafe/jev-1.13`; every run in the paper was served the snapshot `typesafe/jev-1.13-20260917`, which each result
file records. The code reads the API key only from the environment:

```bash
export OPENROUTER_API_KEY=...
```

Every script that queries the API has a `--dry-run` mode that prints its plan and sends nothing. Regenerating the
paper's tables and figures (below) reads only the released result files and never touches the API.

## Data preparation

- **Emails.** The email-comprehension data was written by the authors and is generated in full by the code
  (`flipjev/email_sets.py`, `flipjev/long_emails.py`); no download is needed.
- **Classification datasets.** `python -m scripts.build_datasets` rebuilds the nine datasets (and IMDB with `--imdb`)
  from their pinned public sources into `data/datasets/`. `--check` compares each rebuilt file against
  `data/datasets/checksums.json`. The arXiv abstracts come from the live arXiv API and ship with the repository.
- **Agent trajectories.** The two benchmarks are not shipped; `data/trajectories/README.md` gives the download and
  the expected layout. Set `TRAJECTORY_DATA` if they live elsewhere.

See **Data and licences** below for the source and licence of every dataset.

## Running the experiments

Run from the repository root with `OPENROUTER_API_KEY` set. Each run writes to a directory you choose under `runs/`.
The approximate costs and times are those the paper reports in its appendix; with six to eight parallel workers the
API answers about 65,000 queries per hour.

**RQ1: common prompt injections.** Five fixed sets on the email task, and five strategies on the other datasets.

```bash
python -m scripts.common_injections --target emails --out runs/injections_emails          # about $0.36
python -m scripts.common_injections --target datasets --dataset trec \
    --universal-root runs/universal --out runs/injections_trec                            # part of about $1.1 in total
```

**RQ2: per-document search.** One search of up to 2,000 queries for each target.

```bash
python -m scripts.per_document_search --task email --out runs/email_search               # 47,183 queries, about $2.48
python -m scripts.per_document_search --task dataset --dataset trec --out runs/trec_search
```

The nine-dataset study (per-document searches and universal searches together) costs about $52. An email attack of
2,000 queries takes 1.5–2.6 minutes with six to eight workers.

**RQ3: universal suffix.** One suffix searched on ten training documents and evaluated on held-out documents, with
two controls.

```bash
python -m scripts.universal_search --task email --out runs/email_universal
python -m scripts.universal_search --task dataset --dataset dbpedia_14 --out runs/universal/dbpedia_14
```

Run the universal search for each dataset under one root (`runs/universal/<dataset>`) so that the transfer and the
ablations can read the suffixes.

**RQ4: transfer across datasets and tasks.** Each dataset's universal suffix on every other dataset's held-out set.

```bash
python -m scripts.transfer --universal-root runs/universal --out runs/transfer
```

**Trajectory judge.** The per-target search, the universal suffix, the controls and the re-test on each benchmark,
then the transfer between them. The whole trajectory-judge study costs about $31.

```bash
python -m scripts.trajectory_judge --dataset atbench --out runs/judge
python -m scripts.trajectory_judge --dataset rjudge --out runs/judge
python -m scripts.trajectory_judge --transfer --out runs/judge
```

**Appendix analyses.** The token ablation (about $1.10) and the Choice-answer search (about $2.95).

```bash
python -m scripts.ablations token --universal-root runs/universal --out runs/ablation
python -m scripts.ablations choice --dataset trec --universal-root runs/universal --out runs/choice
```

## Regenerating the paper's tables and figures

Both read only the files under `results/` and make no API call.

```bash
python -m scripts.paper.tables     # the main table (Table 1) and the appendix tables; writes the table bodies to tables/
python -m scripts.paper.figures    # all data figures to figures/ (or one by name, e.g. "figures budget")
```

Each script prints the numbers it produces, so they can be checked against `sections/04_experiments.tex` and the
appendix of the paper.

## Running the tests

```bash
python -m pytest tests          # or: python -m unittest discover -s tests
```

The tests are offline. The trajectory-loader test that needs the benchmarks is skipped when they are not on disk.

## Repository layout

```
flipjev/            the attack and the evaluation
  api.py              the OpenRouter Decisions API client (key from the environment)
  search.py           the elitist evolutionary search and its operators
  tasks.py            the task specification, the four questions and the mutation vocabulary
  text.py             word-level tokens and BPE length
  emails.py           the email-comprehension task
  email_sets.py       the five fixed email sets of RQ1
  long_emails.py      the long email targets of the per-document search
  datasets.py         the nine classification datasets and IMDB, with builders and task specs
  evaluation.py       the held-out evaluation and the universal batch fitness
  study.py            the stages of a study (targets, per-document, universal, controls, transfer)
  trajectories.py     the R-Judge and ATBench500 loaders and the judge's rendering
  insertion.py        where attacker text can go inside a trajectory, and the verified insertion
  judge.py            Jev as a trajectory safety judge: the questions, the verdict and the task
scripts/            entry points for running the experiments and regenerating the tables and figures
data/               the arXiv abstracts, dataset checksums, and download notes for the benchmarks
results/            the released result files (case ids, suffixes, probabilities and scores only)
tests/              offline unit tests
```

The result files keep only case ids, suffixes, probabilities and scores; they carry no document or trajectory text,
so the tables and figures regenerate from them with no dataset on disk.

## Data and licences

Before releasing the result files and suffixes, each dataset's licence was checked, personal information was kept out
(no ECtHR court-decision text is shipped), and only the text needed to replay the results is kept.

| Dataset | Source | Licence | How it is obtained |
|---|---|---|---|
| Email comprehension | written by the authors | released here | generated by the code |
| TREC-QC | `SetFit/TREC-QC` (Hugging Face) | the dataset card states no licence; the original data is offered for research use | builder, pinned revision |
| AG News | `fancyzhx/ag_news` | the source corpus permits non-commercial research use | builder, pinned revision |
| DBpedia-14 | `fancyzhx/dbpedia_14` | CC BY-SA 3.0 | builder, pinned revision |
| Yahoo! Answers | `community-datasets/yahoo_answers_topics` | research-use dataset; the card lists the licence as unknown | builder, pinned revision |
| Patent abstracts | `ccdv/patent-classification` (BigPatent) | CC BY 4.0 | builder, pinned revision |
| arXiv abstracts | arXiv API | arXiv metadata is CC0 | shipped (the live API is not reproducible) |
| 20 Newsgroups | scikit-learn `fetch_20newsgroups` | public, widely redistributed | builder |
| LexGLUE ECtHR | `coastalcph/lex_glue` (`ecthr_a`) | CC BY 4.0 | builder; no court-decision text is shipped |
| arXiv full text | `ccdv/arxiv-classification` | CC BY 4.0 | builder, pinned revision |
| IMDB | `stanfordnlp/imdb` | research-use dataset (licence "other") | builder, pinned revision |
| R-Judge | `github.com/Lordog/R-Judge` | see the benchmark repository | download (`data/trajectories/README.md`) |
| ATBench500 | `AI45Research/ATBench` (Hugging Face) | Apache-2.0 | download (`data/trajectories/README.md`) |

Licences were read from each dataset's own card or source page. The pinned Hugging Face revisions are in
`flipjev/datasets.py`, and the builder verifies each rebuilt file against `data/datasets/checksums.json`.

## Ethics Statement

This work is a robustness evaluation of a commercial model through its public API. The findings were reported to
TypeSafe before publication through the vendor's official security-reporting channel. The report included the
universal suffixes of the nine datasets and of the two trajectory benchmarks, together with the exact input and the
API response for each case, so that every case can be replayed. The suffixes change only the decisions of a typed
decision model; they do not extract data and do not make the model produce harmful text, and they affect only systems
in which the attacker can already write content.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{liu2026flipjev,
  title   = {FlipJev: Fast Black-Box Red Teaming Finds Universal Adversarial Suffixes for Typed Decision Models},
  author  = {Liu, Xiaogeng and Wang, Xinyan and Xiao, Chaowei},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

This code is released under a research-only license. See [LICENSE](LICENSE) for details. The datasets keep their own
licences (see [Data and licences](#data-and-licences)).

**Intended use**: This code is provided for academic research and defensive security purposes only. It is intended to
help the community measure and improve the robustness of typed decision models.

## Contact

- **Project page**: [https://flipjev.github.io](https://flipjev.github.io)
- **Issues**: [GitHub Issues](https://github.com/SaFo-Lab/FlipJev/issues)
- **Email**: xliu316@jhu.edu
