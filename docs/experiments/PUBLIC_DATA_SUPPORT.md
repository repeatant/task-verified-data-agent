# Public SQL evaluation data

This document records dataset preparation **before any model outcomes**. It is
not an accuracy report. Local data and private manifests remain under
`reports/public-data/` and are excluded from Git.

## Source and version

- [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev) publishes a 2026 update
  with 500 SQLite questions. We pin the
  [Hugging Face annotation revision](https://huggingface.co/datasets/birdsql/bird_mini_dev/tree/5b256b5a9c17478f06ed6912612e9a14d17065c8).
- Annotation file: `data/mini_dev_sqlite.jsonl`; SHA-256
  `926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606`.
- Original SQLite databases come from the
  [official BIRD dev archive](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip).
  The Mini-Dev README explicitly permits original dev databases for SQLite.
  This is not the newer Harbor container image, and we do not claim exact
  leaderboard comparability with that environment.
- Download: 346,207,293 bytes; SHA-256
  `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630`.
  The inner archive is `dev_20240627/dev_databases.zip`. Eleven original SQLite
  files were extracted without rewriting tables or rows; per-file hashes are
  recorded in the manifest.
- License: CC BY-SA 4.0. This repository contains preparation/scoring code rather
  than redistributing the dataset or publishing benchmark gold in prompts.

## Declared support and selection

The first static audit used a six-table/eighty-column cap. It yielded only 100
eligible questions across three databases, leaving no disjoint development set.
That audit is preserved as `bird-manifest-v1.json`. Before any model calls, the
cap was changed to eight tables and 120 columns, matching the generic adapter's
bounded scope. No model success/failure was used to select that change.

The final pre-inference policy is:

- Include the **whole database schema**, never gold-selected tables or columns.
- At most eight tables, 120 columns, and 16,000 schema characters.
- One SELECT; at most 500 SQL AST nodes; no WITH, window functions, UNION,
  INTERSECT, EXCEPT, or INTO. Safe scalar functions are explicitly enumerated in
  `support_reasons`; unsupported constructs are disclosed rather than rewritten.
- At most 200 result rows and eight result columns, matching the runtime contract.
- Execute gold SQL on the original file solely to validate the scoring target
  and these result-size limits. Gold SQL and rows never enter `public_input`.
- Previews use table/schema order, at most ten rows per table and a 12,000-character
  row budget. BLOB/oversized rows are omitted and counted, never replaced with
  invented values. Previews are incomplete; they do not replace original DBs.

The resulting support audit has **247 eligible questions out of 500**:

- California schools: 37
- Card games: 55
- Codebase community: 21
- Debit card specializing: 7
- Financial: 17
- Student club: 17
- Thrombosis prediction: 72
- Toxicology: 21

Exclusion reasons are non-exclusive: unsupported SQL construct 155; unknown
table/CTE alias 150; table count 105; unsupported function 32; column count 31;
result rows four; top-level SELECT shape two; result columns one. A CTE may
therefore receive both the construct and table-alias reason. These counts must
not be summed to obtain the number of excluded tasks. All syntactically eligible
gold queries executed successfully; five were excluded by result size.

## Frozen split

Seed: `20261107`. The algorithm shuffles pairs of candidate development databases
with this seed and chooses the first pair that can supply twenty development
tasks while leaving at least five databases and 100 test tasks. Task selection
then cycles through databases using seeded per-database order.

- **Development: 20 questions**, from student club and thrombosis prediction.
- **Test: 100 questions**, from California schools, card games, codebase community,
  debit card specializing, financial, and toxicology.
- Development/test databases and question IDs are disjoint. This supports an
  unseen-database claim relative to this project's development split, **not** a
  claim that a pretrained model has never seen public BIRD data.
- The remaining eligible questions are unselected reserve data. The manifest
  stores all eligibility decisions and selected IDs before inference.

Manifest: `reports/public-data/bird-manifest-v2.json`; SHA-256
`ebd595c6b1be7b128b61f43356498f59a0795355d3de02852700ce601b627406`.
Keep this first manifest immutable. Any changed policy or split needs a new
version and a separate explanation.

## Scoring and trust boundary

`model_inputs[str(question_id)]` is an explicit allowlist: task ID, question,
provided evidence, whole schema, column types, constraints, preview rows, and an
incomplete-preview notice. The separate `eligibility` section contains gold SQL
and expected rows for the evaluator only. Do not send the entire manifest to an
Agent.

The scorer uses the original SQLite file with `mode=ro`, `query_only`, extension
loading disabled, an execution deadline, bounded results, and an authorizer that
rejects writes and external file operations. Evaluation uses the
[official BIRD EX definition](https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py):
set equality of returned rows. Strict multiset equality and ordered equality are
reported separately because official EX ignores duplicate multiplicity and row
order. Neither source-DB EX nor additional generated probes prove correctness
over every possible input database.

Only the declared **100-question supported subset** can be scored by this run.
Results must not be presented as the official full 500-question benchmark or
compared directly with its published leaderboard.

## Reproduction

After obtaining the pinned annotations and original databases in the local data
directory, run:

```bash
python tools/prepare_public_benchmark.py reports/public-data \
  --output reports/public-data/bird-manifest-new.json
pytest tests/test_public_benchmark.py -q
```

The preparation command refuses to overwrite an existing manifest and makes no
model requests. It hashes original files, records support exclusions, and
selects the split without model outcomes.
