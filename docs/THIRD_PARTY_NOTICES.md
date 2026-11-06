# Dependencies and method references

Current Agent orchestration uses LangGraph. Project-specific task guards,
reference execution, evidence handling and evaluation remain separate from
third-party framework capabilities.

## Runtime dependencies

- **LangGraph** — MIT; https://github.com/langchain-ai/langgraph . State graphs
  and persistence for both current verification paths. SQLite checkpoint support
  is installed separately.
- **SQLGlot** — MIT; https://github.com/tobymao/sqlglot . SQL AST dependency.
- **Hypothesis** — MPL-2.0; https://github.com/HypothesisWorks/hypothesis . Data
  generation and shrinking; its code has not been copied or modified.
- **FastAPI** — MIT; https://github.com/fastapi/fastapi . Earlier local API
  example, separate from the current dynamic verification runner.

## Method references

- **SQLDriller** — Apache-2.0; https://github.com/SJTU-IPADS/SQLDriller . Research
  reference for counterexample-oriented SQL diagnosis, not a runtime dependency.
- **SQLancer** — MIT; https://github.com/sqlancer/sqlancer . Database testing
  reference, not installed as this project's runtime.

Reference repositories are not claims of authorship or equivalent results.
Additional dynamic verification references are listed in [DISCOVERY](DISCOVERY.md#方法参考).

## Historical integration

**DB-GPT** — MIT; https://github.com/eosphoros-ai/DB-GPT . Earlier experiments
used its native Agent loop at commit `06a46eea2d248a0f2b3e9096035424df4f2dc2eb`.
The active integration, installation scripts and dependency have been removed.
The [historical pin](history/dbgpt_pin.json) and immutable test snapshot preserve
that provenance; they do not require installing DB-GPT to run the current graphs.
Upstream source was used unmodified in a separate checkout, whose LICENSE must
remain with that checkout. Historical feedback measurements are not reassigned
to LangGraph.

Direct and transitive dependency licenses remain with their distributions.
The project author's own open-source license has not yet been selected; this
notice does not grant a license to the project-specific code.
