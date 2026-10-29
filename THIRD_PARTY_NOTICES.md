# Dependencies and method references

This project extends a pinned upstream Agent loop and uses external libraries.
The project-specific task guard, test adaptation, evidence handling and
evaluation code are distinguished from upstream capabilities in the README.

- **DB-GPT** — MIT; https://github.com/eosphoros-ai/DB-GPT . Native loop pinned
  to `06a46eea2d248a0f2b3e9096035424df4f2dc2eb`. Bootstrapped as a separate
  checkout; selected upstream source hashes are in `docs/research/dbgpt_pin.json`.
  Upstream source has not been modified. Preserve its LICENSE with its checkout.
- **SQLGlot** — MIT; https://github.com/tobymao/sqlglot . SQL AST dependency.
- **Hypothesis** — MPL-2.0; https://github.com/HypothesisWorks/hypothesis . Data
  generation and shrinking dependency; its code has not been copied or modified.
- **LangGraph** — MIT; https://github.com/langchain-ai/langgraph . Used by the
  checkpointed executor and earlier local prototype. Its SQLite checkpoint
  package is installed separately; neither graph wraps the DB-GPT agent loop.
- **FastAPI** — MIT; https://github.com/fastapi/fastapi . Earlier local API.
- **SQLDriller** — Apache-2.0; https://github.com/SJTU-IPADS/SQLDriller . Research
  reference for counterexample-oriented SQL diagnosis, not a runtime dependency.
- **SQLancer** — MIT; https://github.com/sqlancer/sqlancer . Method reference for
  database testing; it is not installed as this project's runtime.

Direct and transitive dependency licenses remain with their distributions.
Reference repositories are not claims of authorship or of equivalent results.
The project author's own open-source license has not yet been selected; this
notice does not grant a license to the project-specific code.
