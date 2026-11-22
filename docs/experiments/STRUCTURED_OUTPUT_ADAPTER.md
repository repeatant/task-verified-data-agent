# Optional structured-output adapter

This follow-up is isolated from the frozen experiment. It has mock-transport and
local validation tests; it has not been enabled for, or credited with improving,
the existing live results.

`QwenGraphModel(..., structured_output=True)` and `QwenReviewModel(...)` request
strict `json_schema` output. The default remains `json_object`. Writer fields and
the review action union are constrained structurally; SQL safety, DSL syntax,
source quotes, actual schema names, PK/FK constraints and stage authorization are
still checked locally. No reference answer or benchmark label enters a schema.

Accounting reserves message content characters plus the compact serialized
`response_format` envelope. Strict-mode events record its character count,
the schema digest and total reserved input characters. Client identities differ
by mode, including injected first-action wrappers, and the schema implementation
is included in the runtime fingerprint.

The existing output-token cap and failure policy remain. There is no automatic
fallback, extra retry or allowance reset on provider rejection or truncation.
Alibaba recommends avoiding output caps to prevent truncated JSON; this project
retains its explicit spending limit, so strict output is not a guarantee against
truncation. See [Qwen structured output](https://help.aliyun.com/zh/model-studio/qwen-structured-output).

## Preselected paired confirmation

The follow-up uses the existing `confirmation-manifest.json`: 40 new questions
from **four** existing BIRD databases. It is not an unseen-database evaluation.
The fixed allocation is California Schools 18, Card Games 18, Codebase Community 2
and Toxicology 2. Database-specific estimates for the last two are very uncertain.
Neither the selection nor the model prompts may be adjusted after observing
confirmation results. Official SQL/results are supplied only to the scorer.

Each question receives one shared initial SQL generated in `json_object` mode.
The two full-system methods then use `json_object` or strict `json_schema` for
their own reference and all later calls. Separate reference generation is part
of the format intervention. Prompts, model, temperature 0, 1,800 output-token cap,
full verification policy and local semantic validators are otherwise unchanged.
Reference preparation allows at most three calls; the verification trajectory
allows at most 12 additional calls, three attempts, three reviews and 40 generated
challenge examples. Both method ordering and lane assignment are seeded.

Primary reporting remains paired correct SQL delivery on the original database,
alongside wrong release, damage to initially correct SQL, rejection, cost and
latency. Initial-reference acceptance is also reported, but combines JSON/DSL/
fixture validity and **is not** a pure schema-compliance or correctness measure.
Completed failures remain failed. A partly written paid stage requires manual
review; it is not silently regenerated. Truncation may advance only to a new cell
through the existing reviewed budget continuation mechanism.

### Execution gates

Run from this isolated worktree. The commands below are a procedure, not evidence
that a provider request or experiment has already run.

```powershell
$baseRun = "../github-publish-v1-20261107/reports/expanded-v1-20261107/runs/main-v1"
../../.venv/Scripts/python.exe tools/evaluate_structured_confirmation.py --base-run $baseRun --freeze-only
# Only after the main/stability campaign has exited and the frozen bundle is reviewed:
../../.venv/Scripts/python.exe tools/evaluate_structured_confirmation.py --base-run $baseRun --smoke
# Continue only if the one real strict-schema compatibility request succeeds:
../../.venv/Scripts/python.exe tools/evaluate_structured_confirmation.py --base-run $baseRun --execute
```

The freeze contains local runtime/requirement hashes, manifest and bounded-source
digests, the external budget driver's full bytes/hash and a protocol snapshot.
The runner dynamically loads that frozen main driver's budget functions, restores
`sys.path` and asserts that every loaded `dataverify` module still belongs to the
isolated worktree. Old and strict modes therefore use the same experimental code.

Paid modes take the same allocation and four lane execution leases as the main
campaign. They require its existing cumulative CNY 120 receipt, reuse the four
CNY 28 ledgers, preserve all earlier charges/unknown reservations and allocate
**zero** extra allowance. The latest `lane_quarantines` chain is carried into
truncation handling. The compatibility request also consumes the existing budget.

For planning only, 12,000 message characters and six calls per method imply about
CNY 26.87 in conservative reservations, including schema overhead and the smoke.
The absolute 1,241-request / 40,000-character worst case is about CNY 168.95;
therefore full completion cannot be promised within the remaining allowance.
The existing per-lane cap stops further calls. No smaller question subset or
failed-cell retry is selected after viewing results to make the study finish.

Offline tests use mock models/transports to verify pairing, failure retention,
schema-accounted reservations, context boundaries and import identity. They do
not establish provider support, latency improvements or better SQL accuracy.
