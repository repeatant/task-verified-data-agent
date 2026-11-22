# Development findings before the expanded test

These observations use development data only. They are not final benchmark
results and must not be used as a résumé performance claim. The prospective
100-question public test split was not inspected for model results in this
review. Development outputs remain preserved in their original run directories.

## Runs reviewed

- `runs/dev-pilot-2`: four development questions, four methods, sixteen records.
- `runs/development-v2`: twenty development questions, four methods, eighty
  records. These questions come from the two development databases: student club
  and thrombosis prediction. The four pilot questions are included again here;
  they are not additional independent examples.

Paths are relative to `reports/expanded-v1-20261107/`. Records contain original
model contexts, generated programs, statuses, scoring, and usage. No results were
deleted or rerolled during this review.

## Observed outcomes

For the twenty-question development run, original-database BIRD EX gives:

- **Direct**: 4/20 correct SQL deliveries; 16/20 delivered in total, including
  twelve wrong releases. Four candidates were refused by the preview executor.
- **Execution-feedback self-repair**: 4/20 correct deliveries; all twenty
  delivered, including sixteen wrong releases. It made the four initially
  non-executable outputs executable without recovering their intended semantics.
- **Two-way protocol**: 1/20 correct deliveries; three delivered in total,
  including two wrong releases. One initially correct SQL was damaged; three
  initially correct tasks were rejected.
- **Full protocol**: 0/20 correct deliveries; one delivered SQL was wrong.
  All four initially correct SQLs were retained as correct candidates but
  rejected from delivery. Preserving a candidate is not successful delivery.

The initial reference plan was valid in only **6/20** tasks. Fourteen tasks
therefore failed before either reference-based protocol could meaningfully
investigate SQL semantics. The full method's final statuses were fourteen
`initial_reference_invalid`, four `needs_clarification`, one `inconclusive`, and
one `reviewed`.

For the same four questions shared with the earlier pilot, there is no demonstrated
increase in correct deliveries: direct and self-repair each delivered one correct
answer in both runs; full and two-way delivered no correct answer in either run.
The twenty-question run expands diagnosis but is not evidence of improvement.

## Failure taxonomy

### 1. Reference-program output contract

Of the fourteen invalid initial references, the terminal errors were:

- `GRAPH_QWEN_RESPONSE_TRUNCATED`: four.
- `REFERENCE_CALL_NOT_ALLOWED`: three.
- `REFERENCE_ONLY_LOCAL_ASSIGNMENTS_ALLOWED`: two.
- One each: `REFERENCE_EVALUATION_ERROR`, `RESULT_SCALAR_REQUIRED`,
  `REFERENCE_OPERATION_NOT_ALLOWED_Slice`, `REFERENCE_COMPREHENSION_TARGET`, and
  `BASIS_NOT_IN_SOURCE`.

The model often produced ordinary Python loops, methods, slices, or return shapes
outside the restricted interpreter's contract. Error feedback already explained
some violations, but valid completion remained unreliable. Requiring the initial
response to contain a computation plus a whole, foreign-key-valid synthetic
database increased both output size and generation obligations.

This is a protocol feasibility problem. It is not evidence that the independent
audit repaired a semantic error, and rejection caused by these errors must not
be presented as successful error detection.

### 2. Missing database value semantics

Development question 396 produced an assumption that inpatient admission is
encoded as `IN`. The original dictionary explicitly defines admission as `+`
for inpatient and `-` for outpatient. The task's column names alone do not
communicate these encodings. The laboratory dictionary also identifies `U-PRO`
as proteinuria stored in a text column, which matters for numeric comparison.

The original database archive includes 75 dictionary CSVs. Supplying a bounded,
question-independent portion of this metadata equally to all methods is justified;
supplying gold-derived predicates or selecting columns using answer SQL is not.

Dictionary metadata is itself imperfect: for example, one expense identifier
description calls it an income identifier. It is context, not an unquestionable
oracle. The system should retain source attribution and report conflicts rather
than treating every description as a hard rule.

### 3. Genuine semantic mistakes after syntax is valid

Examples from development data demonstrate why successful execution is insufficient:

- Question 306: the generated SQL sums remaining budget after an expense join
  without deduplicating member/budget pairs. Multiple expenses charged to one
  budget line can multiply the same remaining balance.
- Question 308: the question's evidence defines enrolled people through the club
  membership list. Both SQL and reference reinterpret enrollment as attendance.
  The reference then records this avoidable misinterpretation as an assumption.
- Question 319: fixing a nonexistent column made SQL executable, but both SQL and
  the reference still counted attendance records instead of distinct members
  across events within a category. A two-way agreement was consequently wrong.
- Question 420: the initial SQL has an unquoted hyphenated identifier and produces
  grouped rows instead of the requested separate patient counts. Repairing only
  syntax is insufficient to establish the requested output contract.

### 4. Agreement on insufficient test inputs

In question 310, the full method retained an inner join that omits budget events
with no attendees. Its reference programs could express zero-attendance output,
but the retained fixtures did not expose the discrepancy, and the bound preview
was empty for the relevant predicate. The three programs agreed on their observed
inputs and an incorrect SQL was published.

The two-way run for that same question generated a zero-attendance example,
changed the join to a left join, and produced a correct final SQL. This is a
concrete counterexample to claiming that an additional blind context always
improves correctness. Test diversity and witness discovery still matter.

### 5. Harmful repair despite an initially correct SQL

Question 297 began with correct SQL. The two-way method changed the source
category to the invented literal `Fundraiser` instead of the original database's
`Fundraising`. The reference and fixtures failed to distinguish that literal
from the intended behavior. It delivered a wrong result on the original database.
The full method withheld the task because the supporting calculations disagreed.

This illustrates a real protection benefit on one case, together with a usability
cost: the full method also withheld the other three initially correct tasks.
Report both effects, not only the protected case.

## Minimal changes justified by these observations

1. Separate initial reference-program generation from optional synthetic-data
   generation. Allow an initial plan to use the bound source without first
   manufacturing every foreign-key dependency. Retain independently generated
   challenge fixtures and executed checks later in the protocol.
2. Provide the original field dictionary through a deterministic, bounded policy
   shared by every method. The implementation in `public_schema_notes.py` accepts
   only archive path, database ID, schema, and character budget. It does not
   accept a question, gold SQL, expected answer, or model result.
3. Keep syntax errors, truncations, schema interpretation errors, false refusals,
   and wrong releases separate in reporting. A syntax fix must not be credited as
   a semantic correction unless original-database or held-out-suite scoring agrees.
4. Re-run a small declared development pilot under the new frozen protocol, then
   validate the complete development set before spending on the frozen test.
   Do not change the official test selection in response to these failures.

The dictionary policy is 2,800 characters, table-balanced round-robin, with value
descriptions prioritized within each table. It matches actual columns only,
labels omissions and truncated descriptions, and exposes `complete=false`.
Observed development coverage is 38/48 described student-club fields and 57/61
thrombosis fields; admission and sex encodings are included. These are context
coverage counts, not evidence of improved model performance.

The next protocol's effects remain to be measured. The current development run is
not healthy enough to support a claim of robust cross-domain correction, and
the independently audited programs do not eliminate correlated semantic errors.
