# Verifier regression workflow

## v25: shared-fact dependency handling and uncertainty context

`shared-fact-reassessment-v25` invalidates and re-evaluates the connected
source-fact component within a scoped group. Previously valid sibling judgments
that mention the same fact cannot be retained while only an event is re-evaluated.
The current-window action aggregate is included when source events are affected.
Edges come from source-defined fact IDs, not prose similarity or score matching.
Independent criteria such as geometry remain retained. If fact conflicts already
exist in the first response they trigger the same bounded joint reassessment.
If the second response remains contradictory, affected observations are excluded
from partial results and no valid group cache is written. The limit remains two
model calls per group. Cross-group/repeat review retains its existing budget.

Scoped response v3 permits valid criterion-level image references on an
`unobserved` judgment. The host stores them separately as `uncertainty_context`,
with empty affirmative evidence refs/timestamps and unchanged null score and
evidence text. This is context for an abstention, not proof of failure or success.
Invalid, duplicate, or out-of-scope refs remain rejected. Fact-level unknown
citations retain their existing contract. This avoids spending a rewrite call
on an abstention just because it points to the images the model could not resolve.
It does not establish that the model's explanation for uncertainty is correct.

`summary.json` now lists `unobserved_judgments` explicitly. Protocol validity does
not imply usable observations for every criterion. `diagnostics.json` includes
`group.shared-facts-N.json` audits: conflicts, source dependencies, and affected
criteria, alongside original/corrected responses. The saved v24 replay tests cover
both reported failures without API calls or video generation. Visual direction
judgments (toward/away, clockwise/counter-clockwise) still need fixed-video labels;
mutually consistent responses are not independent visual verification.

From v24, synchronize `conditioning_verifier.py`, `scoped_judgment.py`, and
`verifier_facts.py` under `src/evovideo_skill/`, plus
`scripts/verifier_regression_suite.py`. Reuse the same three-task canary command
below with a new `outputs/h3_verifier_v25_suite_XXXXXX` directory. Inspect the
summary and keep the automatically generated diagnostics; do not regenerate H3
videos or advance to a full experiment merely because JSON now parses.

## v24: separate format repair from semantic reassessment

`bounded-semantic-reassessment-v24` retains the two-call fixed-window limit.
Missing fields and bad citations still use format-only repair: existing semantic
claims cannot change silently. Recognized conflicting fact value/basis pairs or
action outcomes/components instead use one explicit same-evidence reassessment
of the affected criteria. The invalid verdict is not frozen and no replacement
score is prescribed. Independently valid sibling criteria remain unchanged.
Both responses and the classification are saved; an unresolved conflict still
fails validation. A truthful unknown remains unscored. This is not an independent
second-model confirmation.

The v23 saved canary caught a retry deadlock: `contradicted + not_visible` and
`absent + nonempty matched` failed validation, but format-only feedback also
forbade changing the conflicting claims. The offline regression fixture now
preserves the reported rows and checks retry routing, unchanged sibling scores,
failure after two identical invalid responses, and truthful abstention. Synthetic
follow-up responses in tests exercise control flow; they are not visual labels.

Identity binding v2 also blocks explicit standalone present-tense absence claims
inside an observed actor binding. The reported declaration that the source A
person "is not present" cannot support an observed A binding. This limited text
check does not attempt general contradiction detection or reject time-qualified
occlusion as an identity error.

After synchronizing the five changed runtime modules (`conditioning_verifier.py`,
`criterion_grounding.py`, `scoped_judgment.py`, `verifier_facts.py`, and
`verifier_identity.py`), repeat the same three-task canary in a **new** directory:

```bash
suite_dir="$(mktemp -d outputs/h3_verifier_v24_suite_XXXXXX)"
set -o pipefail
PYTHONPATH=src python scripts/verifier_regression_suite.py \
  --run-dir outputs/h3_story350_v221_smoke15_x25tKA \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --max-groups 3 --execute --output-dir "$suite_dir" \
  2>&1 | tee "${suite_dir}.log"
```

The diagnostics collector from `fd9a472` automatically writes
`$suite_dir/diagnostics.json`. Preserve that file and `summary.json` together.
The verifier protocol bump prevents reusing old verdicts as v24 observations;
it does not invalidate the saved videos used by the development canary.

The v23 protocol (`source-actor-binding-v23`) checks source actor bindings before
using fixed-window scores. One `identity_bindings` object is shared by all
criteria in a request. Its appearance IDs and candidate image citations must
match the source registry. Explicit contradictory label/appearance claims in
the evidence text also block the dependent criteria. A common identity error
gets at most one group reassessment, within the existing two-call group limit.
Unresolved dependencies are withheld, not assigned zero or inverted to success.
Unrelated observations such as scene geometry remain usable. Per-criterion
automatic review does not spend more calls on the same unresolved identity.
The final source check also covers explicit contradictions in global judgments.

The registry currently parses the explicit appearance declaration grammar used
by Story350 (`A is an adult with ... and a ...`, and the distinct-adult variant).
All 350 checked-in tasks contain that grammar, 75 with two actors. This is not a
general natural-language identity resolver. Identical source appearances cannot
establish which actor is which. If the setup contains clothing context, a
conditional default garment is excluded from identity anchors; it must not
override an explicit striped jacket or an apron-removal action. The source
contract is not visual evidence. The
literal-claim check detects unique source appearance anchors in explicit actor
descriptions; it cannot prove all free-text paraphrases are semantically correct.
Visual-model errors can remain even when these checks pass.

## A staged workflow instead of repeated full runs

1. Keep failed raw responses, request manifests and candidate media immutable.
   Add each failure class to offline tests, including positive, negative, unknown,
   swapped-actor and source-label-permutation cases. Never call a model to test a
   deterministic parsing or dependency-propagation fix.
2. Audit a whole saved run offline, rather than requesting one more JSON field
   from the user each time:

   ```bash
   PYTHONPATH=src python scripts/verifier_regression_suite.py \
     --run-dir outputs/h3_story350_v221_smoke15_x25tKA \
     > outputs/verifier_v23_offline_audit.json
   ```

   This makes zero API calls and does not modify old results. Old responses
   without binding fields are unsupported under v23, not failed-video labels.

3. Run a bounded canary on existing media from up to three distinct source tasks:

   ```bash
   suite_dir="$(mktemp -d outputs/h3_verifier_v23_suite_XXXXXX)"
   set -o pipefail
   PYTHONPATH=src python scripts/verifier_regression_suite.py \
     --run-dir outputs/h3_story350_v221_smoke15_x25tKA \
     --config configs/h3_story350_debug.json \
     --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
     --max-groups 3 --execute --output-dir "$suite_dir" \
     2>&1 | tee "${suite_dir}.log"
   ```

   There are at most six model calls, no H3 generation and no memory updates.
   Selection is deterministic, by source task and group, not by quality score.
   Requests left by an interrupted run with no saved evaluation record are
   reported as unavailable. Changed tasks, videos or references stop preflight;
   they are never silently replaced. Provider/evidence/internal errors stop the
   canary; a response-format failure is recorded and other selected tasks can run.

4. Inspect `summary.json` for `canary_gate`, format validity, actual calls, binding coverage,
   withheld identity judgments and per-case errors. `offline_audit.json` captures
   source alias conflicts; each case directory retains requests, raw responses,
   `.identity-0.json` / `.identity-1.json` and normalized results. `case-NNN.log`
   contains complete diagnostic output. `needs_visual_labels` means protocol
   checks passed but fixed visual labels have not validated the canary;
   `blocked_contract_or_identity` means it must not advance. These artifacts are sufficient to debug
   a failed case together, without regenerating the video.
   `binding_coverage` counts only returned identity gates; failed cases without a
   parsed summary are explicitly counted as `cases_without_parsed_summary`.
   A coverage of 1 with two failed cases does not mean the cohort passed.

   New canaries also produce `diagnostics.json`: source contracts, original and
   corrected raw responses, validation errors and identity audits for every case,
   including parser successes that may contain semantic contradictions. For an
   already completed v23 canary, collect the same evidence with zero API calls:

   ```bash
   PYTHONPATH=src python scripts/verifier_regression_suite.py \
     --collect-suite outputs/h3_verifier_v23_suite_QY979F \
     > outputs/h3_verifier_v23_suite_QY979F.diagnostics.json
   ```

   This command needs no API key, source video or original task file and does not
   rewrite the run. It includes text evidence only, not media bytes or transport
   configuration. Missing/truncated artifacts are reported, not silently skipped.
   Share this one diagnostic file instead of repeatedly extracting individual
   criteria or spending more model calls before examining the saved responses.
5. After the canary passes, re-evaluate a fixed small set of complete existing
   videos. Check known visual outcomes, whole-video aggregation, disagreement and
   withheld-sample rates before restarting learning under a new protocol directory.
   A single valid JSON group is not a release gate for a full experiment.

## Reusable visual labels

Unit tests and binding coverage do not measure visual accuracy. Human-check a
small development set once, preserve the media hashes, and reuse those labels
on subsequent changes rather than reviewing each run manually. Include correct,
wrong-actor, wrong-action, occluded and ambiguous cases across different tasks.

The canary accepts `--labels path/to/labels.json`. Its `cases` array contains
`task_key` (from `suite_plan.json`), `video_sha256`, `group`, `annotation_note`
(human provenance/context), and an `expected` map from criterion names to
`{"status": "observed", "score": 0.0}` or
`{"status": "unobserved", "score": null}`. Use the actual reviewed values.
Scores use a fixed absolute tolerance of 0.05. Labels are matched by task, video
hash and group and are never sent to the model. Unlabeled cases are reported
separately; matching formats or model agreement are never counted as labeled
visual successes. These are development regression labels, not held-out gains.

Do not expand budgets or relax thresholds simply to make a regression pass.
When observations remain uncertain, report the exclusion and coverage. Keep
generation failures, transport failures, format errors, identity ambiguity,
source contradictions and genuine low-quality videos as separate outcomes.
