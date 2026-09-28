---
type: Experiment
title: European canonical-pair GPU audit
description: Resumable automated audit of twelve completed European datasets on the existing eight-server Gemma pool.
status: draft
generated: {by: codex, at: '2026-09-27T10:37:00+00:00'}
---
# Scope and borrowed infrastructure

The user authorized reusing the existing eight vLLM servers configured with
0.45 GPU memory utilization. The servers on localhost ports 8600–8607 were
verified through process arguments and `/v1/models`: Gemma
`google/gemma-4-26B-A4B-it`, snapshot
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, context 16,384.
No server was started, stopped or reconfigured. Process evidence is retained in
`../artifacts/european-audit/20260927/borrowed-servers.json`.

The full audit covers the authoritative completed campaign outputs for ca, cs,
de, el, es, et, fr, it, pt-PT, ro, uk and fi: **5,602,597 canonical pairs** across
train, validation and test. It audits each original/corrupted pair once, not the
duplicated task views. Portuguese remains explicitly European Portuguese.
The source manifest derives from the checker_resilient_v1 launch record.

# Shared code and preservation

`scripts/audit_european_pairs.py` reuses `Database` from HRM-Text
`dfm12.audit_full` and the JSON-response parser from `dfm12.jobs`. A DaLA-specific
adapter supplies canonical pairs and a grammar/spelling rubric; it does not
claim these are HRM chat imports or bypass that project's export gates.

Source and manifest hashes, shared-code hashes, client hash, complete prompt,
schema, server identities and concurrency are recorded in the run directory.
All 36 canonical source files are hash-checked before requests. The queue uses
bounded streaming, WAL transactions, persistent cursors, four-attempt retries,
leases and an exclusive client lock. Restart using the exact command in the
launch receipt; the frozen client and configuration must match. SIGTERM drains
in-flight requests. A hard crash may require waiting for the 30-minute leases.

The full client now uses **256 requests per endpoint, 2,048 total** (previously 32/256 and 64/512). No generated
source file is edited. Errors, flags and uncertain responses are retained.
Results are stored in `jobs.sqlite`, keyed by source component, ordinal and pair
ID, with edit metadata retained for later rule/family analysis. `status.json`
reports progress and endpoint counts. Failed requests are never passes.

# Smoke evidence and limits

Before full launch, 288 real pairs (first eight of each language/split) completed
with no failed requests. This is a transport/rubric smoke sample, not a random
quality estimate. A further 36 agent-authored controls covered injected errors,
unchanged sentences and reversed correction direction in all twelve languages.
All 24 unchanged/reversed controls were flagged; all twelve injected-error
controls passed. These controls are not native-validated gold and do not prove
precision. Three infrastructure unit tests passed for uncertainty, language
identity/prompt isolation, resume/retry and checksum mismatch handling.

Agent inspection found judge weaknesses: one Catalan source-error explanation
proposed exactly the same words as its correction; a Romanian control explanation
incorrectly criticized the valid form `Copiii`. Finnish control judgments also
varied in how they discussed colloquial versus standard agreement. Therefore
**pass, flag and review are automated signals, not release acceptance or native
linguistic validation**. In particular, no flagged rows are automatically deleted.
The judge assesses source correctness, whether the corruption introduces an
actual error, grammar/spelling scope, and intended-meaning preservation. It is
instructed to respect valid variants and ignore stylistic preferences. The
input omits the generator's correctness claims and rule names to reduce anchoring.

# Active run

Launched 2026-09-27 at 10:36:48 UTC. Receipt and frozen client:
`../artifacts/european-audit/20260927/launch.json` and
`../artifacts/european-audit/20260927/audit_european_pairs.py`.
Output: `la_output/european_audit/full_20260927/`.
Smoke output: `la_output/european_audit/calibration_20260927/` (historical directory
name; only a smoke test). Control results:
`la_output/european_audit/controls_20260927.json`.
The full audit is ongoing. Final aggregation, adjudication of questionable
judgments, release decisions and uploads remain outstanding.

Startup verification at 10:37:53 UTC observed 6,271 completed judgments and
active responses from all eight endpoints. A few truncated model responses
entered the retry queue; exhaustion remains an explicit failed job requiring
follow-up, never a linguistic judgment. Startup evidence is in
`../artifacts/european-audit/20260927/startup.json`.


# Concurrency increase

On 2026-09-27 the user requested increased concurrency. The original client was
gracefully drained, including waiting for its request timeouts. The restart
preserved **322,083 completed judgments, 317 failed jobs and 1,056 pending jobs**;
no running leases remained. Only `concurrency_per_endpoint` changed in the run
configuration, from 32 to 64. The prompt, model, sources, client code and existing
decisions are unchanged. Both complete configurations and the previous launch
are retained in `../artifacts/european-audit/20260927/resize-64.json`; the active
launch receipt was updated to the new PID and command. This explicit operational
migration preserves the client's strict configuration check on subsequent resumes.
The eight existing vLLM servers remain at 0.45 GPU memory utilization. Other
workloads share this pool, so throughput need not grow proportionally.


## Server outage detected 2026-09-28T13:35:22.567639+00:00

All eight borrowed endpoints (8600–8607) refused connections during a status
check. Gracefully stopped the audit client to prevent consuming retries during
the outage; the borrowed servers were not modified. Preserved 5,518,894
completed judgments and 22,329 failed jobs. Client alive after drain:
False. Evidence: `../artifacts/european-audit/20260927/server-outage.json`.
The main pass is incomplete and its earlier ETA no longer applies. Recovery
requires healthy endpoints and an explicit retry pass for failed jobs, including
connection failures and truncated responses. Original dataset rows remain intact.


## Resume after server restoration — 2026-09-28

At the user's request, verified all eight endpoints again serve the same pinned
Gemma snapshot with a 16,384-token context. Under the exclusive client lock,
archived the complete affected job records transactionally in SQLite table
`outage_retry_history`, then reset 16,658 jobs with connection/disconnection,
timeout or HTTP 500 errors to pending with zero attempts. This includes 16,460
exhausted failed jobs and 198 already-pending transient failures. All 5,518,894
completed judgments were preserved; 5,869 truncation/JSON failures remain for a
separate response-recovery pass. No linguistic judgments were reset.

Restarted the unchanged frozen client with 64 requests per endpoint (512 total),
using the same source hashes, prompt and schema. The operation and individual
reset IDs/errors are recorded in `../artifacts/european-audit/20260927/outage-recovery.json`;
`launch.json` now identifies the resumed process. No vLLM server was modified.


## Increase to 256 clients per server — 2026-09-28

At explicit user request, drained and resumed with 256 requests per endpoint
(2,048 total). Preserved 5,541,209 completed judgments, 5,907 failed jobs and
2,025 pending jobs. The sole code change raises the CLI concurrency ceiling
from 128 to 256; the original frozen client remains intact and the new frozen
client is `audit_european_pairs_256.py`. Updated the concurrency and client hash
in the configuration, retaining both configurations and previous launch in
`../artifacts/european-audit/20260927/resize-256.json`. Three infrastructure tests
passed. Sources, rubric, model, schema and existing judgments are unchanged.
The vLLM servers were not modified; only the audit client was restarted.


## Main pass complete; response recovery — 2026-09-28

The main client exited normally after processing all 36 sources and all
5,602,597 pairs: 5,596,599 judgments and 5,998 exhausted failures. The idle
servers reflected completion, not a stalled queue. Started a targeted recovery
pass for those 5,998 jobs: 5,977 truncated responses, eight malformed JSON
responses and 13 disconnects. Archived their complete prior records in SQLite
`response_retry_history` before resetting attempts. All existing judgments remain
unchanged. The new frozen `audit_european_pairs_recovery.py` changes only the
response allowance from 512 to 2,048 tokens; the rubric/schema/model and
256-per-server concurrency remain unchanged. Both configurations and original
failure counts are retained in `../artifacts/european-audit/20260927/response-recovery.json`.
The active launch receipt now identifies the recovery client. This is automated
review, not linguistic certification; no dataset rows are removed.


## Halve recovery concurrency — 2026-09-28

At the user's request, set recovery concurrency to 128 per server (1,024 total
capacity). The preceding recovery had already finished, adding 1,643 judgments
and leaving 4,355 failed requests: 3,919 truncations, three malformed responses
and 433 disconnects. Archived and requeued only the 433 disconnected requests
in `recovery128_history`; deterministic response failures remain unresolved.
Preserved all 5,598,242 completed judgments. The unchanged recovery client keeps
the 2,048-token response allowance. Both configurations, counts and previous
launch are recorded in `../artifacts/european-audit/20260927/resize-recovery-128.json`.
The live request count is limited by the remaining 433 jobs, below the configured
1,024 maximum. No server settings or dataset rows were changed.


## Exclude unresolved audit responses — 2026-09-28

The user explicitly approved excluding the 3,922 persistent truncation/JSON
failures discussed in the preceding review. These exact jobs are marked in the
SQLite `release_exclusions` table with reason `audit_unresolved`, not linguistic
rejection. Original source datasets, job records, failed responses' error history
and previous retry archives remain intact. The `release_audit_decisions` view
returns completed judgments minus explicit exclusions; its rows are still audit
signals, not automatic release approval.

Machine-readable release exclusion artifacts are in
`la_output/european_audit/full_20260927/release_exclusions/`: complete ID/evidence
JSONL, per-language pair-ID lists and a checksum/count/coverage manifest. Future
release packaging must apply these pair-ID exclusions to every task view and
split. The candidate source files have not been rewritten. The receipt is
`../artifacts/european-audit/20260927/release-exclusions.json`.
This authorization covers the previously discussed 3,922 jobs, not additional
failures produced by the subsequent disconnected-request recovery.

Automatic coverage check across all canonical audit records found **no rule lost
entirely** through these exclusions. There are 5,598,675 candidate pairs remaining
after this exclusion alone; this is not a count of release-approved pairs.
