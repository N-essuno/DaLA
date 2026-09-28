---
type: Experiment
title: European export failures and Greek orthography investigation
description: Unicode case folding causes unlicensed spelling edits and Greek dictionary false rejections; publication status checked independently.
status: draft
generated: {by: codex, at: '2026-09-26T18:28:07.996225+00:00'}
---

# German and Estonian

Both reached the 478,930-pair selection target but failed final validation with
`Edit differs from licensed rule`. Checkpoints remain intact. An exhaustive scan
of screened checkpoint rows found 46 invalid German edits among 499,177 rows
(26 internal deletions, 20 internal transpositions), and one invalid Estonian
edit among 490,002 rows. These are screened-pool counts, not final-selection counts.

The shared morphology adapter case-folds the original before character edits,
turning ß into ss. The character-operation validator operates on lowercase
original text, preserving ß, and correctly rejects the resulting extra changes.
For example, German `Floß → Flss` is not one internal deletion. The Estonian
failure is `Großheide → Grosheide`, a German place name inside Estonian source
text; this also warrants checking proper-name protection.

Do not loosen the validator. The repair should preserve orthographic surface
forms when generating/checking spelling edits, remove or regenerate affected
candidates, and repeat final selection and validation with explicit provenance.

# Greek

The completed local build contains 81,590 pairs from a base source of 226,834
Wikipedia documents, after source exhaustion and legacy-batch recovery. This
is not evidence that good Greek text or corruption resources are intrinsically
insufficient: a confirmed pipeline normalization bug suppresses yield.

Python casefold turns final ς into σ. The current source dictionary gate passes
case-folded words to Hunspell. Direct dictionary probes confirm all nine words
της, τις, τους, ως, επίσης, όπως, καθώς, προς, πόλης are recognized in their
correct surface spelling and rejected after case folding. Frequent recorded
unknowns include τησ (499,302), τισ (170,151), and τουσ (164,930).

The run records 1,190,541 source_dictionary_unknown rejections, versus only 146
no_licensed_corruption rejections. These counters include recovery processing,
not unique sentences; dictionary failures are not all necessarily this bug.
The current evidence establishes a major artificial source filter, but does not
quantify how many additional final pairs a repair will yield.

All eight configured grammar families occur, but the distribution is uneven:
44,664 spelling; 17,694 determiner number; 12,744 determiner case; 4,304 determiner
gender; 887 adjective case; 737 adjective number; 381 adjective gender; 93 verb
person agreement; 86 verb number agreement. Source quality checks and sparse
agreement coverage remain relevant after correcting normalization.

The fix must cover resource preparation as well as runtime lookup:
`scripts/prepare_european_pilots.py` case-folds attested forms, and
`scripts/prepare_unimorph_inputs.py` also case-folds forms. Separate lookup keys
from orthographically faithful output forms, rebuild affected resources, then
measure recovered yield and audit samples. Blindly replacing sigma characters
or relaxing dictionary validation is not a valid repair.

# Publication status and evidence

An authenticated Hugging Face organization inventory found no Finnish, Catalan,
Czech, or Spanish DaLA repository. All four are locally completed at 478,930
pairs each; local completion does not mean publication or human validation.

Evidence is in `../artifacts/european-expansion/export-investigation/`:
`de-summary.json`, `et-summary.json`, the corresponding invalid-edit JSONL files,
`greek-dictionary-probe.json`, and `hf-inventory.json`.

This investigation combines automatic checkpoint validation/dictionary queries
and agent code inspection. No native linguistic validation was performed. No
active generation code, profiles, checkpoints, or exports were changed, and
nothing was uploaded. This describes the initial investigation; the implementation and continuation below supersede that pending repair status.


# Implemented repair and continuation

On 2026-09-26T18:58:00.560436+00:00, installed the tested shared fix after isolated implementation in
`/work/mimir/DaLA-orthography-repair/`. Surface spellings use Unicode lowercase
rather than casefold in morphology lookup, dictionary screening, corruption
generation, observed spelling preparation and edit validation. Lemma identity
keys remain case-folded separately. The reduced rulebook exporter uses the
rulebook's explicit `surface_normalization: unicode_lower` marker; legacy
export normalization remains unchanged when that marker is absent.

UD, UniMorph and nominal preparation now preserve surface orthography. Expansion
preparation rejects incompatible old/new normalization receipts rather than
silently mixing them. German/Estonian/Greek resources were reconstructed from
pinned raw UD training and UniMorph evidence; a parallel legacy reconstruction
matched the prior analyses exactly before compiling the corrected forms. No
cross-language rules or unverified reverse-casefold guesses were introduced.

**188 unit tests passed.** Fresh 100-document pilots independently passed
artifact checksums, licensed edit reconstruction, task views, provenance and
split checks: German 697, Estonian 575, Greek 374 pairs. Tests explicitly cover
ß/ss, final sigma, distinct attestation counts, actual character generation,
reduced-rulebook export, exact quarantine and retained checkpoint lineage.

On the same 100 Greek source documents and seed, the original pipeline retained
37 pairs versus 374 with the repair. Dictionary rejections fell from 600 to 133.
Source grammar/context guards remain active; this is a bounded yield result,
not a full-corpus forecast or a linguistic precision measurement. Diagnostic
pilot v1 exposed the export-normalization issue; it is preserved and superseded
by independently validated `orthography_pilot_v2`.

# Ukrainian checker diagnosis

The failing original contains combining acute accents in `Влади́слава Гому́лки`.
LanguageTool 6.6 removes the accents internally but a `DictionaryMatchFilter`
uses incompatible offsets, raising `StringIndexOutOfBoundsException` (end 124,
text length 123). The logged sentence omits those accents: replaying the log
text succeeds, but replaying the exact checkpoint original returns HTTP 500
twice against a fresh server; an ordinary Ukrainian control returns 200.

The new generic `checker.source_failure_exclusions` setting quarantines only
that exact original SHA256, with a reproduction-evidence reference. It rejects
the candidate before invoking the checker and records
`documented_source_checker_failure`. No rewritten input or fabricated success
is used; all other checker failures still abort rather than bypass screening.
Further distinct LanguageTool crashes would require separate investigation.

A reviewed continuation preserves 10,259 verified candidate batches and
189,558 screened rows, all revalidated against the repaired edit validator.
The source quarantine does not occur in already screened rows. Original per-batch
generator signatures, the complete parent migration and the existing recovery
batch ordering remain intact. The old checkpoints are preserved; no chained
migration is represented as generation equivalence.

# Active operations

Portuguese and Romanian exhausted their source pools at 436,051 and 377,246
pairs respectively. The user accepts these shortfalls; no padding or additional
source expansion was introduced.

French paused at 238,285 retained pairs while the tested changes were installed
in the main repository. Its unchanged code is preserved in
`/work/mimir/DaLA-campaign-v2-frozen/`; checksum equality with its original run
signature was checked. It resumes with the same profile/checkpoints from that
snapshot, avoiding a hot code change to its campaign.

The replacement supervisor runs the same shared pipeline for every language:
German/Estonian/Greek each have 40 parser workers and fresh corrected candidates;
Ukrainian resumes with 24 workers and the exact quarantine; French continues
with its original 12 workers and code. Total configured parser workers: 156.
Language-specific resources remain separate inputs. New production profiles:
`la_output/resources/european-expansion/orthography_v2/CODE/profile.json`.
New DE/ET/EL/UK checkpoints and outputs use `orthography_v2`; earlier versions
remain intact. The three rebuilt languages still target 478,930 pairs each.

Authoritative launch and status: `../artifacts/european-expansion/orthography_v2/launch.json`;
active pointer: `../artifacts/european-expansion/active-campaign.json`.
`scripts/report_european_scale_status.py` accepts this combined launch, including
the completed datasets retained from earlier runs. Evidence directory:
`../artifacts/european-expansion/orthography-repair/`, including exact checker
replays, paired-pilot metrics, independent validation and final test logs.

These are automatic checks and agent implementation inspection. No native
linguistic validation or HF upload was performed in this repair.

Live continuation verification: all 156 parser workers were observed alive.
Ukrainian passed the previously failing batch 10178 (zero-based), with exactly
one documented checker-failure rejection, and is screening subsequent batches.
See `orthography-repair/resume-health.json` for the timestamped receipt.
French replay reporting preserves its pre-pause retained count until reconstruction
catches up; checkpoints and selected originals have not been lost.


# Checker throughput scaling

At 2026-09-26T20:23:00.595166+00:00, the user authorized larger Ukrainian/French allocations, then
explicitly requested continued checker increases while screening is limiting.
A five-second process CPU sample measured about 41 logical cores for the three
builds despite 76 parser workers. French used approximately 18.1 checker cores
and 0.4 parser cores; Ukrainian 7.2 and 2.8; Greek 5.2 and 7.4. Prepared queues
were essentially full, demonstrating screening rather than parser starvation.

French moved from 12 to 20 parser workers and 2 to 4 checker instances (32
screening clients); Ukrainian from 24 to 72 parser workers and 1 to 6 checkers
(48 clients). Greek keeps 40 parser workers and moves from 1 to 4 checkers
(32 clients), after a fresh measurement of 153 prepared batches ahead and
4.4 checker cores. Total configured parser workers: 132. Parser counts are
process allocations, not a claim that all those CPU cores are continuously busy.

`scripts/migrate_execution_profile.py` now permits checker worker/instance
execution settings while retaining all linguistic inputs and code hashes.
Execution-only migrations explicitly preserve generation-upgrade lineage,
per-batch generator signatures and recovery ordering. A regression fixture
checks those properties; all old checkpoint directories remain intact.
French still uses its frozen generator. Greek was untouched during the initial
French/Ukrainian resize and subsequently received its own checked continuation.

The running `scripts/autoscale_european_checkers.py` monitors all three languages
once per minute, waits for checkpoint replay plus a three-minute throughput
window, and increases instances by 50% (at least four initially) when the prepared
queue exceeds 70% of its prefetch window, checker CPU use is substantial and
parsers are not saturated. Each checker receives eight screening clients.
It leaves estimated CPU headroom (projected host use below 80%), checks available
memory, and pauses further growth after two measured windows without at least
10% improvement over the previous pool's throughput. These are operational
heuristics, not a guarantee of linear scaling. Failed checker responses are
never converted into successes. Further crashes still surface as failures.

Each resize verifies process identity, stops only the affected language's child
process tree, verifies and links committed checkpoints, records provenance and
launches a separate continuation. Other language jobs keep running. The active
combined launch tracks those independent supervisors and terminal logs.

Active launch: `../artifacts/european-expansion/checker_scale_v1/launch.json`.
Controller PID/command: `../artifacts/european-expansion/checker-autoscaling/controller-launch.json`.
Controller decisions, measured rates and resize receipts are stored alongside it.
The active-campaign pointer includes the controller receipt, so future pause or
stop operations must stop the controller first to prevent it initiating a resize.
No generation code, corruption policy, source text or quality threshold was
changed during these execution resizes. No upload or human review occurred.


The user subsequently authorized increasing parsers too when they become the
bottleneck. The replacement controller scales either stage by 50%, using busy
parsers plus a depleted prepared queue to identify a parser bottleneck, or a
full prepared queue and busy checkers to identify a checker bottleneck. A full
queue still permits checker growth when both stages are busy. Plateau comparisons
are stage-specific so an unsuccessful checker increase does not prevent a needed
parser increase. Four focused allocation-policy tests pass, including parser
scaling, CPU/memory limits and simultaneous stage activity.

The updated host budget is 90% of available logical CPUs, reserving roughly 10%
for other work; memory checks reserve 5% of host RAM plus a fixed 16 GiB and
estimated incremental demand. Parser growth uses measured maximum worker RSS
with 30% margin; checker growth budgets 3 GiB per new JVM. These are headroom
limits, not an instruction to exhaust RAM or trigger swapping/OOM.

The old checkers-only controller was stopped while idle; the replacement is
registered in the same controller-launch receipt with mode `parser_and_checker`.
All three languages are generating after replay. Observed live allocation:
French 20 parsers/4 checkers/32 screening clients; Ukrainian 72/6/48; Greek
40/4/32. Thus 132 parser processes and 14 checker JVMs are currently alive;
actual CPU use differs from allocated process counts. Further changes will be
recorded automatically by the controller. See `checker_scale_v1/live-allocation.json`.
