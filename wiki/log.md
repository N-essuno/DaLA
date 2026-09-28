# DaLA update log

- 2026-09-21: Renamed the English corpus to **DaLA English — Common Pile**, directory `la_output/english_common_pile_scaled/`, to avoid implying TV2R source material. Updated display metadata, profile, script defaults and documentation; preserved original manifests and evaluation receipts. Data artifacts are unchanged and their checksums were rechecked. The finalizer allows display-name-only profile changes while retaining the original generation-profile hash.

- 2026-09-21: Completed the scale dataset at `la_output/english_tv2r_scale`: 478,930 pairs / 957,860 rows per task (2.67% below the acceptability reference, 2.93% above correction). Splits have 766,288 / 47,892 / 143,680 rows. It realizes 47,073 substitutions and is 12.46 times the prior corpus. All 73 tests, artifact validation, pinned source-span checks and input-hash checks pass; Danish core/profile remains unchanged. Source review flagged 18 erroneous/uncertain originals and four boilerplate cases in 130 pre-curation pairs; affected documents and recurring boilerplate were excluded. All 108 accepted reviewed pairs are unchanged in the final data. The card explicitly preserves provisional checker-screened status; human precision remains unmeasured. See [scale assessment](/pages/english-scale.md).

- 2026-09-21: The scale artifact audit found valid JSON strings containing U+2028 that Python `splitlines()` incorrectly treated as separate JSONL records. Fixed review/validation/upload readers to read physical records; added an export/load/validation regression (73 tests passed). Inspection also found a joined heading/fragment, so final source exclusions now remove ambiguous embedded line separators. Preserved the first written export as pre-audit data before rebuilding.

- 2026-09-21: Full-scale export validation caught two valid token swaps with accented English words (`café`, `façade`) rejected by an ASCII-only independent validator. Aligned its alphabetic-token/whitespace contract with generation and added regression coverage; 72 tests passed. The failed export produced no final dataset. Screened receipts remain reusable under exact, hash-verified validation-only changes.

- 2026-09-21: Started the TV2R-scale English build targeting 492,063 pairs / 984,126 rows per task. Added pinned Global Voices long articles, publisher-license provenance and authors. Shared candidate preparation, indexed equivalent deduplication and checksummed batches support resumption. The 100-document ordinary/batched/resumed comparison matched all 1,550 pairs; 67 tests passed. Initial 70-pair agent review accepted injected errors but flagged 12 originals, recorded as final-selection exclusions. Full-scale results are tracked in the scale evaluation page.

## 2026-09-21

* **Creation**: Established the OKF 0.2 knowledge bundle and [maintenance conventions](/schema.md).
* **Decision**: Recorded the [English contract](/pages/english-contract.md): empirical grammar and spelling, curated Common Pile sources, and both TV2R task views.
* **Implementation**: Added evidence extraction, 92 guarded mappings, pinned source loading, independent-edit composition, document splitting, deduplication, instruction exports and review support.
* **Quality**: Initial source inspection motivated a local LanguageTool gate. Mechanical screening remains distinct from human precision validation.
* **History**: Preserved the superseded EWT pilot and [TV2R findings](/pages/tv2r-and-history.md).

* **Completed build**: Exported 14,103 checked pairs (28,206 rows per task), verified all source spans and artifact integrity, passed 37 tests and recorded [results and limitations](/pages/build-results.md). Three agent-audited source exclusions do not constitute human precision validation.

* **Danish comparison**: Audited active corruption selection: 15 functions, fixed rare-first priority, productive morphological rules, and 74 fixed directed whole-word mappings (45 spelling). Corrected the comparison with the English exact-substitution restriction in [implementation history](/pages/tv2r-and-history.md).

* **Refactor**: Extracted the Danish language pack and generic operators. Frozen-reference testing passed 77,014 rule comparisons and four complete pipeline comparisons with exact row/CSV/RNG equality. Connected productive English grammar through the shared profile API; final amount/diversity/quality assessment follows.

* **Validation finding**: Full productive-English export initially failed on `lectures→lecture`: dictionary forward inflection existed but reverse lemmatization was incomplete. Added a regression and bidirectional dictionary-table validation; no guessed inflection was enabled. Failed staging produced no final dataset.

* **Completed multilingual assessment**: Final productive English build has 18,958 pairs and 2,038 distinct substitutions; 52 tests and full source/artifact checks pass. Family-stratified agent review found three source flags in 70 pairs, excluded in the final build; human precision remains unmeasured. [Results](/pages/multilingual-assessment.md).

- 2026-09-21: Expanded active English spelling from 13 to 72 sourced mappings (66 words), with Cambridge, Merriam-Webster and IELTS IDP provenance. Preserved corpus support counts and used null for list-only evidence. Excluded one checker-undetected candidate. Added evidence-validation and candidate regressions; 55 tests passed. Documented Danish POS-constrained random word deletion/swapping and absence of English random fallback or productive character spelling generators. Completed expanded build: 21,090 pairs, 67 realized spelling mappings; dominant substitution 22.0% overall / 45.3% within spelling. All 72 active spelling mappings passed US/GB screening, dataset validation passed, and a 12-pair agent spot-check found no invalid injected edits. No human precision claim or balancing change.

- 2026-09-21: Added three data-driven productive character mechanisms and two guarded English token fallback operators. Probe caught checker-accepted valid spelling reprogramed; protected variable doubling suffixes and added regression tests. Kept broad legacy POS noise disabled after English spacing and valid-deletion counterexamples. Both dialect lexical evidence is mandatory for productive spelling. Completed final build with 38,434 pairs and 14,912 distinct substitutions. Attested spellings take priority over generated spellings. Six additional source flags from probe/final review were excluded; 55 retained reviewed sources and final edits were rechecked. 65 tests and artifact validation passed. Largest substitution is 9.4% overall; spelling is 74.6% of edits. Danish implementation/profile unchanged against equivalence receipt.

- 2026-09-21: Paused HF release preparation for the requested fresh quality comparison; no upload. Uniformly sampled 100 English final pairs and 100 Danish acceptability negatives, recovering 87 Danish originals by exact paired joins and explicitly checking 13 retrieved deletion/swap counterparts. Agent inspection found English source judgments 80 acceptable / 11 erroneous / 9 uncertain versus Danish 83 / 14 / 3; clearly error-inducing edits in 100 English versus 87 Danish pairs, with six Danish grammar-preserving and seven uncertain transformations. Strict usable pairs: 80 English, 71 Danish. This is not human gold or a statistically established overall ranking. Frozen decisions and provenance are retained in [comparison](pages/language-quality-comparison.md); neither corpus was altered to remove flagged rows. Source screening and English spelling-heavy coverage require further work before a high-quality release claim.

- 2026-09-21: User authorized publication after quality comparison. Uploaded `schneiderkamplab/dala-english-common-pile` publicly with two configurations, 957,860 rows each; all remote file hashes and both streaming loads verified. Card discloses known source-label errors and spelling concentration. See release receipt and [release record](pages/hugging-face-release.md).
- 2026-09-21: User corrected the candidate list to Norwegian/Swedish/Dutch, then selected Dutch. Inspected pinned DynaWord snapshots and actual Dutch data/annotations; began Dutch pack and DynaWord source adapter. Excluded PBL from initial scope due to visible extraction damage; found Dutch LanguageTool uncategorized grammar rules and carrier false positives requiring explicit rule-ID handling and an independent OpenTaal real-word veto.

- 2026-09-21: Implemented Dutch language pack and pinned DynaWord Parquet/annotation adapter. Added OpenTaal veto, Dutch checker rule-ID classification and counterexample guards for die→dat and de reizen→het reizen; fixed source-fragment/quotation/spacing handling and a parser plural false positive. Preserved initial 2,495-pair audit (90/100 strict usable; one supplementary family-label error) and intermediate build. Final pilot: 2,552 pairs, 1,976 distinct substitutions, all six families but 97.0% spelling. New seeded sample: 91/100 strict usable, seven erroneous and two uncertain sources; all 100 injected errors valid, with five previously reviewed originals. Exported 108 agent-accepted pairs separately; neither sample is native-speaker gold. All provenance/task/lexicon checks and 87 tests passed. Recorded limits and next scaling priorities in [Dutch DynaWord](pages/dutch-dynaword.md). No Dutch upload.

- 2026-09-22: Expanded Dutch to 71 entries/nine families using pinned morphology and 50 lexical spellings; selected excellent government and modern EUR-Lex after inspecting six sources. Recovered expanded pilot after fixing dictionary-mapping export validation (96 tests pass): 1,448 pairs, spelling 49.0%, article changes 44.1%. Uniform 200-pair agent review: 187 acceptable sources, nine erroneous, four uncertain; all injected edits valid. All 16 family supplements accepted. Exact ordinary/checkpoint pair equivalence holds on 200 documents. Eligible pool is 10,273 documents; rough source-stratified yield projection is 52k pairs, not English-scale 479k. Recorded remaining production quality/capacity limits and proposed exclusions in [readiness](pages/dutch-scale-readiness.md); no production run or Dutch upload.

- 2026-09-22: User approved the larger Dutch validation build and balance experiment. Added opt-in extraction/pronoun guards and 13 prior audit exclusions; started frozen 1,000-document checkpoint run. Implemented maximal per-split article/spelling caps without changing source pairs. 102 tests pass. Measured 733,575 eligible paragraphs across the selected source pool versus 120,501 at cap 12. Results pending in [larger validation](pages/dutch-larger-validation.md).

- 2026-09-22: Completed 1,000-document Dutch validation: 6,474 screened pairs, 2,325 under per-split article/spelling caps, and 2,312 after excluding 13 newly flagged originals. Fresh uniform audit of 200 previously unreviewed pairs: 188 acceptable sources, seven errors, five uncertain; all injected errors valid. Separate six-pair supplement had one source error. Preserved both measured outputs and audits; final subset is review-informed, not independently re-audited. Full source recovery, spelling vetoes, unchanged-subset/exclusion/cap checks and 102 tests pass. Realized agreement spans 60 verb lemmas and 93 adjective lemmas, but d/dt and relative-pronoun edits remain one example each. Current-depth projection is approximately 48k raw / 16k balanced pairs, not a quota guarantee. [Results](pages/dutch-larger-validation.md). No Dutch upload.

- 2026-09-22: At the user's request, made `nl_scale` the active uncapped Dutch production profile: all eligible paragraphs, other grammar before article-number/article-gender edits, spelling fallback, and no percentage downsampling. Merged all 46 audited source exclusions; retained historical profiles for reproducibility. Targeted paragraph/priority checks pass. End-to-end 20-document probe: 1,019 paragraphs, 342 pairs versus 240 paragraphs/136 pairs for the same capped documents. Article share 36.8% versus 47.8%; full artifact, source-recovery, spelling and exclusion checks pass. Full-corpus generation not launched by this configuration change. [Details](pages/dutch-uncapped.md).

- 2026-09-22: User authorized full-scale Dutch construction with `nl_scale`. Started all 10,273 documents, 103 batches. Added optional identical local checker pooling after measuring the single-process bottleneck; 104 tests pass and 200 real candidates match exactly against single-checker evidence/reasons. A 16-instance cold-cache benchmark achieved 66.2 candidates/s. Production restarted with 32 local instances on the 384-core host; unchanged linguistic checks and source settings, new checkpoint identity. [Run record](pages/dutch-full-scale.md).

- 2026-09-22: User requested 128 concurrency. Stopped the 32-instance attempt cleanly and archived its checkpoints/log; configured 128 local checker instances and 128 concurrent screening workers (two server threads each). Linguistic settings remain unchanged. Restarting the full Dutch build with a fresh checkpoint identity.

- 2026-09-22: Completed full uncapped Dutch build: 188,182 pairs / 376,364 rows per task, 9,537 contributing documents, 64 active rules and 19,892 distinct substitutions. Independent artifact/task/source checks and all 100,269 spelling vetoes pass. Fresh uniform 200-pair agent audit excluding prior reviewed originals: 190 acceptable sources, six erroneous, four uncertain; all 200 injected edits valid. Separate 22-pair supplement entirely accepted. Preserved ten source flags without changing the measured frozen output; no upload. [Results and limitations](pages/dutch-full-scale.md).

- 2026-09-22: User authorized investigating Rechtspraak and Officiële bekendmakingen and starting an extension to English size. Downloaded pinned data/annotations; selected excellent written sources and explicit modern publication headers for official announcements. Added opt-in paragraph-length setting with unchanged defaults, preserved offsets and additional source-risk filters. Started 400-document pilot; prepared deduplicated base-preserving extension/merge toward 478,930 pairs. [Investigation](pages/dutch-legal-extension.md).

- 2026-09-22: Legal-source investigation/pilot completed: 7,685 pairs; first 200-pair audit 180 acceptable sources and all edits valid. Conservative filtering retained 7,543 unchanged pairs. Fresh 50-per-source audit: Rechtspraak 49 acceptable, one erroneous; official announcements 44 acceptable, four erroneous, two uncertain; all 100 edits valid. Excluded all 27 pilot flags, added conservative extraction/segmentation guards and prioritized Rechtspraak 10:1 by document. 108 tests, full base replay and real merge smoke validation pass. Started 128-instance/128-worker extension driver toward 478,930 combined pairs, followed by automated assessment and a fresh sample pending linguistic review. [Details](pages/dutch-legal-extension.md).

- 2026-09-22: User authorized final audit and upload. Audited 200 previously unreviewed pairs from the 478,930-pair Dutch extension: 187 acceptable originals, ten erroneous, three uncertain; all 200 edits valid. Separate 25-pair supplement: one uncertain source, all edits valid. Exported unchanged 478,916-pair release after excluding 14 flags; mechanical validation passed. Preparing `schneiderkamplab/dala-dutch-dynaword` with two chat configurations and explicit quality limitations. [Release record](pages/dutch-hf-release.md).

- 2026-09-22: Published `schneiderkamplab/dala-dutch-dynaword` at revision `a09dffe55c9db956f3ae39ed1916cb2d5dfaff73`: 478,916 pairs, 957,832 chat rows per configuration. Both full local loaders passed; all remote files/sizes/hashes and both remote streaming configurations verified. Card discloses the 187/200 pre-exclusion source audit and 14 removed flags. [Release record](pages/dutch-hf-release.md).
# 2026-09-24 — Six separate language datasets

Accepted separate pl/sv/nb/nn/fo/is datasets; large-language target 478,930 pairs,
smaller-language source exhaustion, and broad grammar/spelling coverage independent
of size. Added pinned corpus/resource inventories, separate standard prompts,
Stanza parser integration, dictionary-screened candidate rules and a six-language
pilot runner. Faroese uses explicit GiellaLT paradigms because FarPaHC lacks lemmas.
Candidate outputs are not independently grammar-checked or release-ready. See
[contract and remaining gates](pages/six-language-expansion.md).

- 2026-09-24: Expanded six-language candidate preparation with SGJP, SALDO, separate Bokmål/Nynorsk Ordbank, BÍN and Faroese GiellaLT normative generation. Fixed Nynorsk tag conversion and syncretism guards; source audits added agreement/extraction checks and optional Polish/Swedish original-sentence LanguageTool screening. Earlier pilots are superseded; all-source audit pilots are running. 119 tests passed before the last audit-driven changes. No new production or HF publication claim. See [six-language expansion](pages/six-language-expansion.md).

- 2026-09-24T06:40:48.770004+00:00: Started six full candidate builds (100 parser workers), with a detached automatic validation/isolation/coverage finalizer. 122 tests, Danish 77,014 comparisons, 206-pair fragmentation/resume equivalence and all six semantic conformance suites pass. Corrected Faroese Wikipedia source tag before launch. Outputs remain candidates awaiting final linguistic audit; no HF upload. See [live run and evidence](pages/six-language-expansion.md#full-candidate-run-scale_v1).

- 2026-09-24T06:46:55.753368+00:00: Early full-run audit flagged two of 36 inspected pairs. Fixed systematic Polish/Icelandic seed-limited surface-analysis gaps by native-lexicon closure; PL/IS scale_v1 superseded by scale_v2. Polish ambiguity regression passes. Swedish resumes after checker transport failure. Added final selection review exclusions, tested on a separate 246→245-pair smoke output. See [audit and replacement runs](pages/six-language-expansion.md#early-scale-audit-and-replacement-runs).

- 2026-09-24T08:15:06.429442+00:00: Status inspection found nb/nn/fo receipt-publication races and pl/sv checker transport failures. Fixed atomic publication/future synchronization and bounded per-request retries; all 125 tests pass. Verifying unchanged checkpoint batches into explicit recovery runs with old/new code provenance; original runs retained. See [recovery](pages/six-language-expansion.md#2026-09-24-checkpoint-recovery).

- 2026-09-24T08:16:11.670446+00:00: All six verified checkpoint sets migrated and recovery_v1 builds/finalizer launched. Fresh 206-pair ordinary/fragmented/resumed outputs equal the pre-fix canonical pairs; 125 tests pass. Current-run pointer recorded; no linguistic changes or uploads.

- 2026-09-24T08:20:21.801627+00:00: Recovery builds continue; screened output realizes 8–12 grammar families per language plus spelling. Audited 65 family-stratified examples: 58 provisional accepts, five rejects, two uncertain; seven flagged originals queued for final exclusion. Not a precision estimate or human validation. See recovery_v1/live-audit artifacts.

- 2026-09-24T09:17:05.606288+00:00: Bokmål completed 478,930 candidate pairs and Faroese exhausted selected sources at 88,237. Both automatic artifact/task/provenance validations passed. Fresh 40-pair agent audit provisionally accepted 39 and flagged one uncertain source; native dictionary evidence resolved another concern without exclusion. Other four builds continue; final isolation/audit pending.

- 2026-09-24T10:37:33.381939+00:00: User requested Nynorsk cap478,930 without deletion. Full871,199-pair pool and checkpoints preserved. Creating separate 80/10/10 capped export and configuring final isolation to select the same cap from the full reservoir. No generation changes.

- 2026-09-24T10:41:55.364186+00:00: Completed and validated separate Nynorsk478,930-pair export (383,144/47,893/47,893). Full871,199-pair pool unchanged; all eight grammar families and spelling retained. Final assembly configured for the same cap.

- 2026-09-24T10:53:05.750871+00:00: Applied Icelandic478,930 stopping target and64 parser workers for each active language (PL/SV/IS). Checksum-verified batches migrated to scaled64_v1 without deleting originals or changing linguistic inputs; builds/finalizer relaunched. Completed datasets remain preserved.

- 2026-09-24T11:02:51.222579+00:00: Found and fixed lazy executor startup on checkpoint-heavy resumes. Explicit prestart is pinned in scaled64_v2 profiles; all126 tests and206-pair equivalence checks pass. Relaunched PL/SV/IS with64 parser slots each and478,930 targets, preserving earlier checkpoints. Final live process-count verification underway.

- 2026-09-24T11:05:23.143417+00:00: Verified64 live parser processes each for PL/SV/IS (192 total). Icelandic stopping cap and all three split targets confirmed. Current-run pointer and PID evidence updated; all existing datasets/checkpoints preserved.

- 2026-09-24T11:31:20.050329+00:00: Paused PL/SV/IS and finalizer at user request; no live owned processes remain. Preserved123,655 /164,392 /128,744 selected pairs and all checkpoint files. Last committed batch checksums verified; resume commands and process history recorded in scaled64_v2/pause.json.

- 2026-09-24T11:38:05.903093+00:00: User requested resume. Verified unchanged code/profiles/resources and worker-pool pins; relaunched PL/SV/IS scaled64_v2 and finalizer from preserved checkpoints with64 parser workers each and478,930 targets. Previous pause/launch receipts archived; nothing deleted.

- 2026-09-24T14:09:14.369762+00:00: Diagnosed PL/SV local-port exhaustion from retained per-batch checker sessions. Added pinned persistent screening threads;128 tests pass and160 live checks have identical responses (161→9 retained sessions with8 threads). Preparing verified-checkpoint PL/SV restart; Icelandic continues unchanged.

- 2026-09-24T14:10:44.513749+00:00: PL/SV resumed as screening_v1 from7,764/7,652 verified screened batches, preserving225,852/336,166 selected pairs. Finalizer restarted with updated run mapping; IS remains live with64 workers. Active launch receipts recorded for both supervisors.

- 2026-09-24: User requested128 parser workers for remaining Polish/Icelandic builds; preserving checkpoints and478,930 caps in scaled128_v1. Swedish finished. Added explicit runner worker-count override; three pool tests passed.

- 2026-09-24: Scheduled user-requested Icelandic128→256 worker handoff after successful Polish completion. Preserves checkpoints, linguistic settings, and478,930 cap; watcher is included in pause receipts. Three orchestration tests passed.

- 2026-09-26: Surveyed and ranked23 additional European languages for DaLA grammar/spelling readiness. Recorded primary resources, observed versus synthetic evidence, MultiGEC redistribution limits and provisional family targets. Recommended first group: German, Czech, Ukrainian, Italian, Spanish, Estonian. Resource research only; no new language precision validation, builds or uploads.

- 2026-09-26: User confirmed promoting French to the first implementation group (tier A), alongside German, Czech, Ukrainian, Italian, Spanish and Estonian. Updated wiki and structured survey; provisional rank7 and unresolved access/linguistic checks retained.
- 2026-09-26: Reassessed Portuguese varieties separately: pt-PT11/B, pt-BR12/B (24 ranked language/variety entries). Added Brazilian spelling research, constructed benchmark, CoGrOO and clean-source leads; distinguished observed errors, constructed pairs, scored essays and corpus size. Preserved French promotion. Agent research only; no human linguistic validation or production-yield claim.

- 2026-09-26T06:29:53.028942+00:00: Completed twelve European candidate pilots (de, fr, es, it, cs, pt-PT, fi, et, ca, el, ro, uk), 9,346 pairs. Reused pair pipeline/MorphologyPack; extracted shared compiler, added annotated-source and configurable lexical backends, isolated language inputs and standard checks. Inspected212 pairs (192 provisional accepts,8 rejects,12 uncertain), excluded20 originals and rebuilt. All twelve artifacts validated;139 tests pass; existing Nynorsk mappings/semantics preserved. No native linguistic validation or upload. Small/missing coverage and split cells are explicit in the pilot concept.

## 2026-09-26 — CPU expansion of European pilots

Continued all authorized non-GPU preparation using the shared pipeline. Downloaded all twelve CPU parser packages, pinned independent Wikipedia/Europarl source pools and language-specific UniMorph evidence. Added a shared canonical source adapter, unknown/ambiguous-analysis vetoes and opt-in exact-offset isolation of contracted sentences. Mined training-only German/Czech/Ukrainian corrections; agent-reviewed 47/50/2 nonword maps respectively. Kept checker-authored fixtures distinct from observed data. Source-only diagnostics passed automatic validation; expanded diagnostics and independent checker outcomes are recorded in [the experiment](pages/european-cpu-expansion.md). No native validation, GPU inference or publication.

Completed CPU expansion assessment: **3,032 selected candidate pairs** (1,081 grammar, 1,951 spelling), splits **2,442/253/337**, all twelve languages nonempty in all splits. Greek used 1,000 articles and pt-PT 252 sitting groups; the original annotated pilots remain preserved. Final source review and observed-map priority were applied through shared generation/export code. **158 tests passed**, twelve selected datasets passed mechanical validation, pinned resources verified, Nynorsk mappings unchanged. Observed-map export regression tests cover capitalized forms and mapping-only rulebooks. Native validation and documented family/source coverage gaps remain. See `wiki/artifacts/european-expansion/current.json` and [the experiment](pages/european-cpu-expansion.md).

## 2026-09-26 — Grammar-family gap diagnosis

Distinguished six zero-mapping families from 22 configured but unrealized families. Identified Finnish POS, Romanian set-valued case and Ukrainian feature-applicability bottlenecks; recorded a read-only Ukrainian compiler counterfactual and finite-agreement context restrictions. Proposed targeted fixes and per-gate diagnostics; no rulebooks/datasets changed. See [CPU expansion](pages/european-cpu-expansion.md).


## 2026-09-26 — Grammar coverage fixes and complete CPU input preparation

Implemented opt-in set-valued morphology, conditional feature applicability,
Finnish attributive pronoun licensing, conservative noun subjects, explicit
pronoun features and per-gate diagnostics in the shared pipeline. Language
inventories remain separate inputs. Added pinned French/Portuguese nominal
lexicons with ambiguity vetoes and independent standard dictionary checks.
Fixed a native Voikko concurrency failure and verified serial/parallel replay.

Completed targeted CPU pilots and independent final exports: 30,897 pairs,
97/98 configured grammar-family cells; Czech person agreement remains unselected
after source-quality exclusions. Preserved all earlier versions. Recorded 140
agent judgments and prepared 518 blank native-review rows. Prepared all 87 pinned
Wikipedia shards and all eligible Portuguese Europarl sittings: 13,123,130 full
source documents, with resumable 478,930-pair CPU profiles for each language.
Full-size generation was not launched. No GPU use or upload. 165 tests passed;
Danish tested-engine hashes and Nynorsk mappings remain unchanged; Finnish
checkpoint replay reproduced exact pairs. See [coverage preparation](pages/european-coverage-preparation.md)
and `artifacts/european-expansion/coverage_final/` for receipts and limitations.


## 2026-09-26 — Full European candidate generation launched

Started `european_scale_v1` for all twelve languages with 192 CPU parser workers
in total (16/language), one parser thread each, no document cap and 478,930-pair
per-language targets. Fresh profiles and checkpoints preserve earlier results.
Reused checksum-pinned parser startup and persistent screening helpers; their
three focused tests passed. Supervisor startup was confirmed for all twelve
language jobs. Run receipt: `artifacts/european-expansion/european_scale_v1/launch.json`.
Generation is running; these are not completed or human-validated datasets.

Startup health check confirmed all twelve languages producing checkpoints, all 192 parser workers alive, and 17,952 retained pairs at 2026-09-26T10:36:24.700122+00:00. Interval throughput is recorded in the run's status history; startup rates are not a production ETA.

- 2026-09-26: Inspected production yield differences. Confirmed whole-sentence
  rejection of multiword expansions and overlapping opaque-sentence/length
  counters; corrected interpretation of reported sentence throughput. Recorded
  shared-parser recovery and profiling recommendations in European coverage
  preparation. Active builds and pinned generation code remain unchanged.

- 2026-09-26: Implemented opt-in contraction recovery in an isolated code snapshot,
  preserving all active production code hashes. Added exact-span and controller
  guards, separate diagnostic caches and optional reuse of the checker pool.
  Paired diagnostics recover additional French, Italian, Greek and pt-PT pairs.
  Audited 120 checker-rejected and 33 recovered examples as agent inspection;
  recorded source exclusions without changing active profiles. See European
  contraction recovery for counts and limitations.

- 2026-09-26: Contraction recovery validation completed: 172 tests passed; all four candidate exports passed independent validation. Saved a patch that passes apply-check, four separate next-run profiles, and a one/two-server checker compatibility result (40 identical diagnostics). Main production code remains unchanged.

- 2026-09-26T13:26:13.690537+00:00: Installed the remaining shared throughput improvements; all twelve paired parser checks and 178 tests passed. Stopped v1 with 1,797,691 retained pairs, migrated verified batches with explicit generator provenance, and launched v2 with 192 parser workers across eleven languages. Finnish stays complete; Portuguese recovers from its exhausted legacy source pass. See European campaign upgrade.

- 2026-09-26T13:27:54.712596+00:00: Verified v2 resume health: all 192 parser workers alive; all eleven resumed languages generating after replay; Portuguese recovery pass active. 1,820,152 retained pairs including completed Finnish; no reported failures.

- 2026-09-26T18:28:07.996225+00:00: Investigated German/Estonian final validation failures: 46/1 invalid screened edits caused by ß case folding. Confirmed Greek final-sigma dictionary false rejections and case-folded resource forms. Checked HF inventory: Finnish, Catalan, Czech and Spanish remain local only. Recorded evidence and repair scope in European orthography investigation; running code and datasets unchanged.

- 2026-09-26T18:58:00.560436+00:00: Installed shared orthographic normalization and rebuilt pinned DE/ET/EL evidence. 188 tests and three independent pilot export validations passed; paired Greek yield 37→374 on 100 documents. Reproduced Ukrainian accent/LanguageTool offset crash, quarantined only its exact original, preserved 189,558 screened rows with lineage. Launched corrected DE/ET/EL builds, Ukrainian continuation and unchanged frozen-code French resume (156 configured parser workers). Portuguese/Romanian completed at accepted shortfalls 436,051/377,246. See European orthography investigation.

- 2026-09-26T18:59:33.225120+00:00: Confirmed all 156 replacement parser workers alive and Ukrainian screening beyond its original failure, with exactly one documented source-checker quarantine. Status reporting preserves French counts during unchanged-checkpoint replay.

- 2026-09-26T20:23:00.595166+00:00: Measured checker bottlenecks (~41 actual logical cores, 76 allocated parsers). Resized FR to 20 parsers/4 checkers, UK to 72/6, and Greek to 40/4 with preserved checkpoints and lineage. Started user-authorized adaptive checker scaling on measured queues, CPU headroom and throughput. Active campaign is checker_scale_v1; controller is registered in the active pointer.

- 2026-09-26T20:25:44.174998+00:00: Extended authorized adaptive scaling to parser bottlenecks as well as checkers. Replacement controller grows the limiting stage under a 90% host CPU budget and memory headroom, with stage-specific plateau detection. Four focused policy tests pass. Verified FR20/4, UK72/6, EL40/4 parser/checker processes alive and generating; prior checkpoints retained.

- 2026-09-27T05:26:40.369389+00:00: Implemented shared fail-closed checker resilience after Ukrainian stopped at 236,958 pairs. Live replay of the exact crashing sentence now rejects/audits it and continues to a successful control without caching failure. Preparing preserved-checkpoint continuation with 128 parsers, 16 checker JVMs and 128 screening clients; outage and failure-budget guards remain hard failures. See checker resilience.

- 2026-09-27T05:28:24.755321+00:00: 201 tests passed. Verified and preserved 13,736 candidate batches / 248,382 screened rows, and launched Ukrainian checker_resilient_v1 with 128 parsers / 16 checkers / 128 screening clients. All 16 checkers started; checkpoint reconstruction is pending. Registered Ukrainian-only adaptive capacity controller in the active pointer.

- 2026-09-27T05:29:07.272063+00:00: Verified all 128 Ukrainian parser workers and 16 checkers alive; checkpoint reconstruction is advancing with 236,958 retained pairs preserved.

- 2026-09-27T10:30:20.772554+00:00: Confirmed Ukrainian completed at 478,930 pairs and passed export mechanical checks. One automatically rejected checker-crashing sentence; no parser/checker workers remain. All twelve European candidate datasets locally complete (PT 436,051, RO 377,246; ten others 478,930 each). Release audits/uploads remain pending.

## 2026-09-27 — Borrow existing GPU pool for twelve-language audit

Confirmed eight live Gemma servers at localhost:8600–8607 with GPU memory
utilization 0.45; reused without server changes. Implemented a canonical-pair
adapter around the existing HRM persistent audit queue. Completed a 288-pair
smoke test and 36 agent-authored controls; three infrastructure tests passed.
Found model explanation errors during agent inspection, so judgments remain
review signals and cannot automatically remove rows or approve a release.
Started a resumable audit of all 5,602,597 pairs with 32 clients per endpoint.
See [European pair audit](pages/european-pair-audit.md) for receipts and limits.

## 2026-09-27 — Increase audit concurrency

At user request, drained and resumed the canonical-pair audit with 64 clients per
server, 512 total. Preserved 322,083 completed judgments and all pending/failed
jobs; archived the configuration-only migration and previous launch receipt.
The existing eight vLLM servers were left unchanged. See
[European pair audit](pages/european-pair-audit.md).


## Server outage detected 2026-09-28T13:35:22.567639+00:00

All eight borrowed endpoints (8600–8607) refused connections during a status
check. Gracefully stopped the audit client to prevent consuming retries during
the outage; the borrowed servers were not modified. Preserved 5,518,894
completed judgments and 22,329 failed jobs. Client alive after drain:
False. Evidence: `artifacts/european-audit/20260927/server-outage.json`.
The main pass is incomplete and its earlier ETA no longer applies. Recovery
requires healthy endpoints and an explicit retry pass for failed jobs, including
connection failures and truncated responses. Original dataset rows remain intact.

## 2026-09-28 — Resume audit after server restoration

Verified all eight existing endpoints and resumed the unchanged audit client at
64 requests/server. Archived and reset 16,658 transient-error jobs (including
16,460 exhausted failures); preserved all 5,518,894 completed judgments.
The remaining 5,869 truncation/JSON failures still need separate recovery.
See [European pair audit](pages/european-pair-audit.md) and its recovery receipt.

## 2026-09-28 — Increase audit to 256 requests per server

User-requested increase to 2,048 concurrent requests total. Raised the CLI
ceiling, preserved the earlier frozen client and archived configuration changes.
Gracefully drained and resumed with all 5,541,209 completed judgments intact.
Three infrastructure tests passed. See [European pair audit](pages/european-pair-audit.md).

## 2026-09-28 — Main audit complete; recover exhausted responses

Confirmed the main audit finished all 5,602,597 pairs, with 5,596,599 judgments
and 5,998 exhausted failures. Archived and requeued only failures with a larger
2,048-token response allowance, preserving existing judgments and prior errors.
See [European pair audit](pages/european-pair-audit.md) for provenance.

## 2026-09-28 — Halve recovery concurrency

Set 128 requests per server as requested. The preceding pass had already exited;
archived and requeued 433 disconnected requests, preserving 5,598,242 judgments.
The 3,922 truncation/JSON failures remain unresolved. See
[European pair audit](pages/european-pair-audit.md).

## 2026-09-28 — Exclude unresolved pairs from release

At user request, marked the 3,922 discussed persistent truncation/JSON failures
as `audit_unresolved` release exclusions; preserved all originals and audit
history. Wrote machine-readable pair-ID exclusions for future task exports.
See [European pair audit](pages/european-pair-audit.md).

## 2026-09-28 — Prepare audited DaLA releases and DFM12 additions

Prepared and mechanically validated twelve passed-only European HF packages
(4,738,657 pairs). Inventoried the 14 existing audited DFM12 DaLA packages for
8 earlier languages. Began isolated training-only DFM12 screening/tokenization
using shared code, with a new guarded local-additions build interface. No upload
or live sampled-corpus replacement. See [audited release integration](pages/audited-release-integration.md).


Completion: all 24 new DFM12 task components are registered and tokenized:
15,194,140 train rows, 1,265,448,830 tokens, zero tokenization drops. All twelve
HF packages retain every parent corruption family. Final manifest and source
hash checks passed; upload remains pending.

## 2026-09-28 — Publish twelve accepted-only European DaLA packages

User explicitly authorized upload. All twelve public Hub repositories verified
by exact inventory, size and content hash, covering 4,738,657 retained pairs.
Corrected Portuguese card taxonomy to pt + BCP47 pt-PT without changing data;
updated DFM12 metadata pins. All twenty non-Danish language variants now have
published accepted-only versions. See [release integration](pages/audited-release-integration.md).

## 2026-09-28 — Commit multilingual implementation and evidence

Prepared the `multilingual` branch for publication: language-driven pipeline,
Danish compatibility, multilingual resources, tests, audit/release scripts and
OKF evidence. All 205 tests passed with `/tmp/dala-six-venv/bin/python -m unittest
discover -s tests`; wiki validation passed. Generated datasets, upload staging,
bytecode and bulky runtime logs remain local/ignored. Durable Hub publication
receipts and the DFM12 completion summary are copied into
`artifacts/european-release/`. HRM-Text integration changes belong to its separate
repository and are not included in this DaLA commit; that worktree also contains
unrelated uncommitted DFM12 framework and training changes.
