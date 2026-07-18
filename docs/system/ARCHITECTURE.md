# Architecture

## V1 Principle

Noeris should be built as an evidence pipeline, not a chat agent with tools glued on.

The smallest credible system is:

- deterministic enough to replay
- modular enough to swap components
- strict about verification before publishing memos

## Core Loop

1. Ingest new papers, repos, benchmarks, and evals.
2. Normalize them into structured research objects.
3. Build topic-local research memory with claims, evidence refs, open questions, and contradictions.
4. Rank candidate hypotheses.
5. Generate bounded experiment specs.
6. Execute experiments in reproducible runtimes.
7. Verify whether the cycle produced empirical evidence or only planning artifacts.
8. Publish research memos with artifact references and explicit risks.

## Component Boundaries

- `source_provider`
  collects papers, repos, benchmarks, and other research inputs
- `research_memory`
  derives structured claims and later becomes the claim/method graph layer
- `hypothesis_planner`
  proposes candidate ideas from the topic and current memory
- `experiment_planner`
  converts ranked hypotheses into bounded experiment specs
- `experiment_executor`
  runs or simulates experiments and captures outcomes
- `verifier`
  blocks publication if evidence is missing or the cycle is incomplete
- `memo_writer`
  turns the cycle into a human-readable report with machine-usable structure

## State Model

The core state unit is a `ResearchCycle`.

It should include:

- topic
- research context
- hypotheses
- experiment specs
- experiment results

This is the minimum state needed to:

- replay a cycle
- inspect where a claim came from
- compare planned vs executed work
- decide whether a memo is publishable

## Verification Gates

Every cycle should pass explicit gates before it is treated as evidence-backed.

Current gate shape:

- sources present
- claims present
- hypotheses present
- experiments present
- results recorded
- empirical execution attached

Later gate shape should add:

- source freshness
- citation coverage
- experiment artifact integrity
- baseline comparison completeness
- contradiction and regression checks
- novelty scoring confidence

## Execution Model

V1 should be synchronous and local-first.

That means:

- no distributed orchestration yet
- no queueing system yet
- no long-running multi-agent runtime yet
- no autonomous background loops until the single-cycle contract is solid

## 0research Challenger Boundary

Noeris can now turn matching world-model hypotheses into a deterministic batch
of at most five complete kernel configurations. This is a proposal boundary,
not an evaluator:

- only parameters declared by the operator are copied into a challenger;
- list-valued hypotheses expand into scalar, executable configurations over a
  complete validated baseline;
- out-of-domain and shared-memory-invalid configurations are rejected;
- every usable hypothesis must carry durable source references;
- the output binds generator identity, rationale, evidence references, and the
  full configuration into a content-addressed ID;
- the output cannot contain budgets, corpora, scores, evaluator settings,
  promotion authority, or measured outcomes.

The trusted 0brain controller separately pins the generator and knob allowlist,
injects a sealed manifest plus disjoint evaluation corpora, and grants only
draft-PR authority. GPU benchmarking and grading remain independent of Noeris.

The file transport is the generator-only `0research-export` command. It reads
an explicit world-model snapshot, source-reference map, complete baseline, and
shape; then emits a canonical JSON array that can be passed directly to
0brain's `improvement-project` command:

```bash
research-engine 0research-export \
  --world-model world-model.json --source-refs source-refs.json \
  --baseline baseline.json --shape shape.json \
  --operator matmul --hardware H100 \
  --generator-id noeris.world-model-v1 \
  --generator-digest sha256:<pinned-source-digest> \
  --output challengers.json
```

The command fails if it cannot produce at least one valid challenger. It never
runs a benchmark, updates the world model, or supplies evaluation and promotion
policy.

### Untrusted kernel evidence proposals

`zero_research_tournament._build_untrusted_tournament_proposal` is a private,
dependency-injected adapter-development seam for one controller-planned
allocation. It requires four independent
inputs: the exact projected candidate, the content-addressed plan, the exact
signed private controller envelope, and the runtime environment identity. It
reverifies the controller signature and candidate/authorization digests, then
rederives every seed and per-case arm order from the signed nonce before calling
an in-process GPU runner.

The runner must return reference correctness, two byte-identical same-input
outputs, and the full controller-requested timing sample vector. The retained
evidence includes every raw sample plus GPU UUID, driver, CUDA, Python, Torch,
Triton, software-image, repository-tree, evaluator, and zero-dollar usage
identities. An independently allowed worker must sign the exact evidence and
device identity before the builder returns it. The result is the distinct
`noeris-kernel-tournament-proposal-v1` contract with an immutable
`acceptedBy0brain: false` marker. It is not production evidence, and 0brain must
reject the proposal schema categorically rather than trusting that marker.

Kaggle kernels cannot receive independently protected `/etc` policy mounts, an
ephemeral `/run/secrets` signing-key mount, or a controller-pinned OCI image.
`zero_research_kaggle_capture` is therefore the provider-compatible execution
boundary. It accepts only candidate, authorization, plan, allocation ID, output
directory, and Kaggle kernel ref; it has no signer, key, policy, oracle, or
observer surface. It emits the unsigned and categorically unaccepted
`noeris-kaggle-allocation-capture-v1` contract plus raw artifacts and a plainly
labelled `noeris-kaggle-self-report-v1`. The self-report has
`independentlyObserved: false`; it is not zero-dollar evidence.

Real provider dispatch uses the separate v2 capsule path. A signed tournament
round binds a reviewed `noeris-kaggle-execution-template-v1` file-tree digest,
not the digest of the final package. After the plan exists, the controller
builds `execution-capsule.json` with the exact candidate, signed controller
envelope, plan, allocation, Kaggle ref, and the complete canonical template
manifest. The final provider package is then the fixed private/offline metadata,
that inert capsule, and the exact template files. This ordering is deliberately
acyclic: the signed dispatch can bind the plan, template, capsule, final package,
and runtime-release digests without requiring a package to contain its own
digest.

Before CUDA initialization, the v2 capture wrapper requires canonical capsule
and metadata bytes, recomputes the capsule and template digests, selects exactly
one matching v2 plan round, hashes every template file, and rejects any missing,
extra, traversing, symbolic-link, or unsupported package entry. Its capture is
`noeris-kaggle-allocation-capture-v2` and binds the exact capsule and template
digests. The template tree is the executed code identity. Repository commit and
tree values remain signed provenance; the offline provider does not pretend to
verify them through an unavailable `.git` checkout. The legacy v1 Python seam
remains for migration tests only and must not be accepted by real v2
dispatch/intake.

The capture records Kaggle's observable `BUILD_DATE` and `GIT_COMMIT`, exact
Python/Torch/Triton/CUDA versions, and a recomputable
`noeris-kaggle-runtime-v1` fingerprint. It deliberately does not assert an OCI
image digest. A trusted external collector must map those observable release
markers to a controller-approved Kaggle release manifest and independently
observed provider status before an isolated worker attestor may sign proposal
or artifact receipts. `zero_research_kaggle_worker` remains the fixed-mount
contract/reference implementation, but must not be dispatched to Kaggle with a
self-materialized signing secret or policy and called independently protected.

Inputs for the correctness oracle use `pinned-float64-matmul-v1`. For each
tensor, SHA-256 counter blocks over the exact domain, case seed, tensor name,
and counter yield bits in least-significant-bit-first order. Bits map to
`{-1,+1}`, then both input tensors are scaled by the same exactly representable
power of two. The reference is an exact signed-int64 dot product converted to
float64 and scaled by the squared power of two. Reference and two separately
executed outputs are retained as little-endian float64 bytes. Timing samples are
retained as positive integer nanoseconds; the proposal carries their exact
millisecond conversion.

The capture pins T4 execution, git commit, tracked-tree content digest, six
allowed Triton knobs, randomized arm order, warmups, samples, and hard shape,
memory, FLOP, raw-artifact, verifier-series, and wall-clock ceilings. It stages
owner-only files and publishes the allocation directory atomically with Linux
`renameat2(RENAME_NOREPLACE)`. A retry reverifies the exact input digests,
plan-derived result order and seeds, runtime fingerprint, code identity,
self-report, raw paths, timing bounds, and every artifact byte before returning
without another GPU run. Every artifact path is rooted at the planned
allocation ID, allowing a controller to assemble distinct downloaded trees
without rewriting capture bytes or permitting cross-allocation aliases.

Capture outputs remain categorically unaccepted. The external attestor must
derive the existing `noeris-kernel-tournament-proposal-v1` and
`noeris-kernel-allocation-artifacts-v1` contracts without changing captured raw
paths. The separate 0brain series verifier may emit accepted evidence only
after it proves unique allocations,
distinct GPU UUIDs and Kaggle refs, independent oracle and controller-observer
receipts, reproduced identities, and whole-series zero-dollar compliance. Only
then may at least three allocations inform a learning decision. Neither the
worker proposal nor accepted evidence may train a model, open a PR, merge,
deploy, or publish on its own.

## What To Defer

- generalized multi-domain research
- complex multi-agent society simulations
- pre-training infrastructure
- hosted collaboration layer
- heavy graph database choices
- autonomous self-modifying planner loops

## Initial Build Order

1. stable research object schema
2. explicit component interfaces
3. single-topic research cycle with verification gates
4. arXiv and GitHub ingestion
5. artifact-backed experiment result format
6. first claim/method memory layer
7. continuous topic monitoring
