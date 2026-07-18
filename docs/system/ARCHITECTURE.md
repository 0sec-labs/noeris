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

The fixed-policy `zero_research_kaggle_worker` is the production raw-artifact
producer for one allocation. Its public CLI permits only candidate,
authorization, plan, allocation ID, output directory, and Kaggle kernel ref;
trust paths, namespaces, principal, signing key, and image identity are fixed.
The controller and worker allowed-signers files must be root-protected and
key-disjoint. The worker key is an ephemeral secret whose public fingerprint is
pinned separately. The worker cannot load reference-oracle or controller
usage-observer keys and cannot emit either receipt.

Inputs for the correctness oracle use `pinned-float64-matmul-v1`. For each
tensor, SHA-256 counter blocks over the exact domain, case seed, tensor name,
and counter yield bits in least-significant-bit-first order. Bits map to
`{-1,+1}`, then both input tensors are scaled by the same exactly representable
power of two. The reference is an exact signed-int64 dot product converted to
float64 and scaled by the squared power of two. Reference and two separately
executed outputs are retained as little-endian float64 bytes. Timing samples are
retained as positive integer nanoseconds; the proposal carries their exact
millisecond conversion.

The worker pins the T4 runtime, software-image digest, git commit, tracked-tree
content digest, six allowed Triton knobs, randomized arm order, warmups, samples,
and hard shape, memory, FLOP, raw-artifact, verifier-series, wall-clock, and
zero-dollar ceilings. It stages owner-only files and publishes the allocation
directory atomically with Linux `renameat2(RENAME_NOREPLACE)`. A retry verifies
the canonical proposal, signatures, receipt bindings, usage receipt, and every
raw byte digest before returning the retained result without another GPU run.
Every signed artifact path is rooted at the planned allocation ID
(`<allocation-id>/usage.json` and `<allocation-id>/raw/...`). This lets a
controller assemble three independently downloaded allocation trees beneath one
series root without rewriting signed receipts or allowing one allocation to
alias another allocation's bytes.

Its outputs remain only `noeris-kernel-tournament-proposal-v1`,
`noeris-kernel-allocation-artifacts-v1`, `noeris-kaggle-usage-v1`, and raw
artifacts. They are categorically unaccepted. The separate 0brain series
verifier may emit accepted evidence only after it proves unique allocations,
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
