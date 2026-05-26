# ScienceAgentBench Local Scaffold

This is an experimental local-only scaffold for issue #117 and the broader
ScienceAgentBench direction in #85. It does not run ScienceAgentBench tasks and
does not claim any ScienceAgentBench score.

## What Exists

- `research_engine.science_agent_bench.TaskRegistry` registers deterministic
  science task specs.
- `ScienceTaskSpec` captures a task id, discipline, task type, input payload,
  expected output, and enabled strategies.
- `ExperimentDatabase` records strategy results and extracts reusable
  cross-task memory items.
- `ScienceStrategyRouter` chooses among `decompose_subtasks`, `direct_solve`,
  and `retrieve_and_adapt` without an LLM, cost model, or paid compute.
- `DeterministicScienceRunner` executes a toy bioinformatics normalization
  fixture and records the result.

The fixture is intentionally small: normalize `[2, 4, 6]` to a target sum of
`24`, yielding `[4, 8, 12]`. Its purpose is CI validation of the Noeris-style
loop, not benchmark performance.

## Kernel Search Mapping

| Noeris kernel search | Local science scaffold |
|---|---|
| `TritonOperatorSpec` / operator registry | `ScienceTaskSpec` / `TaskRegistry` |
| `ConfigDatabase` records measured configs | `ExperimentDatabase` records strategy attempts |
| Selector/router chooses configs | `ScienceStrategyRouter` chooses task strategy |
| Reusable insights from prior runs | `ScienceMemoryItem` extracted from successful tasks |
| GPU runner | Deterministic local fixture runner |

## Deferred

- Real ScienceAgentBench task loading and gold-file evaluation.
- LLM proposal, tool-use, and code-generation loops.
- Docker/E2B sandbox execution for arbitrary task code.
- Cost modeling. v0 strategies are cheap and deterministic.
- Any leaderboard, success-rate, or score claim.

## Next Benchmark Integration Step

For #85, the next concrete step is to add an adapter that converts one
ScienceAgentBench-Lite task fixture into `ScienceTaskSpec`, runs it in an
offline sandbox, and compares the produced artifact to the benchmark gold
output. Only after that adapter exists should the repo report task-level
ScienceAgentBench success or failure.
