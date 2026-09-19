# Scenario and evidence matrix

| Scenario | Test / evidence | Status |
|---|---|---|
| Spherical scale, opposite direction, zero/nonfinite input | `test_spherical.py` | Automated |
| Missing facet, incompatible family, invalid weights | `test_spherical.py` | Automated |
| Full / compact payload round trip and malformed input | `test_spherical.py`, `test_activation.py` | Automated |
| Identical hashes for different entities | `test_activation.py` | Exact constraint enforced |
| Activation threshold, empty shortlist, no answer | `test_activation.py`, `verify_spherical.py` | Automated and replayed |
| Graph cycles, forward/reverse dependency traversal | 100 seeded random graphs in `test_activation.py` | Independent oracle comparison |
| Stale source, same-revision content change, revision-only update | `test_spherical.py` | Automated |
| Unauthorized dependency and no information in failure | `test_activation.py` | Automated local-role check |
| Contradictory evidence and exact packet budget | `test_spherical.py` | Automated |
| Structured missing claims, wrong entity, stale metadata | `test_requirements.py`, `structured-v1` | Regression and prospective fixtures |
| Text retrieval across domains | `spherical-public-v1`, `residual-v1/v2`, 2,029 queries | Binary-only loses; richer residual refinement matches dense |
| Fixed 256-bit allocation | `stamp256-v1`, 21 training runs | Equal allocation retained; 32 raw bytes, metadata external |
| Stateful handoffs and code contracts | `workflow-efficiency-v1` through `v4` | Final selected16/16 vs full10/16; typed deterministic rendering |
| Read-only retrieval concurrency | `scale-comparison-v1` | 1k/10k/100k rows and1/4 workers; Faiss controls; no distributed-service claim |
| Multi-view versus multi-direction equal 256-bit representation | `spherical-v1`, `spherical-v2` | Procedural supplied-facet experiment |
| Learned relevance training and validation-only tuning | Six exported spherical models; `verify_spherical.py` retrains | Reproducible; synthetic domain only |
| Text generation with complete and incomplete packets | `spherical-audit-v1`, `spherical-guarded-v1` | Actual local Qwen2.5-1.5B inference |
| Constrained code generation | `local-tasks-v1`, `local-tasks-v2` | Actual local model, tiny AST grammar; no arbitrary execution |
| Image / speech / video generation | `multimodal-v2`, 36 calls, 18 pairs | All output hashes match; unchanged generator tokens; routing overhead |
| Real repository coding, cross-team coordination | No independent completed task benchmark | Open gate |
| Automatic facets / inferred relationships | Not implemented | Open research task |
| ARM/mobile power, networked agents and production scale | Not measured | Open gate |
| Internal attention/KV-cache acceleration | Not implemented | Research hypothesis only |

All LLM test counts report both calls and unique cases. Repeated calls and
projection seeds are not additional independent tasks. Timing is machine-specific;
the replay verifier reproduces rankings and grades, not wall-clock durations.
