# wm-agent Agent Benchmark

| Run | Pearson | Spearman | Discover hit | Safe hit | Risk lift | Risk reduction | Agent ms | Exhaustive ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4.2-canonical-val | 0.6376 | 0.6175 | 0.6875 | 0.6875 | 0.7665 | 0.35 | 0.2537 | 634.9 |
| V4.2-canonical-test | 0.6704 | 0.6201 | 0.75 | 0.5 | 1.006 | 0.3585 | 0.1708 | 639.2 |
| V4.2-strict-OOD-FT7 | 0.2348 | 0.2998 | 0.5 | 0.3125 | 0.2771 | 0.2808 | 0.2202 | 500 |

## Risk continuum vs reference points

Mean horizon-10 risk score on the same 162-candidate event space.
Discover policies should push toward `oracle_high`; safety policies toward `oracle_low`.
The FT-7-only column is the natural rule-based strawman.

| Run | Oracle low | Agent safe | Random | FT-7 heuristic | Agent discover | Oracle high | Discover gain over FT-7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4.2-canonical-val | 0.1809 | 0.2109 | 0.3245 | 0.5025 | 0.5732 | 0.5982 | +73.9% |
| V4.2-canonical-test | 0.1357 | 0.1725 | 0.2689 | 0.436 | 0.5395 | 0.5589 | +84.2% |
| V4.2-strict-OOD-FT7 | 0.2189 | 0.2674 | 0.3717 | 0.4326 | 0.4747 | 0.5374 | +40.2% |

*Discover gain over FT-7* = (agent_discover − ft7) / (oracle_high − ft7). Reads as: "how much of the FT-7 → oracle gap did the agent close?"
