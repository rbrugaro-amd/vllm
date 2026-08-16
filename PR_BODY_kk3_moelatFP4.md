# [Kimi-K3][ROCm] MXFP4 for the latent-MoE projections

## What

Kimi-K3's latent MoE brackets its routed experts with two dense
`ReplicatedLinear` projections — `routed_expert_down_proj` (7168→3584, before
the router) and `routed_expert_up_proj` (3584→7168, after experts + all-reduce
+ norm). The routed experts ship as MXFP4, but these two are stored bf16 and
were built with `quant_config=None`, so they run bf16. They are quantizable
from the same checkpoint; ATOM does exactly this.

## Design

Uses the extension point `Mxfp4Config` documents for itself — *"Subclasses
override `get_name()` and `override_quantization_method()` to register
themselves as the handler for a specific checkpoint format"* — mirroring
`GptOssMxfp4Config`:

- `KimiK3Mxfp4Config(Mxfp4Config)` claiming only `model_type == "kimi_k3"`, so
  it cannot affect gpt-oss or any other MXFP4 producer. Its `LinearBase` branch
  returns the existing `Mxfp4OnlineLinearMethod` (#49347) for the two latent
  projections and defers to `super()` otherwise.
- `models/config.py`: K3's existing compressed-tensors → mxfp4 rewrite now
  points at `"kimi_k3_mxfp4"`.
- `quantization/__init__.py`: register it.
- `kimi_k3/amd/linear.py`: pass `quant_config` to the two projections.

No new env var, no user-facing config, no new online method. Gated on a native
MXFP4 linear kernel (not emulation) being available.

Also fixes an unprotected invariant in `ROCmLatentMoERunner` — see below.

## Measured — and the honest conclusion is that this does not pay on K3

MI355X gfx950, TP=8, real weights, `_LatFP4v2` image.

**Kernel level**, these two shapes, best FP4 path vs bf16:

| M | 1–64 | 256 | 1024 | 4096 | 8192 |
|---|---|---|---|---|---|
| down_proj | 0.5–0.9× | 1.48× | 2.48× | 1.91× | 2.06× |
| up_proj | 0.5–0.9× | 1.28× | 2.09× | 2.35× | 2.41× |

Crossover ≈ M=256. Five GEMM variants were compared (plain/preshuffled Triton
W4A4, ASM W4A4, plain/preshuffled W4A16). No single one wins everywhere, and
W4A16 — the obvious candidate to remove the decode-side activation-quant cost —
does not rescue decode and is ~19× worse at M=8192.

**End to end**, conc 8:

| workload | throughput | TTFT | TPOT | E2EL |
|---|---|---|---|---|
| prefill-heavy (ISL 68000 / OSL 350) | 1.003× | 1.002× | 1.005× | 1.004× |
| decode-heavy (ISL 2000 / OSL 2000) | **0.892×** | **0.690×** | **0.961×** | **0.890×** |

Prefill-heavy is indistinguishable from noise, because these two projections
are only ~6% of prefill compute. Decode-heavy is an **11% throughput
regression**.

The production agentic workload this targets is decode-dominated: measured
`vllm:prefix_cache_hits/queries` of 85.6–92.6% puts prefill at ~30–34% of
engine time at conc 8. So the expected net effect there is negative.

**Accuracy**, gsm8k 5-shot, full 1319, greedy, same image both arms:

| arm | strict-match | flexible-extract |
|---|---|---|
| bf16 | 0.9689 ± 0.0048 | 0.9689 ± 0.0048 |
| MXFP4 | 0.9644 ± 0.0051 | 0.9636 ± 0.0052 |

−0.45 pp, inside the CI — though both FP4 runs landed below the control rather
than straddling it, so a small real degradation cannot be excluded.

## Why post it anyway

The mechanism is reusable for any partially-quantized checkpoint, and it is the
first example of a model-scoped `Mxfp4Config` subclass making a per-layer
precision decision. I would rather have the design reviewed than sit on it.
If the conclusion is "correct mechanism, wrong model," that is a useful outcome.

## The `ROCmLatentMoERunner` fix (independent of the above)

`_tail_shardable` is evaluated in `__init__`, before
`process_weights_after_loading`, so it cannot see whether the up-projection
ended up quantized. `_shard_up_proj_tail` then indexes `.weight` as a dense
`(N, K)` matrix of the activation dtype and folds it into a `beta=1` epilogue —
with a packed MXFP4 weight that silently reinterprets packed values as
activations. Nothing guards it today. This re-checks lazily and falls back to
the replicated up-projection.

Worth noting for anyone quantizing that layer later: the sharded tail is
column-parallel, so it costs 1/TP of the up-proj GEMM. At M=8192 bf16-sharded
(~38 µs) beats FP4-replicated (~125 µs) by 3×. Quantizing the up-projection is
only worthwhile if sharding is preserved — i.e. shard first, then quantize, so
each rank quantizes its own `(N/TP, K)` slice. Not attempted here.

## Not done

- No autotuned aiter configs. The preshuffled Triton GEMM has no tuned entry
  for these shapes (`GEMM-AFP4WFP4_PRESHUFFLED-N=3584-K=7168` and
  `N=7168-K=3584` are absent from every aiter version), so it runs the default
  heuristic. Tuning could recover maybe 20–30% of the kernel, which maps to
  ~0.3% end-to-end on the best case — not enough to change the verdict.
- No shard-aware FP4 up-projection (above).
- Not validated on a main-based build: the measurements were taken with an
  equivalent runtime patch on `cb8104839c`, which predates #49347.
