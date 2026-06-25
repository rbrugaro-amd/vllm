# vLLM `torch.compile` & Fusions on MI355 — Presentation Outline

> **Audience:** AMD performance engineers, already strong on PyTorch CUDA graphs and compilation.
> **Goal:** Comprehensive talk, intro → deep technical detail, on how vLLM uses `torch.compile`
> to deliver perf uplifts (esp. fusions), using a real fusion PR
> ([#39242 — MLA dual RMSNorm fusion](https://github.com/vllm-project/vllm/pull/39242)) as the worked example.
>
> **This file is an outline only.** We iterate on container ideas/concepts here, then drill into
> each section before writing final slide content.

---

## 0. Framing & Logistics (1–2 slides)

- One-line thesis: *"In vLLM, you make the model fast by teaching the compiler, not by rewriting the model."*
- What the audience will be able to do afterward:
  - Read a vLLM compile pipeline dump and know what each stage produced.
  - Understand where a fusion pass plugs in and why it's safe.
  - Write/debug a fusion pass (using PR #39242 as the template).
- Roadmap slide (mirror of sections below).

---

## Part I — Foundations

### 1. CUDA Graphs vs. `torch.compile` — what each buys you (2–3 slides)

> Topic: *cuda graph vs. torch compile - benefits of each.*

- **The two distinct problems:**
  - CPU launch overhead (host can't enqueue kernels fast enough) → **CUDA Graphs**.
  - Suboptimal/unfused kernels + memory round-trips → **`torch.compile` / Inductor**.
- **CUDA Graphs:** capture a kernel launch sequence once, replay with ~zero CPU overhead.
  - Constraints: static shapes, fixed memory addresses (must copy inputs into persistent buffers),
    captures only CUDA work (no CPU branching), unsafe raw API.
- **`torch.compile` (Inductor):** generates optimized (often fused) Triton/C++ kernels;
  reduces *kernel count* and *memory traffic*, can autotune.
- **They are complementary, not either/or** — vLLM uses both, layered.
- Decision framing: graphs attack *overhead*; compile attacks *kernel quality*. Most wins need both.
- **Deep-dive candidates:** memory-address constraint of graphs; why attention is hard to capture;
  autotuning cost/benefit.

### 2. vLLM's optimization philosophy (2–3 slides)

> Topic: *VLLM optimization philosophy: rely on the compiler instead of editing the model everywhere;
> contrast with SGLang (briefly).*

- **Separation of concerns:** model code stays clean & readable (math only);
  optimizations live in compiler passes (`PassConfig`, Inductor passes).
- Why this matters: one fusion pass benefits *every* model that hits the pattern;
  no copy-paste of fused kernels across N model files; no broken layer abstractions.
- vLLM IR reinforces this: model calls high-level ops (`ir.ops.rms_norm`), kernel choice deferred to compiler.
- **Contrast w/ SGLang (brief, non-inflammatory):** SGLang leans more on hand-written/templated fused
  kernels and model-level integration; vLLM bets on a compiler pipeline + pattern-matching passes as the
  primary uplift mechanism. Trade-off: compiler approach = generality + maintainability vs. hand-kernel =
  max control per case. *(Keep to 1 slide; don't litigate.)*
- Tie-in: this is exactly why your AMD work ships as a *pass* (PR #39242), not a model edit.
- **Deep-dive candidates (`PassConfig` + optimization levels):**
  - The 4 levels: `-O0` (nothing), `-O1` (compile + PIECEWISE cudagraphs + fast fusions),
    `-O2` (default: + FULL_AND_PIECEWISE + more fusions/ranges), `-O3` (= O2 today, room for experimental).
  - A level is just a *preset of underlying flags*; user-set flags always override the level default.
  - **Auto-enable is conditional, not unconditional.** Fusions turn on only when their preconditions hold:
    platform (e.g. `fuse_mla_dual_rms_norm`, `fuse_rope_kvcache`, `fuse_act_padding` are ROCm/AITER-only),
    and "custom op in use" guards (`fuse_norm_quant`/`fuse_act_quant` only fire when a custom kernel is
    active — otherwise Inductor's own fusion is better, so vLLM leaves them off).
  - **Important nuance to teach: some fusions have NO default at any level (always off) and must be forced
    via config.** Examples: `fuse_attn_quant`, `enable_qk_norm_rope_fusion`, `fuse_minimax_qk_norm`,
    `enable_sp`, `fuse_gemm_comms`. Reasons vary — needs full-graph visibility (`splitting_ops=[]`),
    known perf regressions on some HW, or model/TP-specific. So `-O3` does **not** turn these on;
    you must opt in explicitly, e.g. `-cc.pass_config.fuse_attn_quant=True`.
  - Takeaway framing: "level = safe defaults; the long tail of situational fusions is opt-in."

---

## Part II — How vLLM Compilation Actually Works

### 3. Piecewise compilation vs. piecewise CUDA graph (2 slides)

> Topic: *piecewise compilation vs. piecewise graph.*

- The unifying idea: **split the full graph at attention ops** (`splitting_ops`).
- **Piecewise compilation:** Inductor compiles each inter-attention subgraph independently;
  attention runs as an opaque custom op (`unified_attention_with_output`).
- **Piecewise CUDA graph:** capture graphs only for the compute-between-attentions pieces;
  attention executes eager → keeps attention flexible while removing overhead elsewhere.
- Why attention is the natural seam: its output shares the query's shape; everything between
  attentions is token-wise and graph-friendly.
- Mention **full CUDA graph** mode (incl. attention) when the backend is graph-compatible (decode/MOE wins).

- **Sub-slide — "attention is special" along TWO independent axes (clear up a common confusion):**
  - **Axis A — Inductor *compilation*:** attention is *always* excluded. It's wrapped as an opaque
    custom op (`unified_attention_with_output` / MLA equivalent) so Dynamo doesn't trace into it and
    Inductor doesn't codegen it (data-dependent KV-cache access, complex shapes, backend-specific kernels).
    This is independent of CUDA graphs.
  - **Axis B — CUDA graph *capture*:** attention *can* be captured — **conditionally**, based on backend
    capability and batch shape. This is the piecewise-vs-full CUDA graph choice.
  - **Concrete AMD example (resolves "but I see AITER MLA inside a CUDA graph in my traces!"):**
    - Default `cudagraph_mode = FULL_AND_PIECEWISE`: **full** CUDA graph for *uniform decode*,
      **piecewise** (attention eager) for prefill/mixed.
    - AITER MLA backend capability = `UNIFORM_SINGLE_TOKEN_DECODE` → its assembly kernel **is** captured
      in a full CUDA graph during `query_len==1` decode, but runs eager/piecewise for prefill/mixed.
      Both observations are true and consistent.
  - **So what attention "cannot do":** be captured into one static graph valid across *prefill/mixed/
    non-uniform* batches. A captured graph bakes in fixed addresses **and** fixed launch geometry
    (grid dims, kernel selection); prefill/mixed varies per step (seqlens, #tokens, #reqs) so one graph
    can't replay them. Decode is the easy case because it's regular (1 token/seq, fixed-shape metadata
    copied into persistent buffers).
  - The `AttentionCGSupport` ladder: `ALWAYS` (e.g. FA3, Triton) > `UNIFORM_BATCH` (FlashMLA, FlashInferMLA,
    AITER FlashAttn) > `UNIFORM_SINGLE_TOKEN_DECODE` (AITER MLA, CUTLASS MLA) > `NEVER`. Unsupported modes
    auto-downgrade (e.g. `FULL` → `FULL_AND_PIECEWISE`).

- **Deep-dive candidates:**
  - **[KEEP]** `compile_sizes` (feeds Inductor: specialized + autotuned kernels per size) vs
    `cudagraph_capture_sizes` (feeds graph capture; real batches padded up). Independent decisions.
  - **[KEEP]** fine-grained buffer management → why attention takes its **output buffer as an input**
    (fixed addresses for replay).
  - **[BACKUP]** `CudagraphDispatcher` + `BatchDescriptor` dispatch keys (runtime "which graph?",
    priority `FULL` > `PIECEWISE` > `NONE`).
  - **[BACKUP]** `use_inductor_graph_partition` (Inductor-side split; lets full-graph passes like
    attn-quant fusion / sequence parallelism coexist with piecewise CUDA graphs; experimental, torch≥2.9).

### 4. The compilation flow with code transformation (3–5 slides — centerpiece)

> Topic: *detail compilation flow w/ code transformation.* (Use the ASCII pipeline from prompt.txt.)

- Walk the pipeline stage-by-stage (the diagram you already drafted):
  1. **Dynamo** — Python forward → FX graph (`transformed_code.py`, `computation_graph.py`).
  2. **Pre-grad passes** — high-level FX rewrites (`BEFORE_PRE_GRAD.*.py`, `patterns.*.py`).
  3. **Functionalization + decomposition** (AOTAutograd) — remove mutation (`__compiled_fn_*`).
  4. **Before split** — one giant functional graph.
  5. **The split** — partition into compilable vs. fallback (attention) subgraphs.
  6. **After split** — post-split graph.
  7. **Inductor** — lower to Triton kernels (`kernel_*.py`, `full_code_for_forward_*`, `.best_config`).
  → GPU execution.
- Emphasize: **vLLM-compile is NOT stock `torch.compile`** — it's a custom compiler built on internal
  PyTorch Compile APIs (full-graph capture, custom split, custom passes, own cache).
- Call out **where fusions live**: pre-grad and post-grad custom passes (this is the hook for Part IV).
- Compilation cache: factors in the hash (configs, PyTorch config, traced files); copyable cache dir;
  "all compilation finishes before serving" guarantee.
- **Deep-dive candidates:**
  - **[KEEP]** why **batch size (#tokens) is the only symbolic dim** — weights/buffers static; one dynamic
    artifact reused for all sizes.
  - **[KEEP, light]** dynamic-shape modes & vLLM guard dropping: `backed` (default, guards droppable,
    0/1-specializes) vs `unbacked` (no guards, may pick conservative paths) vs `backed_size_oblivious`
    (middle ground). vLLM assumes guards are safe to drop.
  - **[BACKUP]** `unbacked` subtleties (data-dependent errors, general-path clones, missed optimizations);
    debugging with `TORCH_LOGS=+dynamic` + `compilation_metrics`.

### 5. vLLM IR — what it is and why it was added (3 slides)

> Topic: *vllm IR, why it was added.*

- **The gap it fills:** between low-level `torch` ops and vLLM layers (RMSNorm, quant) — a *functional IR*
  / FX "dialect".
- **Separation of semantics vs. implementation vs. dispatch:**
  - `@register_op` defines semantics (native torch reference).
  - `register_impl` registers kernels per provider (`native`, `vllm_c`, `aiter`, `triton_*`, OOT).
  - Priority lists (`--ir-op-priority.<op>=aiter,vllm_c,native`) pick the kernel late in compilation.
- **Why it helps fusions (key for this talk):** passes match against *one* high-level op, not against
  every low-level kernel variant or functionalization shape. (This is precisely what makes PR #39242's
  pattern simple.)
- Eager/compile consistency; `maybe_inplace` overload + donated-input tracking + clone elimination
  (zero-copy in-place kernels).
- **AMD angle:** `aiter` provider, ROCm default priority lists, registering AMD kernels OOT.
- **Deep-dive candidates:**
  - **[KEEP]** `supports_args` checked with **fake tensors** at compile, **real tensors** in eager →
    same dispatch logic both modes (only inspect dtype/params, never values/batch size). Heart of
    eager-compile consistency.
  - **[KEEP, one-liner]** `VLLM_BATCH_INVARIANT=1` → auto-selects batch-invariant kernels for
    reproducible numerics (costs perf; matters for eval/debug).
  - **[BACKUP]** IR lowering (`VllmIRLoweringPass`) inserts clones for in-place kernels →
    `UnsafeCloneEliminationPass` removes them using donated-input info → zero-copy in-place execution.

### 6. Nuances: control flow, black-box ops, and the dynamic-shape contract (2–3 slides)

> Topic: *what happens when code branches on batch size (torch.cond); vLLM's "black box" operator mechanism.*

- **The full-graph requirement:** model forward must capture into a single graph dynamic on #tokens.
  Branching on a dynamic shape (`if x.size(0) % 128 == 0`) → guards/constraint violations / silent bugs.
- **Two escape hatches:**
  1. Rewrite to avoid the branch.
  2. **Wrap the branchy/opaque logic into a custom operator** — Dynamo does not trace *into* custom ops,
     so it appears as one node ("black box").
- **Why register a black-box op (benefits / when to use):**
  - Hides un-traceable logic (data-dependent control flow, complex external kernels like attention).
  - Defines a clean fusion/replacement target (your fused op *is* a black box op).
  - Controls the split boundary (`splitting_ops`).
- Note `torch.cond` as the structured-control-flow alternative when you must keep multiplexing in-graph.
- The dynamic-shape contract: vLLM drops guards by assumption; modes (`backed`/`unbacked`/
  `backed_size_oblivious`) exist to debug when that assumption breaks.
- **Deep-dive candidates:** how `unified_attention_with_output` is a black box; output-as-input buffer trick.

---

## Part III — Worked Example: PR #39242 (MLA Dual RMSNorm Fusion)

> The methodology demo. Everything above converges here. This is the user's own merged PR
> (ROCm/AITER, DeepSeek-V3 / Kimi-K2, MI355X).

### 7. The opportunity (1–2 slides)

- MLA attention runs **two separate RMS norms per layer** (q_a_layernorm + kv_a_layernorm).
  Kimi-K2 = 61 layers → 122 norm launches/forward.
- Goal: fuse the pair into **one** `fused_qk_rmsnorm` HIP kernel (AITER) → 2 launches → 1 per layer.
- Why a *pass* and not a model edit (callback to philosophy, Part 2).
- Note (honest framing): native `rms_norm` lets Inductor auto-fuse these; this pass targets the
  **AITER custom-op** case Inductor can't fuse on its own.

### 8. The pattern, in graph terms (2 slides)

- Show unfused FX subgraph (from the pass docstring):

```text
gemm -> split_with_sizes([q_dim, kv_dim])
    +-- q_c     -> vllm_ir.rms_norm(q_c, q_w, eps)
    +-- kv_lora -> split_with_sizes([kv_c_dim, k_pe_dim])
                    +-- kv_c -> vllm_ir.rms_norm(kv_c, kv_w, eps)
                    +-- k_pe   (passthrough)
```

- Show fused result → single `fused_mla_dual_rms_norm(q_c, q_w, kv_c, kv_w, eps, eps)`.
- Highlight: the pattern matches the **connected subgraph** (both splits + both norms + k_pe passthrough),
  and it matches on **`vllm_ir.rms_norm`** (Part 5 payoff — one clean target).

### 9. Implementing the pass (2–3 slides)

- The two pieces every fusion needs:
  1. **A fused custom op** — `fused_mla_dual_rms_norm` in `vllm/_aiter_ops.py` wrapping AITER's
     `fused_qk_rmsnorm` (a "black box op", Part 6).
  2. **A PatternMatcher pass** — `MLADualRMSNormFusionPass` / `MLADualRMSNormPattern` using
     `register_replacement(pattern, replacement, inputs, fwd_only, pm_pass)`.
- `pattern` fn = the unfused subgraph; `replacement` fn = the fused op; example inputs trace both.
- Register for each epsilon (1e-5, 1e-6) — show why (eps is baked into the traced pattern).
- Wiring: `PassConfig.fuse_mla_dual_rms_norm` flag, registration in `pass_manager.py`,
  auto-enable on ROCm/AITER at O1+.
- **Deep-dive candidates:** `register_replacement` internals; topological-ordering pitfall called out in
  PR review; why the rewrite moved from manual graph-walk to pattern matching; `uuid()`/cache hashing.

### 10. Verifying & measuring (1–2 slides)

- Correctness: unit test (`test_fuse_mla_dual_rms_norm`) — pattern fires, ops replaced, numerics match;
  GSM8K accuracy gate (0.94 ≥ 0.90).
- Pass fired on all TP workers: `MLADualRMSNormFusionPass: fused 1 q/kv norm pair(s)`.
- Trace evidence: two CK RMSNorm kernels → one `aiter::fused_qk_rmsnorm_kernel`.
- Perf table: Kimi-K2-Thinking-MXFP4, TP=4, **MI355X**, ~1.02x geomean (be honest about magnitude;
  the *method* generalizes, this op is small).

---

## Part IV — Practical Track (config + debug BKM)

### 11. Compilation config cheat-sheet (1–2 slides)

> Topic: *example compile config and what each means / when needed.*

- Optimization levels `-O0..-O3` and what they flip on.
- Key knobs:
  - `-cc.mode` (NONE / STOCK_TORCH_COMPILE / VLLM_COMPILE), `-cc.backend` (inductor/eager).
  - `cudagraph_mode`, `cudagraph_capture_sizes`, `compile_sizes`.
  - `pass_config` fusion flags (e.g. `fuse_mla_dual_rms_norm`, `fuse_norm_quant`, `fuse_rope_kvcache`).
  - `--ir-op-priority.<op>=aiter,vllm_c,native`.
  - `dynamic_shapes_config.type`, `compile_cache_save_format=unpacked`.
- Dot-notation vs. JSON form; explicit user settings override level defaults.

### 12. Debug BKM (2 slides) — *live demo with PR #39242*

> Topic: *what are DEBUG BKM; possibly run live with the PR.*

- **tlparse** (`TORCH_TRACE=<dir>` → `tlparse`): see every stage + generated kernels.
- **Subsystem isolation table** (turn off one thing at a time):
  `--enforce-eager`, `-cc.mode=0`, `-cc.cudagraph_mode=NONE`, `-cc.backend=eager`,
  `-cc.ir_enable_torch_wrap=False`.
- `VLLM_LOGGING_LEVEL=DEBUG` (kernel selection, cache hits, re-enables inductor asserts).
- `compile_cache_save_format=unpacked` → editable Inductor code (breakpoints/prints).
- Dynamic-shape debugging: `TORCH_LOGS=+dynamic`, `compilation_metrics`, stricter shape modes.
- Cache busting: `VLLM_DISABLE_COMPILE_CACHE=1`, `rm -rf ~/.cache/vllm`, `rm -rf /tmp/torchinductor_$(whoami)`.
- **Demo script idea:** run Kimi-K2/DeepSeek with the pass on MI355, show tlparse before/after fusion +
  the `fused 1 q/kv norm pair(s)` log line + the single AITER kernel in the trace.

---

## 13. Wrap-up (1 slide)

- Recap thesis: model stays clean; compiler delivers the perf; fusions are pattern-matched passes.
- The repeatable methodology (spot pattern → fused black-box op → pattern pass → verify → measure).
- AMD takeaways: `aiter` provider + ROCm pass surface = where to land MI355 kernel wins.
- Pointers: `docs/design/torch_compile.md`, `vllm_ir.md`, `fusions.md`, `debug_vllm_compile.md`,
  the [blog post](https://blog.vllm.ai/2025/08/20/torch-compile.html).

---

## Appendix / Backup slides (optional)

- Compilation cache key internals.
- `maybe_inplace` memory-savings walkthrough.
- Full vLLM IR pipeline summary diagram.
- Sequence parallelism / AsyncTP (other fusion families) for context.
- Full `PassConfig` fusion reference table (from `fusions.md`).

---

### Open questions to resolve before building slides

1. **Talk length / slide budget?** (Drives how many deep-dives we keep vs. move to appendix.)
2. **Live demo or recorded?** Do we have MI355 + Kimi-K2/DeepSeek access at presentation time?
3. **How hard to lean on the SGLang contrast?** (Recommend: light touch, 1 slide.)
4. **Second fusion example?** (e.g. `fuse_rope_kvcache` is also ROCm/AITER — good AMD companion if time.)
5. **Depth on dynamic shapes / guard dropping?** (Could be its own 2–3 slide deep-dive or appendix.)
