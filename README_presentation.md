# vLLM torch.compile & Fusions — Presentation (work in progress)

Scratch workspace for an AMD MI355 technical talk on vLLM's `torch.compile`
integration and fusion passes, built around PR
[#39242](https://github.com/vllm-project/vllm/pull/39242) (MLA dual RMSNorm fusion).

## Files

- `prompt.txt` — original brief / requirements + the hand-drawn pipeline sketch.
- `presentation_outline.md` — full talk outline (container ideas, `[KEEP]`/`[BACKUP]` tags, deep-dive notes).
- `gen_slides.py` — reproducible deck generator (python-pptx) that fills the AMD template.
- `potx_to_pptx.py` — converts the `.potx` template to `.pptx` (python-pptx can't open `.potx`).
- `Corporate White Background.potx` — AMD corporate template (source of branding/layouts).
- `vllm_compile_fusions_amd.pptx` — the generated deck (current output).

## Rebuild

```bash
# from the repo root
python3 -m venv .slides-venv
.slides-venv/bin/pip install python-pptx
.slides-venv/bin/python potx_to_pptx.py        # -> corporate_template.pptx
.slides-venv/bin/python gen_slides.py          # -> vllm_compile_fusions_amd.pptx
```

`gen_slides.py` reads `corporate_template.pptx` (the converted template) and the
in-repo diagram `docs/assets/design/debug_vllm_compile/design_diagram.png`, and
must be run from the repo root so those relative paths resolve.

## Status / TODO

- Done: Parts I–IV core slides (foundations, compile flow, vLLM IR,
  functionalization/`maybe_inplace`, the PR #39242 walkthrough, config cheat-sheet).
- Next: Debug BKM slide (wire in `tlparse_inductor.png` + "what's readable vs opaque"
  note + subsystem-isolation table), closing slide, and the `[BACKUP]` appendix.

Note: `.slides-venv/` and the regenerated `corporate_template.pptx` are intentionally
not committed (the venv is local; the `.pptx` is produced by `potx_to_pptx.py`).
