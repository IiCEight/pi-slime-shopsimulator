# Changes from upstream

This file is a compact overview, not an exhaustive upstream diff.

## Slime integration

- Added the `examples/ShopSimulator/` SFT, GRPO and eval-only workflow.
- Added Pi multi-turn tool harness and ShopSimulator adapter.
- Kept the four parent task pools and only the `sft_512`, `rl_500` and
  `official_test_200` experiment slices.
- Removed fixed-revision, manifest, SHA and historical-artifact gates from the
  public example; launchers now use the supplied files and checkpoints.
- Added Qwen3.5 loss-mask handling, including exclusion of the template-injected empty think block.
- Reduced evaluation finalization to an identity-agnostic metric summary.
- Added Qwen3.5 checkpoint conversion and launcher compatibility fixes.

Historical experiment results are shown only in the repository root `README.md`.

## ShopSimulator integration

The modified ShopSimulator source is intentionally not copied into this workspace. `shopsimulator_patch/shopsimulator-slime-integration.patch` is the single reproducible patch; the repository root `README.md` covers environment isolation, deterministic pricing, text-environment adaptation, runtime dependencies and API changes.
