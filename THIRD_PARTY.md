# Third-party components

The repository-root MIT License applies only to original material that this
project has the right to license. Third-party components retain their own
licenses and terms; the root license does not override them.

## Slime

- Upstream: https://github.com/THUDM/slime
- Snapshot base revision: `624b824a898ab0ec1fcb4d373004c7f3852bf515`
- License: Apache-2.0
- Vendored source: `slime/`
- Local modifications: see `CHANGES.md`

The upstream license is preserved at `slime/LICENSE`. The reviewed modification scope is summarized in `CHANGES.md` and in the repository-root `README.md`, using the fixed snapshot revision above as the comparison baseline.

## ShopSimulator

- Upstream: https://github.com/ShopAgent-Team/ShopSimulator.git
- Upstream base revision: `51bb26012cee31aea7ac26177c5ffe807026ac07`
- Tested modified revision: `3ab366b2982e9ffa59957086d0845f955ef2245b`
- Vendored source: no
- Patch: `shopsimulator_patch/shopsimulator-slime-integration.patch`
- Modification description: `README.md`

The upstream repository did not declare an explicit software redistribution license when this workspace was created. This repository does not grant rights to the upstream ShopSimulator source. Obtain permission from its copyright holders before redistributing the complete modified source.

## Pi

- Upstream: https://github.com/earendil-works/pi.git
- Package: `@earendil-works/pi-coding-agent`
- Version: `0.84.2`
- Node.js: `>=22.19.0`
- License: MIT
- Vendored source: no

Pi is installed as a pinned external dependency. The Shop-specific extension remains in the Slime integration code.

## Qwen3.5-2B

- Base model: https://huggingface.co/Qwen/Qwen3.5-2B
- License: Apache-2.0
- SFT model: https://huggingface.co/mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep
- GRPO/RL model: https://huggingface.co/mrzhao13/qwen3.5-2b-shopsimulator-grpo-rl500-1ep
- Historical teacher dataset (private): https://huggingface.co/datasets/mrzhao13/pi-slime-shopsimulator-sft-512

The base and trained model weights are not vendored in this Git repository. The model cards record their source, license, training details and artifact identities. Access to the private historical teacher dataset is not required because this repository includes the task slice and the complete collection/preparation workflow.
