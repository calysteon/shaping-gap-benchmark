Third-party provenance
======================

The experiment uses the public Qwen/Qwen3-0.6B-MLX-4bit checkpoint, pinned to
revision 173234aa840d113125e9f2271100ddbaf16c9620. It derives from Qwen3-0.6B.
The full checkpoint is excluded. The supplied measured down columns, calibrated
representations and native observations are derived from that frozen checkpoint.

Qwen model terms and attribution are retained in ``third-party/QWEN-LICENSE``.
The derived arrays have been extracted or transformed for this benchmark; they
are not an unchanged full Qwen checkpoint. Model sources:

* https://huggingface.co/Qwen/Qwen3-0.6B-MLX-4bit
* https://huggingface.co/Qwen/Qwen3-0.6B

Native execution imports MLX and MLX-LM as dependencies. This artifact does not
vendor those libraries. Their project sources are:

* https://github.com/ml-explore/mlx
* https://github.com/ml-explore/mlx-lm

The manuscript bibliography identifies the research building blocks used by the
method. Its reference implementation was written for this research project.

Checkpoint 03 model substrate
-----------------------------

Qwen/Qwen3-1.7B-MLX-4bit, revision
21457c6f51ed54a7c16e988c0844db973815c137, is additionally used in Checkpoint 03.
The upstream model card identifies Apache-2.0. The existing Qwen license copy
is retained at third-party/QWEN-LICENSE. Model weights are downloaded separately.
Checkpoint 03 also uses Qwen/Qwen3-0.6B-MLX-4bit at its recorded pinned revision.

Checkpoint 05 extracted component banks
---------------------------------------

Checkpoint 05 includes selected materialized gate/up rows, normalization-folded
rows and down columns derived from both pinned Qwen3 checkpoints above, with
model revisions and numerical fidelity in data/banks/manifest.json. These small
derived banks are redistributed with the retained upstream Qwen license and
attribution. They are transformed arrays, not unchanged full source models.
The separate Gaussian controls are generated with matched row norms.

Checkpoint 06 recruitment pools and application data
----------------------------------------------------

Checkpoint 06 redistributes selected frozen pretrained parameter rows under the
same Qwen3 model provenance and terms listed above and in Checkpoint 05. Its
pool manifests record exact source parameter digests, unit coordinates and
original-module numerical fidelity. These arrays are selected component banks,
not complete source language models.

The UCI Appliances Energy Prediction dataset is by Luis Candanedo (2017),
DOI https://doi.org/10.24432/C5VC8G, licensed under Creative Commons Attribution
4.0 International: https://creativecommons.org/licenses/by/4.0/.
Unmodified source CSV and ZIP, download receipt and attribution are in
``checkpoints/06-cumulative-contract-interfaces/data/application/``.
The forecast study uses univariate appliance consumption from one house.
Derived forecasts and numerical features are identified separately.
