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
