Shaping Gap Benchmark
====================

An evolving research benchmark for constructing executable computations from
acquired interfaces to ordinary frozen pretrained components.

**Latest: Checkpoint 03 — Preserved neural augmentation and reusable observation interfaces.**
`Read Paper III <checkpoints/03-preserved-neural-augmentation/paper/shaping-gap-benchmark-iii.pdf>`_
or `reproduce the results <checkpoints/03-preserved-neural-augmentation/README.rst>`_.

Research checkpoints
--------------------

Each checkpoint is introduced in one publication commit with its own paper,
code, protocols, evidence and measured contribution. Earlier papers and evidence
retain their original paths. Follow the
`commit history <https://github.com/calysteon/shaping-gap-benchmark/commits/main/>`_
and `machine-readable index <CHECKPOINTS.json>`_ to track advances.

.. list-table:: Published research checkpoints
   :header-rows: 1
   :widths: 10 40 30 20

   * - Checkpoint
     - Advance
     - Paper and evidence
     - Publication record
   * - **03 · Latest**
     - Acquired execution updates paired with shared observation interfaces;
       old-program retention and fresh-stream processing quality over two sizes.
     - `Paper III <checkpoints/03-preserved-neural-augmentation/paper/shaping-gap-benchmark-iii.pdf>`_
       · `Code, data and reproduction <checkpoints/03-preserved-neural-augmentation/README.rst>`_
     - November 13, 2026 · `Checkpoint commits <https://github.com/calysteon/shaping-gap-benchmark/commits/main/checkpoints/03-preserved-neural-augmentation>`_
   * - **02 · Scaling and functions**
     - Response-moment evaluation scales from 22 to 512 components;
       fixed programs implement complete finite Boolean functions.
     - `Paper II <checkpoints/02-scaling-and-functions/paper/shaping-gap-benchmark-ii.pdf>`_
       · `Code, data and reproduction <checkpoints/02-scaling-and-functions/README.rst>`_
     - November 12, 2026 · `f558ee9 <https://github.com/calysteon/shaping-gap-benchmark/commit/f558ee9f3cda6c2fdc734d9d5ecd8ee3b6cd63d9>`_
   * - **01 · Foundation**
     - Effect interfaces, incoming-context sensitivity and compatibility-guided
       execution of original frozen neural components.
     - `Paper I <paper/shaping-gap-benchmark.pdf>`_
       · `Original artifact guide <checkpoints/01-effect-interfaces/README.rst>`_
     - November 12, 2026 · `20b5a2a <https://github.com/calysteon/shaping-gap-benchmark/commit/20b5a2a8d63998b13e78041d0d3b7ec7812779e7>`_

What checkpoint 03 establishes
------------------------------

* **Shared compatibility:** one acquired observation interface per execution
  update retains 113/131 fresh-reference-reliable old program instances,
  versus 90/131 with direct reuse. Eleven of twelve charts retain every such program.
* **Repeated quality gains:** two primary updates improve the finite nonlinear
  temporal score in all three blocks while preserving old programs. One 1.7B
  case also exceeds both matched controls in all three blocks.
* **Complete evidence:** all twelve cases and signed doses are released. The mean
  primary quality change is negative, making context-sensitive update selection
  the next research threshold.
* **Native fidelity and costs:** 4,032 saved outputs independently replay bit
  exactly. The ledger charges 1,413,312 incremental native sequences, inherited
  source work separately, interface storage, fitting and verification.

.. image:: checkpoints/03-preserved-neural-augmentation/results/report/program-retention.png
   :alt: Existing-program retention aggregated over twelve charts, direct versus shared transport
   :width: 100%

Start with the latest checkpoint
--------------------------------

From the repository root, in an environment with NumPy and Matplotlib::

    python3 -m pip install -r checkpoints/03-preserved-neural-augmentation/requirements-analysis.txt
    python3 checkpoints/03-preserved-neural-augmentation/code/replay.py
    python3 checkpoints/03-preserved-neural-augmentation/code/report.py

These commands load no model and write under the checkpoint's ``runs/`` directory.
The `checkpoint guide <checkpoints/03-preserved-neural-augmentation/README.rst>`_
explains full native reproduction, inherited fixtures and measurement scope.
Earlier commands remain in each historical checkpoint guide.

Authors and citation
--------------------

Nathaniel Oh and Paul Attie. Cite the paper associated with the checkpoint used.
Each checkpoint has a citation record; `CITATION.cff <CITATION.cff>`_ indexes the
current paper. The papers are publicly available here; arXiv identifiers will be
added when assigned. Model provenance and third-party terms are documented in
`THIRD_PARTY_NOTICES.rst <THIRD_PARTY_NOTICES.rst>`_. Weights are downloaded separately.
