Shaping Gap Benchmark
====================

Learning executable improvements over frozen pretrained computation
-----------------------------------------------------------------

**The central result: a runtime can acquire an execution change together with
an interface that keeps existing programs usable, then retain improved
computation on fresh inputs.**

Shaping Gap Benchmark develops runtimes over existing pretrained neural
computation. Native experience acquires execution changes; paired trajectories
acquire shared compatibility interfaces; outcome feedback determines which
construction to retain. The pretrained weights remain fixed.

**Latest: Checkpoint 04 — Experience-acquired execution and compatibility reuse.**
`Read Paper IV <checkpoints/04-experience-acquired-reuse/paper/shaping-gap-benchmark-iv.pdf>`_
· `Code, data and reproduction <checkpoints/04-experience-acquired-reuse/README.rst>`_
· `Full results <checkpoints/04-experience-acquired-reuse/results/report/FEEDBACK-RESULTS.txt>`_

Experience → acquire an execution/interface pair → lock the construction → reuse
it on untouched inputs. Checkpoint 04 closes this finite cycle in known runtime
contexts, with acquisition and lifecycle costs recorded.

.. image:: checkpoints/04-experience-acquired-reuse/results/report/checkpoint-overview.png
   :alt: Experience-acquired execution/interface cycle, two strict success contexts and old-program retention across all twelve contexts
   :width: 100%

Measured advance
----------------

On the nonlinear temporal-computation benchmark, two retained constructions
improve processing quality on both fresh validation streams while preserving
all eligible old programs:

.. list-table:: Two strict joint-success contexts
   :header-rows: 1
   :widths: 25 25 25 25

   * - Frozen model / context
     - Fresh-stream quality gains
     - Old programs with shared interface
     - Old programs with direct reuse
   * - 0.6B / 111005
     - **+3.95%, +4.39%**
     - **13/13**
     - 9/13
   * - 1.7B / 111004
     - **+5.09%, +4.96%**
     - **18/18**
     - 10/18

Across **all twelve contexts**, five updates are retained and seven keep baseline.
Four selected updates improve quality on both streams with full old-program
retention; two meet the stricter 1% gain threshold in both. Pooled mean score
changes are **+0.114 and +0.076**, and **140/141** old program instances remain
usable, versus 116/141 without the shared interfaces. One selected update
regresses and loses a program; every case remains in the published evidence.

This is learning and reuse within experienced runtime contexts. The separate
predictor evaluated on unseen runtime geometry misses its aggregate transfer
criterion. The percentages above measure a finite temporal-processing score.
Pretrained weights, observation interfaces and readout coefficients stay frozen
during final reuse validation.

**Verification and costs:** 5,040 saved native outputs independently reproduce
bit exactly; 1,008 score vectors and 387 seals replay. The ledger charges
1,852,268 new two-position native sequences, including unsuccessful trials,
controls, padding and verification, with inherited acquisition reported
separately. Interface, policy, readout and chart storage are explicit.

Start with the current result
-----------------------------

From the repository root::

    python3 -m pip install -r checkpoints/04-experience-acquired-reuse/requirements-analysis.txt
    python3 checkpoints/04-experience-acquired-reuse/code/replay.py
    python3 checkpoints/04-experience-acquired-reuse/code/report.py

Replay loads no model. The `checkpoint guide <checkpoints/04-experience-acquired-reuse/README.rst>`_
also documents original-weight verification and complete native acquisition in
a separate workspace. New outputs go under the checkpoint's ``runs/`` directory.

Research checkpoints
--------------------

Each checkpoint is an advancement with its own paper, code, protocols, evidence
and publication commit. Earlier papers and evidence retain their original paths.
Follow the `commit history <https://github.com/calysteon/shaping-gap-benchmark/commits/main/>`_
and `machine-readable index <CHECKPOINTS.json>`_ to track the buildup.

.. list-table:: Published research checkpoints
   :header-rows: 1
   :widths: 10 40 30 20

   * - Checkpoint
     - Advance
     - Paper and evidence
     - Publication record
   * - **04 · Central result**
     - Experience acquires a retained execution/interface pair; two fresh-stream
       joint successes, positive pooled quality and 140/141 old instances retained.
     - `Paper IV <checkpoints/04-experience-acquired-reuse/paper/shaping-gap-benchmark-iv.pdf>`_
       · `Code, data and reproduction <checkpoints/04-experience-acquired-reuse/README.rst>`_
     - November 13, 2026 · `Checkpoint commits <https://github.com/calysteon/shaping-gap-benchmark/commits/main/checkpoints/04-experience-acquired-reuse>`_
   * - **03 · Execution and compatibility**
     - Preserved neural augmentation and shared observation interfaces establish
       the execution/compatibility mechanism over two frozen model sizes.
     - `Paper III <checkpoints/03-preserved-neural-augmentation/paper/shaping-gap-benchmark-iii.pdf>`_
       · `Code, data and reproduction <checkpoints/03-preserved-neural-augmentation/README.rst>`_
     - November 13, 2026 · `17822c9 <https://github.com/calysteon/shaping-gap-benchmark/commit/17822c9fab3291c91fe476d0a04bf869c0244e15>`_
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

The next research threshold
----------------------------

Start from an already improved runtime and acquire a second retained gain,
composing compatibility interfaces into one fixed-size map while charging
reorganization, failed trials and maintenance. This tests cumulative development
from the demonstrated learning-and-reuse cycle. Stronger regression control and
transfer to unseen runtimes remain explicit parts of the research program.

Authors and citation
--------------------

Nathaniel Oh and Paul Attie. Cite the paper associated with the checkpoint used.
`CITATION.cff <CITATION.cff>`_ points to the current paper. Papers are publicly
available here; arXiv identifiers will be added when assigned. Model provenance
and licensing are in `THIRD_PARTY_NOTICES.rst <THIRD_PARTY_NOTICES.rst>`_.
Weights are downloaded separately.
