Shaping Gap Benchmark
====================

Companion artifact for **Shaping Gap Benchmark: Effect Interfaces for
Compositional Execution of Frozen Neural Computation**, by Nathaniel Oh and
Paul Attie.

The paper studies a runtime that acquires reusable descriptions of native
components in an ordinary frozen pretrained model and uses those descriptions
to construct computations satisfying explicit neural contracts. The retained
interface describes amplitude, accumulated readout effect and incoming-context
sensitivity. A temporary program executes original components on live state.

The first study fixes a 22-component bank, three local activation domains and
288 later contracts. Nine controls share the native checker and 512-LPE online
search allowance. Acquisition, search and deployment costs are reported together.

Paper and artifact
------------------

* ``paper/shaping-gap-benchmark.pdf``: the paper.
* ``paper/source/``: TeX, bibliography, tables, figures and figure/table generators.
* ``code/``: acquisition, compatibility planning, native execution and verification.
* ``protocols/``: source-acquisition and final-benchmark protocols.
* ``data/source/``: sealed native observations and acquired interface arrays.
* ``data/benchmark/``: recorded contracts, roots, selected programs and result receipts.
* ``data/accounting/``: common inventory-discovery accounting inputs.
* ``data/provenance.json`` and ``MANIFEST.json``: checkpoint and artifact hashes.

Quick reproduction without the checkpoint
-------------------------------------------

Install NumPy and Matplotlib in your Python environment::

    python3 -m pip install -r requirements-analysis.txt
    python3 code/replay.py
    python3 code/report.py

The replay uses NumPy only and loads no checkpoint. It checks all 2,592 selected
programs against the recorded final readouts and contract decisions using the
structure-calibrated finite bank. Expected contract agreement is 2,592/2,592,
with maximum absolute readout error approximately 3.3021e-6. The acquired payload
is 415,840 bytes, including directions, coordinates and anchors. Its scope is
the finite bank and fixed readouts in the three recorded domains.

The report regenerates lifecycle ledgers, paired outcomes, cumulative curves
and figures from the recorded observations. Outputs go under ``runs/``; commands
do not overwrite the supplied evidence.

Native benchmark reproduction
-------------------------------

Native execution uses Apple Silicon/Metal, MLX 0.32.4 and MLX-LM 0.32.2. Install
``requirements-native.txt`` in a suitable virtual environment. Download the
public ``Qwen/Qwen3-0.6B-MLX-4bit`` checkpoint at revision
``173234aa840d113125e9f2271100ddbaf16c9620``. The loaded-parameter digest is
recorded in the protocol. The checkpoint itself is not bundled.

Run the complete final study from the supplied sealed source interfaces::

    python3 code/run.py --checkpoint /path/to/qwen3-0.6b-4bit
    python3 code/verify.py --checkpoint /path/to/qwen3-0.6b-4bit --run runs/benchmark

The driver refuses to overwrite an existing destination. It records all native
primitives and branches, recomputes live neural activities and checks original
parameter hashes. The verifier replays every selected program, samples 512
branches from the complete generated trace, and checks 24 source directions
against direct native program autodiff. Running the verifier on the supplied
``data/benchmark`` checks selected programs and source directions; branch
sampling additionally requires a generated complete trace.

No hosted inference API, tokenizer, language prompt or original model forward
pass is used. The model supplies frozen parameter-backed components.

To reacquire the source interfaces before reproducing the final study::

    python3 code/acquire.py --checkpoint /path/to/qwen3-0.6b-4bit
    python3 code/run.py --checkpoint /path/to/qwen3-0.6b-4bit --source runs/source --output runs/reacquired-benchmark

The source command performs the recorded native acquisition, fitting,
structured calibration and diagnostic validation. It omits the earlier
development construction batch. Final benchmark replay checks that inherited
source arrays match the recorded hashes; exact replay is tied to the recorded
numerical environment. Changes to data or algorithms should be recorded as a
new experiment.

Recorded result
----------------

The 32-context endpoint interface satisfies 100/288 contracts versus 80/288 for
native local sensitivity, with 15.76% less lifecycle native-query work. The
differential interface satisfies 104; query-matched endpoints satisfy 105;
conditioning removal and permutation satisfy 105 and 108. The explicitly
structure-calibrated reference satisfies 129 with 21.39% less lifecycle
native-query work than native local sensitivity. All nine controls appear in
the paper and recorded summary.

LPE counts native linear-projection and transpose-projection queries. Active
time, scalar planning and memory are separate measurements. The full frozen
checkpoint was loaded during native evaluation; interface payload is not total
process memory. Scientific validation and replay costs are separately recorded.

Research direction
--------------------

This first paper establishes the interface, executable construction mechanism
and measured frontier. The follow-on questions are decision-relevant acquisition,
continual consolidation under a retained-state budget, and externally grounded
capabilities across larger banks and pretrained architectures.

Citation
---------

Nathaniel Oh and Paul Attie. 2026. *Shaping Gap Benchmark: Effect Interfaces for
Compositional Execution of Frozen Neural Computation.* See the included paper
and ``CITATION.cff``.

Model provenance and third-party terms are documented in
``THIRD_PARTY_NOTICES.rst``. The full model checkpoint is downloaded separately.
