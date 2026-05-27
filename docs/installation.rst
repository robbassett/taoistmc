Installation
============

Requirements
------------

- Python ≥ 3.10
- NumPy ≥ 1.26
- SciPy ≥ 1.17
- Numba ≥ 0.65 (for JIT compilation and optional CUDA support)
- Pydantic ≥ 2.11
- Typer ≥ 0.25 (CLI)
- PyYAML ≥ 6.0
- joblib ≥ 1.5 (parallel processing)

Installation from Source
------------------------

Clone the repository and install in editable/development mode:

.. code-block:: bash

   git clone https://github.com/robbassett/TAOIST_MC.git
   cd TAOIST_MC
   pip install .

For development with docs and testing extras:

.. code-block:: bash

   pip install ".[dev,docs]"

GPU Support
-----------

GPU acceleration requires:

- NVIDIA GPU with CUDA capability
- CUDA Toolkit (matching your GPU driver)
- Numba will automatically detect CUDA availability

If no GPU is detected or CUDA is unavailable, the package falls back to
CPU execution without error. Set ``use_gpu=True`` in your ``TaoistConfig``
to attempt GPU usage.