Quick Start
===========

Python API Example
------------------

.. code-block:: python

   import taoistmc as tmc
   import numpy as np
   import matplotlib.pyplot as plt

   from taoistmc.config import PowerLawSegment, SightlineConfig, TaoistConfig

   # Define piecewise f(N) segments for IGM
   igm_low = PowerLawSegment(
       log_N_min=12.0, log_N_max=15.2,
       beta=1.635, log_A=9.305, gamma=2.5
   )
   igm_high = PowerLawSegment(
       log_N_min=15.2, log_N_max=21.0,
       beta=1.463, log_A=7.542, gamma=1.0
   )

   # Optional CGM segment
   cgm_seg = PowerLawSegment(
       log_N_min=13.0, log_N_max=21.0,
       beta=1.381, log_A=6.716, gamma=1.0
   )

   sightline_cfg = SightlineConfig(
       igm_segments=[igm_low, igm_high],
       cgm_segments=[cgm_seg]
   )

   config = TaoistConfig(
       sightline_config=sightline_cfg,
       delta_wav=0.25,
       use_gpu=False,   # Set True if you have CUDA
       n_jobs=-1        # Use all CPU cores
   )

   # Run simulation
   tao = tmc.TaoistMc(config)
   taus = tao.run(z_em=2.5, n_sightlines=200)

   # taus shape: (n_sightlines, n_wavelengths)
   mean_transmission = np.mean(np.exp(-taus), axis=0)

   # Plot
   plt.plot(tao.wav / (1 + 2.5), mean_transmission)
   plt.xlabel(r'$\lambda_{\rm rest}$ (Å)')
   plt.ylabel(r'$\langle T_{\rm IGM}\rangle$')
   plt.show()

Command Line Interface
----------------------

Generate a starter configuration:

.. code-block:: bash

   taoistmc init --output my_config.yaml

Run a simulation (automatically caches results):

.. code-block:: bash

   taoistmc run --config my_config.yaml -n 100 3.0

See ``taoistmc run --help`` and ``taoistmc init --help`` for all options.

Configuration via YAML
----------------------

After running ``taoistmc init``, edit the generated ``config.yaml`` to customize
your physics parameters. The file uses the same structure as the Pydantic models.

Caching Behavior
----------------

Results are saved under ``taoist_runs/zXpX/`` (e.g., ``z2p500`` for z=2.5).
If you request more sightlines than already exist with matching physics,
only the delta is generated. This makes iterative workflows very efficient.