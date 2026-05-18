![Da Way](docs/tmc.png)

Install:

clone from this branch then run from `TAOIST_MC` directory:

```bash
pip install .
```

NOTE - `taoistmc` now includes saving/loading functionality. By default run outputs will be saved in `./taoist_runs/zXpX/` where zXpX is the redshift of the run (e.g. run at z_em=2.4 would be `./taoist_runs/z2p4`). If subsequent runs are performed at the same redshift with the same config it will only generate new sightlines if these are required. 

In the event that there are existing sightlines that can be loaded, this will happen automatically when calling `TaoistMc.run()` and will be appended to the outputs of this method. Note that all available data will be returned, i.e. if you request 50 sightlines, but 100 are avaialble, the method will return all 100.

Demo code

```python
import taoistmc as tmc
import numpy as np
import matplotlib.pyplot as plt

from taoistmc.config import PowerLawSegment, SightlineConfig, TaoistConfig

if __name__ == "__main__":

    igm_low = PowerLawSegment(log_N_min=12.0, log_N_max=15.2, beta=1.635, log_A=9.305, gamma=2.5)
    igm_high = PowerLawSegment(log_N_min=15.2, log_N_max=21.0, beta=1.463, log_A=7.542, gamma=1.0)
    cgm_seg = PowerLawSegment(log_N_min=13.0, log_N_max=21.0, beta=1.381, log_A=6.716, gamma=1.0)

    config = SightlineConfig(
        igm_segments=[igm_low, igm_high],
        cgm_segments=[cgm_seg]
    )

    full_config = TaoistConfig(
        sightline_config=config,
        delta_wav=0.25
    )

    F = plt.figure()
    ax = F.add_subplot(111)
    for z in (1.5,2.5,3.5):
        tao = tmc.TaoistMc(full_config)
        output = tao.run(z, 200)

        taum = np.mean(np.exp(-output),axis=0)
        ax.plot(tao.wav/(1.+z),taum,lw=1)
    [ax.axvline(x=_x,c='r',ls='--',lw=.3) for _x in [911.75,1216.]]
    ax.set_ylim(-.1,1.1)
    ax.set_xlim(799,1249)
    ax.set_xlabel(r'$\lambda_{rest}$',fontsize=18)
    ax.set_ylabel(r'$T_{IGM}$',fontsize=18)
    plt.show()

```

`taoistmc` now includes the ability to run from config `yaml` and a handy CLI. The CLI has two methods, `init` and `run`. You can call `--help` for each method to get more info, e.g.:

```bash
taoistmc run --help
```

The `init` method will create a new `config.yaml` with parameters matched to Steidel et al. 2018:

```bash
taoistmc init
```

The `run` method will run a set of sightlines at the specified redshift. By default 100 sightlines will be generated, however this is easily modified:

```bash
taoistmc run -n 50 2.4
```

Above will run 50 sightlines at redshift 2.4.

... full documention coming soon ...