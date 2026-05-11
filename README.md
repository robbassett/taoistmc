![Da Way](docs/tmc.png)

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
        sightline_config=config
    )

    F = plt.figure()
    ax = F.add_subplot(111)
    for z in (1.5,2.5,3.5):
        tao = tmc.TaoistMc(z, full_config)
        output = tao.run(200)

        taum = np.mean(np.exp(-output),axis=0)
        ax.plot(tao.wav/(1.+z),taum,lw=1)
    [ax.axvline(x=_x,c='r',ls='--',lw=.3) for _x in [911.75,1216.]]
    ax.set_ylim(-.1,1.1)
    ax.set_xlim(799,1249)
    ax.set_xlabel(r'$\lambda_{rest}$',fontsize=18)
    ax.set_ylabel(r'$T_{IGM}$',fontsize=18)
    plt.show()

```