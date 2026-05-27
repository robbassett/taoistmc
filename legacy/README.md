![Logo by James Josephides Swinburne Astronomy Productions](/TMClogo.png)

*Logo by James Josephides, Swinburne Astronomy Productions*

TrAnsmission Of IoniSing lightT - Monte Carlo: Simulated IGM UV Transmission

This code uses hydrogen absorption system statistics from the literature (redshift and column density distribution functions)
to simulate the transmission of ionizing radiation through Hydrogen in the IGM for sources at a specified redshift. 
The primary outputs of TAOIST-MC are ensembles of IGM transmission functions at UV wavelengths, i.e. T(lambda).

The main use case for TAOIST-MC to is produce Monte Carlo simulations of the expected observed ionising
flux from sources of ionizing radiation (star-forming galaxies or AGN) by coupling the IGM transmission functions produced
with model spectra of high redshift sources produced with population synthesis (e.g. BPASS). For example, using 10,000 IGM
sightlines from TAOIST-MC at z > 3.4 coupled to a BPASS model, one can estimate the detected flux in the photometric u-band
and compare with known sources from your favourite deep survey. In this way, for individual
detection of ionizing flux at high redshift, TAOIST-MC can be used to predict the probability distribution of ionising
photon escape in a more meaningful way than simply assuming a mean IGM transmission function for all galaxies, thus accounting
for possible biases for detections of ionizing radiation towards high transmission sightlines. Below is Figure one from Bassett et al. 2021 illustrating the basic outputs of TOAIST-MC.

![Figure 1 of Bassett et al. 2021](Bassett2021_F1.png)

[More detailed description of TOAIST-MC can be found in Section 2 of Bassett et al. 2021 (soon to be published in Monthly Notices of the Royal Astronomical Society)](https://arxiv.org/abs/2101.00727)

Development of TAOIST-MC is ongoing with plans to make the code more user friendly with a number of example script to be
provided. In the future, there are also plans to include Helium ionization, a process that may play an important role at 
higher photon energies.

# Installation (requires python 3+)

To install TAOIST-MC navigate to the parent directory where you would like TAOIST-MC to live and clone the repository there:

bash$git clone https://github.com/robbassett/TAOIST_MC

Now TAOIST\_MC is setup to run from within the TAOIST\_MC folder. If you want to import TAOIST_MC from another location, simply add the TAOIST\_MC folder to your path. For bash shell this can be done by adding the following line to your .bashrc file:

export PATH=/path-to-TMC/TAOIST_MC/:$PATH

where "/path-to-TMC/" is the full path to the folder containing the TAOIST\_MC folder on YOUR system. If not using bash, Google how to do this for your shell.





Comments, questions, concerns? Email me at: rbassett.astro@gmail.com
