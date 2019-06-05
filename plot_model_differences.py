import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d


sns.set(color_codes=True)
colorL = sns.color_palette("Paired")
LL = ['-', ':', '-.', '--']

sta = np.loadtxt('HSC_chem_Hf.csv', delimiter=',', skiprows=1,
                 usecols=range(5))
staB = [40, 106, 118, sta.shape[0]]

arb = np.loadtxt('arb_Hf.csv', delimiter=',', skiprows=1)
arbB = [0, 54, 61, arb.shape[0]]

pau = np.loadtxt('Hf_summ.txt', delimiter=',', skiprows=1)
pauB = [0, 228, 278, pau.shape[0]]

quant = ['C_p', 'H', 'S', 'G']

plt.figure(figsize=[8, 4])

for ii in range(3):

    ax = plt.subplot(1, 3, ii+1)

    for jj in range(4):

        Tsta = sta[staB[ii]:staB[ii+1], 0]
        Asta = sta[staB[ii]:staB[ii+1], jj+1]

        Tarb = arb[arbB[ii]:arbB[ii+1], 0]
        Aarb = arb[arbB[ii]:arbB[ii+1], jj+1]

        Tpau = pau[pauB[ii]:pauB[ii+1], 0]
        Apau = pau[pauB[ii]:pauB[ii+1], jj+1]

        # develop interpolant model for pau
        Apau_ = interp1d(Tpau, Apau,
                         kind='linear', bounds_error=False,
                         fill_value='extrapolate')

        Esta = -100*(Apau_(Tsta)-Asta)/Apau_(Tsta)
        Earb = -100*(Apau_(Tarb)-Aarb)/Apau_(Tarb)

        labsta = r"$%s$ HSC Chem" % quant[jj]
        labarb = r"$%s$ Arb2014" % quant[jj]

        ax.plot(Tsta, Esta, c='k', ls=LL[jj],
                alpha=.9, label=labsta)
        ax.plot(Tarb, Earb, c='g', ls=LL[jj],
                alpha=.9, label=labarb)

        ax.set_xlim(Tarb.min(), Tarb.max())
        ax.locator_params(nbins=3, axis='x')

    box = ax.get_position()

    if ii == 0:
        ax.set_ylabel('% deviation', fontsize='large')

        ax.set_position([box.x0,
                         box.y0 + box.height * 0.3,
                         box.width*.9,
                         box.height * 0.8])
    if ii == 1:
        ax.set_xlabel('T (K)', fontsize='large')

        ax.set_position([box.x0 + box.width*.05,
                         box.y0 + box.height * 0.3,
                         box.width*.95,
                         box.height * 0.8])
    if ii == 2:
        ax.set_position([box.x0 + box.width*.2,
                         box.y0 + box.height * 0.3,
                         box.width*.9,
                         box.height * 0.8])

        ax.legend(loc='lower right', shadow=False,
                  fontsize=11, ncol=4, fancybox=False,
                  bbox_to_anchor=(1.1, -.5))

    plt.tick_params(axis='both', labelsize='large')

plt.savefig('model_differences.png')

plt.show()
