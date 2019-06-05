import matplotlib.pyplot as plt
import importlib
import os
import pickle
import seaborn as sns
import numpy as np
from scipy.integrate import cumtrapz


def plot_prediction_all(flattrace, Tt, At, It,
                        pltper, fnames,
                        yerr=None, PerrS=None,
                        colorLL=None, CI=None,
                        xlim=None, ylim=None, pltname=None,
                        xlabel=None, ylabel=None,
                        legend_loc=None, leg_ncol=None):

    """import feval functions"""
    feval_Cp_alpha = importlib.import_module(fnames[0]).feval_Cp_plt
    feval_H_alpha = importlib.import_module(fnames[0]).feval_H_plt
    deb_Cp = importlib.import_module(fnames[0]).deb_Cp

    feval_Cp_beta = importlib.import_module(fnames[1]).feval_Cp_plt
    feval_H_beta = importlib.import_module(fnames[1]).feval_H_plt

    feval_Cp_liquid = importlib.import_module(fnames[2]).feval_Cp_plt
    feval_H_liquid = importlib.import_module(fnames[2]).feval_H_plt

    markermat = ['o', 'v', 'p',
                 's', '>', 'P',
                 '*', '<', 'X',
                 'D', 'd', '^',
                 'o', 'v', 'p',
                 's', '>', 'P',
                 '*', '<', 'X',
                 'D', 'd', '^']

    """define plotting points"""
    Tplt1 = np.linspace(1e-10, 30, 30)[:-1]
    Tplt2 = np.linspace(30, 298.15, 100)[:-1]
    Tplt3 = np.linspace(298.15, 2016, 100)
    T_alp = np.concatenate([Tplt1, Tplt2, Tplt3])
    T_bet = np.linspace(2016, 2502, 50)
    T_liq = np.linspace(2502, 4900, 100)
    Tplt = np.concatenate([T_alp, T_bet, T_liq])
    nplt = Tplt.size

    """obtain the parameter traces, truncate
    the traces for each phase to the minimum
    number of samples in any trace"""
    skip = 1
    smin = np.min([flattrace[0].shape[0],
                   flattrace[1].shape[0],
                   flattrace[2].shape[0]])
    flattrace_alp = flattrace[0][:smin:skip, :]
    flattrace_bet = flattrace[1][:smin:skip, :]
    flattrace_liq = flattrace[2][:smin:skip, :]

    """compute Cp, H, S and G for Tplt points. This calculation
    must be split up by phase for Cp and H"""

    # compute the debye term for the alpha phase
    deb = np.zeros((flattrace_alp.shape[0], T_alp.size))
    for ii in range(T_alp.size):
        means = deb_Cp(flattrace_alp[:, 0], T_alp[ii])
        deb[:, ii] = means

    Cp_alp = feval_Cp_alpha(flattrace_alp, T_alp, deb)
    H_alp = feval_H_alpha(flattrace_alp, T_alp, deb)

    # compute Cp and H for the beta phase
    Cp_bet = feval_Cp_beta(flattrace_bet, T_bet)
    H_bet = feval_H_beta(flattrace_bet, T_bet)

    # compute Cp and H for the liquid phase
    Cp_liq = feval_Cp_liquid(flattrace_liq, T_liq)
    H_liq = feval_H_liquid(flattrace_liq, T_liq)

    # compile Cp and H for all phases
    Cp = np.concatenate([Cp_alp, Cp_bet, Cp_liq], axis=1)
    H = np.concatenate([H_alp, H_bet, H_liq], axis=1)

    # compute S for the alpha phase
    S_alp = cumtrapz(y=Cp_alp/T_alp, x=T_alp, axis=-1, initial=0)
    S_alp -= S_alp[..., 0, None]

    # compute S for the beta phase
    S_bet = cumtrapz(y=Cp_bet/T_bet, x=T_bet, axis=-1, initial=0)
    S_bet += S_alp[..., -1, None]
    S_bet += (H_bet[:, 0] - H_alp[:, -1])[..., None]/T_bet[0]

    # compute S for the liquid phase
    S_liq = cumtrapz(y=Cp_liq/T_liq, x=T_liq, axis=-1, initial=0)
    S_liq += S_bet[..., -1, None]
    S_liq += (H_liq[:, 0] - H_bet[:, -1])[..., None]/T_liq[0]

    S = np.concatenate([S_alp, S_bet, S_liq], axis=1)

    G = H - Tplt*S

    Av = [Cp, H, S, G]

    """print a summary file with tabulated values"""

    summary = np.zeros((nplt, 5))
    summary[:, 0] = Tplt
    summary[:, 1] = np.mean(Cp, 0)
    summary[:, 2] = np.mean(H, 0)
    summary[:, 3] = np.mean(S, 0)
    summary[:, 4] = np.mean(G, 0)
    header = 'T (K),Cp (J/mol K),H-H298.15 (J/mol),S-S0 (J/mol K),G-H298.15 (J/mol)'
    np.savetxt('Hf_summ.txt', summary, header=header, delimiter=',')

    """prepare all scheduled plots"""

    listcomp = [[ii, jj] for ii in range(len(pltper))
                for jj in range(pltper[ii])]

    for ii, jj in listcomp:

        plt.figure(figsize=[7, 4.5])

        # colorL contains a unique color for each dataset
        if colorLL is None:
            colorL = sns.color_palette("hls",
                                       np.unique(It[ii]).size)

        # if errors are not provided, get bounds from the T points
        if xlim is None or xlim[ii] is None or xlim[ii][jj] is None:
            spc = .15
            xPlb = Tt[ii].min()-spc*(Tt[ii].max()-Tt[ii].min())
            xPub = Tt[ii].max()+spc*(Tt[ii].max()-Tt[ii].min())
        else:
            xPlb, xPub = xlim[ii][jj]

        if It[ii] is not None:
            kk = 0
            for name in np.unique(It[ii]):
                x = Tt[ii][It[ii] == name]
                y = At[ii][It[ii] == name]
                marker = markermat[kk]
                color = colorL[kk]
                label = name

                # PerrS[ii][jj] is a list of datasets to plot errorbars
                # for. If plterr is set to False, no errorbars are
                # plotted. plterr is only True if the dataset
                # label in in PerrS[ii][jj]
                if PerrS is None or PerrS[ii] is None or PerrS[ii][jj] is None:
                    plterr = False
                else:
                    if label in PerrS[ii][jj]:
                        plterr = True
                    else:
                        plterr = False

                # only plot data points if they fall within
                # the x limits
                if np.any((xPlb <= x)*(x <= xPub)):
                    if plterr:
                        plt.errorbar(x, y,
                                     yerr=yerr[ii][np.array(It[ii]) == name],
                                     marker=marker, markersize=6,
                                     color=color, linestyle='',
                                     alpha=.8, label=label)
                    else:
                        plt.plot(x, y,
                                 marker=marker, markersize=6,
                                 color=color, linestyle='',
                                 alpha=.8, label=label)

                kk += 1

        if CI is None or CI[ii] is None or CI[ii][jj] is None:
            CI_ = 95
        else:
            CI_ = CI[ii][jj]

        low, mid, high = np.percentile(Av[ii],
                                       [0.5*(100-CI_),
                                        50,
                                        100-0.5*(100-CI_)],
                                       axis=0)

        plt.fill_between(Tplt, low, high,
                         alpha=0.3, facecolor='b',
                         label='%s%% CI' % CI_)

        plt.plot(Tplt, mid, c='b', alpha=.9, label='This work')

        stan = np.loadtxt('HSC_chem_Hf.csv', delimiter=',', skiprows=1,
                          usecols=range(5))
        Tstan = stan[:, 0]
        Astan = stan[:, ii+1]
        plt.plot(Tstan, Astan, c='k', alpha=.9, label='HSC Chem',
                 linestyle=':')

        arb = np.loadtxt('arb_Hf.csv', delimiter=',', skiprows=1)
        Tarb = arb[:, 0]
        Aarb = arb[:, ii+1]
        plt.plot(Tarb, Aarb, c='g', alpha=.9, label='Arb2014',
                 linestyle='--')

        plt.xlim(xPlb, xPub)

        if ylim is None or ylim[ii] is None or ylim[ii][jj] is None:
            # spacing factor outside of the x limit ranges
            spc = .15

            # restrict range of evaluating y limits to the
            # x range
            low_ = low[(Tplt > xPlb)*(Tplt < xPub)]
            high_ = high[(Tplt > xPlb)*(Tplt < xPub)]

            yPlb = low_.min()-spc*(high_.max()-low_.min())
            yPub = high_.max()+spc*(high_.max()-low_.min())
        else:
            yPlb, yPub = ylim[ii][jj]
        plt.ylim(yPlb, yPub)

        if leg_ncol is not None and \
           leg_ncol[ii] is not None and \
           leg_ncol[ii][jj] is not None:
            ncol = leg_ncol[ii][jj]
        else:
            ncol = 2

        if legend_loc is not None and \
           legend_loc[ii] is not None and \
           legend_loc[ii][jj] is not None:
            legloc = legend_loc[ii][jj]
        else:
            legloc = 'lower right'

        plt.legend(loc=legloc, shadow=False,
                   fontsize=12, ncol=ncol, fancybox=False)

        if xlabel is None or xlabel[ii] is None or xlabel[ii][jj] is None:
            plt.xlabel('x', fontsize='large')
        else:
            plt.xlabel(xlabel[ii][jj], fontsize='large')

        if ylabel is None or ylabel[ii] is None or ylabel[ii][jj] is None:
            plt.ylabel('y', fontsize='large')
        else:
            plt.ylabel(ylabel[ii][jj], fontsize='large')

        plt.tick_params(axis='both', labelsize='large')

        plt.tight_layout()

        if pltname is not None and \
           pltname[ii] is not None and \
           pltname[ii][jj] is not None:
            plt.savefig(pltname[ii][jj] + "_pred.png")


if __name__ == '__main__':

    sns.set(color_codes=True)

    """import traces from pkl files"""
    fnames = ['alpha_quart_debye', 'beta_quad', 'liquid_lin']
    flattrace_l = []
    Tt_Cp, At_Cp, Et_Cp, Etr_Cp, It_Cp = [], [], [], [], []
    Tt_H, At_H, Et_H, Etr_H, It_H = [], [], [], [], []

    for name in fnames:
        with open(name + '.pkl', 'rb') as buff:
            D = pickle.load(buff)

            if D['sampler'] == 'pymultinest':
                flattrace = D['rawtrace']
            else:
                """remove the tuning samples from the raw trace
                (nwalkers, nlinks, dim)"""
                trace = D['rawtrace'][:, -D['nlinks']:, :]
                """obtain a flattened version of the chain"""
                flattrace = trace.reshape((D['nlinks']*D['nwalkers'],
                                           len(D['pname'])))
            flattrace_l += [flattrace]

            Tt_Cp += list(D['Tt_Cp'])
            At_Cp += list(D['At_Cp'])
            Et_Cp += list(D['Et_Cp'])
            It_Cp += list(np.array(D['name_list_Cp'])[D['It_Cp']])
            Etr_Cp += list(D['Etr_Cp'])

            Tt_H += list(D['Tt_H'])
            At_H += list(D['At_H'])
            Et_H += list(D['Et_H'])
            It_H += list(np.array(D['name_list_H'])[D['It_H']])
            Etr_H += list(D['Etr_H'])

    Tt_Cp = np.array(Tt_Cp)
    At_Cp = np.array(At_Cp)
    Et_Cp = np.array(Et_Cp)
    It_Cp = np.array(It_Cp)
    Etr_Cp = np.array(Etr_Cp)
    Tt_H = np.array(Tt_H)
    At_H = np.array(At_H)
    Et_H = np.array(Et_H)
    It_H = np.array(It_H)
    Etr_H = np.array(Etr_H)

    Tt_l = [Tt_Cp, Tt_H, None, None]
    At_l = [At_Cp, At_H, None, None]
    Et_l = [Et_Cp, Et_H, None, None]
    It_l = [It_Cp, It_H, None, None]
    Etr_l = [Etr_Cp, Etr_H, None, None]
    pltper = [5, 4, 1, 1]
    xlim = [[(1e-10, 30), (1e-10, 2100), (1e-10, 4900),
             (1950, 2550), (2200, 4900)],
            [(1e-10, 2100), (1e-10, 4900),
             (1950, 2550), (2200, 4900)],
            [(1e-10, 4900)],
            [(1e-10, 4900)]]
    ylim = [[None, None, None, (31, 45), (30, 65)],
            None,
            None,
            None]
    PerrS = [[('Bur1958'), ('Fil1971'), None, ('Mil2006S2'), ('Kor2005')],
             [('Cag2008'), None, ('Kats1985'), ('Ros2001')],
             None, None]
    on = 'allphase'
    pltname = [[on + 'Cp_all_lowT', on + 'Cp_all_alpha',
                on + 'Cp_all', on + 'Cp_all_beta',
                on + 'Cp_all_liquid'],
               [on + 'H_all_alpha', on + 'H_all',
                on + 'H_all_beta', on + 'H_all_liquid'],
               [on + 'S_all'],
               [on + 'G_all']]
    xlabel = [5*[r"$T (K)$"],
              4*[r"$T (K)$"],
              [r"$T (K)$"],
              [r"$T (K)$"]]
    ylabel = [5*[r"$C_p \left(J {mol}^{-1} K^{-1}\right)$"],
              4*[r"$H \left(J {mol}^{-1} \right)$"],
              [r"$S \left(J {mol}^{-1} K^{-1}\right)$"],
              [r"$G \left(J {mol}^{-1} \right)$"]]
    legend_loc = [['upper center', None, None,
                   'upper center', 'upper left'],
                  4*['upper left'],
                  [None],
                  ['lower left']]
    leg_ncol = [[3, 3, 3, 2, 3],
                4*[2],
                [2],
                [2]]

    plot_prediction_all(flattrace_l, Tt_l, At_l, It_l,
                        pltper, fnames,
                        yerr=Etr_l, PerrS=PerrS,
                        xlim=xlim, ylim=ylim, pltname=pltname,
                        xlabel=xlabel, ylabel=ylabel,
                        legend_loc=legend_loc, leg_ncol=leg_ncol)

    plt.show()
