import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import sys
import os
import pickle
import seaborn as sns
import scipy.stats as ss
import numpy as np
import core_compute as cc
import core_plot as cp


def feval_Cp_plt(param, T):

    b_1 = param[..., 1, None]
    b_2 = param[..., 2, None]
    b_3 = param[..., 3, None]

    A = b_1 + b_2*T + b_3*T**2

    return A


def feval_H_plt(param, T):

    b_0 = param[..., 0, None]
    b_1 = param[..., 1, None]
    b_2 = param[..., 2, None]
    b_3 = param[..., 3, None]

    A = b_0 + b_1*T + 0.5*b_2*T**2 + (1./3.)*b_3*T**3

    return A


def likelihood(param, D):
    """
    compute the log likelihood for a set of datapoints given
    a parameterization
    """
    dA_Cp = D['At_Cp']-feval_Cp_plt(param, D['Tt_Cp'])
    dA_H = D['At_H']-feval_H_plt(param, D['Tt_H'])

    # obtain the hyperparameter vectors for Cp and H
    nhyp_H = len(D['name_list_H'])
    hyp_H = param[-nhyp_H:]

    nhyp_Cp = len(D['name_list_Cp'])
    hyp_Cp = param[-(nhyp_H+nhyp_Cp):-nhyp_H]

    if np.any(hyp_Cp <= 0) or np.any(hyp_H <= 0):
        return -np.inf

    hypvec_Cp = np.zeros(D['Tt_Cp'].shape)
    for ii in range(nhyp_Cp):
        hypvec_Cp[D['It_Cp'] == ii] = hyp_Cp[ii]

    hypvec_H = np.zeros(D['Tt_H'].shape)
    for ii in range(nhyp_H):
        hypvec_H[D['It_H'] == ii] = hyp_H[ii]

    dof = 2+1e-6
    prob_Cp = ss.t.logpdf(dA_Cp, dof, loc=0, scale=D['Et_Cp']/hypvec_Cp).sum()
    prob_H = ss.t.logpdf(dA_H, dof, loc=0, scale=D['Et_H']/hypvec_H).sum()
    prob = prob_Cp + prob_H

    if np.isnan(prob):
        return -np.inf

    return prob


def get_data(name_list, phase):

    Tt, At, Et, It = [], [], [], []

    """load data from the text files"""
    wd = os.getcwd()
    os.chdir('data_process')

    for ii in range(len(name_list)):
        f = open('%s.csv' % name_list[ii], 'r')
        lines = list(f)
        f.close()
        for jj in range(1, len(lines)):
            tmp = lines[jj].split()
            if tmp[3] == phase:
                Tt += [tmp[0]]
                At += [tmp[1]]
                Et += [tmp[2]]
                It += [ii]

    Tt = np.array(Tt).astype(float)
    At = np.array(At).astype(float)
    Et = np.array(Et).astype(float)
    It = np.array(It).astype(int)

    sorting = np.argsort(Tt)
    Tt = Tt[sorting]
    At = At[sorting]
    Et = Et[sorting]
    It = It[sorting]

    os.chdir(wd)

    return Tt, At, Et, It


def WP(msg, filename):
    """
    Summary:
        This function takes an input message and a filename, and appends that
        message to the file. This function also prints the message
    Inputs:
        msg (string): the message to write and print.
        filename (string): the full name of the file to append to.
    Outputs:
        both prints the message and writes the message to the specified file
    """
    fil = open(filename, 'a')
    print(msg)
    fil.write(msg)
    fil.write('\n')
    fil.close()


if __name__ == '__main__':
    """initialize important variables"""
    sns.set(color_codes=True)
    np.random.seed(1)

    """either load the trace and parameters
    or compute from scratch"""
    if len(sys.argv) > 1:
        # load the trace and the model
        with open(sys.argv[1], 'rb') as buff:
            D = pickle.load(buff)

    else:
        # for convenience, store all important variables in dictionary
        D = {}

        # save the current file name
        D['fname'] = sys.argv[0]

        # outname is the name for plots, etc
        D['outname'] = D['fname'][:-3]

        # set up a log file
        D['wrt_file'] = D['outname'] + '.txt'
        fil = open(D['wrt_file'], 'w')
        fil.close()

        # name_list: list of the the names of the datasets
        # D['name_list'] = ['Ade1952', 'Aru1972', 'Bur1958',
        #                   'Cag2008', 'Cez1974', 'Col1971',
        #                   'Fie1961', 'Fil1971', 'Gol1970',
        #                   'Haw1963', 'Kat1985', 'Kne1963',
        #                   'Kor2005', 'McC1964', 'Mil2006S1',
        #                   'Mil2006S2', 'Par2003', 'Pel1971',
        #                   'Ros2001', 'Wol1957']
        D['name_list_Cp'] = ['Cez1974', 'Fil1971', 'Mil2006S1',
                             'Mil2006S2', 'Par2003', 'Pel1971']
        D['name_list_H'] = ['Cag2008', 'Kat1985', 'Ros2001']
        D['name_list'] = D['name_list_Cp']+D['name_list_H']
        nds = len(D['name_list'])

        D['phase'] = 'beta'

        data = get_data(D['name_list_Cp'], D['phase'])
        D['Tt_Cp'], D['At_Cp'] = data[0], data[1]
        D['Et_Cp'], D['It_Cp'] = data[2], data[3]
        data = get_data(D['name_list_H'], D['phase'])
        D['Tt_H'], D['At_H'] = data[0], data[1]
        D['Et_H'], D['It_H'] = data[2], data[3]

        D['likelihood'] = likelihood

        # define the prior distributions
        D['npar_model'] = 4
        D['distV'] = D['npar_model']*['uniform'] + nds*['expon']

        if os.path.exists(D['outname'] + '_prior.csv'):
            print('prior suggestions loaded')
            nxtprior = np.loadtxt(D['outname'] + '_prior.csv')
            D['locV'] = list(nxtprior[0, :D['npar_model']])
            D['scaleV'] = list(nxtprior[1, :D['npar_model']])
        else:
            D['locV'] = [-1e5, 0, -1e-1, -1e-4]
            D['scaleV'] = [2e5, 100, 2e-1, 2e-4]

        D['locV'] += nds*[None]
        D['scaleV'] += nds*[None]
        D['cV'] = (D['npar_model']+nds)*[None]

        # sampler: select a type of sampler to evaluate the posterior
        # distribution
        D['sampler'] = 'pymultinest'

        """set up the proper set of variable names for the problem
        of interest"""
        D['pname'] = ['b_0', 'b_1', 'b_2', 'b_3']
        D['pname_plt'] = ['b_0', 'b_1', 'b_2', 'b_3']

        for ii in range(nds):
            D['pname'] += ['alpha_%s' % D['name_list'][ii]]
            D['pname_plt'] += ['\\alpha_{%s}' % D['name_list'][ii]]

        D['nparam'] = len(D['pname'])

        """run the MH algorithm to sample posterior distribution"""

        if D['sampler'] == 'kombine':
            D = cc.sampler_kombine(D)
        elif D['sampler'] == 'emcee':
            D = cc.sampler_emcee(D)
        elif D['sampler'] == 'pymultinest':
            D = cc.sampler_multinest(D)
        else:
            print('invalid sampler selected')
            sys.exit()

        """calculate rescaled errors"""
        if D['sampler'] == 'pymultinest':
            flattrace = D['rawtrace']

        else:
            """remove the tuning samples from the raw trace
            (nwalkers, nlinks, dim)"""
            trace = D['rawtrace'][:, -D['nlinks']:, :]
            """obtain a flattened version of the chain"""
            flattrace = trace.reshape((D['nlinks']*D['nwalkers'],
                                       len(D['pname'])))

        nhyp_H = len(D['name_list_H'])
        hyp_H = np.mean(flattrace[:, -nhyp_H:], 0)
        print(nhyp_H)
        print(hyp_H.shape)

        nhyp_Cp = len(D['name_list_Cp'])
        hyp_Cp = np.mean(flattrace[:, -(nhyp_H+nhyp_Cp):-nhyp_H], 0)

        hypvec_Cp = np.zeros(D['Tt_Cp'].shape)
        for ii in range(nhyp_Cp):
            hypvec_Cp[D['It_Cp'] == ii] = hyp_Cp[ii]

        hypvec_H = np.zeros(D['Tt_H'].shape)
        for ii in range(nhyp_H):
            hypvec_H[D['It_H'] == ii] = hyp_H[ii]

        D['Etr_Cp'] = D['Et_Cp']/hypvec_Cp
        D['Etr_H'] = D['Et_H']/hypvec_H

        # save the trace and the posterior samples
        with open(D['outname'] + '.pkl', 'wb') as buff:
            pickle.dump(D, buff)

    """perform post-processing and analyses on the sampled chains"""
    if D['sampler'] == 'pymultinest':
        flattrace = D['rawtrace']

    else:

        """remove the tuning samples from the raw trace
        (nwalkers, nlinks, dim)"""
        trace = D['rawtrace'][:, -D['nlinks']:, :]

        """obtain a flattened version of the chain"""
        flattrace = trace.reshape((D['nlinks']*D['nwalkers'], len(D['pname'])))

        """compute convergence diagnostics"""

        # Rhat (Gelman, 2014.) diagnoses convergence by checking the mixing
        # of the chains as well as their stationarity. Rhat should be less than
        # 1.1 for each variable of interest
        Rhat = cc.gelman_diagnostic(trace, D['pname'])
        msg = "Rhat: %s" % Rhat
        cc.WP(msg, D['wrt_file'])

        # neff (Gelman, 2014.) gives the effective number of samples for
        # each variable of interest. It should be greater than 10
        # for each variable
        neff = cc.effective_n(trace, D['pname'])
        msg = "effective sample size: %s" % neff
        cc.WP(msg, D['wrt_file'])

        cp.plot_chains(D['rawtrace'], flattrace, D['nlinks'], D['pname'],
                       D['pname_plt'], pltname=D['outname'])
        cp.plot_squiggles(D['rawtrace'], 0, 1, D['pname_plt'],
                          pltname=D['outname'])

    """perform various analyses"""
    msg = "sampling time: " + str(D['sampling_time']) + " seconds"
    WP(msg, D['wrt_file'])

    msg = "model evidence: " + str(D['lnZ']) + \
          " +/- " + str(D['dlnZ'])
    cc.WP(msg, D['wrt_file'])

    cc.coef_summary(flattrace, D['pname'], D['wrt_file'])

    nxtprior = np.zeros((2, D['nparam']))
    nxtprior[0, :] = np.mean(flattrace, 0) - 5*np.std(flattrace, 0)
    nxtprior[1, :] = 10*np.std(flattrace, 0)
    np.savetxt(D['outname'] + '_prior.csv', nxtprior)

    cp.plot_hist(flattrace, D['pname'], D['pname_plt'], pltname=D['outname'])

    cp.plot_cov(flattrace, D['pname_plt'], pltname=D['outname'],
                tight_layout=False)

    """configure model prediction plots for Cp, H, S and G"""
    name_list_l = [D['name_list_Cp'], D['name_list_H'], None, None]
    Tt_l = [D['Tt_Cp'], D['Tt_H'], None, None]
    At_l = [D['At_Cp'], D['At_H'], None, None]
    Etr_l = [D['Etr_Cp'], D['Etr_H'], None, None]
    It_l = [D['It_Cp'], D['It_H'], None, None]
    pltper = [1, 1, 0, 0]
    xlim = [[(1800, 2600)], [(1800, 2600)], None, None]
    ylim = [[(24, 40)], [(47000, 85000)], None, None]
    on = D['outname']
    pltname = [[on + 'Cp', on + 'Cp_close', on + 'Cp_vclose'],
               [on + 'H'],
               [on + 'S'],
               [on + 'G']]
    xlabel = [[r"$T (K)$"],
              [r"$T (K)$"],
              [r"$T (K)$"],
              [r"$T (K)$"]]
    ylabel = [[r"$C_p \left(J {mol}^{-1} K^{-1}\right)$"],
              [r"$H \left(J {mol}^{-1} \right)$"],
              [r"$S \left(J {mol}^{-1} K^{-1}\right)$"],
              [r"$G \left(J {mol}^{-1} \right)$"]]
    legend_loc = [[None],
                  [None],
                  [None],
                  [None]]

    cp.plot_prediction_all(flattrace, name_list_l,
                           Tt_l, At_l, It_l, pltper,
                           feval_Cp_plt, feval_H_plt,
                           yerr=Etr_l,
                           xlim=xlim, ylim=ylim, pltname=pltname,
                           xlabel=xlabel, ylabel=ylabel,
                           legend_loc=legend_loc)
