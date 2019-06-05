import sys
import pickle
import seaborn as sns
import scipy.stats as ss
import numpy as np
import core_compute as cc
import core_plot as cp
import matplotlib.pyplot as plt


def feval_Cp(param, T, D=None):

    theta = param[..., 0]
    a = param[..., 1]
    b = param[..., 2]

    R = 8.314459848  # J/mol*K
    frac = theta/T
    expf = np.exp(frac)
    ein = 3*R*(frac**2)*(expf/(expf-1)**2)
    A = ein + a*T + b*T**2

    return A


def feval_H(param, T, D=None):

    theta = param[..., 3]
    a = param[..., 4]
    b = param[..., 5]

    R = 8.314459848  # J/mol*K
    ein = 3*R*theta/(np.exp(theta/T)-1.)
    A = ein + .5*a*T**2 + (1./3.)*b*T**3
    
    T298 = 298.15
    ein298 = 3*R*theta/(np.exp(theta/T298)-1.)
    A298 = ein298 + .5*a*T298**2 + (1./3.)*b*T298**3

    A -= A298

    return A


def likelihood(param, D):
    """
    compute the log likelihood for a set of datapoints given
    a parameterization
    """
    Aest_Cp = feval_Cp(param, D['Tt_Cp'])
    Aest_H = feval_H(param, D['Tt_H'])

    if param[0] <= 0:
        return -np.inf

    prob_CP = ss.norm.logpdf(Aest_Cp, loc=D['At_Cp'], scale=D['Et_Cp']).sum()
    prob_H = ss.norm.logpdf(Aest_H, loc=D['At_H'], scale=D['Et_H']).sum()
    prob = prob_CP + prob_H

    if np.isnan(prob):
        return -np.inf

    return prob


def read_data(param_true):

    SD_Cp = .7
    SD_H = 450

    nCp = 100
    Tt_Cp = np.linspace(1, 75, nCp)
    At_Cp = feval_Cp(param_true, Tt_Cp)
    Et_Cp = SD_Cp*(At_Cp/At_Cp.max())
    At_Cp += ss.norm.rvs(loc=0, scale=Et_Cp, size=nCp)
    It_Cp = 0*np.ones(Tt_Cp.shape)

    nH = 100
    Tt_H = np.linspace(300, 1800, nH)
    At_H = feval_H(param_true, Tt_H) + \
        ss.norm.rvs(loc=0, scale=SD_H, size=nH)
    Et_H = SD_H*np.ones(Tt_H.shape)
    It_H = 0*np.ones(Tt_H.shape)

    return Tt_Cp, Tt_H, At_Cp, At_H, Et_Cp, Et_H, It_Cp, It_H


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

    np.random.seed(0)

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

        D['param_true'] = np.array([150, .006, 7e-7, 150, .006, 7e-7])

        data = read_data(D['param_true'])
        D['Tt_Cp'], D['Tt_H'] = data[0], data[1]
        D['At_Cp'], D['At_H'] = data[2], data[3]
        D['Et_Cp'], D['Et_H'] = data[4], data[5]
        D['It_Cp'], D['It_H'] = data[6], data[7]

        D['likelihood'] = likelihood

        D['distV'] = ['uniform', 'uniform', 'uniform',
                      'uniform', 'uniform', 'uniform']
        D['locV'] = [145, 0.003, -1e-4,
                     0, 0, -5e-6]
        D['scaleV'] = [155, 0.006, 2e-4,
                       750, 0.009, 1e-5]
        D['cV'] = 6*[None]

        D['dim'] = len(D['distV'])

        # name_list: list of the the names of the datasets
        D['name_list'] = ['synthetic data']

        # sampler: select a type of sampler to evaluate the posterior
        # distribution
        D['sampler'] = 'pymultinest'

        """set up the proper set of variable names for the problem
        of interest"""
        D['pname'] = ['theta_Cp', 'a_Cp', 'b_Cp',
                      'theta_H', 'a_H', 'b_H']
        D['pname_plt'] = ['\\theta_{C_p}', 'a_{C_p}', 'b_{C_p}',
                          '\\theta_{H}', 'a_{H}', 'b_{H}']

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
        WP(msg, D['wrt_file'])

        # neff (Gelman, 2014.) gives the effective number of samples for
        # each variable of interest. It should be greater than 10
        # for each variable
        neff = cc.effective_n(trace, D['pname'])
        msg = "effective sample size: %s" % neff
        WP(msg, D['wrt_file'])

        cp.plot_chains(D['rawtrace'], flattrace, D['nlinks'], D['pname'],
                       D['pname_plt'], pltname=D['outname'])
        cp.plot_squiggles(D['rawtrace'], 0, 1, D['pname_plt'], pltname=D['outname'])

    """perform various analyses"""
    msg = "sampling time: " + str(D['sampling_time']) + " seconds"
    cc.WP(msg, D['wrt_file'])

    msg = "model evidence: " + str(D['lnZ']) + \
          " +/- " + str(D['dlnZ'])
    cc.WP(msg, D['wrt_file'])
	
    cp.plot_hist(flattrace, D['pname'], D['pname_plt'],
                 param_true=D['param_true'], pltname=D['outname'])

    cc.coef_summary(flattrace, D['pname'], D['outname'])

    cp.plot_cov(flattrace[:, :3], D['pname_plt'][3:],
	            param_true=D['param_true'][3:],
                figsize=[5.5, 5.5], sciform=True,
				pltname=D['outname'] + '_Cp')

    cp.plot_cov(flattrace[:, 3:], D['pname_plt'][3:],
	            param_true=D['param_true'][3:],
                figsize=[5.5, 5.5], sciform=True,
				pltname=D['outname'] + '_H')

    cp.plot_prediction(flattrace, D['name_list'],
                       D['Tt_Cp'], D['At_Cp'], D['It_Cp'],
                       feval_Cp, D, xlim=[1, 1000], ylim=[-5, 45],
                       xlabel=r"$T \, (K)$",
                       ylabel=r"$C_p \, \left(J \, {mol}^{-1} K^{-1}\right)$",
                       param_true=D['param_true'],
					   pltname=D['outname'] + 'Cp')
    cp.plot_prediction(flattrace, D['name_list'],
                       D['Tt_Cp'], D['At_Cp'], D['It_Cp'],
                       feval_Cp, D, xlim=[1, 80], ylim=[-2, 21],
                       xlabel=r"$T \, (K)$",
                       ylabel=r"$C_p \, \left(J \, {mol}^{-1} K^{-1}\right)$",
                       param_true=D['param_true'],
					   pltname=D['outname'] + 'Cp_close')

    cp.plot_prediction(flattrace, D['name_list'],
                       D['Tt_H'], D['At_H'], D['It_H'],
                       feval_H, D, xlim=[1, 1850], ylim=[-10000, 55000],
                       xlabel=r"$T \, (K)$",
                       ylabel=r"$H \, \left(J \, {mol}^{-1} \right)$",
                       param_true=D['param_true'],
					   pltname=D['outname'] + '_H')
    cp.plot_prediction(flattrace, D['name_list'],
                       D['Tt_H'], D['At_H'], D['It_H'],
                       feval_H, D, xlim=[1, 500], ylim=[-8000, 8000],
                       xlabel=r"$T \, (K)$",
                       ylabel=r"$H \, \left(J \, {mol}^{-1} \right)$",
                       param_true=D['param_true'],
					   pltname=D['outname'] + 'H_close')

    plt.show()
