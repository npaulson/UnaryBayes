import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import sys
import pickle
import seaborn as sns
import scipy.stats as ss
import numpy as np
import core_compute as cc
import core_plot as cp


def feval(param, T, D):

    A = np.zeros((np.atleast_2d(param)[..., 0]*T).shape)
    for ii in range(D['order']+1):
        A += np.atleast_2d(param)[..., ii]*T**ii

    return A


def likelihood(param, D):
    """
    compute the log likelihood for a set of datapoints given
    a parameterization
    """
    dA = D['At']-feval(param, D['Tt'], D)

    # make sure the likelihood is -inf if the scale <= 0
    if param[-1] <= 0:
        return -np.inf

    prob = ss.norm.logpdf(dA, loc=0, scale=param[-1]).sum()

    if np.isnan(prob):
        return -np.inf

    return prob


def read_data(param_true):

    npts = 10
    Tt = ss.uniform.rvs(loc=0, scale=1, size=npts)
    Aerr = ss.norm.rvs(loc=0, scale=param_true[2], size=npts)
    At = param_true[0] + param_true[1]*Tt + Aerr
    At[4] += 0.5
    Et = np.zeros(At.shape)
    It = np.zeros(At.shape)
    It[4] = 1

    return Tt, At, Et, It


def start_pos(size, order):

    pos = np.zeros((size, order+2))
    for ii in range(order+1):
        pos[:, ii] = ss.uniform.rvs(loc=-2, scale=4, size=size)
    pos[:, -1] = ss.uniform.rvs(loc=0, scale=1, size=size)

    return pos


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

        D['param_true'] = [1, 1, .05]

        D['Tt'], D['At'], D['Et'], D['It'] = read_data(D['param_true'])

        D['likelihood'] = likelihood

        # define the prior distributions
        D['distV'] = ['uniform', 'uniform', 'uniform']
        D['locV'] = [-2, -2, 0]
        D['scaleV'] = [4, 4, 1]
        D['cV'] = [None, None, None]
        D['dim'] = len(D['distV'])

        # name_list: list of the the names of the datasets
        D['name_list'] = ['synthetic data', 'outlier']

        # typ: whether this is a linear, quadratic or cubic regression
        D['order'] = 1

        # sampler: select a type of sampler to evaluate the posterior
        # distribution
        D['sampler'] = 'pymultinest'

        """set up the proper set of variable names for the problem
        of interest"""
        D['pname'] = []
        D['pname_plt'] = []
        for ii in range(D['order']+1):
            D['pname'] += ['theta_' + str(ii)]
            D['pname_plt'] += ['\\theta_%s' % str(ii)]
        D['pname'] += ['sigma']
        D['pname_plt'] += ['\\sigma']

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
        cc.WP(msg, D['wrt_file'])

        # neff (Gelman, 2014.) gives the effective number of samples for
        # each variable of interest. It should be greater than 10
        # for each variable
        neff = cc.effective_n(trace, D['pname'])
        msg = "effective sample size: %s" % neff
        cc.WP(msg, D['wrt_file'])

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

    bounds = ((.25, 2.1), (-.8, 2.2), (0, .6))
    cp.plot_cov(flattrace, D['pname_plt'], param_true=D['param_true'],
                bounds=bounds, figsize=[5.5, 5.5], pltname=D['outname'])

    cp.plot_prediction(flattrace, D['name_list'],
                       D['Tt'], D['At'], D['It'], feval, D,
                       colorL=['k', 'r'],
                       param_true=D['param_true'],
                       ylim=(1.1, 2.2), pltname=D['outname'])

    plt.show()
