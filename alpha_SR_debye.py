import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import pickle
import seaborn as sns
import scipy.stats as ss
import numpy as np
import core_compute as cc
import core_plot as cp
from scipy.integrate import simps, cumtrapz


def deb_Cp(theta, T):
    T = np.array(T)
    T[T < 1e-70] = 1e-70
    # ub: array of upper bounds for integral
    TT = np.array(theta)[..., None] / \
        np.array(T)[None, ...]

    # nx: number of steps in x integration
    nx = 100

    # x: array for variable of integration
    # integration will be performed along
    # last axis of array
    x = np.ones(list(TT.shape)+[nx]) * \
        np.linspace(0, 1, nx)[None, ...]
    x *= x*TT[..., None]

    R = 8.314459848  # J/mol*K
    expx = np.exp(x)

    # if any elements of expx are infinite or equal to 1,
    # replace them with zero. This doesn't change the result
    # of the integration and avoids numerical issues
    expx[expx > 1e100] = 0
    expx[expx - 1 < 1e-100] = 0

    # perform integration over
    # the equispace data points along the last
    # axis of the arrays
    integrand = (x**4)*expx / (expx-1.)**2
    integral = simps(y=integrand, x=x, axis=-1)

    return np.squeeze(9*R*((1/TT)**3)*integral)


def feval_Cp(param, T):

    theta = param[..., 0]
    beta1 = param[..., 1, None]
    beta2 = param[..., 2, None]
    tau = param[..., 3, None]
    gamma = param[..., 4, None]

    # R = 8.314459848  # J/mol*K
    # frac = theta/T
    # expf = np.exp(frac)
    # lowT = 3*R*(frac**2)*(expf/(expf-1)**2)
    lowT = deb_Cp(theta, T)

    ineq1 = tau - gamma > T
    ineq2 = (tau-gamma <= T)*(T <= tau+gamma)
    ineq3 = tau + gamma < T
    term1 = beta1*T
    term2 = beta2*(T - tau + gamma)**2/(4*gamma)
    term3 = beta2*(T - tau)
    bcm = ineq1*term1 + \
        ineq2*(term1 + term2) + \
        ineq3*(term1 + term3)

    A = lowT + bcm

    return A


def feval_Cp_plt(param, T, deb):

    # theta = param[..., 0, None]
    beta1 = param[..., 1, None]
    beta2 = param[..., 2, None]
    tau = param[..., 3, None]
    gamma = param[..., 4, None]

    """Cp for alpha phase"""

    # R = 8.314459848  # J/mol*K
    # frac = theta/T
    # expf = np.exp(frac)
    # lowT = 3*R*(frac**2)*(expf/(expf-1)**2)
    lowT = deb

    ineq1 = tau - gamma > T
    ineq2 = (tau-gamma <= T)*(T <= tau+gamma)
    ineq3 = tau + gamma < T
    term1 = beta1*T
    term2 = beta2*(T - tau + gamma)**2/(4*gamma)
    term3 = beta2*(T - tau)
    bcm = ineq1*term1 + \
        ineq2*(term1 + term2) + \
        ineq3*(term1 + term3)

    A = lowT + bcm

    return A


def feval_H(param, T):

    theta = param[..., 0, None]
    beta1 = param[..., 1, None]
    beta2 = param[..., 2, None]
    tau = param[..., 3, None]
    gamma = param[..., 4, None]

    """compute the enthalpy for the alpha phase"""

    # R = 8.314459848  # J/mol*K
    # lowT = 3*R*theta/(np.exp(theta/T)-1.)

    # add on 298.15K to T so that H_298.15 = 0 is enforced
    T_ = np.array(list(T) + [298.15])
    T = np.atleast_1d(T)
    T_ = np.atleast_1d(T_)

    thetam = np.mean(theta)

    # first create equispaced temps at which to eval Cp
    T_v1 = np.linspace(1e-10, thetam/8, 30)[:-1]
    T_v2 = np.linspace(thetam/8, 3*thetam, 50)[:-1]
    T_v3 = np.linspace(3*thetam, 2100, 20)
    T_v = np.concatenate([T_v1, T_v2, T_v3])

    # evaluate Debye-Cp term at equispaced points
    DebCp_v = deb_Cp(theta, T_v)
    # evaluate Debye-Cp term at actual temps
    DebCp = deb_Cp(theta, T_)

    # array for H-Debye terms
    DebH = np.zeros((theta.size, T_.size))

    # split it up by each temp
    for ii in range(T_.size):
        # identify number of Temps in T_v less than actual
        # temp
        idx = np.sum(T_v < T_[ii])

        T__ = np.zeros((idx+1))

        T__[:idx+1] = T_v[:idx+1]

        DebCp_ = np.zeros((theta.size, idx+1))
        DebCp_[..., :idx+1] = DebCp_v[..., :idx+1]

        # last temp and Cp are for the actual temp
        # of interest
        T__[-1] = T_[ii]
        DebCp_[..., -1] = DebCp[..., ii]

        # perform numerical integration
        DebH_ = np.squeeze(simps(y=DebCp_, x=T__, axis=-1))
        DebH[:, ii] = DebH_

    # we subtract debH at 298.15K from debH at all other temps
    lowT = np.squeeze(DebH[..., :-1]) - np.squeeze(DebH[..., -1])

    ineq1 = tau - gamma > T
    ineq2 = (tau-gamma <= T)*(T <= tau+gamma)
    ineq3 = tau + gamma < T
    term1 = beta1*0.5*T**2
    term2 = beta2*(T - tau + gamma)**3/(12*gamma)
    term3 = beta2*0.5*(T**2 - 2*tau*T + tau**2 - gamma**2 + (4./3.)*gamma**2)
    bcm = ineq1*term1 + \
        ineq2*(term1 + term2) + \
        ineq3*(term1 + term3)

    A = lowT + bcm

    return A


def feval_H_plt(param, T, deb):
    beta1 = param[..., 1, None]
    beta2 = param[..., 2, None]
    tau = param[..., 3, None]
    gamma = param[..., 4, None]

    """compute the enthalpy for the alpha phase"""

    # R = 8.314459848  # J/mol*K
    # lowT = 3*R*theta/(np.exp(theta/T)-1.)

    lowT = cumtrapz(y=deb, x=T, axis=-1, initial=0)

    ineq1 = tau - gamma > T
    ineq2 = (tau-gamma <= T)*(T <= tau+gamma)
    ineq3 = tau + gamma < T
    term1 = beta1*0.5*T**2
    term2 = beta2*(T - tau + gamma)**3/(12*gamma)
    term3 = beta2*0.5*(T**2 - 2*tau*T + tau**2 - gamma**2 + (4./3.)*gamma**2)
    bcm = ineq1*term1 + \
        ineq2*(term1 + term2) + \
        ineq3*(term1 + term3)

    A = np.squeeze(lowT) + bcm

    T298idx = T == 298.15
    A -= A[..., T298idx]

    return A


def likelihood(param, D):
    """
    compute the log likelihood for a set of datapoints given
    a parameterization
    """
    dA_Cp = D['At_Cp']-feval_Cp(param, D['Tt_Cp'])
    dA_H = D['At_H']-feval_H(param, D['Tt_H'])

    # obtain the hyperparameter vectors for Cp and H
    nhyp_H = len(D['name_list_H'])
    hyp_H = param[-nhyp_H:]

    nhyp_Cp = len(D['name_list_Cp'])
    hyp_Cp = param[-(nhyp_H+nhyp_Cp):-nhyp_H]

    if param[0] <= 0 or np.any(hyp_Cp <= 0) or np.any(hyp_H <= 0):
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


def custom(cube, cube_p, D):
    cube_p_gamma = ss.uniform.ppf(cube[4],
                                  loc=D['locV'][4],
                                  scale=cube_p[3])
    return cube_p_gamma


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
        D['name_list_Cp'] = ['Ade1952', 'Aru1972', 'Bur1958',
                             'Cez1974', 'Col1971', 'Fil1971',
                             'Kne1963', 'McC1964','Mil2006S1',
                             'Mil2006S2', 'Pel1971', 'Wol1957']
        D['name_list_H'] = ['Cag2008', 'Fie1961', 'Gol1970',
                            'Haw1963', 'Kat1985']
        D['name_list'] = D['name_list_Cp']+D['name_list_H']
        nds = len(D['name_list'])

        D['phase'] = 'alpha'

        data = get_data(D['name_list_Cp'], D['phase'])
        D['Tt_Cp'], D['At_Cp'] = data[0], data[1]
        D['Et_Cp'], D['It_Cp'] = data[2], data[3]
        data = get_data(D['name_list_H'], D['phase'])
        D['Tt_H'], D['At_H'] = data[0], data[1]
        D['Et_H'], D['It_H'] = data[2], data[3]

        D['likelihood'] = likelihood
		D['custom'] = custom

        # define the prior distributions
        D['npar_model'] = 5
        D['distV'] = 4*['uniform'] + 1*['custom'] + nds*['expon']

        if os.path.exists(D['outname'] + '_prior.csv'):
            print('prior suggestions loaded')
            nxtprior = np.loadtxt(D['outname'] + '_prior.csv')
            D['locV'] = list(nxtprior[0, :D['npar_model']])
            D['scaleV'] = list(nxtprior[1, :D['npar_model']])
        else:
            D['locV'] = [0, -.1, -.1, 0, 0]
            D['scaleV'] = [700, .2, .2, 2000, 1000]

        D['locV'] += nds*[None]
        D['scaleV'] += nds*[None]
        D['cV'] = (D['npar_model']+nds)*[None]
        D['dim'] = len(D['distV'])

        # sampler: select a type of sampler to evaluate the posterior
        # distribution
        D['sampler'] = 'pymultinest'

        """set up the proper set of variable names for the problem
        of interest"""
        D['pname'] = ['theta', 'beta1', 'beta2', 'tau', 'gamma']
        D['pname_plt'] = ['\\theta', '\\beta_1', '\\beta_2',
                          '\\tau', '\\gamma']

        for ii in range(nds):
            D['pname'] += ['alpha_%s' % D['name_list'][ii]]
            D['pname_plt'] += ['\\alpha_{%s}' % D['name_list'][ii]]

        # print(D['pname'])

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
    It_l = [D['It_Cp'], D['It_H'], None, None]
    pltper = [3, 1, 1, 1]
    xlim = [[(1e-10, 3000), (1e-10, 250), (1e-10, 30)],
            [(1e-10, 3000)],
            [(1e-10, 3000)],
            [(1e-10, 3000)]]
    ylim = [[None, (-2, 27), (-.5, 5)],
            [None],
            [None],
            [None]]
    on = D['outname']
    pltname = [[on + 'Cp', on + 'Cp_close', on + 'Cp_vclose'],
               [on + 'H'],
               [on + 'S'],
               [on + 'G']]
    xlabel = [3*[r"$T (K)$"],
              [r"$T (K)$"],
              [r"$T (K)$"],
              [r"$T (K)$"]]
    ylabel = [3*[r"$C_p \left(J {mol}^{-1} K^{-1}\right)$"],
              [r"$H \left(J {mol}^{-1} \right)$"],
              [r"$S \left(J {mol}^{-1} K^{-1}\right)$"],
              [r"$G \left(J {mol}^{-1} \right)$"]]
    legend_loc = [[None, None, 'upper left'],
                  ['upper left'],
                  [None],
                  [None]]

    cp.plot_prediction_all(flattrace, name_list_l,
                           Tt_l, At_l, It_l,
                           pltper,
                           feval_Cp_plt, feval_H_plt,
                           deb_Cp=deb_Cp,
                           xlim=xlim, ylim=ylim, pltname=pltname,
                           xlabel=xlabel, ylabel=ylabel,
                           legend_loc=legend_loc)
