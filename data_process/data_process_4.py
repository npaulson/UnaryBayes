import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d


if __name__ == '__main__':

    sns.set(color_codes=True)

    """create linear interpolants for the temperature standard conversion
    scales"""
    raw = np.loadtxt('T27_T90.csv', delimiter=',', skiprows=0)
    del_T90_T27 = interp1d(raw[:, 0], raw[:, 1],
                           kind='linear', bounds_error=False,
                           fill_value='extrapolate')

    raw = np.loadtxt('T48_T90.csv', delimiter=',', skiprows=0)
    del_T90_T48 = interp1d(raw[:, 0], raw[:, 1],
                           kind='linear', bounds_error=False,
                           fill_value='extrapolate')

    raw = np.loadtxt('T68_T90.csv', delimiter=',', skiprows=0)
    del_T90_T68 = interp1d(raw[:, 0], raw[:, 1],
                           kind='linear', bounds_error=False,
                           fill_value='extrapolate')

    """create a quadratic spline interpolant for the Cp, H data from
    Brown2005"""
    Zr_alpha = np.loadtxt('Zr_alpha_arb2013.csv',
                          delimiter=',', skiprows=1)
    f_Cp_alpha = interp1d(Zr_alpha[:, 0], Zr_alpha[:, 1], kind='quadratic',
                          fill_value='extrapolate')

    f_H_alpha = interp1d(Zr_alpha[:, 0], Zr_alpha[:, 2],
                         kind='quadratic', bounds_error=False,
                         fill_value='extrapolate')

    Zr_beta = np.loadtxt('Zr_beta_arb2013.csv',
                         delimiter=',', skiprows=1)
    f_Cp_beta = interp1d(Zr_beta[:, 0], Zr_beta[:, 1], kind='quadratic',
                         fill_value='extrapolate')

    f_H_beta = interp1d(Zr_beta[:, 0], Zr_beta[:, 2],
                        kind='quadratic', bounds_error=False,
                        fill_value='extrapolate')

    Zr_liquid = np.loadtxt('Zr_liquid_arb2013.csv',
                           delimiter=',', skiprows=1)
    f_Cp_liquid = interp1d(Zr_liquid[:, 0], Zr_liquid[:, 1], kind='quadratic',
                           fill_value='extrapolate')

    f_H_liquid = interp1d(Zr_liquid[:, 0], Zr_liquid[:, 2],
                          kind='quadratic', bounds_error=False,
                          fill_value='extrapolate')

    x_a = np.linspace(0, 2056, 300)
    x_b = np.linspace(1850, 2450, 300)
    x_l = np.linspace(2260, 5000, 300)

    plt.figure()
    plt.plot(x_a, f_Cp_alpha(x_a), 'r-', label='alpha, Hf temps')
    plt.plot(x_b, f_Cp_beta(x_b), 'g-', label='beta, Hf temps')
    plt.plot(x_l, f_Cp_liquid(x_l), 'b-', label='liquid, Hf temps')
    plt.plot(Zr_alpha[:, 0], Zr_alpha[:, 1], 'ro', label='alpha')
    plt.plot(Zr_beta[:, 0], Zr_beta[:, 1], 'go', label='beta')
    plt.plot(Zr_liquid[:, 0], Zr_liquid[:, 1], 'bo', label='liquid')
    plt.xlabel(r"$T (K)$")
    plt.ylabel(r"$C_p \left(J {mol}^{-1} K^{-1}\right)$")
    plt.legend()

    plt.figure()
    plt.plot(x_a, f_H_alpha(x_a), 'r-', label=', Hf temps')
    plt.plot(x_b, f_H_beta(x_b), 'g-', label=', Hf temps')
    plt.plot(x_l, f_H_liquid(x_l), 'b-', label=', Hf temps')
    plt.plot(Zr_alpha[:, 0], Zr_alpha[:, 2], 'ro', label='alpha')
    plt.plot(Zr_beta[:, 0], Zr_beta[:, 2], 'go', label='beta')
    plt.plot(Zr_liquid[:, 0], Zr_liquid[:, 2], 'bo', label='liquid')
    plt.xlabel(r"$T (K)$")
    plt.ylabel(r"$H \left(J {mol}^{-1} \right)$")
    plt.legend()

    """load up the Excel workbook"""
    wbook = pd.ExcelFile('Hafnium_v6.xlsx')
    names = wbook.sheet_names
    print(names)

    for name in names:

        dat = pd.read_excel(wbook, name)

        authors = dat.loc[0, 'Authors']
        year = np.int32(dat.loc[0, 'Year'])
        typ = dat.loc[0, 'type']
        x_Zr = dat.loc[0, '% Zr Content']/100.
        ZrCor = dat.loc[0, 'Zr corrected']
        errtyp = dat.loc[0, 'error type provided']

        print('\n' + authors + ' ' + str(year))

        Phase = np.array(dat.iloc[:, 0])  # phase names
        Torig = np.array(dat.iloc[:, 3])  # orig. temp. (K)
        Aorig = np.array(dat.iloc[:, 4])  # orig. response var.
        Eorig = np.array(dat.iloc[:, 5])  # orig. err. (with orig. def.)

        """correct temperatures due to changing temperature
        scales.
        See: [include ref]"""
        if year <= 1948:
            T = del_T90_T27(Torig) + Torig
            print('27-90')
        elif year <= 1968:
            T = del_T90_T48(Torig) + Torig
            print('48-90')
        elif year <= 1990:
            T = del_T90_T68(Torig) + Torig
            print('68-90')
        else:
            T = Torig
            print('no temperature scale conversion')

        pct_dT = 100*np.abs(T-Torig)/T
        print('max temp difference: ' + str(pct_dT.max()))

        """if not already performed, correct Cp or H for
        Zr content
        See: [include ref]"""
        A_ = np.zeros(Aorig.shape)
        aI = Phase == 'alpha'
        bI = Phase == 'beta'
        lI = Phase == 'liquid'

        if ZrCor == 'no' and typ == 'Cp':
            A_[aI] = f_Cp_alpha(Torig[aI])
            A_[bI] = f_Cp_beta(Torig[bI])
            A_[lI] = f_Cp_liquid(Torig[lI])
            A = (Aorig - x_Zr*A_)/(1-x_Zr)
            print('Zr correction for Cp')
        elif ZrCor == 'no' and typ == 'H':
            A_[aI] = f_H_alpha(Torig[aI])
            A_[bI] = f_H_beta(Torig[bI])
            A_[lI] = f_H_liquid(Torig[lI])
            A = (Aorig - x_Zr*A_)/(1-x_Zr)
            print('Zr correction for H')
        else:
            A = Aorig
            print('no Zr correction')

        pct_dA = 100*np.abs(A-Aorig)/A
        print('max property difference: ' + str(pct_dA.max()))

        """compute the errors in the response variable"""
        if errtyp in ['absolute', 'unknown', 'none']:
            # in this case the error is assumed to be an absolute
            # percentage bounds on the error. We then assume that
            # the true value is uniformly distributed between the
            # bounds. We convert this to a 1-sigma standard error
            # according to the GUM standard eq. (7)
            E = np.sqrt(((0.01*Eorig*A)**2)/3)
        elif errtyp == '2-sigma':
            E = 0.5*0.01*Eorig*A
        elif errtyp == '1-sigma':
            E = 0.01*Eorig*A
        else:
            print('invalid error type specified')
            sys.exit()

        plt.figure(figsize=[7, 4.5])

        plt.plot(Torig, Aorig, 'ks', alpha=.8, label='uncorrected')
        plt.plot(T, Aorig, 'ro', alpha=.8, label='correct T')
        plt.plot(Torig, A, 'bd', alpha=.8, label='correct Zr')
        plt.errorbar(T, A, E, color='g', marker='v', alpha=.8,
                     linestyle='', label='correct both')
        plt.xlabel(r"$T (K)$", fontsize='large')

        if typ == 'Cp':
            plt.ylabel(r"$C_p \left(J {mol}^{-1} K^{-1}\right)$",
                       fontsize='large')
        elif typ == 'H':
            plt.ylabel(r"$H \left(J {mol}^{-1} \right)$",
                       fontsize='large')

        plt.tick_params(axis='both', labelsize='large')

        plt.title(authors + ' ' + str(year))

        plt.legend(shadow=False,
                   fontsize=12, fancybox=False)

        plt.tight_layout()

        alldata = np.concatenate([[T], [A], [E], [Phase]]).T
        print(alldata.shape)
        if typ == 'Cp':
            header = 'T (K),Cp (J/mol*K)'
        elif typ == 'H':
            header = 'T (K),H (J/mol)'
        np.savetxt(name+'.csv', alldata,
                   fmt='%.18e %.18e %.18e %s',
                   delimiter=' ', header=header)

    plt.show()
