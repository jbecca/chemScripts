#! /usr/bin/python

from __future__ import division, print_function
import sys, os
from prep import abs_file_path
from chem import collect
from textwrap import dedent


def main():
    """\
    Extract and plot data from an output file.  It will guess what type
    of data you want displayed unless you request a specific property

    See man page for more info.
    """
    from argparse import ArgumentParser, RawDescriptionHelpFormatter, SUPPRESS
    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s 1.1')
    parser.add_argument('-p', '--polarizability', action='store_const',
                        const='pol', dest='mode', help='Plot both the real '
                        'and imaginary parts of the polarizability together.')
    parser.add_argument('--ORD', action='store_const',
                        const='ord', dest='mode', help='Plot both the real '
                        'and imaginary parts of the optical rotation together.')
    parser.add_argument('--cd', action='store_const',
                        const='cd', dest='mode', help='Plot a circular dichroism '
                        'spectrum based on optical rotatory strengths.')
    parser.add_argument('-a', '--absorbance', action='store_const',
                        const='abs', dest='mode', help='Plot an absorbance '
                        'spectrum, either from the imaginary part of the '
                        'polarizability, or from the excitation energies.')
    parser.add_argument('-t', '--tpa', action='store_const',
                        const='tpa', dest='mode', help='Plot a two-photon '
                        'absorbance spectrum using a TPA calculation.')
    parser.add_argument('-th', '--thpa', action='store_const',
                        const='3pa', dest='mode', help='Plot a three-photon '
                        'absorbance spectrum using a 3PA calculation.')
    parser.add_argument('-H', '--hyperraman', action='store_const', dest='mode',
                        const='hyperraman', help='Plot the hyper-Raman spectrum.'  
                        ' Must be from derivatives.  Cannot be used with guess'
                        ' and plot functionality right now.')
    parser.add_argument('-sh', '--2ndhyperraman', action='store_const', dest='mode',
                        const='2ndhyperraman', help='Plot the second hyper-Raman'
                        ' spectrum. Must be from derivatives.  Cannot be used with'
                        ' guess and plot functionality right now.')
    parser.add_argument('-r', '--raman', action='store_const', dest='mode',
                        const='raman', help='Plot the Raman spectrum.  Can '
                        'be from a Raman calculation or from derivatives.')
    parser.add_argument('--IR', action='store_const', dest='mode',
                        const='IR', help='Plot the IR spectrum.  This does not '
                        'work with guess and plot.')
    parser.add_argument('--ROA', action='store_const', dest='mode',
                        const='ROA', help='Plot the ROA spectrum.')
    parser.add_argument('--field', action='store_const', dest='mode',
                        const='field', help='Plot electric fields of a Dim system.')
    parser.add_argument('-D', '--dir', help='If the property to plot requires '
                        'collecting from a directory, this will specify some '
                        'directory that is not the current directory.  The '
                        "default is '.'.", default='.')
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help='Write a .mpl.py file that can be edited to '
                        'customize your plot and run with the python '
                        'interpriter.  The name is based on the input name')
    parser.add_argument('--debug', help='Stops execution for collection '
                        'errors', action='store_true', default=False)
    parser.add_argument('--plane', action='store', default='z',
                        help='Specify which direction to set as 0 when'
                        'plotting the electric field.  Must be x, y, or z.  '
                        'Defaults to z.')
    parser.add_argument('files', help='The files to plot.', nargs='+')
    args = parser.parse_args()

    # Collect the data
    data = []
    for f in args.files:
        try:
            if args.debug:
                data.append(collect(f, raise_err=True))
            else:
                data.append(collect(f))
        except IOError:
            print('Skipping', f, file=sys.stderr)
            continue
            
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    #plt.rc('font', **{'family':'serif', 'serif': ['Times'], 'size': 20})

    # Loop over all the given files and collect
    for d in data:

        # Determine what type of plot to make.
        if args.mode:
            string = option_plot(d, args)
        else:
            string = guess_and_plot(d, args)

        # Execute the returned commands
        execute(string, d, args)


#####################
# END OF MAIN PROGRAM
#####################

def guess_and_plot(d, args):
    '''Guess based on the calculation type what to plot.'''
    if 'DIM' in d.calctype and d.dim_dipoles != None:
        return plot_field(d, args)
    elif 'RAMAN' in d.calctype:
        return plot_raman(d, args)
    elif 'FREQUENCIES' in d.calctype:
        # Since args.dir is always the current directory now, plot_IR 
        # will never happen here...
        if args.dir:
            return plot_raman(d, args)
        else:
            return plot_IR(d, args)
    elif 'POLARIZABILITY' in d.calctype or 'OPTICAL ROTATION' in d.calctype:
        return plot_pol(d, args)
    elif 'TPA' in d.calctype:
        return plot_tpa(d, args)
    elif '3PA' in d.calctype:
        return plot_3pa(d, args)
    elif 'EXCITATIONS' in d.calctype:
        return plot_abs(d, args)
    else:
        sys.exit('File data in {0} is not yet implemented'.format(d.filename))

def option_plot(d, args):
    '''Plot based on a given option.'''
    if args.mode == 'pol':
        return plot_pol(d, args)
    elif args.mode == 'ord':
        return plot_ord(d, args)
    elif args.mode == 'raman':
        return plot_raman(d, args)
    elif args.mode == 'IR':
        return plot_IR(d, args)
    elif args.mode == 'tpa':
        return plot_tpa(d, args)
    elif args.mode == '3pa':
        return plot_tpa(d, args)
    elif args.mode == 'abs':
        return plot_abs(d, args)
    elif args.mode == 'field':
        return plot_field(d, args)
    elif args.mode == 'hyperraman':
        return plot_hyperraman(d, args)
    elif args.mode == '2ndhyperraman':
        return plot_2ndhyperraman(d, args)
    elif args.mode == 'ROA':
        return plot_vroa(d, args)
    elif args.mode == 'cd':
        return plot_cd(d, args)

def execute(string, d, args):
    import matplotlib.pyplot as plt
    '''Execute either the plotting or the printing.'''
    # Either plot for screen or save as text file for editing
    if args.save:
        # Save as text to file.  First define heading
        head = dedent('''\
        from __future__ import division, print_function
        import sys, os
        from chem import collect
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style('whitegrid')
        sns.set_context('talk', font_scale=1.5)

        # Set to render text with LaTeX, edit font properties, define figure
        plt.rc('text.latex', preamble='\usepackage{{sfmath}}')
        plt.rc('text', usetex=True)
        # plt.rc('font', **{{'family':'serif', 'serif': ['Times'], 'size': 20}})
        params = {{'axes.labelsize': 20,
                  'font.size' : 20,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'font.sans-serif' : 'Helvetica'}}
        plt.rcParams.update(params)
        fig = plt.figure()

        # Collect the data
        d = collect('{0}')

        # Plot your data
        '''.format(abs_file_path(d.filename)))
        tail = dedent('''\

        # The below command puts plot on the screen
        plt.show()
        # The below command saves to file (edit extention if you wish)
        #fig.savefig('{0}.{1}', transparent=False, format='{1}')
        '''.format(os.path.splitext(d.filename)[0], 'png'))
        fname = '.'.join([os.path.splitext(d.filename)[0], 'mpl.py'])
        with open(fname, 'w') as f:
            print(head, string, tail, sep='', file=f)
    else:
        # Plot using the returned string and exec
        fig = plt.figure()
        exec string in globals(), locals()
        plt.show()

def plot_abs(d, args):
    '''Return a string capable of plotting the absorbance spectrum.'''
    if 'EXCITATIONS' in d.calctype:
        string = dedent('''\
        from chem.constants import HART2NM, HART2EV, PI
        from numpy import linspace
        from mfunc import sum_lorentzian

        # gamma = half width at half max
        gamma = 0.0037

        # Define the points to plot, then plot
        domain = linspace(0.5*d.excitation_energies[0], d.excitation_energies[-1]*1.5, 2000)
        values = sum_lorentzian(domain, peak=d.excitation_energies,
                                height=d.oscillator_strengths, hwhm=gamma)
        sub = fig.add_subplot(111)
        # Plot in units of molar absorptivity
        sub.plot(HART2NM / domain,  values / ( 0.0348 * HART2EV ), 'b-', lw=2)
        stickscale = 1.0 / ( gamma * PI * 0.0348 * HART2EV )
        # Plot in units of absorbance cross section (Angstrom^2 / molecule)
        #sub.plot(HART2NM / domain,  
        #         values * 1000.0 / ( 26153.21 * 0.0348 * HART2EV ), 
        #         'b-', lw=2)
        #stickscale = 1000.0 / ( 26153.21 * gamma * PI * 0.0348 * HART2EV )
        # Stick spectra
        sub.stem(HART2NM / d.excitation_energies, 
                 d.oscillator_strengths*stickscale, 'k-', 'k ', 'k ')

        # Titles, labels, and limits
        #sub.set_title('Title of Figure')
        #sub.set_xlabel(r'$\lambda$ (nm)')
        sub.set_xlabel(r'$\mathrm{{Wavelength}}$ ($\mathrm{{nm}}$)')
        # In molar absorptivity
        sub.set_ylabel(r'$\epsilon$ $\\Big(\\times 10^{-3} \\frac{\mathrm{{L}}}'
                                       r'{\mathrm{{mol}}\cdot\mathrm{{cm}}}\Big)$')
        # In absorbance cross section
        #sub.set_ylabel(r'$\sigma_{\mathrm{{OPA}}}$ $\\Big(\mathrm{{\AA}}^2/\mathrm{{molecule}}\\Big)$')
        sub.set_xlim(300, 800)
        #sub.set_ylim(low, high)
        ''')
    elif set(['POLARIZABILITY', 'FD']) <= d.calctype:
        string = dedent('''\
        from scipy.interpolate import InterpolatedUnivariateSpline
        from chem.constants import HART2NM
        from numpy import linspace

        # Define the points to plot, smooth points, then plot
        x = d.e_frequencies
        try:
            y = d.isotropic(qm=True, dim=True).imag
        except (ValueError,TypeError,AssertionError):
            try:
                y = d.isotropic(qm=False,dim=True).imag
            except (ValueError,TypeError,AssertionError):
                y = d.isotropic(qm=True, dim=False).imag    
        fitcoeff = InterpolatedUnivariateSpline(x, y)
        domain = linspace(x[0], x[-1], 2000)
        values = fitcoeff(domain)
        sub = fig.add_subplot(111)
        sub.plot(HART2NM / domain, values, 'b-', lw=2)

        # Titles, labels, and limits
        #sub.set_title('Title of Figure')
        sub.set_xlabel(r'$\lambda$ (nm)')
        sub.set_ylabel(r'$\sigma(\omega)$ (a.u.)')
        sub.set_xlim(300, 800)
        #sub.set_ylim(low, high)
        ''')
    else:
        sys.exit('Cannot plot absorbance for file {0}'.format(d.filename))
    return string

def plot_tpa(d, args):
    '''Return a string to plot two-photon absorbance.'''
    if 'TPA' in d.calctype:
        string = dedent('''\
        from chem.constants import HART2NM, HART2EV, PI
        from numpy import linspace
        from mfunc import sum_lorentzian

        # gamma = half width at half max
        gamma = 0.0037

        # Define the points to plot, then plot
        domain = linspace(0.5*d.excitation_energies[0], d.excitation_energies[-1]*1.5, 2000)
        values = sum_lorentzian(domain, peak=d.excitation_energies,
                                height=d.linear_sigma_tpa, hwhm=gamma)
        sub = fig.add_subplot(111)
        # Plot in units of TPA cross section (x10^{-50} cm^4 sec/photon)
        sub.plot(HART2NM / domain,  
                 values, 
                 'b-', lw=2)
        stickscale = 1.0 / ( gamma * PI )
        # Stick spectra
        sub.stem(HART2NM / d.excitation_energies, 
                 d.linear_sigma_tpa*stickscale, 'k-', 'k ', 'k ')

        # Titles, labels, and limits
        #sub.set_title('Title of Figure')
        #sub.set_xlabel(r'$\lambda$ (nm)')
        sub.set_xlabel(r'$\mathrm{{Wavelength}}$ ($\mathrm{{nm}}$)')
        # In two-photon absorbance cross section
        sub.set_ylabel(r'$\sigma_{\mathrm{{TPA}}}$ $\Big(\\times 10^{-50} '
                       r'\\frac{\mathrm{{cm}}^4 \mathrm{{s}}}{\mathrm{{photon}}}\Big)$')
        sub.set_xlim(300, 800)
        ''')
    else:
        sys.exit('Cannot plot TPA for file {0}'.format(d.filename))
    return string

def plot_3pa(d, args):
    '''Return a string to plot three-photon absorbance.'''
    if '3PA' in d.calctype:
        string = dedent('''\
        from chem.constants import HART2NM, HART2EV, PI
        from numpy import linspace
        from mfunc import sum_lorentzian

        # gamma = half width at half max
        gamma = 0.0037

        # Define the points to plot, then plot
        domain = linspace(0.5*d.excitation_energies[0], d.excitation_energies[-1]*1.5, 2000)
        values = sum_lorentzian(domain, peak=d.excitation_energies,
                                height=d.linear_sigma_3pa, hwhm=gamma)
        sub = fig.add_subplot(111)
        # Plot in units of 3PA cross section (x10^{-82} cm^6 sec^2/photon)
        sub.plot(HART2NM / domain,  
                 values, 
                 'b-', lw=2)
        stickscale = 1.0 / ( gamma * PI )
        # Stick spectra
        sub.stem(HART2NM / d.excitation_energies, 
                 d.linear_sigma_3pa*stickscale, 'k-', 'k ', 'k ')

        # Titles, labels, and limits
        #sub.set_title('Title of Figure')
        sub.set_xlabel(r'$\mathrm{{Wavelength}}$ ($\mathrm{{nm}}$)')
        # In three-photon absorbance cross section
        #sub.set_ylabel(r'$\sigma_{3PA}$ $\Big(\\times 10^{-82} '
        #               r'\\frac{\mathrm{{cm}}^6 \mathrm{{s}}^2}{\mathrm{{photon}}}\Big)$')
        sub.set_ylabel(r'$\sigma_{\mathrm{{3PA}}}$ $\Big(\\times 10^{-82} '
                       r'\\frac{\mathrm{{cm}}^6 \mathrm{{s}}^2}{\mathrm{{photon}}}\Big)$')
        sub.set_xlim(300, 800)
        ''')
    else:
        sys.exit('Cannot plot 3PA for file {0}'.format(d.filename))
    return string

def plot_pol(d, args):
    '''Return a string to plot polarizability.'''
    if set(['POLARIZABILITY', 'FD']) <= d.calctype or set(['OPTICAL ROTATION', 'FD']) <= d.calctype:
        string = dedent('''\
        from scipy.interpolate import InterpolatedUnivariateSpline
        from chem.constants import HART2EV
        from numpy import linspace, array
        import chem
        import numpy as np

        # Define the points to plot, smooth points, then plot
        rvalues = ivalues = None
        if 'FREQRANGE' in d.subkey:
            # The case that "freqrange" is in the input file and
            # there are multiple results in one single output file 
            x = d.e_frequencies
            try:
                y = d.isotropic()
            except ValueError:
                y = d.isotropic(dim=True)
        else:
            # The case that there is only one result in each output and
            # collect results from all the outputs in one directory.
            # Grab all the files in the current directory
            files = os.listdir('.') 
            #Sort the files
            files.sort() 
            #Make a list for frequency, "x"; polarizaiblity, "y"
            x = [] 
            y = [] 
            for file in files:
                # Grab the files we are interested in
                if file[-4:] == '.out': 
                   # Collect all the info. in the files
                   f = chem.collect(file) 
                   # Get freq as x-axis, average pol as y-axis
                   x.append(f.e_frequencies[0]) 
                   y.append(f.isotropic()[0]) 
            # Make an array for each of them
            x = np.array(x)
            y = np.array(y)

        domain = linspace(x[0], x[-1], 2000)
        # Smooth and plot
        sub = fig.add_subplot(111)
        # Real values.  Comment this out to do only imaginary
        fitcoeff = InterpolatedUnivariateSpline(x, y.real)
        rvalues = fitcoeff(domain)
        # Imag values.  Comment this out to do only real
        fitcoeff = InterpolatedUnivariateSpline(x, y.imag)
        ivalues = fitcoeff(domain)
        # Both real and imaginary is plotted differently from only real or imag
        if rvalues is not None and ivalues is not None:
            l1, = sub.plot(domain*HART2EV, rvalues, 'g-', lw=2,
                          label=r'$\\alpha^R(\omega)$')
            sub2 = sub.twinx()
            l2, = sub2.plot(domain*HART2EV, ivalues, 'b-', lw=2,
                           label=r'$\\alpha^I(\omega)$')

            # Titles, labels, and limits
            sub.legend([l1, l2], [l1.get_label(), l2.get_label()],
                       fancybox=True, shadow=True, prop={'size':16},
                       loc='best')
            sub.set_ylabel(r'$\\alpha^R(\omega)$ (a.u.)')
            sub2.set_ylabel(r'$\\alpha^I(\omega)$ (a.u.)')
            #sub.set_ylim(low, high)
            #sub2.set_ylim(low, high)
        else:
            if rvalues is not None:
                sub.plot(domain*HART2EV, rvalues, 'g-', lw=2)

                # Titles, labels, and limits
                sub.set_ylabel(r'$\\alpha^R(\omega)$ (a.u.)')
            elif ivalues is not None:
                sub.plot(domain*HART2EV, ivalues, 'b-', lw=2)

                # Titles, labels, and limits
                sub.set_ylabel(r'$\\alpha^I(\omega)$ (a.u.)')
            #sub.set_ylim(low, high)
        #sub.set_title('Title of Figure')
        sub.set_xlim(domain[0]*HART2EV, domain[-1]*HART2EV)
        sub.set_xlabel(r'Frequency (eV)')
        ''')
    else:
        sys.exit('Cannot plot polarizability for file {0}'.format(d.filename))
    return string

def plot_ord(d, args):
    '''Return a string to plot ORD and CD from G-tensor of OptRot tensor.'''
    if (d.ord != None or d.gtensor != None) and d.e_frequencies != None:
        string = dedent('''\
        from scipy.interpolate import InterpolatedUnivariateSpline
        from chem.constants import HART2WAVENUM, HART2NM
        from numpy import linspace, array, einsum, argsort

        # Define the points to plot, smooth points, then plot
        rvalues = ivalues = None
        x = d.e_frequencies
        if (d.ord!=None) and (len(d.ord)==len(d.e_frequencies)):
            beta = einsum('iaa->i', d.ord) / 3.
        else:
            beta = einsum('iaa,i->i', d.gtensor, 1/x) * (-1./3.)

        # Sort values in order of increasing energy
        sort = argsort(x)
        x = x[sort]
        beta = beta[sort]

        y1 = 1.343e-6 * (HART2WAVENUM(x))**2 * beta.real
        y2 = 1.343e-6 * (HART2WAVENUM(x))**2 * beta.imag / 3298.8

        domain = linspace(x[0], x[-1], 2000)

        # Smooth and plot
        sub = fig.add_subplot(111)
        sub.axhline(linewidth=1, color='k')
        # Real values.  Comment this out to do only imaginary
        fitcoeff = InterpolatedUnivariateSpline(x, y1)
        rvalues = fitcoeff(domain)
        # Imag values.  Comment this out to do only real
        fitcoeff = InterpolatedUnivariateSpline(x, y2)
        ivalues = fitcoeff(domain)
        # Both real and imaginary is plotted differently from only real or imag
        if rvalues is not None and ivalues is not None:
            l1, = sub.plot(HART2NM(domain), rvalues, 'g-', lw=2,
                          label=r'$[\\phi]$')
            l3, = sub.plot(HART2NM(x), y1, 'go')
            sub2 = sub.twinx()
            l2, = sub2.plot(HART2NM(domain), ivalues, 'b-', lw=2,
                           label=r'$\\Delta\\epsilon$')
            l4, = sub2.plot(HART2NM(x), y2, 'bo')

            # Titles, labels, and limits
            sub.legend([l1, l2], [l1.get_label(), l2.get_label()],
                       fancybox=True, shadow=True, prop={'size':16},
                       loc='best')
            sub.set_ylabel(r'$[\\phi]_{\\lambda}$  $(deg.\\ cm^2/dmol)$')
            sub2.set_ylabel(r'$\\Delta\\epsilon$  $(l\\ mol^{-1} cm^{-1})$')

            y1max = rvalues.max() * 1.1
            y1min = rvalues.min() * 1.1
            r1 = abs(y1min) / y1max

            y2min = ivalues.min() * 1.1
            y2max = ivalues.max() * 1.1
            r2 = abs(y2min) / y2max

            ruse = max(r1, r2)

            y1min = - ruse * y1max
            y2min = - ruse * y2max

            sub.set_ylim(y1min, y1max)
            sub2.set_ylim(y2min, y2max)

        else:
            if rvalues is not None:
                sub.plot(HART2NM(domain), rvalues, 'g-', lw=2)
                sub.plot(HART2NM(x), y1, 'go')

                # Titles, labels, and limits
                sub.set_ylabel(r'$[\\phi(\\lambda)]$  $(deg.\\ cm^2/dmol)$')
            elif ivalues is not None:
                sub.plot(HART2NM(domain), ivalues, 'b-', lw=2)
                sub.plot(HART2NM(x), y2, 'bo')

                # Titles, labels, and limits
                sub.set_ylabel(r'$\\Delta\\epsilon$  $(l\\ mol^{-1} cm^{-1})$')
            #sub.set_ylim(low, high)
        #sub.set_title('Title of Figure')
        sub.set_xlim(HART2NM(domain).min(), HART2NM(domain).max())
        sub.set_xlabel(r'$\\lambda$ (nm)')
        ''')
    else:
        sys.exit('Cannot plot ORD/CD for file {0}'.format(d.filename))
    return string

def plot_raman(d, args):
    if args.dir and args.dir != '.':
       temp = "dir='{0}'".format(args.dir)
    else:
       temp = ''
    if 'RAMAN' in d.calctype or 'FREQUENCIES' in d.calctype:
        string = dedent('''\
        from mfunc import sum_lorentzian
        from chem.constants import PI
        from numpy import linspace
        if 'RAMAN' in d.calctype:
            pass
        else:
            d.collect_raman_derivatives({0})
        # Set the full width at half max and the factor to scale the peaks by
        # Another valid choice is fwhm = 10
        fwhm = 20
        scaleexp = 32 # as in 1E{{scale}}
        scale = 10**scaleexp
        raman_intensity = d.cross_section()
        # Define the points to plot, then plot
        domain = linspace(0, d.v_frequencies[-1]*1.5, num=2000)
        y = sum_lorentzian(domain, d.v_frequencies, raman_intensity, fwhm=fwhm)
        sub = fig.add_subplot(111)
        sub.plot(domain, y*scale, 'r')
        # Comment the below three lines to not plot the sticks
        # fwhm is converted to hwhm. pi is for normalization
        stickscale = scale / ( ( fwhm / 2 ) * PI )
        sub.stem(d.v_frequencies, raman_intensity*stickscale, 'k-', 'k ', 'k ')

        # Title, lables, and limits
        #sub.set_title('Title')
        #lab = r'Differential Cross-Section $\\frac{{d\sigma}}{{d\Omega}}$ '
        #lab += r'($10^{{-'+str(scaleexp)+r'}}\\frac{{cm^2}}{{sr}}$)'
        lab = r'$\mathrm{{Differential Cross-Section}}$ $\\frac{{d\sigma}}{{d\Omega}}$ '
        lab += r'($\\times 10^{{-'+str(scaleexp)+r'}}\\frac{{\mathrm{{cm}}^2}}{{\mathrm{{sr}}}}$)'
        sub.set_ylabel(lab)
        #sub.set_xlabel(r'Wavenumber (cm$^{{-1}}$)')
        sub.set_xlabel(r'$\mathrm{{Wavenumber}}$ ($\mathrm{{cm}}^{{-1}}$)')
        sub.set_xlim(0, d.v_frequencies[-1]*1.5)
        '''.format(temp))
    else:
        sys.exit('Cannot plot Raman for file {0}'.format(d.filename))
    return string

def plot_vroa(d, args):
    if 'VROA' in d.calctype or 'FREQUENCIES' in d.calctype:
        string = dedent('''\
        from mfunc import sum_lorentzian
        from chem.constants import PI
        from numpy import linspace
        from numpy import asarray
        if 'VROA' in d.calctype:
            pass
        else:
            d.collect_roa_derivatives()
            d.calc_roa_intensities()
        # Set the full width at half max and the factor to scale the peaks by
        # Another valid choice is fwhm = 10
        fwhm = 20
        # Define the points to plot, then plot
        domain = linspace(0, d.v_frequencies[-1]*1.5, num=2000)
        intensities = asarray(d.vroa_intensities['180deg'][:])
        y = sum_lorentzian(domain, d.v_frequencies, intensities, fwhm=fwhm)
        sub = fig.add_subplot(111)
        sub.plot(domain, y, 'r')

        # Title, lables, and limits
        #sub.set_title('Title')
        #lab = r'Intensity  '
        #lab += r'($10^{{3}} \\frac{{\\AA^2}}{{amu}}$)'
        lab = r'$\mathrm{{Intensity}}$  '
        lab += r'($\\times 10^{{3}} \\frac{{\mathrm{{\AA}}^2}}{{\mathrm{{amu}}}}$)'
        sub.set_ylabel(lab)
        #sub.set_xlabel(r'Wavenumber (cm$^{{-1}}$)')
        sub.set_xlabel(r'$\mathrm{{Wavenumber}}$ ($\mathrm{{cm}}^{{-1}}$)')
        sub.set_xlim(0, d.v_frequencies[-1]*1.5)
        ''')
    else:
        sys.exit('Cannot plot VROA for file {0}'.format(d.filename))
    return string

def plot_cd(d, args):
    '''Return a string to plot circular dichroism based on optical rotatory
       strength.'''
    if 'CD SPECTRUM' in d.calctype:
        string = dedent('''\
        from chem.constants import HART2NM, HART2EV, PI
        from numpy import linspace
        from mfunc import sum_lorentzian

        # gamma = half width at half max
        gamma = 0.0037

        # Define the points to plot, then plot
        domain = linspace(0.5*d.excitation_energies[0], d.excitation_energies[-1]*1.5, 2000)
        values = sum_lorentzian(domain, peak=d.excitation_energies,
                                height=d.opt_rot_strengths, hwhm=gamma)
        sub = fig.add_subplot(111)
        # Plot in units of optical rotatory strength
        sub.axhline(linewidth=1, color='k')
        sub.plot(HART2NM / domain, values,  'b-', lw=2)
        stickscale = 1.0 / ( gamma * PI )
        # Stick spectra
        sub.stem(HART2NM / d.excitation_energies, 
                 d.opt_rot_strengths*stickscale, 'k-', 'k ', 'k ')

        # Titles, labels, and limits
        #sub.set_title('Title of Figure')
        #sub.set_xlabel(r'$\lambda$ (nm)')
        sub.set_xlabel(r'$\mathrm{{Wavelength}}$ ($\mathrm{{nm}}$)')
        # In optical rotatory strength
        #sub.set_ylabel(r'Optical Rotatory Strength $\Big(\\times 10^{-40} '
        #               r'esu^2 cm^2\Big)$')
        sub.set_ylabel(r'$\mathrm{{Optical Rotatory Strength}}$ $\Big(\\times 10^{-40} '
                       r'\mathrm{{esu}}^2 \mathrm{{cm}}^2\Big)$')
        sub.set_xlim(300, 800)
        ''')
    else:
        sys.exit('Cannot plot circular dichroism for file {0}'.format(d.filename))
    return string

def plot_hyperraman(d, args):
    if args.dir and args.dir != '.':
        temp = "dir='{0}', hpol=True".format(args.dir)
    else:
        temp = 'hpol=True'
    if 'FREQUENCIES' in d.calctype:
        string = dedent('''\
        from mfunc import sum_lorentzian
        from chem.constants import PI
        from numpy import linspace

        d.collect_raman_derivatives({0})
        # Set the full width at half max and the factor to scale the peaks by
        # Another valid choice is fwhm = 10
        fwhm = 20
        scaleexp = 64 # as in 1E{{scale}}
        scale = 10**scaleexp
        hyperraman_intensity = d.hyperraman_cross_section()
        # Define the points to plot, then plot
        domain = linspace(0, d.v_frequencies[-1]*1.5, num=2000)
        y = sum_lorentzian(domain, d.v_frequencies, hyperraman_intensity, fwhm=fwhm)
        sub = fig.add_subplot(111)
        sub.plot(domain, y*scale, 'r')
        # Comment the below three lines to not plot the sticks
        # fwhm is converted to hwhm. pi is for normalization
        stickscale = scale / ( ( fwhm / 2 ) * PI )
        sub.stem(d.v_frequencies, hyperraman_intensity*stickscale, 'k-', 'k ', 'k ')

        # Title, lables, and limits
        #sub.set_title('Title')
        #lab = r'$\\frac{{d\sigma^{{HRS}}}}{{d\Omega}}$ '
        #lab += r'($10^{{-'+str(scaleexp)+r'}}\\frac{{cm^4\\, s}}{{photon\, sr}}$)'
        lab = r'$\\frac{{d\sigma^{{\mathrm{{HRS}}}}}}{{d\Omega}}$ '
        lab += r'($\\times 10^{{-'+str(scaleexp)+r'}}\\frac{{\mathrm{{cm}}^4\\, \mathrm{{s}}}}{{\mathrm{{photon}}\, \mathrm{{sr}}}}$)'
        sub.set_ylabel(lab)
        #sub.set_xlabel(r'Wavenumber (cm$^{{-1}}$)')
        sub.set_xlabel(r'$\mathrm{{Wavenumber}}$ ($\mathrm{{cm}}^{{-1}}$)')
        sub.set_xlim(0, d.v_frequencies[-1]*1.5)
        '''.format(temp))
    else:
        sys.exit('Cannot plot hyper-Raman for file {0}'.format(d.filename))
    return string

def plot_2ndhyperraman(d, args):
    if args.dir and args.dir != '.':
        temp = "dir='{0}', shpol=True".format(args.dir)
    else:
        temp = 'shpol=True'
    if 'FREQUENCIES' in d.calctype:
        string = dedent('''\
        from mfunc import sum_lorentzian
        from chem.constants import PI
        from numpy import linspace

        d.collect_raman_derivatives({0})
        # Set the full width at half max and the factor to scale the peaks by
        # Another valid choice is fwhm = 10
        fwhm = 20
        scaleexp = 96 # as in 1E{{scale}}
        scale = 10**scaleexp
        secondhyperraman_intensity = d.secondhyperraman_cross_section() 
        # Define the points to plot, then plot
        domain = linspace(0, d.v_frequencies[-1]*1.5, num=2000)
        y = sum_lorentzian(domain, d.v_frequencies, secondhyperraman_intensity, fwhm=fwhm)
        sub = fig.add_subplot(111)
        sub.plot(domain, y*scale, 'r')
        # Comment the below three lines to not plot the sticks
        # fwhm is converted to hwhm. pi is for normalization
        stickscale = scale / ( ( fwhm / 2 ) * PI )
        sub.stem(d.v_frequencies, secondhyperraman_intensity*stickscale, 'k-', 'k ', 'k ')

        # Title, lables, and limits
        #sub.set_title('Title')
        #lab = r'$\\frac{{d\sigma^{{2HRS}}}}{{d\Omega}}$ '
        #lab += r'($10^{{-'+str(scaleexp)+r'}}\\frac{{cm^6\\, s^2}}{{photon^2\, sr}}$)'
        lab = r'$\\frac{{d\sigma^{{\mathrm{{2HRS}}}}}}{{d\Omega}}$ '
        lab += r'($\\times 10^{{-'+str(scaleexp)+r'}}\\frac{{\mathrm{{cm}}^6\\ \mathrm{{s^2}}}}{{\mathrm{{photon^2}}\, \mathrm{{sr}}}}$)'
        sub.set_ylabel(lab)
        #sub.set_xlabel(r'Wavenumber (cm$^{{-1}}$)')
        sub.set_xlabel(r'$\mathrm{{Wavenumber}}$ ($\mathrm{{cm}}^{{-1}}$)')
        sub.set_xlim(0, d.v_frequencies[-1]*1.5)
        '''.format(temp))
    else:
        sys.exit('Cannot plot second hyper-Raman for file {0}'.format(d.filename))
    return string

def plot_IR(d, args):
    if 'FREQUENCIES' in d.calctype:
        string = dedent('''\
        from mfunc import sum_lorentzian
        from chem.constants import PI
        from numpy import linspace

        # Set the full width at half max and the factor to scale the peaks by
        # Another valid choice is fwhm = 10
        fwhm = 20
        # Define the points to plot, then plot
        domain = linspace(0, d.v_frequencies[-1]*1.5, num=2000)
        y = sum_lorentzian(domain, d.v_frequencies, d.IR, fwhm=fwhm)
        sub = fig.add_subplot(111)
        sub.plot(domain, y, 'r')
        # Comment the below three lines to not plot the sticks
        # fwhm is converted to hwhm. pi is for normalization
        stickscale = 1 / ( ( fwhm / 2 ) * PI )
        sub.stem(d.v_frequencies, d.IR*stickscale, 'k-', 'k ', 'k ')

        # Title, lables, and limits
        #sub.set_title('Title')
        #lab = r'IR intensity '
        #lab += r'($\\frac{{km}}{{mol}}$)'
        lab = r'$\mathrm{{IR intensity}}$ '
        lab += r'($\\frac{{\mathrm{{km}}}}{{\mathrm{{mol}}}}$)'
        sub.set_ylabel(lab)
        #sub.set_xlabel(r'Wavenumber (cm$^{{-1}}$)')
        sub.set_xlabel(r'$\mathrm{{Wavenumber}}$ ($\mathrm{{cm}}^{{-1}}$)')
        sub.set_xlim(0, d.v_frequencies[-1]*1.5)
        ''')
    else:
        sys.exit('Cannot plot IR for file {0}'.format(d.filename))
    return string

def plot_field(d, args):
    center = d.find_center(qm=False, dim=True)
    maxdist = d.maxdist(qm=False, dim=True)
    if args.plane == 'x':
        a = 0
        b = (center[1]-maxdist, center[1]+maxdist, 2*maxdist/200)
        c = (center[2]-maxdist, center[2]+maxdist, 2*maxdist/200)
    elif args.plane == 'y':
        a = (center[0]-maxdist, center[0]+maxdist, 2*maxdist/200)
        b = 0
        c = (center[2]-maxdist, center[2]+maxdist, 2*maxdist/200)
    elif args.plane == 'z':
        a = (center[0]-maxdist, center[0]+maxdist, 2*maxdist/200)
        b = (center[1]-maxdist, center[1]+maxdist, 2*maxdist/200)
        c = 0
    else:
        sys.exit('Invalid plane for file {0}'.format(d.filename))
    area = '{0}, {1}, {2}'.format(a, b, c)
    if 'FD' in d.calctype:
        type = 'FD scattered'
    else:
        type = 'static scattered'
    string = dedent('''\
    from numpy import array
    from chem.drawing import drawField
    efield = array([])
    atomfield = []
    distance = []
    for i in [0, 1, 2]:
        A, B, tmp = drawField(d, {0}, type='{1}', freq=0,
                              dir=i, scale=1.2, test=True)
        if i == 0:
            efield = tmp
        else:
            efield = efield + tmp
    efield = efield/3.0
    from matplotlib import ticker, cm 
    sub = fig.add_subplot(111)
    pallete = cm.jet 
    cont = sub.contourf(A, B, efield,
                        locator=ticker.MaxNLocator(500),
                        cmap=pallete)
    CB = fig.colorbar(cont)
    CB.set_label('Electric Field Magnitude^4 (Atomic Units)')
    '''.format(area, type))
    
    return string    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
