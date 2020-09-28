#!/usr/bin/env python

from __future__ import print_function, division
import sys, os

def main():
    '''\
    DESCRIPTION

    This program will create the numerical differentiation input files required
    to perform an analysis using numerical three-point differentiation.
    This is done by using a frequencies calculation as a source file, and
    creating input files for the plus and minus directions of each normal mode,
    which can then be used to determine whatever normal mode dependent property
    you are interested in.  The property calculated depends on the template file
    used.

    The input files are named using the convention 'mode####.##-[mp].inp',
    where ####.## is the vibrational frequency to two decimal places, and [mp]
    is either m or p, depending on minus or plus direction.

    AUTHORS: Seth M. Morton & Justin Moore
    '''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s 2.4')
    parser.add_argument('-t', '--template', help='The template file whose '
                        'keys are to be copied into the new mode files. '
                        'If not given, the program will look for a file in '
                        "the current folder named 'template.ext', where 'ext'"
                        'is the extension that corresponds to the program the'
                        'template is for.')
    parser.add_argument('--low', help='The low mode to include. The '
                        'default is %(default)s cm-1.', default=float(500),
                        type=float)
    parser.add_argument('--high', help='The high mode to include. The '
                        'default is %(default)s cm-1.', default=float(1800),
                        type=float)
    parser.add_argument('-q', '--qmcharge', help='The QM system total charge',
                        default=None)
    parser.add_argument('-a', '--atombasis', help='The basis set for each '
                        'atom in the system, entered in \'atom basis\' format.',
                        nargs='+', default=None)
    parser.add_argument('--stepsize', help='The normal mode step-size. The default '
                        'is 0.01.', default=0.01, type=float)
    parser.add_argument('freqfile', help='The frequency file to base the '
                        'derivatives on.')
    args = parser.parse_args()

    #  Verify options and create head and tail.
    template, source, low, high, qmcharge, atombasis = initiallize(args, args.freqfile)

    # Create the input files for the various vibrational modes
    create_inputs(template, args, source, low, high, qmcharge, atombasis)


def initiallize(args, freqfile):   
    '''Checks the options and sets up the calculation according to them.'''
    from chem import collect
    from prep import range_check, file_safety_check

    # Collect data from the source file
    source = collect(freqfile)

    # Ensure correctness of source file
    assert 'FREQUENCIES' in source.calctype, (
                                 source.filename+' is not a FREQUENCIES file!')

    # Verify Range
    try:
        low, high = range_check(args.low, args.high)
    except ValueError as v:
        raise ValueError ('Error in --low and --high: '+str(v))

    # Determine a template name and make sure it exists
    if args.template:
        template = args.template
        try:
            file_safety_check(template)
        except IOError:
            raise IOError ('Template file does not exist')
    else:
        try: # ADF
            file_safety_check('template.run')
        except IOError:
            try: # NWChem
                file_safety_check('template.nw')
            except IOError:
                try: # Dalton
                    file_safety_check('template.dal')
                except IOError:
                    raise IOError ('Template file does not exist')
                else:
                    template = 'template.dal'
            else:
                template = 'template.nw'
        else:
            template = 'template.run'

    # Determine the QM charge of the system for Dalton
    if args.qmcharge == None:
        qmcharge = '0.0'
    else:
        qmcharge = args.qmcharge
 
    # Determine if an "atombasis" is being used for a Dalton 
    # calculation (i.e. different basis set for each atom).
    if args.atombasis == None:
        atombasis = None
    elif args.atombasis != None:
        atombasis = {}
        for elem in range(len(args.atombasis)):
            temp = args.atombasis[elem].split()
            # Check if the basis set requested by the user involves an ECP.
            if len(temp) == 3: # Has ECP
                atombasis[temp[0]] = temp[1] + ' ' + temp[2]
            else: # No ECP
                atombasis[temp[0]] = temp[1]

    return template, source, low, high, qmcharge, atombasis


def create_inputs(template, args, source, low, high, qmcharge, atombasis):
    '''Create the Raman input files based on the source file.'''

    # Check to see if this is a calculation of the two-photon transition
    # moments.  Currently this is written for handling Dalton jobs. 
    fh = open(template)
    l = [x.rstrip() for x in fh.readlines()]
    if '.TWO-PHOTON' in l:
        tpa = True
    else:
        tpa = False

    # Stepsize
    sR = args.stepsize # default is 0.01

    # Define the extention and numbering for the input files
    ext = os.path.splitext(template)[1]

    # Keep track of number of files skipped
    skipped = { 'range' : 0, 'negative' : 0 }
    tot = 0

    # Initialize a few variables for loop over modes below
    previous_mode = ''
    degeneracy = 0
    deg_list = ('', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')

    # Loop over each normal mode
    for i in range(source.nmodes):
    
        mode = source.v_frequencies[i]
        
        tot += 1
        # Skip if less than zero or not in specified range
        if mode <  0.0:
            skipped['negative'] += 1
            continue
        if mode < low or mode > high:
            skipped['range'] += 1
            continue
    
        # Convert mode to string, rounding to 2 decimal places
        strmode = '{0:.2f}'.format(round(mode, 2))

        # If there are degenerate modes, account for this with deg_list
        if previous_mode == strmode or previous_mode[:-2] == strmode:
            degeneracy += 1
            strmode = '_'.join([strmode, deg_list[degeneracy]])
        else:
            degeneracy = 0

        # Loop over minus, then plus
        for mp in ('m', 'p'):

            # I DO NOT THINK WE NEED THIS b/c deg_list 
            # If there a degenerate modes, account for this with a b
            #if os.path.exists('mode' + strmode + '-' + mp + ext):
            #    strmode += '_b'

            # Modified to handle two-photon absorption jobs from Dalton.
            # To prevent overwriting other calculations, we store 
            # two-photon absorption jobs with the name "tpa_mode" instead
            # of "mode".
            if tpa:
                fname = 'tpa_mode' + strmode + '-' + mp + ext
            else:
                fname = 'mode' + strmode + '-' + mp + ext
                
            # Open file, then print head
            #with open('mode' + strmode + '-' + mp + ext, 'w') as f:
            with open(fname, 'w') as f:
    
                # Copy the frequeincies object to a new object
                new = source.copy()

                # For each atom, calculate either the plus or minus direction
                # of coordinate for the particular normalized normal mode.
                # Replace the coordinates
                nmode = new.normal_modes[i]
                if mp == 'm':
                    new.coordinates = new.coordinates - nmode * sR
                else:
                    new.coordinates = new.coordinates + nmode * sR
    
                # Copy the template and print to file
                new.copy_template(template=template, file=f, 
                                  charge=qmcharge, basis=atombasis)

        # Save this mode
        previous_mode = strmode

    # Inform user of skipped normal modes.
    if skipped['negative']:
        print(skipped['negative'], 'imaginary normal mode(s) skipped.')
    if skipped['range']:
        print(skipped['range'], 'out of '+str(tot)+' normal mode(s) skipped.')


if __name__ == '__main__':
    try:
        main()
    except (IOError, ValueError) as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        sys.exit(1)
