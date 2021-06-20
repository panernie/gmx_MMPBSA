"""
This module provides a means for users to take advantage of gmx_MMPBSA's parsing
ability. It exposes the free energy data (optionally to numpy arrays) so that
users can write a simple script to carry out custom data analyses, leveraging
the full power of Python's extensions, if they want (e.g., numpy, scipy, etc.)
"""

# ##############################################################################
#                           GPLv3 LICENSE INFO                                 #
#                                                                              #
#  Copyright (C) 2020  Mario S. Valdes-Tresanco and Mario E. Valdes-Tresanco   #
#  Copyright (C) 2014  Jason Swails, Bill Miller III, and Dwight McGee         #
#                                                                              #
#   Project: https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA                  #
#                                                                              #
#   This program is free software; you can redistribute it and/or modify it    #
#  under the terms of the GNU General Public License version 3 as published    #
#  by the Free Software Foundation.                                            #
#                                                                              #
#  This program is distributed in the hope that it will be useful, but         #
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY  #
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License    #
#  for more details.                                                           #
# ##############################################################################

from copy import deepcopy
from GMXMMPBSA import infofile, main, amber_outputs
from GMXMMPBSA.exceptions import SetupError, NoFileExists
from GMXMMPBSA.fake_mpi import MPI

import pandas as pd
from pathlib import Path
import parmed
import os
import numpy as np
from types import SimpleNamespace

__all__ = ['load_gmxmmpbsa_info']

make_array = lambda x: np.fromiter(x, float)
make_array_len = lambda x: np.zeros(x, float)

class mmpbsa_data(dict):
    """ Main class that holds all of the Free Energy data """
    def __init__(self, app):
        super(mmpbsa_data, self).__init__()
        """ Load data from an info object """
        if not isinstance(app, main.MMPBSA_App):
            raise TypeError('mmpbsa_data can only take an MMPBSA_App!')
        # Loop through all of the data
        if not hasattr(app, 'calc_types'):
            raise SetupError('Output files have not yet been parsed!')
        # Now load the data into the dict
        self.mutants = {}
        has_mutant = False
        # See if we are doing stability
        self.stability = app.stability
        # Now load the data
        for key in app.calc_types:
            if key == 'qh':
                continue
            self[key] = {}
            if key == 'ie':
                if not self.stability:
                    self[key] = {'data': app.calc_types[key].data['data'], 'value': app.calc_types[key].data[
                        'iedata'].avg(), 'ie_startframe': app.calc_types[key].data['frames'][-app.calc_types[
                        key].data['ieframes']], 'ieframes': app.calc_types[key].data['ieframes'], 'ie_segment':
                        app.INPUT['ie_segment']}
                continue
            self[key]['complex'] = {dkey: make_array(app.calc_types[key]['complex'].data[dkey])
                        for dkey in app.calc_types[key]['complex'].data}
            if not self.stability:
                self[key]['receptor'] = { dkey: make_array(app.calc_types[key]['receptor'].data[dkey])
                            for dkey in app.calc_types[key]['receptor'].data}
                self[key]['ligand'] = {dkey: make_array(app.calc_types[key]['ligand'].data[dkey])
                           for dkey in app.calc_types[key]['ligand'].data}
                self[key]['delta'] = {dkey: make_array(app.calc_types[key]['delta'].data[dkey])
                           for dkey in app.calc_types[key]['delta'].data}

        # Are we doing a mutant?
        if app.calc_types.mutants:
            for mut_sys, mutants_items in app.calc_types.mutants.items():
                self.mutants[mut_sys] = {}
                for key in mutants_items:
                    if key == 'qh':
                        continue
                    self.mutants[mut_sys][key] = {}
                    if key == 'ie':
                        if not self.stability:
                            self.mutants[mut_sys][key] = {
                                'data': app.calc_types.mutants[mut_sys][key].data['data'],
                                'value': app.calc_types.mutants[mut_sys][key].data['iedata'].avg(),
                                'ie_startframe': app.calc_types[key].data['frames'][-app.calc_types[key].data[
                                    'ieframes']],
                                'ieframes': app.calc_types[key].data['ieframes'],
                                'ie_segment':app.INPUT['ie_segment']}
                        continue

                    self.mutants[mut_sys][key]['complex'] = {
                        dkey: make_array(app.calc_types.mutants[mut_sys][key]['complex'].data[dkey])
                        for dkey in app.calc_types.mutants[mut_sys][key]['complex'].data}

                    if not self.stability:
                        self.mutants[mut_sys][key]['receptor'] = {
                            dkey: make_array(app.calc_types.mutants[mut_sys][key]['receptor'].data[dkey])
                            for dkey in app.calc_types.mutants[mut_sys][key]['receptor'].data}

                        self.mutants[mut_sys][key]['ligand'] = {
                            dkey: make_array(app.calc_types.mutants[mut_sys][key]['ligand'].data[dkey])
                            for dkey in app.calc_types.mutants[mut_sys][key]['ligand'].data}

                        self.mutants[mut_sys][key]['delta'] = {
                            dkey: make_array(app.calc_types.mutants[mut_sys][key]['delta'].data[dkey])
                            for dkey in app.calc_types.mutants[mut_sys][key]['delta'].data}


    def __iadd__(self, other):
        """
        Adding one to another extends every array. The way we do this depends on
        whether we're using numpy arrays or
        """
        return self._add_numpy(other)

    def _add_numpy(self, other):
        """
        If we have numpy available, we need to extend every array in a numpy-valid way
        """
        used_keys = []
        for key in self:
            used_keys.append(key)
            try:
                for dkey in self[key]['complex']:
                    np.append(self[key]['complex'][dkey], other[key]['complex'][dkey])
                for dkey in self[key]['receptor']:
                    np.append(self[key]['receptor'][dkey], other[key]['receptor'][dkey])
                for dkey in self[key]['ligand']:
                    np.append(self[key]['ligand'][dkey], other[key]['ligand'][dkey])
            except KeyError:
                pass
        for key in other:
            if key in used_keys:
                continue
            # If we didn't have a particular calc type, copy that array in here
            self[key] = deepcopy(other[key])
        # Check mutant statuses. If the other has mutant and I don't, copy other
        # If we both have mutant, combine.  If only I do, already done
        if self.mutants and not other.mutants:
            self.mutants = deepcopy(other.mutants)
        elif self.mutants:
            used_keys_mutant = []
            for mut_sys, mutants_items in self.mutants.items():
                for key in mutants_items:
                    used_keys_mutant.append(key)
                    try:
                        for dkey in self[key]['complex']:
                            np.append(self[key]['complex'][dkey], other[key]['complex'][dkey])
                        for dkey in self[key]['receptor']:
                            np.append(self[key]['receptor'][dkey], other[key]['receptor'][dkey])
                        for dkey in self[key]['ligand']:
                            np.append(self[key]['ligand'][dkey], other[key]['ligand'][dkey])
                    except KeyError:
                        pass

            for mut_sys, mutants_items in other.mutants.items():
                for key in mutants_items:
                    if key in used_keys_mutant:
                        continue
                    self.mutants[mut_sys][key] = deepcopy(other.mutants[mut_sys][key])

class APIDecompOut(amber_outputs.DecompOut):

    def __init__(self, basename, res_info, app): #surften, num_files, verbose, nframes, prmtop):

        surften = app.INPUT['surften']
        num_files = app.mpi_size
        verbose = app.INPUT['dec_verbose']
        nframes = app.numframes
        prmtop = app.FILES.complex_prmtop

        amber_outputs.DecompOut.__init__(self, basename, prmtop, surften, False, num_files, verbose)
        self.array_data = {}
        # Make a new dict for all printed tokens (TDC,SDC,BDC)
        for key in self.allowed_tokens:
            self.array_data[key] = {}
        for i in range(nframes):
            for key in self.allowed_tokens:
                for _ in range(self.num_terms):
                    rnum, internal, vdw, eel, pol, sas, tot = self.get_next_term(key)
                    rnum = res_info[rnum - 1]
                    if rnum not in self.array_data[key]:
                        self.array_data[key][rnum] = {}
                        for k in ('int', 'vdw', 'eel', 'pol', 'sas', 'tot'):
                            self.array_data[key][rnum][k] = make_array_len(nframes)
                    self.array_data[key][rnum]['int'][i] = internal
                    self.array_data[key][rnum]['vdw'][i] = vdw
                    self.array_data[key][rnum]['eel'][i] = eel
                    self.array_data[key][rnum]['pol'][i] = pol
                    self.array_data[key][rnum]['sas'][i] = sas
                    self.array_data[key][rnum]['tot'][i] = tot

class APIPairDecompOut(amber_outputs.PairDecompOut):

    def __init__(self, basename, res_info, app):

        surften = app.INPUT['surften']
        num_files = app.mpi_size
        verbose = app.INPUT['dec_verbose']
        nframes = app.numframes
        prmtop = app.FILES.complex_prmtop

        amber_outputs.DecompOut.__init__(self, basename, prmtop, surften, False, num_files, verbose)
        self.array_data = {}
        # Make a new dict for all printed tokens (TDC,SDC,BDC)
        for key in self.allowed_tokens:
            self.array_data[key] = {}

        for i in range(nframes):
            for key in self.allowed_tokens:
                for _ in range(self.num_terms):
                    rnum, rnum2, internal, vdw, eel, pol, sas, tot = self.get_next_term(key)
                    rnum = res_info[rnum - 1]
                    rnum2 = res_info[rnum2 - 1]
                    if rnum not in self.array_data[key]:
                        self.array_data[key][rnum] = {}
                    if rnum2 not in self.array_data[key][rnum]:
                        self.array_data[key][rnum][rnum2] =  {}
                        for k in ('int', 'vdw', 'eel', 'pol', 'sas', 'tot'):
                            self.array_data[key][rnum][rnum2][k] = make_array_len(nframes)
                    self.array_data[key][rnum][rnum2]['int'][i] = internal
                    self.array_data[key][rnum][rnum2]['vdw'][i] = vdw
                    self.array_data[key][rnum][rnum2]['eel'][i] = eel
                    self.array_data[key][rnum][rnum2]['pol'][i] = pol
                    self.array_data[key][rnum][rnum2]['sas'][i] = sas
                    self.array_data[key][rnum][rnum2]['tot'][i] = tot


def _mask2reslist(mask, com_str, app):
    mol = {}
    rmstr = app.INPUT[mask].strip(':')
    ress = []
    for x in rmstr.split(','):
        if len(x.split('-')) > 1:
            start, end = x.split('-')
            ress.extend(range(int(start), int(end) + 1))
        else:
            ress.append(int(x))
    for rn in ress:
        residue = com_str.residues[rn - 1]
        icode = ':' + f'{residue.insertion_code}' if f'{residue.insertion_code}' else ''
        mol[rn] = (f"{residue.chain}:{residue.name}:{residue.number}" + icode)

    return mol


def _transform_from_lvl_models(nd):
    data = []
    for k, v in nd.items():
        for k1, v1 in v.items():
                data.append([(k, k1), v1])
    return data


def _transform_from_lvl_decomp(nd):
    data = []
    for k, v in nd.items(): # model
        for k1, v1 in v.items(): # mol
            for k2, v2 in v1.items(): # TDC, SDC, BDC
                for k3, v3 in v2.items(): # residue
                    for k4, v4 in v3.items(): # residue in per-wise or terms in per-res
                        if isinstance(v4, dict): # per-wise
                            for k5, v5 in v4.items():
                                data.append([(k, k1, k2, k3, k4, k5), v5])
                        else:
                            data.append([(k, k1, k2, k3, k4), v4])
    return data


def _transform_mmpbsa_data(data, app):
    # TODO: include rmsd and Rg for FES?
    mmpbsa_data = {'energy': None, 'decomp': None, 'ie': None, 'nmode': None, 'qh': None}

    interval = app.INPUT['interval']
    start = app.INPUT['startframe']
    frames = [x for x in range(start, start + app.numframes * interval, interval)]
    nmode_frames = [x for x in range(app.INPUT['nmstartframe'], app.INPUT['nmstartframe'] +
                                              app.numframes_nmode * app.INPUT['nminterval'],
                                              app.INPUT['nminterval'])]
    end = app.INPUT['endframe']

    for key, value in data.items():
        if key in ['gb', 'pb', 'rism gf', 'rism std']:
            e_data = {(key,) + x: v for x, v in _transform_from_lvl_models(value)}
            mmpbsa_data['energy'] = pd.DataFrame(e_data, index=frames)
        elif key == 'decomp':
            d_data = {x: v for x, v in _transform_from_lvl_decomp(value)}
            mmpbsa_data['decomp'] = pd.DataFrame(d_data, index=frames)
        elif key == 'ie':
            ie_data = {x: v for x, v in value.items() if x == 'data'}
            ie_rest = {x: v for x, v in value.items() if x != 'data'}
            mmpbsa_data['ie'] = {'data': pd.DataFrame(ie_data, index=frames)}
            mmpbsa_data['ie'].update(ie_rest)
        elif key in ['nmode' 'qh']:
            e_data = {(key,) + x: v for x, v in _transform_from_lvl_models(value)}
            if key == 'nmode':
                mmpbsa_data['nmode'] = pd.DataFrame(e_data, index=nmode_frames)
            else:
                mmpbsa_data['qh'] = pd.DataFrame(e_data)
    return mmpbsa_data


def _get_delta_decomp(df: pd.DataFrame):
    for x in df.columns.levels[0]:
        rest = df[x]['complex'] - pd.concat([df[x]['receptor'], df[x]['ligand']], axis=1)
        rest.columns = pd.MultiIndex.from_product([[x], ['delta']] + rest.columns.levels)
        df = pd.concat([df, rest], axis=1)
    return df


def load_gmxmmpbsa_info(fname, make_df=True):
    """
    Loads up an gmx_MMPBSA info file and returns a mmpbsa_data instance with all
    of the data available in numpy arrays if numpy is available. The returned
    object is a mmpbsa_data instance.

    change the structure to get more easy way to graph per residue

    mmpbsa_data attributes:
    -----------------------
       o  Derived from "dict"
       o  Each solvent model is a dictionary key for a numpy array (if numpy is
          available) or array.array (if numpy is unavailable) for each of the
          species (complex, receptor, ligand) present in the calculation.
       o  The alanine scanning mutant data is under another dict denoted by the
          'mutant' key.

    Data Layout:
    ------------
               Model     |  Dictionary Key    |  Data Keys Available
       -------------------------------------------------------------------
       Generalized Born  |  'gb'              |  EGB, ESURF, *
       Poisson-Boltzmann |  'pb'              |  EPB, EDISPER, ECAVITY, *
       3D-RISM (GF)      |  'rism gf'         |
       3D-RISM (Standard)|  'rism std'        |
       Normal Mode       |  'nmode'           |
       Quasi-harmonic    |  'qh'              |

    * == TOTAL, VDW, EEL, 1-4 EEL, 1-4 VDW, BOND, ANGLE, DIHED

    The keys above are entries for the main dict as well as the sub-dict whose
    key is 'mutant' in the main dict.  Each entry in the main (and mutant sub-)
    dict is, itself, a dict with 1 or 3 keys; 'complex', 'receptor', 'ligand';
    where 'receptor' and 'ligand' are missing for stability calculations.
    If numpy is available, all data will be numpy.ndarray instances.  Otherwise,
    all data will be array.array instances.

    All of the objects referenced by the listed 'Dictionary Key's are dicts in
    which the listed 'Data Keys Available' are keys to the data arrays themselves

    Examples:
    ---------
       # Load numpy for our analyses (optional)
       import numpy as np

       # Load the _MMPBSA_info file:
       mydata = load_mmpbsa_info('_MMPBSA_info')

       # Access the complex GB data structure and calculate the autocorr. fcn.
       autocorr = np.correlate(mydata['gb']['complex']['TOTAL'],
                               mydata['gb']['complex']['TOTAL'])

       # Calculate the standard deviation of the alanine mutant receptor in PB
       print mydata.mutant['pb']['receptor']['TOTAL'].std()
    """
    if not isinstance(fname, Path):
        fname = Path(fname)

    if not fname.exists():
        raise NoFileExists("cannot find %s!" % fname)
    os.chdir(fname.parent)
    app = main.MMPBSA_App(MPI)
    info = infofile.InfoFile(app)
    info.read_info(fname)
    app.normal_system = app.mutant_system = None
    app.parse_output_files()
    return_data = mmpbsa_data(app)
    # Since Decomp data is parsed in a memory-efficient manner (by not storing
    # all of the data in arrays, but rather by printing each data point as it's
    # parsed), we need to handle the decomp data separately here
    # Open Complex fixed structure to assign per-(residue/wise) residue name
    try:
        complex_str = parmed.read_PDB(app.FILES.complex_fixed)
    except:
        complex_str = parmed.read_PDB(app.FILES.prefix + 'COM.pdb')
    # Get receptor and ligand masks
    rec = _mask2reslist('receptor_mask', complex_str, app)
    lig = _mask2reslist('ligand_mask', complex_str, app)

    com = rec.copy()
    com.update(lig)
    com_res_info = [value for key, value in sorted(com.items())]
    rec_res_info = [value for key, value in rec.items()]
    lig_res_info = [value for key, value in lig.items()]

    if app.INPUT['decomprun']:
        # Simplify the decomp class instance creation
        if app.INPUT['idecomp'] in (1, 2):
            DecompClass = lambda x, part: APIDecompOut(x, part, app)
        else:
            DecompClass = lambda x, part: APIPairDecompOut(x, part, app)

        if not app.INPUT['mutant_only']:
            # Do normal GB
            if app.INPUT['gbrun']:
                return_data['decomp'] = {'gb' : {}}
                return_data['decomp']['gb']['complex'] = DecompClass(app.FILES.prefix + 'complex_gb.mdout',
                                                                     com_res_info).array_data
                if not app.stability:
                    return_data['decomp']['gb']['receptor'] = DecompClass(app.FILES.prefix + 'receptor_gb.mdout',
                                                                          rec_res_info).array_data
                    return_data['decomp']['gb']['ligand'] = DecompClass(app.FILES.prefix + 'ligand_gb.mdout',
                                                                        lig_res_info).array_data
            # Do normal PB
            if app.INPUT['pbrun']:
                return_data['decomp'] = {'pb' : {}}
                return_data['decomp']['pb']['complex'] = DecompClass(app.FILES.prefix + 'complex_pb.mdout',
                                                                     com_res_info).array_data
                if not app.stability:
                    return_data['decomp']['pb']['receptor'] = DecompClass(app.FILES.prefix + 'receptor_pb.mdout',
                                                                          rec_res_info).array_data
                    return_data['decomp']['pb']['ligand'] = DecompClass(app.FILES.prefix + 'ligand_pb.mdout',
                                                                        lig_res_info).array_data
        if app.INPUT['alarun']:
            for mut_sys, mut_mask_ndx in zip(app.INPUT['mutants_labels'].split(','),
                                             app.INPUT['mutants_mask'].split(',')):

                mut_com = com.copy()
                if int(mut_mask_ndx) in mut_com:
                    old_name = mut_com[int(mut_mask_ndx)].split(':')
                    old_name[1] = app.INPUT['mutant']
                    mut_com[int(mut_mask_ndx)] = ':'.join(old_name)
                mut_rec = rec.copy()
                if int(mut_mask_ndx) in mut_rec:
                    old_name = mut_rec[int(mut_mask_ndx)].split(':')
                    old_name[1] = app.INPUT['mutant']
                    mut_rec[int(mut_mask_ndx)] = ':'.join(old_name)

                mut_lig = lig.copy()
                if int(mut_mask_ndx) in mut_lig:
                    old_name = mut_lig[int(mut_mask_ndx)].split(':')
                    old_name[1] = app.INPUT['mutant']
                    mut_lig[int(mut_mask_ndx)] = ':'.join(old_name)

                mut_com_res_info = [value for key, value in sorted(mut_com.items())]
                mut_rec_res_info = [value for key, value in mut_rec.items()]
                mut_lig_res_info = [value for key, value in mut_lig.items()]

                # Do mutant GB
                if app.INPUT['gbrun']:
                    return_data.mutants[mut_sys]['decomp'] = {'gb' : {}}
                    return_data.mutants[mut_sys]['decomp']['gb']['complex'] = DecompClass(
                        app.FILES.prefix + f'mutant_complex_gb_{mut_sys}.mdout', mut_com_res_info).array_data
                    if not app.stability:
                        return_data.mutants[mut_sys]['decomp']['gb']['receptor'] = DecompClass(
                            app.FILES.prefix + f'mutant_receptor_gb_{mut_sys}.mdout', mut_rec_res_info).array_data
                        return_data.mutants[mut_sys]['decomp']['gb']['ligand'] = DecompClass(
                            app.FILES.prefix + f'mutant_ligand_gb_{mut_sys}.mdout', mut_lig_res_info).array_data
                # Do mutant PB
                if app.INPUT['pbrun']:
                    return_data.mutants[mut_sys]['decomp'] = {'pb' : {}}
                    return_data.mutants[mut_sys]['decomp']['pb']['complex'] = DecompClass(
                        app.FILES.prefix + f'mutant_complex_pb_{mut_sys}.mdout', mut_com_res_info).array_data
                    if not app.stability:
                        return_data.mutants[mut_sys]['decomp']['pb']['receptor'] = DecompClass(
                            app.FILES.prefix + f'mutant_receptor_pb_{mut_sys}.mdout', mut_rec_res_info).array_data
                        return_data.mutants[mut_sys]['decomp']['pb']['ligand'] = DecompClass(
                            app.FILES.prefix + f'mutant_ligand_pb_{mut_sys}.mdout', mut_lig_res_info).array_data

    app_namespace = SimpleNamespace(FILES=app.FILES, INPUT=app.INPUT, numframes=app.numframes,
                                    numframes_nmode=app.numframes_nmode)
    if not make_df:
        return return_data, app_namespace

    output_data = type('calc_type_dict', (dict,), {'mutants': {}})()
    #
    output_data.update(_transform_mmpbsa_data(return_data, app))
    output_data.mutants.update({
        mut_sys: _transform_mmpbsa_data(value, app)
        for mut_sys, value in return_data.mutants.items()
    })
    # if decomp clacluate the delta energy
    if output_data['decomp'] is not None:
        output_data['decomp'] = _get_delta_decomp(output_data['decomp'])
        # Iterate over all mutants
        if output_data.mutants:
            for mut_sys in output_data.mutants:
                output_data.mutants[mut_sys]['decomp'] = _get_delta_decomp(output_data.mutants[mut_sys]['decomp'])
    return output_data, app_namespace

