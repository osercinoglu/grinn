from prody import *
from prody import LOGGER
import numpy as np
import itertools
from itertools import islice
import concurrent.futures
import pyprind
import signal
# GromacsWrapper is only import here since source_gmxrc must be run first.
import gromacs
import gromacs.environment
#print("GROMACS_DIR:", gmx_env_vars.get("GROMACS_DIR"))
from contextlib import contextmanager
import os, sys, pickle, shutil, time, subprocess, panedr, pandas, glob
import logging
from scipy.sparse import lil_matrix
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import mdtraj as md
import pandas as pd
import networkx as nx
import tqdm
import argparse

# Global variable to store the process group ID
pgid = os.getpgid(os.getpid())

# Directly modifying logging level for ProDy to prevent printing of noisy debug/warning
# level messages on the terminal.
LOGGER._logger.setLevel(logging.FATAL)

def create_logger(outFolder, noconsoleHandler=False):
    """
    Create a logger with specified configuration.

    Parameters:
    - outFolder (str): The folder where log files will be saved.
    - noconsoleHandler (bool): Whether to add a console handler to the logger (default is False).

    Returns:
    - logger (logging.Logger): The configured logger object.
    """
    # If the folder does not exist, create it
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
    # Configure logging format
    loggingFormat = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    logFile = os.path.join(os.path.abspath(outFolder), 'calc.log')
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(loggingFormat, datefmt='%d-%m-%Y:%H:%M:%S')

    # Create console handler and set level to DEBUG
    if not noconsoleHandler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(logFile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def getRibeiroOrtizNetwork(pdb, df_intEn, includeCovalents=True, intEnCutoff=1, startFrame=0, residue_indices=None):
    sys = parsePDB(pdb)
    if residue_indices is None:
        sys_sel = sys.select("all")
        resIndices = np.unique(sys_sel.getResindices())
    else:
        resIndices = np.array(sorted(residue_indices))
        sys_sel = sys.select(' or '.join([f"resindex {i}" for i in resIndices]))
    numResidues = len(resIndices)
    resNames = sys_sel.getResnames()
    resNums = sys_sel.getResnums()
    chains = sys_sel.getChids()
    rname_rnum_ch = ['_'.join(map(str, [resNames[i], resNums[i], chains[i]])) for i in range(numResidues)]

    # Robustly find the pair column
    pair_col = None
    for col in ['Pair_indices', 'pair']:
        if col in df_intEn.columns:
            pair_col = col
            break
    if pair_col is None:
        raise ValueError("Could not find a 'Pair_indices' or 'pair' column in the input DataFrame.")

    # Identify frame columns: all columns between pair_col and the first annotation column
    frame_start = list(df_intEn.columns).index(pair_col) + 1
    annotation_cols = ['res1_index', 'res2_index', 'res1_chain', 'res2_chain', 'res1_resnum', 'res2_resnum', 'res1_resname', 'res2_resname', 'res1', 'res2']
    frame_end = len(df_intEn.columns)
    for ann in annotation_cols:
        if ann in df_intEn.columns:
            frame_end = list(df_intEn.columns).index(ann)
            break
    frame_cols = df_intEn.columns[frame_start:frame_end]
    numFrames = len(frame_cols)

    nx_list = []
    for m in range(startFrame, numFrames):
        frame_col = frame_cols[m]
        network = nx.Graph()
        for j in range(numResidues):
            network.add_node(j + 1, label=rname_rnum_ch[j])
        resIntEnMat = np.zeros((numResidues, numResidues))
        for _, row in df_intEn.iterrows():
            pair = row[pair_col]
            resindex_1 = int(pair.split('-')[0])
            resindex_2 = int(pair.split('-')[1])
            try:
                value = float(row[frame_col])
            except Exception:
                continue
            resIntEnMat[resindex_1, resindex_2] = value
            resIntEnMat[resindex_2, resindex_1] = value
        resIntEnMatNegFavor = np.where(resIntEnMat < 0, np.abs(resIntEnMat), 0)
        max_abs = np.max(np.abs(resIntEnMatNegFavor))
        X = resIntEnMatNegFavor / max_abs if max_abs != 0 else resIntEnMatNegFavor
        X = np.clip(X, 0, 0.99)
        if includeCovalents:
            for i in range(numResidues - 1):
                res1 = sys.select('resindex %i' % resIndices[i])
                res2 = sys.select('resindex %i' % resIndices[i + 1])
                if (res1.getChids()[0] == res2.getChids()[0]) and (res1.getSegindices()[0] == res2.getSegindices()[0]):
                    network.add_edge(i + 1, i + 2, weight=X[i, i + 1], distance=1 - float(X[i, i + 1]))
        for i in range(numResidues):
            for j in range(numResidues):
                if not includeCovalents and abs(i - j) == 1:
                    continue
                if not network.has_edge(i + 1, j + 1):
                    if abs(float(resIntEnMat[i, j])) >= abs(intEnCutoff):
                        if X[i, j] < 0.01:
                            continue
                        network.add_edge(i + 1, j + 1, weight=X[i, j], distance=1 - float(X[i, j]))
        nx_list.append(network)
    return nx_list

def compute_pen_and_bc(
    pdb_file, 
    int_en_csv, 
    out_folder, 
    intEnCutoff_values=[1.0], 
    include_covalents_options=[True, False],
    logger=None,
    source_sel="all",
    target_sel="all"
):
    df_intEn = pd.read_csv(int_en_csv)
    if 'Unnamed: 0' in df_intEn.columns:
        df_intEn = df_intEn.drop(columns=['Unnamed: 0'])
    if 'Pair_indices' in df_intEn.columns:
        df_intEn = df_intEn.rename(columns={'Pair_indices': 'pair'})

    # Get union of residue indices from source_sel and target_sel
    sys = parsePDB(pdb_file)
    source_indices = set(sys.select(source_sel).getResindices())
    target_indices = set(sys.select(target_sel).getResindices())
    residue_indices = sorted(source_indices | target_indices)

    all_bc_results = []
    for include_covalents in include_covalents_options:
        for intEnCutoff in intEnCutoff_values:
            logger and logger.info(f"Creating PENs: include_covalents={include_covalents}, intEnCutoff={intEnCutoff}")
            nx_list = getRibeiroOrtizNetwork(
                pdb_file, df_intEn, 
                includeCovalents=include_covalents, 
                intEnCutoff=intEnCutoff,
                residue_indices=residue_indices
            )
            for frame_idx, G in enumerate(nx_list):
                # Save network
                gml_path = os.path.join(
                    out_folder, 
                    f"pen_cov{include_covalents}_cutoff{intEnCutoff}_frame{frame_idx}.gml"
                )
                nx.write_gml(G, gml_path)
                # Compute BCs
                bc_dict = nx.betweenness_centrality(G)
                bc_df = pd.DataFrame({
                    'Residue': list(bc_dict.keys()),
                    'BC': list(bc_dict.values()),
                    'Frame': frame_idx,
                    'include_covalents': include_covalents,
                    'intEnCutoff': intEnCutoff,
                })
                # Add node labels
                bc_df['Label'] = [G.nodes[j]['label'] for j in bc_df['Residue'].values]
                all_bc_results.append(bc_df)
    # Concatenate all results
    df_bc = pd.concat(all_bc_results, axis=0, ignore_index=True)
    df_bc.to_csv(os.path.join(out_folder, "pen_betweenness_centralities.csv"), index=False)
    logger and logger.info(f"Saved all PENs and BCs to {out_folder}")

def run_gromacs_simulation(pdb_filepath, mdp_files_folder, out_folder, ff_folder, nofixpdb, gpu, solvate, npt, logger, nt=1):
    """
    Run a GROMACS simulation workflow.

    Parameters:
    - pdb_filepath (str): The path to the input PDB file.
    - mdp_files_folder (str): The folder containing the MDP files.
    - out_folder (str): The folder where output files will be saved.
    - nofixpdb (bool): Whether to fix the PDB file using pdbfixer (default is True).
    - logger (logging.Logger): The logger object for logging messages.
    - nt (int): Number of threads for GROMACS commands (default is 1).
    - ff_folder (str): The folder containing the force field files (default is None).

    Returns:
    - None
    """

    gromacs.environment.flags['capture_output'] = "file"
    gromacs.environment.flags['capture_output_filename'] = os.path.join(out_folder, "gromacs.log")

    logger.info(f"Running GROMACS simulation for PDB file: {pdb_filepath}")

    if nofixpdb:
        fixed_pdb_filepath = pdb_filepath
    else:
        # Fix PDB file
        fixer = PDBFixer(filename=pdb_filepath)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        pdb_filename = os.path.basename(pdb_filepath)
        fixed_pdb_filepath = os.path.join(out_folder, "protein.pdb")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdb_filepath, 'w'))
        logger.info("PDB file fixed.")
        system = parsePDB(fixed_pdb_filepath)
        writePDB(fixed_pdb_filepath, system.select('protein'))

    if ff_folder is not None:
        ff = ff_folder
    else:
        ff = "amber99sb-ildn"

    if gpu:
        gpu="gpu"
    else:
        gpu="cpu"

    # Run GROMACS commands
    try:
        gromacs.pdb2gmx(f=fixed_pdb_filepath, o=os.path.join(out_folder, "protein.pdb"), 
                        p=os.path.join(out_folder, "topol.top"), i=os.path.join(out_folder,"posre.itp"),
                          ff=ff, water="tip3p", heavyh=True, ignh=True)
        logger.info("pdb2gmx command completed.")
        next_pdb = "protein.pdb"

        index_group_select = 'Protein'
        index_group_name = "Protein"
        gromacs.make_ndx(f=os.path.join(out_folder, next_pdb), o=os.path.join(out_folder, "index.ndx"), input=('q'))

        shutil.copy(os.path.join(out_folder, "topol.top"), os.path.join(out_folder, "topol_dry.top"))
        logger.info("Topology file copied.")

        if solvate:
            gromacs.editconf(f=os.path.join(out_folder, next_pdb), n=os.path.join(out_folder, "index.ndx"), 
                             o=os.path.join(out_folder, "boxed.pdb"), bt="cubic", c=True, d=1.0, princ=True, input=('0','0','0'))
            logger.info("editconf command completed.")
            gromacs.solvate(cp=os.path.join(out_folder, "boxed.pdb"), cs="spc216", p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "solvated.pdb"))
            logger.info("solvate command completed.")
            gromacs.grompp(f=os.path.join(mdp_files_folder, "ions.mdp"), c=os.path.join(out_folder, "solvated.pdb"), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "ions.tpr"))
            logger.info("grompp for ions command completed.")
            gromacs.genion(s=os.path.join(out_folder, "ions.tpr"), o=os.path.join(out_folder, "solvated_ions.pdb"), p=os.path.join(out_folder, "topol.top"), neutral=True, conc=0.15, input=('SOL','q'))
            logger.info("genion command completed.")
            next_pdb = "solvated_ions.pdb"
        else:
            gromacs.editconf(f=os.path.join(out_folder, next_pdb), n=os.path.join(out_folder, 'index.ndx'), 
                             o=os.path.join(out_folder, "boxed.pdb"), bt="cubic", c=True, box=[999,999,999], princ=True, input=(index_group_name, index_group_name, index_group_name))
            logger.info("editconf command completed.")
            next_pdb = "boxed.pdb"
        
        if next_pdb == "solvated_ions.pdb":
            gromacs.grompp(f=os.path.join(mdp_files_folder, "minim.mdp"), c=os.path.join(out_folder, next_pdb), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "minim.tpr"))
        if next_pdb == "boxed.pdb":
            gromacs.grompp(f=os.path.join(mdp_files_folder, "minim_vac.mdp"), c=os.path.join(out_folder, next_pdb), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "minim.tpr"))

        logger.info("grompp for minimization command completed.")
        gromacs.mdrun(deffnm="minim", v=True, c=os.path.join(out_folder, "minim.pdb"), s=os.path.join(out_folder,"minim.tpr"), 
                      e=os.path.join(out_folder,"minim.edr"), g=os.path.join(out_folder,"minim.log"), 
                      o=os.path.join(out_folder,"minim.trr"), x=os.path.join(out_folder,"minim.xtc"), nt=nt, nb=gpu, pin='on') 
        logger.info("mdrun for minimization command completed.")
        gromacs.trjconv(f=os.path.join(out_folder, 'minim.pdb'),o=os.path.join(out_folder, 'minim.pdb'), s=os.path.join(out_folder, next_pdb), input=('0','q'))
        logger.info("trjconv for minimization command completed.")
        next_pdb = "minim.pdb"
        gromacs.trjconv(f=os.path.join(out_folder,next_pdb),o=os.path.join(out_folder, "traj.xtc"))

        if npt:
            gromacs.grompp(f=os.path.join(mdp_files_folder, "npt.mdp"), c=os.path.join(out_folder, next_pdb), 
                           r=os.path.join(out_folder, next_pdb), p=os.path.join(out_folder, "topol.top"), o=os.path.join(out_folder, "npt.tpr"), maxwarn=10)
            logger.info("grompp for NPT command completed.")
            gromacs.mdrun(deffnm="npt", v=True, c=os.path.join(out_folder, "npt.pdb"), s=os.path.join(out_folder,"npt.tpr"), nt=nt, pin='on', 
            x=os.path.join(out_folder, "npt.xtc"), e=os.path.join(out_folder, "npt.edr"), o=os.path.join(out_folder, "npt.trr"))
            logger.info("mdrun for NPT command completed.")
            gromacs.trjconv(f=os.path.join(out_folder, 'npt.pdb'), o=os.path.join(out_folder, 'npt.pdb'), s=os.path.join(out_folder, 'solvated_ions.pdb'), input=('0','q'))
            logger.info("trjconv for NPT command completed.")
            gromacs.trjconv(s=os.path.join(out_folder, 'npt.tpr'), f=os.path.join(out_folder, 'npt.xtc'), o=os.path.join(out_folder, 'traj.xtc'), input=(index_group_name,))
            logger.info("trjconv for NPT to XTC conversion command completed.")
            next_pdb = "npt.pdb"

        gromacs.trjconv(f=os.path.join(out_folder, next_pdb), o=os.path.join(out_folder, 'system_dry.pdb'), s=os.path.join(out_folder, next_pdb), n=os.path.join(out_folder, 'index.ndx'), input=(index_group_name,))
        logger.info(f"trjconv for {next_pdb} to DRY PDB conversion command completed.")
        
        gromacs.trjconv(f=os.path.join(out_folder, 'traj.xtc'), o=os.path.join(out_folder, 'traj_dry.xtc'), s=os.path.join(out_folder, 'system_dry.pdb'), 
                        n=os.path.join(out_folder, 'index.ndx'), input=(index_group_name,))
        logger.info(f"trjconv for traj.xtc to traj_dry.xtc conversion command completed.")

        # Convert npt.xtc to npt.dcd
        traj = md.load(os.path.join(out_folder, 'traj_dry.xtc'), top=os.path.join(out_folder, "system_dry.pdb"))
        traj.save_dcd(os.path.join(out_folder, 'traj_dry.dcd'))

        logger.info("GROMACS simulation completed successfully.")
    except Exception as e:
        logger.error(f"Error encountered during GROMACS simulation: {str(e)}")

# A method for suppressing terminal output temporarily.
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def filterInitialPairsSingleCore(args):
    outFolder = args[0]
    pairs = args[1]
    initPairFilterCutoff = args[2]

    with suppress_stdout():
        system = parsePDB(os.path.join(outFolder, "system_dry.pdb"))

    # Define a method for initial filtering of a single pair.
    def filterInitialPair(pair):
        com1 = calcCenter(system.select("resindex %i" % pair[0]))
        com2 = calcCenter(system.select("resindex %i" % pair[1]))
        dist = calcDistance(com1, com2)
        if dist <= initPairFilterCutoff:
            return pair
        else:
            return None

    # Get a list of included pairs after initial filtering.
    filterList = []
    progbar = pyprind.ProgBar(len(pairs))
    for pair in pairs:
        filtered = filterInitialPair(pair)
        if filtered is not None:
            filterList.append(pair)
        progbar.update()

    return filterList

def perform_initial_filtering(outFolder, source_sel, target_sel, initPairFilterCutoff, numCores, logger):
    """
    Perform initial filtering of residue pairs based on distance.

    Parameters:
    - outFolder (str): The folder where output files will be saved.
    - initPairFilterCutoff (float): The distance cutoff for initial filtering.
    - numCores (int): The number of CPU cores to use for parallel processing.
    - logger (logging.Logger): The logger object for logging messages.

    Returns:
    - initialFilter (list): A list of residue pairs after initial filtering.
    """
    logger.info("Performing initial filtering...")

    # Get the path to the PDB file (system.pdb) from outFolder
    pdb_file = os.path.join(outFolder, "system_dry.pdb")

    # Parse PDB file
    system = parsePDB(pdb_file)
    numResidues = system.numResidues()
    source = system.select(source_sel)
    target = system.select(target_sel)

    sourceResids = np.unique(source.getResindices())
    numSource = len(sourceResids)

    targetResids = np.unique(target.getResindices())
    numTarget = len(targetResids)

    # Generate all possible unique pairwise residue-residue combinations
    pairProduct = itertools.product(sourceResids, targetResids)
    pairSet = set()
    for x, y in pairProduct:
        if x != y:
            pairSet.add(frozenset((x, y)))

    # Prepare a pairSet list
    pairSet = [list(pair) for pair in list(pairSet)]

    # Get a list of pairs within a certain distance from each other, based on the initial structure.
    initialFilter = []

    # Split the pair set list into chunks according to number of cores
    # Reduce numCores if necessary.
    if len(pairSet) < numCores:
        numCores = len(pairSet)
    
    pairChunks = np.array_split(list(pairSet), numCores)

    # Start a concurrent futures pool, and perform initial filtering.
    with concurrent.futures.ProcessPoolExecutor(numCores) as pool:
        try:
            initialFilter = pool.map(filterInitialPairsSingleCore, [[outFolder, pairChunks[i], initPairFilterCutoff] for i in range(0, numCores)])
            initialFilter = list(initialFilter)
            
            # initialFilter may contain empty lists, remove them.
            initialFilter = [sublist for sublist in initialFilter if sublist]

            # Flatten the list of lists
            if len(initialFilter) > 1:
                initialFilter = np.vstack(initialFilter)
        finally:
            pool.shutdown()

    initialFilter = list(initialFilter)
    initialFilter = [pair for pair in initialFilter if pair is not None]
    logger.info('Initial filtering... Done.')
    logger.info('Number of interaction pairs selected after initial filtering step: %i' % len(initialFilter))

    initialFilterPickle = os.path.join(os.path.abspath(outFolder), "initialFilter.pkl")
    with open(initialFilterPickle, 'wb') as f:
        pickle.dump(initialFilter, f)

    return initialFilter

# A method to get a string containing chain or seg ID, residue name and residue number
# given a ProDy parsed PDB Atom Group and the residue index
def getChainResnameResnum(pdb,resIndex):
	# Get a string for chain+resid+resnum when supplied the residue index.
	selection = pdb.select('resindex %i' % resIndex)
	chain = selection.getChids()[0]
	chain = chain.strip(' ')
	segid = selection.getSegnames()[0]
	segid = segid.strip(' ')

	resName = selection.getResnames()[0]
	resNum = selection.getResnums()[0]
	if chain:
		string = ''.join([chain,str(resName),str(resNum)])
	elif segid:
		string = ''.join([segid,str(resName),str(resNum)])
	return [chain,segid,resName,resNum,string]

def process_chunk(i, chunk, outFolder, top_file, pdb_file, xtc_file):
    mdpFile = os.path.join(outFolder, f'interact{i}.mdp')
    tprFile = mdpFile.rstrip('.mdp') + '.tpr'
    edrFile = mdpFile.rstrip('.mdp') + '.edr'

    gromacs.environment.flags['capture_output'] = "file"
    gromacs.environment.flags['capture_output_filename'] = os.path.join(outFolder, f"gromacs_interaction{i}.log")

    gromacs.grompp(f=mdpFile, n=os.path.join(outFolder, 'interact.ndx'), p=top_file, c=pdb_file, o=tprFile, maxwarn=20)
    gromacs.mdrun(s=tprFile, c=pdb_file, e=edrFile, g=os.path.join(outFolder, f'interact{i}.log'), nt=1, rerun=xtc_file)

    return edrFile, chunk

def calculate_interaction_energies(outFolder, initialFilter, numCoresIE, logger):
    """
    Calculate interaction energies for residue pairs.

    Parameters:
    - outFolder (str): The folder where output files will be saved.
    - numCoresIE (int): The number of CPU cores to use for interaction energy calculation.
    - logger (logging.Logger): The logger object for logging messages.

    Returns:
    - edrFiles (list): List of paths to the EDR files generated during calculation.
    """
    logger.info("Calculating interaction energies...")

    gromacs.environment.flags['capture_output'] = "file"
    gromacs.environment.flags['capture_output_filename'] = os.path.join(outFolder, "gromacs.log")

    # Read necessary files from outFolder
    pdb_file = os.path.join(outFolder, 'system_dry.pdb')
    top_file = os.path.join(outFolder, 'topol_dry.top')
    xtc_file = os.path.join(outFolder, 'traj_dry.xtc')

    # Modify atom serial numbers to account for possible PDB files with more than 99999 atoms
    system = parsePDB(pdb_file)
    system.setSerials(np.arange(1, system.numAtoms() + 1))

    system_dry = system.select('protein or nucleic or lipid or hetero and not water and not resname SOL and not ion')
    system_dry = system_dry.select('not resname SOL')

    indicesFiltered = np.unique(np.hstack(initialFilter))
    allSerials = {}

    for index in indicesFiltered:
        residue = system_dry.select('resindex %i' % index)
        lenSerials = len(residue.getSerials())
        if lenSerials > 14:
            residueSerials = residue.getSerials()
            allSerials[index] = [residueSerials[i:i + 14] for i in range(0, lenSerials, 14)]
        else:
            allSerials[index] = np.asarray([residue.getSerials()])

    # Write a standard .ndx file for GMX
    filename = os.path.join(outFolder, 'interact.ndx')
    gromacs.make_ndx(f=os.path.join(outFolder, 'system_dry.pdb'), o=filename, input=('q',))

    # Append our residue groups to this standard file!
    with open(filename, 'a') as f:
        for key in allSerials:
            f.write('[ res%i ]\n' % key)
            if type(allSerials[key][0]).__name__ == 'ndarray':
                for line in allSerials[key][0:]:
                    f.write(' '.join(list(map(str, line))) + '\n')
            else:
                f.write(' '.join(list(map(str, allSerials))) + '\n')

    # Write the .mdp files necessary for GMX
    mdpFiles = []

    # Divide pairsFiltered into chunks so that each chunk does not contain
    # more than 200 unique residue indices.
    pairsFilteredChunks = []
    if len(np.unique(np.hstack(initialFilter))) <= 60:
        pairsFilteredChunks.append(initialFilter)
    else:
        i = 2
        maxNumRes = len(np.unique(np.hstack(initialFilter)))
        while maxNumRes >= 60:
            pairsFilteredChunks = np.array_split(initialFilter, i)
            chunkNumResList = [len(np.unique(np.hstack(chunk))) for chunk in pairsFilteredChunks]
            maxNumRes = np.max(chunkNumResList)
            i += 1

    for pair in initialFilter:
        if pair not in np.vstack(pairsFilteredChunks):
            logger.exception('Missing at least one residue in filtered residue pairs. Please contact the developer.')
        
    i = 0
    for chunk in pairsFilteredChunks:
        filename = str(outFolder)+'/interact'+str(i)+'.mdp'
        f = open(filename,'w')
        #f.write('cutoff-scheme = group\n')
        f.write('cutoff-scheme = Verlet\n')
        #f.write('epsilon-r = %f\n' % soluteDielectric)

        chunkResidues = np.unique(np.hstack(chunk))

        resString = ''
        for res in chunkResidues:
            resString += 'res'+str(res)+' '

        #resString += ' SOL'

        f.write('energygrps = '+resString+'\n')

        # Add energygroup exclusions.
        #energygrpExclString = 'energygrp-excl ='

        # GOTTA COMMENT OUT THE FOLLOWING DUE TO TOO LONG LINE ERROR IN GROMPP
        # for key in allSerials:
        # 	energygrpExclString += ' res%i res%i' % (key,key)

        #energygrpExclString += ' SOL SOL'
        #f.write(energygrpExclString)

        f.close()
        mdpFiles.append(filename)
        i += 1

    def start_subprocess(command):
        return subprocess.Popen(command, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def terminate_process_group(pgid):
        os.killpg(pgid, signal.SIGTERM)

    def parallel_process_chunks(pairsFilteredChunks, outFolder, top_file, pdb_file, xtc_file, numCoresIE, logger):
        edrFiles = []
        pairsFilteredChunksProcessed = []

        max_workers = min(numCoresIE, len(pairsFilteredChunks))  # Adjust max_workers to a smaller number if needed

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_chunk, i, chunk, outFolder, top_file, pdb_file, xtc_file)
                for i, chunk in enumerate(pairsFilteredChunks)
            ]

            def signal_handler(sig, frame):
                print('Signal caught. Shutting down...')
                executor.shutdown(wait=False)
                for future in futures:
                    if future.running():
                        # Attempt to kill the process group of the future
                        try:
                            pid = future.result().pid
                            pgid = os.getpgid(pid)
                            terminate_process_group(pgid)
                        except Exception as e:
                            logger.error(f"Error terminating process group: {e}")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            j = 0
            for future in concurrent.futures.as_completed(futures):
                edrFile, chunk = future.result()
                edrFiles.append(edrFile)
                pairsFilteredChunksProcessed.append(chunk)
                j += 1

                logger.info('Completed calculation percentage: ' + str((j) / len(futures) * 100))

        return edrFiles, pairsFilteredChunksProcessed
    
    edrFiles, pairsFilteredChunksProcessed = parallel_process_chunks(pairsFilteredChunks, outFolder, top_file, pdb_file, xtc_file, numCoresIE, logger)
    return edrFiles, pairsFilteredChunksProcessed

def parse_interaction_energies(edrFiles, pairsFilteredChunks, outFolder, logger):
    """
    Parse interaction energies from EDR files and save the results.

    Parameters:
    - edrFiles (list): List of paths to the EDR files.
    - outFolder (str): The folder where output files will be saved.
    - logger (logging.Logger): The logger object for logging messages.
    """

    system = parsePDB(os.path.join(outFolder, 'system_dry.pdb'))
    
    
    logger.info('Parsing GMX energy output... This may take a while...')
    df = panedr.edr_to_df(os.path.join(outFolder, 'interact0.edr'))
    logger.info('Parsed 1 EDR file.')

    for i in range(1, len(edrFiles)):
        edrFile = edrFiles[i]
        df_pair = panedr.edr_to_df(edrFile)

        # Remove already parsed columns
        df_pair_columns = df_pair.columns
        df_pair = df_pair[list(set(df_pair_columns) - set(df.columns))]

        df = pd.concat([df, df_pair], axis=1)
        logger.info('Parsed %i out of %i EDR files...' % (i + 1, len(edrFiles)))

    logger.info('Collecting energy results...')
    energiesDict = dict()

    for i in range(len(pairsFilteredChunks)):
        pairsFilteredChunk = pairsFilteredChunks[i]
        energiesDictChunk = dict()

        for pair in pairsFilteredChunk:
            #res1_string = getChainResnameResnum(system, pair[0])[-1]
            #res2_string = getChainResnameResnum(system, pair[1])[-1]
            energyDict = dict()

            # Lennard-Jones Short Range interaction
            column_stringLJSR1 = 'LJ-SR:res%i-res%i' % (pair[0], pair[1])
            column_stringLJSR2 = 'LJ-SR:res%i-res%i' % (pair[1], pair[0])
            if column_stringLJSR1 in df.columns:
                column_stringLJSR = column_stringLJSR1
            elif column_stringLJSR2 in df.columns:
                column_stringLJSR = column_stringLJSR2
            else:
                logger.warning(f'Pair {column_stringLJSR1} or {column_stringLJSR2} was not found in the pair interaction '
                                 'energy output.')
                continue

            # Lennard-Jones 1-4 interaction
            column_stringLJ141 = 'LJ-14:res%i-res%i' % (pair[0], pair[1])
            column_stringLJ142 = 'LJ-14:res%i-res%i' % (pair[1], pair[0])
            if column_stringLJ141 in df.columns:
                column_stringLJ14 = column_stringLJ141
            elif column_stringLJ142 in df.columns:
                column_stringLJ14 = column_stringLJ142
            else:
                logger.warning(f'Pair {column_stringLJ141} or {column_stringLJ142} was not found in the pair interaction '
                                 'energy output.')
                continue

            # Coulombic Short Range interaction
            column_stringCoulSR1 = 'Coul-SR:res%i-res%i' % (pair[0], pair[1])
            column_stringCoulSR2 = 'Coul-SR:res%i-res%i' % (pair[1], pair[0])
            if column_stringCoulSR1 in df.columns:
                column_stringCoulSR = column_stringCoulSR1
            elif column_stringCoulSR2 in df.columns:
                column_stringCoulSR = column_stringCoulSR2
            else:
                logger.warning(f'Pair {column_stringCoulSR1} or {column_stringCoulSR2} was not found in the pair interaction '
                                 'energy output.')
                continue

            # Coulombic Short Range interaction
            column_stringCoul141 = 'Coul-14:res%i-res%i' % (pair[0], pair[1])
            column_stringCoul142 = 'Coul-14:res%i-res%i' % (pair[1], pair[0])
            if column_stringCoul141 in df.columns:
                column_stringCoul14 = column_stringCoul141
            elif column_stringCoul142 in df.columns:
                column_stringCoul14 = column_stringCoul142
            else:
                logger.warning(f'Pair {column_stringCoul141} or {column_stringCoul142} was not found in the pair interaction '
                                 'energy output.')
                continue

            # Convert energy units from kJ/mol to kcal/mol
            kj2kcal = 0.239005736
            enLJSR = np.asarray(df[column_stringLJSR].values) * kj2kcal
            enLJ14 = np.asarray(df[column_stringLJ14].values) * kj2kcal
            enLJ = [enLJSR[j] + enLJ14[j] for j in range(len(enLJSR))]
            energyDict['VdW'] = enLJ

            enCoulSR = np.asarray(df[column_stringCoulSR].values) * kj2kcal
            enCoul14 = np.asarray(df[column_stringCoul14].values) * kj2kcal
            enCoul = [enCoulSR[j] + enCoul14[j] for j in range(len(enCoulSR))]
            energyDict['Elec'] = enCoul

            energyDict['Total'] = [energyDict['VdW'][j] + energyDict['Elec'][j] for j in range(len(energyDict['VdW']))]

            #key1 = res1_string + '-' + res2_string
            #key1 = key1.replace(' ', '')
            #key2 = res2_string + '-' + res1_string
            #key2 = key2.replace(' ', '')
            #energiesDictChunk[key1] = energyDict
            #energiesDictChunk[key2] = energyDict

            # Also use residue indices - may come handy later on for some analyses
            key1_alt = str(pair[0]) + '-' + str(pair[1])
            energiesDictChunk[key1_alt] = energyDict

        energiesDict.update(energiesDictChunk)
        logger.info('Collected %i out of %i results' % (i + 1, len(pairsFilteredChunks)))

    logger.info('Collecting results...')

    # Prepare data tables from parsed energies and save to files
    total_data = {}
    elec_data = {}
    vdw_data = {}

    # Collect data into dictionaries
    for key, value in energiesDict.items():
        total_data[key] = value['Total']
        elec_data[key] = value['Elec']
        vdw_data[key] = value['VdW']

    # Convert dictionaries to DataFrames using pd.concat
    df_total = pd.DataFrame(total_data)
    df_elec = pd.DataFrame(elec_data)
    df_vdw = pd.DataFrame(vdw_data)

    # If necessary, copy the DataFrames to defragment them
    df_total = df_total.copy()
    df_elec = df_elec.copy()
    df_vdw = df_vdw.copy()

    # Take transpose of the DataFrames
    df_total = df_total.transpose()
    df_elec = df_elec.transpose()
    df_vdw = df_vdw.transpose()

    # Reset index to avoid pairs being used as index
    df_total.reset_index(inplace=True)
    df_elec.reset_index(inplace=True)
    df_vdw.reset_index(inplace=True)

    def supplement_df(df, system):
        # Rename the first column to 'Pair_indices'

        df.rename(columns={df.columns[0]: 'Pair_indices'}, inplace=True)

        # Extract {res1_index}-{res2_index} from 'Pair_indices', convert them to integers and store them in two separate columns
        df['res1_index'] = df['Pair_indices'].apply(lambda x: int(x.split('-')[0]))
        df['res2_index'] = df['Pair_indices'].apply(lambda x: int(x.split('-')[1]))

        # Find chain ID, residue number, and residue one letter code for each res1 and for each res2, and assign them to new columns
        df['res1_chain'] = df['res1_index'].apply(lambda x: system.select('resindex ' + str(x)).getChids()[0])
        df['res1_chain'] = df['res1_index'].apply(lambda x: system.select('resindex ' + str(x)).getChids()[0])
        df['res2_chain'] = df['res2_index'].apply(lambda x: system.select('resindex ' + str(x)).getChids()[0])
        df['res1_resnum'] = df['res1_index'].apply(lambda x: system.select('resindex ' + str(x)).getResnums()[0])
        df['res2_resnum'] = df['res2_index'].apply(lambda x: system.select('resindex ' + str(x)).getResnums()[0])
        df['res1_resname'] = df['res1_index'].apply(lambda x: system.select('resindex ' + str(x)).getResnames()[0])
        df['res2_resname'] = df['res2_index'].apply(lambda x: system.select('resindex ' + str(x)).getResnames()[0])

        # Merge res1_resname, res1_resnum, and _res1_chain into a new column, do the same for res2
        df['res1'] = df['res1_resname'] + df['res1_resnum'].astype(str) + '_' + df['res1_chain']
        df['res2'] = df['res2_resname'] + df['res2_resnum'].astype(str) + '_' + df['res2_chain']

        return df
    
    # Supplement the DataFrames with additional information
    df_total = supplement_df(df_total, system)
    df_elec = supplement_df(df_elec, system)
    df_vdw = supplement_df(df_vdw, system)

    logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnTotal.csv'))
    df_total.to_csv(os.path.join(outFolder, 'energies_intEnTotal.csv'))
    logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnElec.csv'))
    df_elec.to_csv(os.path.join(outFolder, 'energies_intEnElec.csv'))
    logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnVdW.csv'))
    df_vdw.to_csv(os.path.join(outFolder, 'energies_intEnVdW.csv'))

    logger.info('Pickling results...')

    # Split the dictionary into chunks for pickling
    def chunks(data, SIZE=10000):
        it = iter(data)
        for i in range(0, len(data), SIZE):
            yield {k: data[k] for k in islice(it, SIZE)}

    enDicts = list(chunks(energiesDict, 1000))

    intEnPicklePaths = []

    # Pickle the chunks
    for i in range(len(enDicts)):
        fpath = os.path.join(outFolder, 'energies_%i.pickle' % i)
        with open(fpath, 'wb') as file:
            logger.info('Pickling to energies_%i.pickle...' % i)
            pickle.dump(enDicts[i], file)
            intEnPicklePaths.append(fpath)

    logger.info('Pickling results... Done.')

def cleanUp(outFolder, logger):
    """
    Clean up the output folder by removing unnecessary files.

    Parameters:
    - outFolder (str): The folder where output files will be saved.
    """
    # Cleaning up the output folder
    logger.info('Cleaning up...')

    # Delete all NAMD-generated energies file from output folder
    for item in glob.glob(os.path.join(outFolder, '*_energies.log')):
        os.remove(item)

    for item in glob.glob(os.path.join(outFolder, 'gromacs_*.log')):
        os.remove(item)

    for item in glob.glob(os.path.join(outFolder, '*temp*')):
        os.remove(item)

    # Delete all GROMACS-generated energies file from output folder
    for item in glob.glob(os.path.join(outFolder, 'interact*')):
        os.remove(item)

    for item in glob.glob(os.path.join(outFolder, '*.trr')):
        os.remove(item)

    if os.path.exists(os.path.join(outFolder, 'traj.dcd')):
        os.remove(os.path.join(outFolder, 'traj.dcd'))

    for item in glob.glob(os.path.join(os.getcwd(), '#*#')):
        os.remove(item)

    for item in glob.glob(os.path.join(outFolder, '#*#')):
        os.remove(item)

    logger.info('Cleaning up... completed.')

def test_grinn_inputs(pdb_file, out_folder, ff_folder=None, init_pair_filter_cutoff=10, 
                     nofixpdb=False, top=None, toppar=None, traj=None, nointeraction=False, 
                     gpu=False, solvate=False, npt=False, source_sel="all", target_sel="all", 
                     nt=1, noconsole_handler=False, include_files=None,
                     create_pen=False, pen_cutoffs=[1.0], pen_include_covalents=[True, False]):
    """
    Test and validate inputs for the gRINN workflow.
    
    Returns:
    - (bool, list): (is_valid, list_of_errors)
    """
    errors = []
    warnings = []
    
    # Test required arguments
    if not pdb_file:
        errors.append("ERROR: PDB file path is required")
    elif not os.path.exists(pdb_file):
        errors.append(f"ERROR: PDB file '{pdb_file}' does not exist")
    elif not pdb_file.endswith('.pdb'):
        warnings.append(f"WARNING: PDB file '{pdb_file}' does not have .pdb extension")
    
    if not out_folder:
        errors.append("ERROR: Output folder path is required")
    
    # Test optional file arguments
    if top and not os.path.exists(top):
        errors.append(f"ERROR: Topology file '{top}' does not exist")
    
    if traj:
        if not os.path.exists(traj):
            errors.append(f"ERROR: Trajectory file '{traj}' does not exist")
        elif not traj.endswith(('.xtc', '.trr', '.dcd')):
            warnings.append(f"WARNING: Trajectory file '{traj}' has unusual extension")
    
    if ff_folder and not os.path.exists(ff_folder):
        errors.append(f"ERROR: Force field folder '{ff_folder}' does not exist")
    
    if toppar and not os.path.exists(toppar):
        errors.append(f"ERROR: Toppar folder '{toppar}' does not exist")
    
    # Test include_files
    if include_files:
        for f in include_files:
            if not os.path.exists(f):
                errors.append(f"ERROR: Include file '{f}' does not exist")
    
    # Test numeric parameters
    try:
        cutoff = float(init_pair_filter_cutoff)
        if cutoff <= 0:
            errors.append(f"ERROR: init_pair_filter_cutoff must be positive, got {cutoff}")
    except (TypeError, ValueError):
        errors.append(f"ERROR: init_pair_filter_cutoff must be numeric, got {init_pair_filter_cutoff}")
    
    try:
        nt_val = int(nt)
        if nt_val <= 0:
            errors.append(f"ERROR: nt (threads) must be positive, got {nt_val}")
    except (TypeError, ValueError):
        errors.append(f"ERROR: nt must be an integer, got {nt}")
    
    # Test PEN parameters
    if pen_cutoffs:
        for i, cutoff in enumerate(pen_cutoffs):
            try:
                c = float(cutoff)
                if c <= 0:
                    errors.append(f"ERROR: PEN cutoff must be positive, got {c} at position {i}")
            except (TypeError, ValueError):
                errors.append(f"ERROR: PEN cutoff must be numeric, got {cutoff} at position {i}")
    
    # Test selections
    if source_sel or target_sel:
        try:
            from prody import parsePDB
            sys = parsePDB(pdb_file)
            
            if source_sel and source_sel != "all":
                try:
                    sel = sys.select(source_sel)
                    if sel is None:
                        errors.append(f"ERROR: source_sel '{source_sel}' selects no atoms")
                except Exception as e:
                    errors.append(f"ERROR: Invalid source_sel syntax: {str(e)}")
            
            if target_sel and target_sel != "all":
                try:
                    sel = sys.select(target_sel)
                    if sel is None:
                        errors.append(f"ERROR: target_sel '{target_sel}' selects no atoms")
                except Exception as e:
                    errors.append(f"ERROR: Invalid target_sel syntax: {str(e)}")
        except Exception as e:
            warnings.append(f"WARNING: Could not parse PDB for selection validation: {str(e)}")
    
    # Test logical combinations
    if traj and not top:
        errors.append("ERROR: Trajectory file provided but no topology file")
    
    if toppar and not top:
        warnings.append("WARNING: Toppar folder provided but no topology file")
    
    if nointeraction and create_pen:
        errors.append("ERROR: Cannot create PEN without calculating interactions (nointeraction=True)")
    
    # Test GROMACS functionality
    print("\nTesting GROMACS functionality...")
    gromacs_errors = test_gromacs_functionality(pdb_file, top, traj, ff_folder)
    errors.extend(gromacs_errors)
    
    # Print results
    print("="*60)
    print("gRINN Input Validation Report")
    print("="*60)
    
    if not errors and not warnings:
        print("✓ All inputs valid!")
    else:
        if errors:
            print(f"\n❌ Found {len(errors)} error(s):")
            for e in errors:
                print(f"  {e}")
        
        if warnings:
            print(f"\n⚠️  Found {len(warnings)} warning(s):")
            for w in warnings:
                print(f"  {w}")
    
    print("="*60)
    
    return len(errors) == 0, errors


def test_gromacs_functionality(pdb_file, top=None, traj=None, ff_folder=None):
    """
    Test if GROMACS can actually process the input files.
    
    Returns:
    - errors (list): List of GROMACS-related errors
    """
    errors = []
    temp_dir = None
    
    try:
        import tempfile
        import gromacs
        
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp(prefix="grinn_test_")
        
        # Test 1: Check if GROMACS is available
        try:
            result = gromacs.gmx_version()
            print(f"✓ GROMACS found: {result}")
        except Exception as e:
            errors.append(f"ERROR: GROMACS not found or not working: {str(e)}")
            return errors
        
        # Test 2: Test PDB file with gmx editconf (quick structure check)
        if pdb_file and os.path.exists(pdb_file):
            try:
                test_out = os.path.join(temp_dir, "test.pdb")
                gromacs.editconf(f=pdb_file, o=test_out, princ=True)
                print(f"✓ PDB file is readable by GROMACS")
            except Exception as e:
                errors.append(f"ERROR: GROMACS cannot read PDB file: {str(e)}")
        
        # Test 3: If topology provided, test with gmx grompp
        if top and os.path.exists(top):
            try:
                # Create a minimal mdp file for testing
                test_mdp = os.path.join(temp_dir, "test.mdp")
                with open(test_mdp, 'w') as f:
                    f.write("integrator = md\n")
                    f.write("nsteps = 0\n")
                    f.write("cutoff-scheme = Verlet\n")
                
                test_tpr = os.path.join(temp_dir, "test.tpr")
                # Copy PDB to temp dir to ensure paths work
                import shutil
                temp_pdb = os.path.join(temp_dir, "system.pdb")
                shutil.copy(pdb_file, temp_pdb)
                
                gromacs.grompp(f=test_mdp, c=temp_pdb, p=top, o=test_tpr, maxwarn=10)
                print(f"✓ Topology file is valid and compatible with PDB")
            except Exception as e:
                error_msg = str(e)
                if "atom name" in error_msg.lower():
                    errors.append(f"ERROR: Topology and PDB atom names don't match: {error_msg}")
                elif "residue" in error_msg.lower():
                    errors.append(f"ERROR: Topology and PDB residues don't match: {error_msg}")
                else:
                    errors.append(f"ERROR: GROMACS cannot process topology with PDB: {error_msg}")
        
        # Test 4: If trajectory provided, test with gmx check
        if traj and os.path.exists(traj):
            try:
                # Use gmx check to verify trajectory
                result = gromacs.check(f=traj)
                print(f"✓ Trajectory file is valid")
            except Exception as e:
                errors.append(f"ERROR: GROMACS cannot read trajectory file: {str(e)}")
        
        # Test 5: If custom force field provided, check if it has required files
        if ff_folder and os.path.exists(ff_folder):
            required_ff_files = ['forcefield.itp', 'aminoacids.rtp']
            missing_ff_files = []
            for ff_file in required_ff_files:
                if not os.path.exists(os.path.join(ff_folder, ff_file)):
                    missing_ff_files.append(ff_file)
            
            if missing_ff_files:
                errors.append(f"ERROR: Force field folder missing required files: {', '.join(missing_ff_files)}")
            else:
                print(f"✓ Force field folder contains required files")
        
    except ImportError:
        errors.append("ERROR: Could not import GROMACS - is it properly installed?")
    except Exception as e:
        errors.append(f"ERROR: Unexpected error during GROMACS testing: {str(e)}")
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
    
    return errors


def run_grinn_workflow(pdb_file, out_folder, ff_folder, init_pair_filter_cutoff, nofixpdb=False, top=False, toppar=False, 
                       traj=False, nointeraction=False, gpu=False, solvate=False, npt=False, source_sel="all", target_sel="all", 
                       nt=1, noconsole_handler=False, include_files=False, create_pen=False, pen_cutoffs=[1.0], 
                       pen_include_covalents=[True, False], test_only=False):
    
    # If test_only flag is set, just validate inputs and exit
    if test_only:
        is_valid, errors = test_grinn_inputs(
            pdb_file, out_folder, ff_folder, init_pair_filter_cutoff, nofixpdb, top, toppar, 
            traj, nointeraction, gpu, solvate, npt, source_sel, target_sel, nt, 
            noconsole_handler, include_files, create_pen, pen_cutoffs, pen_include_covalents
        )
        if not is_valid:
            print("\n❌ Workflow cannot proceed due to errors.")
            sys.exit(1)
        else:
            print("\n✓ All checks passed! Workflow can proceed.")
            sys.exit(0)
    
    start_time = time.time()  # Start the timer

    # Find the folder of the current script
    script_folder = os.path.dirname(os.path.realpath(__file__))
    # mdp_files_folder is the mdp_files folder in the script folder
    mdp_files_folder = os.path.join(script_folder, 'mdp_files')

    # If source_sel is None, set it to an appropriate selection
    if source_sel is None:
        source_sel = "not water and not resname SOL and not ion"

    # If target_sel is None, set it to an appropriate selection
    if target_sel is None:
        target_sel = "not water and not resname SOL and not ion"

    if type(source_sel) == list:
        if len(source_sel) > 1:
            source_sel = ' '.join(source_sel)
        else:
            source_sel = source_sel[0]

    if type(target_sel) == list:
        if len(target_sel) > 1:
            target_sel = ' '.join(target_sel)
        else:
            target_sel = target_sel[0]

    logger = create_logger(out_folder, noconsole_handler)
    logger.info('### gRINN workflow started ###')
    # Print the command-line used to call this workflow to the log file
    logger.info('gRINN workflow was called as follows: ')
    logger.info(' '.join(sys.argv))

    # If any include files are listed. 
    if include_files:
        logger.info('Include files provided. Copying include files to output folder...')
        for include_file in include_files:
            shutil.copy(include_file, os.path.join(out_folder, os.path.basename(include_file)))

    # If a force field folder is provided
    if ff_folder:
        logger.info('Force field folder provided. Using provided force field folder.')
        logger.info('Copying force field folder to output folder...')
        ff_folder_basename = os.path.basename(ff_folder)
        shutil.copytree(ff_folder, os.path.join(out_folder, ff_folder_basename), dirs_exist_ok=True)

    # Check whether a topology file as well as toppar folder is provided
    if top:
        logger.info('Topology file provided. Using provided topology file.')
        logger.info('Copying topology file to output folder...')
        shutil.copy(top, os.path.join(out_folder, 'topol_dry.top'))

        if toppar:
            logger.info('Toppar folder provided. Using provided toppar folder.')
            logger.info('Copying toppar folder to output folder...')
            shutil.copytree(toppar, os.path.join(out_folder, 'toppar'))

        logger.info('Copying input pdb_file to output_folder as "system.pdb"...')
        shutil.copy(pdb_file, os.path.join(out_folder, 'system_dry.pdb'))

        # Check whether also a trajectory file is provided
        if traj:
            logger.info('Trajectory file provided. Using provided trajectory file.')
            logger.info('Copying trajectory file to output folder...')
            shutil.copy(traj, os.path.join(out_folder, 'traj_dry.xtc'))
        else:
            logger.info('Generating traj.xtc file from input pdb_file...')
            gromacs.trjconv(f=os.path.join(out_folder, 'system_dry.pdb'), o=os.path.join(out_folder, 'traj_dry.xtc'))

    else:
        run_gromacs_simulation(pdb_file, mdp_files_folder, out_folder, ff_folder, nofixpdb, gpu, solvate, npt, logger, nt)

    if nointeraction:
        logger.info('Not calculating interaction energies as per user request.')
    else:
        initialFilter = perform_initial_filtering(out_folder, source_sel, target_sel, init_pair_filter_cutoff, 4, logger)
        edrFiles, pairsFilteredChunks = calculate_interaction_energies(out_folder, initialFilter, nt, logger)
        parse_interaction_energies(edrFiles, pairsFilteredChunks, out_folder, logger)

    # --- PEN analysis ---
    if create_pen:
        logger.info('Starting PEN (Protein Energy Network) analysis...')
        pen_csv = os.path.join(out_folder, 'energies_intEnTotal.csv')
        pdb_path = os.path.join(out_folder, 'system_dry.pdb')
        if os.path.exists(pen_csv) and os.path.exists(pdb_path):
            compute_pen_and_bc(
                pdb_file=pdb_path,
                int_en_csv=pen_csv,
                out_folder=out_folder,
                intEnCutoff_values=pen_cutoffs,
                include_covalents_options=pen_include_covalents,
                logger=logger,
                source_sel=source_sel,
                target_sel=target_sel
            )
        else:
            logger.warning("PEN input files not found, skipping PEN analysis.")

    cleanUp(out_folder, logger)
    elapsed_time = time.time() - start_time  # Calculate the elapsed time    
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    logger.info('Elapsed time: {:.2f} seconds'.format(elapsed_time))
    logger.info('### gRINN workflow completed successfully ###')
    # Clear handlers to avoid memory leak
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Run gRINN workflow")
    parser.add_argument("pdb_file", type=str, help="Input PDB file")
    parser.add_argument("out_folder", type=str, help="Output folder")
    parser.add_argument("--nofixpdb", action="store_true", help="Fix PDB file using pdbfixer")
    parser.add_argument("--initpairfiltercutoff", type=float, default=10, help="Initial pair filter cutoff (default is 10)")
    parser.add_argument("--nointeraction", action="store_true", help="Do not calculate interaction energies")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for non-bonded interactions in GROMACS commands")
    parser.add_argument("--solvate", action="store_true", help="Run solvation")
    parser.add_argument("--npt", action="store_true", help="Run NPT equilibration")
    parser.add_argument("--source_sel", nargs="+", type=str, help="Source selection")
    parser.add_argument("--target_sel", nargs="+", type=str, help="Target selection")
    parser.add_argument("--nt", type=int, default=1, help="Number of threads for GROMACS commands (default is 1)")
    parser.add_argument("--noconsole_handler", action="store_true", help="Do not add console handler to the logger")
    parser.add_argument("--ff_folder", type=str, help="Folder containing the force field files")
    parser.add_argument('--top', type=str, help='Topology file')
    parser.add_argument('--toppar', type=str, help='Toppar folder')
    parser.add_argument('--traj', type=str, help='Trajectory file')
    parser.add_argument('--include_files', nargs='+', type=str, help='Include files')
    # PEN-specific arguments
    parser.add_argument('--create_pen', action='store_true', help='Create Protein Energy Networks (PENs) and calculate betweenness centralities')
    parser.add_argument('--pen_cutoffs', nargs='+', type=float, default=[1.0], help='List of intEnCutoff values for PEN construction')
    parser.add_argument('--pen_include_covalents', nargs='+', type=lambda x: (str(x).lower() == 'true'), default=[True, False], help='Whether to include covalent bonds in PENs (True/False, can be multiple)')
    # Add test-only flag
    parser.add_argument("--test-only", action="store_true", 
                       help="Only test input validity and GROMACS compatibility without running the workflow")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(0)
    return parser.parse_args()

def main():
    args = parse_args()
    run_grinn_workflow(
        args.pdb_file, args.out_folder, args.ff_folder, args.initpairfiltercutoff, 
        args.nofixpdb, args.top, args.toppar, args.traj, args.nointeraction, 
        args.gpu, args.solvate, args.npt, args.source_sel, args.target_sel, 
        args.nt, args.noconsole_handler, args.include_files,
        create_pen=args.create_pen,
        pen_cutoffs=args.pen_cutoffs,
        pen_include_covalents=args.pen_include_covalents,
        test_only=getattr(args, 'test_only', False)  # Use getattr for compatibility
    )

if __name__ == "__main__":
    def global_signal_handler(sig, frame):
            print('Signal caught in main. Exiting...')
            sys.exit(0)

    signal.signal(signal.SIGINT, global_signal_handler)
    signal.signal(signal.SIGTERM, global_signal_handler)
    main()
