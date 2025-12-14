# Standard library imports
import argparse
import concurrent.futures
import contextlib
import gc
import glob
import io
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from itertools import islice

# Third-party imports
import mdtraj as md
import networkx as nx
import numpy as np
import pandas as pd
import panedr
import pyprind
import tqdm
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from prody import *
from prody import LOGGER, parsePDB, writePDB, calcCenter, calcDistance
from scipy.sparse import lil_matrix

# GromacsWrapper is imported here since source_gmxrc must be run first
import gromacs
import gromacs.environment

# Optional imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Suppress pandas dtype warnings and other noisy warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*dtype.*')
warnings.filterwarnings('ignore', message='.*DtypeWarning.*')
warnings.filterwarnings('ignore', message='.*ParserWarning.*')
warnings.filterwarnings('ignore', message='.*PerformanceWarning.*')

pd.set_option('display.max_info_columns', 0)
pd.set_option('display.max_info_rows', 0)

# Global variable to store the process group ID
pgid = os.getpgid(os.getpid())


def get_trajectory_frame_count(traj_path, stop_after=None):
    """Return number of frames in a trajectory file.

    If `stop_after` is provided, stops early once the count reaches or exceeds it.
    This avoids scanning the whole file when only enforcing an upper limit.
    """
    if not traj_path:
        raise ValueError("Trajectory path is required")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(traj_path)

    ext = os.path.splitext(traj_path)[1].lower()
    chunk_size = 500
    n_frames = 0

    if ext == '.xtc':
        if not hasattr(md, 'formats') or not hasattr(md.formats, 'XTCTrajectoryFile'):
            raise ImportError("mdtraj XTCTrajectoryFile reader is not available")

        with md.formats.XTCTrajectoryFile(traj_path, mode='r') as f:
            while True:
                try:
                    xyz, *_rest = f.read(chunk_size)
                except EOFError:
                    break

                if xyz is None or not hasattr(xyz, 'shape') or xyz.shape[0] == 0:
                    break

                n_frames += int(xyz.shape[0])
                if stop_after is not None and n_frames >= stop_after:
                    return n_frames

        return n_frames

    if ext == '.trr':
        if not hasattr(md, 'formats') or not hasattr(md.formats, 'TRRTrajectoryFile'):
            raise ImportError("mdtraj TRRTrajectoryFile reader is not available")

        with md.formats.TRRTrajectoryFile(traj_path, mode='r') as f:
            while True:
                try:
                    result = f.read(chunk_size)
                except EOFError:
                    break

                if not result:
                    break

                xyz = result[0]
                if xyz is None or not hasattr(xyz, 'shape') or xyz.shape[0] == 0:
                    break

                n_frames += int(xyz.shape[0])
                if stop_after is not None and n_frames >= stop_after:
                    return n_frames

        return n_frames

    # Fallback: try mdtraj generic loader (may be memory-heavy)
    traj = md.load(traj_path)
    n_frames = int(getattr(traj, 'n_frames', 0))
    return n_frames

def parse_topology_includes(topology_file, logger=None):
    """
    Parse a GROMACS topology file to find all #include statements.
    
    Parameters:
    - topology_file (str): Path to the topology file
    - logger: Optional logger for output messages
    
    Returns:
    - list: List of included file paths found in the topology
    """
    include_files = []
    
    if not os.path.exists(topology_file):
        if logger:
            logger.warning(f"Topology file not found: {topology_file}")
        return include_files
    
    try:
        topology_dir = os.path.dirname(os.path.abspath(topology_file))
        
        with open(topology_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith(';') or not line:
                    continue
                
                # Look for #include statements
                if line.startswith('#include'):
                    # Extract the filename from #include "filename" or #include <filename>
                    if '"' in line:
                        # Format: #include "filename"
                        start = line.find('"') + 1
                        end = line.rfind('"')
                        if start > 0 and end > start:
                            include_path = line[start:end]
                    elif '<' in line and '>' in line:
                        # Format: #include <filename>  
                        start = line.find('<') + 1
                        end = line.rfind('>')
                        if start > 0 and end > start:
                            include_path = line[start:end]
                    else:
                        continue
                    
                    # Convert relative paths to absolute paths
                    if not os.path.isabs(include_path):
                        # Normalize path by removing leading ./ or ../
                        include_path = os.path.normpath(include_path)
                        # Remove leading ./ if present
                        if include_path.startswith('./'):
                            include_path = include_path[2:]
                        include_path = os.path.join(topology_dir, include_path)
                    
                    include_files.append(include_path)
        
        if logger:
            logger.info(f"Found {len(include_files)} include files in topology")
            for inc_file in include_files:
                logger.debug(f"  Include file: {inc_file}")
        
        return include_files
        
    except Exception as e:
        if logger:
            logger.error(f"Error parsing topology file {topology_file}: {str(e)}")
        return include_files

def find_gromacs_data_directories(gromacs_dir=None, logger=None):
    """
    Find GROMACS data directories containing force field files.
    
    Parameters:
    - gromacs_dir (str): Optional path to GROMACS installation directory
    - logger: Optional logger for output messages
    
    Returns:
    - list: List of directories where GROMACS force field data might be located
    """
    potential_dirs = []
    
    # 1. Use user-provided GROMACS directory if available
    if gromacs_dir and os.path.exists(gromacs_dir):
        if logger:
            logger.debug(f"Using user-provided GROMACS directory: {gromacs_dir}")
        # Check common subdirectories where force fields are stored
        for subdir in ['share/gromacs/top', 'share/top', 'top', 'data/top']:
            potential_dir = os.path.join(gromacs_dir, subdir)
            if os.path.exists(potential_dir):
                potential_dirs.append(potential_dir)
                if logger:
                    logger.debug(f"Found GROMACS data directory: {potential_dir}")
    
    # 2. Check GROMACS_DIR environment variable (most reliable)
    gromacs_env_dir = os.environ.get('GROMACS_DIR')
    if gromacs_env_dir and os.path.exists(gromacs_env_dir):
        if logger:
            logger.debug(f"Found GROMACS_DIR environment variable: {gromacs_env_dir}")
        # Check common subdirectories where force fields are stored
        for subdir in ['share/gromacs/top', 'share/top', 'top', 'data/top']:
            potential_dir = os.path.join(gromacs_env_dir, subdir)
            if os.path.exists(potential_dir):
                potential_dirs.append(potential_dir)
                if logger:
                    logger.debug(f"Found GROMACS data directory from GROMACS_DIR: {potential_dir}")
    
    # 3. Check GMXDATA environment variable
    gmxdata = os.environ.get('GMXDATA')
    if gmxdata and os.path.exists(gmxdata):
        potential_dirs.append(gmxdata)
        if logger:
            logger.debug(f"Found GROMACS data directory from GMXDATA: {gmxdata}")
    
    # 4. Try to get GROMACS installation directory from gmx command
    try:
        result = subprocess.run(['gmx', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Look for data directory in gmx output
            for line in result.stdout.split('\n'):
                if 'Data prefix:' in line:
                    data_dir = line.split('Data prefix:')[1].strip()
                    if os.path.exists(data_dir):
                        potential_dirs.append(data_dir)
                        if logger:
                            logger.debug(f"Found GROMACS data directory from gmx --version: {data_dir}")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        if logger:
            logger.debug("Could not run 'gmx --version' to detect GROMACS installation")
    
    # 5. Check common GROMACS installation locations as fallback
    gromacs_base_paths = [
        '/usr/local/gromacs',      # Standard installation
        '/opt/gromacs',            # Host-mounted GROMACS
        '/usr/share/gromacs',      # System package installation
        '/opt/conda/share/gromacs', # Conda installation
    ]
    
    for base_path in gromacs_base_paths:
        # Common subdirectories where force fields are stored
        for subdir in ['share/gromacs/top', 'share/top', 'top', 'data/top']:
            potential_dir = os.path.join(base_path, subdir)
            if os.path.exists(potential_dir):
                potential_dirs.append(potential_dir)
                if logger:
                    logger.debug(f"Found GROMACS data directory: {potential_dir}")
    
    # Remove duplicates while preserving order
    unique_dirs = []
    for d in potential_dirs:
        if d not in unique_dirs:
            unique_dirs.append(d)
    
    if logger and unique_dirs:
        logger.info(f"Found {len(unique_dirs)} GROMACS data directories")
    elif logger:
        logger.warning("No GROMACS data directories found")
    
    return unique_dirs

def find_file_in_gromacs_data(filename, gromacs_data_dirs=None, gromacs_dir=None, logger=None):
    """
    Search for a file in GROMACS data directories.
    
    Parameters:
    - filename (str): The filename to search for (can be a relative path like 'amber99sb-ildn.ff/forcefield.itp')
    - gromacs_data_dirs (list): Optional list of GROMACS data directories to search
    - gromacs_dir (str): Optional path to GROMACS installation directory
    - logger: Optional logger for output messages
    
    Returns:
    - str: Full path to the file if found, None otherwise
    """
    if gromacs_data_dirs is None:
        gromacs_data_dirs = find_gromacs_data_directories(gromacs_dir, logger)
    
    for data_dir in gromacs_data_dirs:
        potential_path = os.path.join(data_dir, filename)
        if os.path.exists(potential_path):
            if logger:
                logger.debug(f"Found {filename} in GROMACS data: {potential_path}")
            return potential_path
    
    return None

def auto_detect_and_copy_topology_dependencies(topology_file, out_folder, gromacs_dir=None, logger=None):
    """
    Automatically detect and copy include files and toppar dependencies from a topology file.
    
    Parameters:
    - topology_file (str): Path to the topology file
    - out_folder (str): Output directory where dependencies will be copied
    - gromacs_dir (str): Optional path to GROMACS installation directory
    - logger: Optional logger for output messages
    
    Returns:
    - bool: True if successful, False if critical errors occurred
    
    Raises:
    - RuntimeError: If critical files cannot be found or copied
    """
    if not topology_file or not os.path.exists(topology_file):
        if logger:
            logger.info("No topology file provided or topology file doesn't exist - skipping dependency detection")
        return True
    
    try:
        if logger:
            logger.info("Automatically detecting topology file dependencies...")
        
        topology_dir = os.path.dirname(os.path.abspath(topology_file))
        include_files = parse_topology_includes(topology_file, logger)
        
        if not include_files:
            if logger:
                logger.info("No include files found in topology file")
            return True
        
        # Find GROMACS data directories once for efficiency
        gromacs_data_dirs = find_gromacs_data_directories(gromacs_dir, logger)
        
        # Categorize files into include files and potential toppar directory
        actual_include_files = []
        potential_toppar_dirs = set()
        gromacs_provided_files = []
        
        for include_path in include_files:
            if os.path.exists(include_path):
                actual_include_files.append(include_path)
                
                # Check if this file is in a potential toppar directory
                include_dir = os.path.dirname(include_path)
                if include_dir != topology_dir:
                    potential_toppar_dirs.add(include_dir)
            else:
                # File doesn't exist at absolute path - try relative to topology directory
                rel_path = os.path.join(topology_dir, os.path.basename(include_path))
                if os.path.exists(rel_path):
                    actual_include_files.append(rel_path)
                else:
                    # Check if this file exists in GROMACS data directories
                    # Extract the relative path from the original include_path
                    if topology_dir in include_path:
                        # Get the relative part after topology_dir
                        relative_include = os.path.relpath(include_path, topology_dir)
                    else:
                        # Use the basename or relative path as-is
                        relative_include = os.path.basename(include_path)
                        # If it looks like a force field path, try to preserve directory structure
                        if '/' in include_path and ('.ff/' in include_path or include_path.endswith('.itp')):
                            # Extract force field directory and file pattern
                            path_parts = include_path.split('/')
                            for i, part in enumerate(path_parts):
                                if part.endswith('.ff'):
                                    relative_include = '/'.join(path_parts[i:])
                                    break
                    
                    gromacs_file_path = find_file_in_gromacs_data(relative_include, gromacs_data_dirs, gromacs_dir, logger)
                    
                    if gromacs_file_path:
                        gromacs_provided_files.append((include_path, gromacs_file_path, relative_include))
                        if logger:
                            logger.info(f"Include file found in GROMACS installation: {relative_include}")
                    else:
                        error_msg = f"Required include file not found: {include_path}"
                        if logger:
                            logger.error(error_msg)
                            logger.error(f"  Searched in topology directory: {topology_dir}")
                            logger.error(f"  Searched in GROMACS data directories: {gromacs_data_dirs}")
                            if gromacs_dir:
                                logger.error(f"  Used GROMACS directory: {gromacs_dir}")
                            if os.environ.get('GROMACS_DIR'):
                                logger.error(f"  GROMACS_DIR environment variable: {os.environ.get('GROMACS_DIR')}")
                        raise RuntimeError(error_msg)
        
        # Copy individual include files that are local to the topology
        if actual_include_files:
            if logger:
                logger.info(f"Copying {len(actual_include_files)} local include files to output directory...")
            
            for include_file in actual_include_files:
                try:
                    dest_path = os.path.join(out_folder, os.path.basename(include_file))
                    shutil.copy2(include_file, dest_path)
                    if logger:
                        logger.debug(f"  Copied: {os.path.basename(include_file)}")
                except Exception as e:
                    error_msg = f"Failed to copy include file {include_file}: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                    raise RuntimeError(error_msg)
        
        # Report files found in GROMACS installation (no need to copy)
        if gromacs_provided_files:
            if logger:
                logger.info(f"Found {len(gromacs_provided_files)} include files in GROMACS installation (no need to copy):")
                for orig_path, gromacs_path, rel_path in gromacs_provided_files:
                    logger.info(f"  {rel_path} -> {gromacs_path}")
        
        # Handle potential toppar directories (only for local files)
        if potential_toppar_dirs:
            if logger:
                logger.info(f"Found {len(potential_toppar_dirs)} potential local toppar directories")
            
            for toppar_dir in potential_toppar_dirs:
                # Skip directories that are part of GROMACS installation
                is_gromacs_dir = False
                for gromacs_data_dir in gromacs_data_dirs:
                    if toppar_dir.startswith(gromacs_data_dir):
                        is_gromacs_dir = True
                        if logger:
                            logger.info(f"Skipping GROMACS system directory: {toppar_dir}")
                        break
                
                if is_gromacs_dir:
                    continue
                
                try:
                    toppar_basename = os.path.basename(toppar_dir)
                    dest_toppar = os.path.join(out_folder, toppar_basename)
                    
                    if os.path.exists(dest_toppar):
                        if logger:
                            logger.info(f"Toppar directory already exists: {toppar_basename}")
                        continue
                    
                    if logger:
                        logger.info(f"Copying local toppar directory: {toppar_basename}")
                    
                    shutil.copytree(toppar_dir, dest_toppar, dirs_exist_ok=True)
                    
                except Exception as e:
                    error_msg = f"Failed to copy toppar directory {toppar_dir}: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                    raise RuntimeError(error_msg)
        
        if logger:
            logger.info("Topology dependencies successfully detected and processed")
        
        return True
        
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = f"Unexpected error during topology dependency detection: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)

def recreate_topology_file(structure_file, out_folder, force_field, water_model, ff_folder=None, logger=None):
    """
    Recreate topology file using pdb2gmx from structure file.
    
    Parameters:
    - structure_file (str): Path to the structure file (PDB or GRO)
    - out_folder (str): Output directory for topology files
    - force_field (str): Force field to use
    - water_model (str): Water model to use
    - ff_folder (str): Optional custom force field folder
    - logger: Optional logger for output messages
    
    Returns:
    - bool: True if successful, False otherwise
    """
    try:
        if logger:
            logger.info("Recreating topology file using pdb2gmx...")
            logger.info(f"  Structure file: {structure_file}")
            logger.info(f"  Force field: {force_field}")
            logger.info(f"  Water model: {water_model}")
            if ff_folder:
                logger.info(f"  Custom FF folder: {ff_folder}")
        
        # Prepare force field parameter
        if ff_folder:
            ff_param = ff_folder
        else:
            ff_param = force_field
        
        # Output files
        output_pdb = os.path.join(out_folder, "protein_processed.pdb")
        output_top = os.path.join(out_folder, "topol_dry.top")
        output_itp = os.path.join(out_folder, "posre.itp")
        
        # Set GROMACS environment flags for file output
        gromacs.environment.flags['capture_output'] = "file"
        gromacs.environment.flags['capture_output_filename'] = os.path.join(out_folder, "pdb2gmx_topology_recreation.log")
        
        # Run pdb2gmx to recreate topology
        try:
            gromacs.pdb2gmx(
                f=structure_file,
                o=output_pdb,
                p=output_top,
                i=output_itp,
                ff=ff_param,
                water=water_model,
                heavyh=True,
                ignh=True
            )
            
            if os.path.exists(output_top):
                if logger:
                    logger.info("Topology file successfully recreated")
                    logger.info(f"  Generated topology: {output_top}")
                    logger.info(f"  Processed structure: {output_pdb}")
                return True
            else:
                if logger:
                    logger.error("pdb2gmx completed but topology file was not created")
                return False
                
        except Exception as gmx_error:
            if logger:
                logger.error(f"pdb2gmx failed: {str(gmx_error)}")
                logger.error("This could be due to:")
                logger.error("  1. Incorrect force field specification")
                logger.error("  2. Incompatible structure format")
                logger.error("  3. Missing residues or atoms in the structure")
                logger.error("  4. Unsupported residue types")
            return False
            
    except Exception as e:
        if logger:
            logger.error(f"Error in topology recreation: {str(e)}")
        return False

def handle_missing_topology_flexible(structure_file, out_folder, traj=None, top=None, 
                                   force_field='amber99sb-ildn', water_model='tip3p', ff_folder=None, 
                                   recreate_topology=False, logger=None):
    """
    Flexibly handle missing topology files by recreating topology from structure.
    
    This function handles various scenarios:
    1. Structure + trajectory available -> recreate topology with specified parameters
    2. Existing topology but user wants to recreate it
    
    Parameters:
    - structure_file (str): Path to structure file
    - out_folder (str): Output directory
    - traj (str): Optional trajectory file path
    - top (str): Optional existing topology file path
    - force_field (str): Force field to use for recreation
    - water_model (str): Water model to use for recreation
    - ff_folder (str): Optional custom force field folder
    - recreate_topology (bool): Force recreation even if topology exists
    - logger: Optional logger for output messages
    
    Returns:
    - dict: Status information about the topology handling
    """
    result = {
        'topology_available': False,
        'topology_created': False,
        'method_used': None
    }
    
    try:
        topology_path = os.path.join(out_folder, 'topol_dry.top')
        
        # Check if we need to handle missing topology
        has_existing_topology = top or os.path.exists(topology_path)
        
        if has_existing_topology and not recreate_topology:
            if logger:
                logger.info("Topology file is available and recreation not requested")
            result['topology_available'] = True
            result['method_used'] = 'existing'
            return result
        
        # Structure file available - recreate topology with provided parameters
        if structure_file and os.path.exists(structure_file):
            if logger:
                logger.info("Structure file available - recreating topology with specified parameters")
                logger.info(f"Using force field: {force_field}")
                logger.info(f"Using water model: {water_model}")
            
            if recreate_topology_file(structure_file, out_folder, force_field, water_model, ff_folder, logger):
                result['topology_created'] = True
                result['topology_available'] = True
                result['method_used'] = 'structure_recreation'
                return result
        
        # If we get here, topology creation failed
        if logger:
            logger.error("Failed to create topology file")
            logger.error("Available options:")
            logger.error("  1. Provide a topology file directly with --top")
            logger.error("  2. Use --ensemble_mode for multi-model PDB files")
            logger.error("  3. Ensure structure file is compatible with pdb2gmx")
            logger.error("  4. Check force field and water model specifications")
        
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Error in flexible topology handling: {str(e)}")
        return result


def parse_structure_file(filepath):
    """
    Parse structure file (PDB or GRO format) using ProDy.
    For GRO files, converts to PDB using gmx editconf first.
    
    Parameters:
    - filepath (str): Path to the structure file (.pdb or .gro)
    
    Returns:
    - ProDy AtomGroup: Parsed structure
    
    Raises:
    - ValueError: If file format is not supported or parsing fails
    """
    try:
        # Determine file format from extension
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.pdb':
            return parsePDB(filepath)
        elif file_ext == '.gro':
            # Convert GRO to PDB using gmx editconf
            
            # Create temporary PDB file
            temp_pdb = filepath.replace('.gro', '_temp.pdb')
            
            try:
                # Use gmx editconf to convert GRO to PDB
                gromacs.editconf(f=filepath, o=temp_pdb)
                
                # Parse the converted PDB file
                structure = parsePDB(temp_pdb)
                
                # Clean up temporary file
                if os.path.exists(temp_pdb):
                    os.remove(temp_pdb)
                
                return structure
                
            except Exception as e:
                # Clean up temporary file if it exists
                if os.path.exists(temp_pdb):
                    os.remove(temp_pdb)
                raise ValueError(f"Failed to convert GRO file to PDB: {str(e)}")
        else:
            # Try to determine format by attempting to parse
            try:
                # First try PDB (more common)
                return parsePDB(filepath)
            except:
                # For unknown extensions, try GRO conversion
                try:
                    
                    # Create temporary PDB file
                    temp_pdb = filepath + '_temp.pdb'
                    
                    # Use gmx editconf to convert to PDB
                    gromacs.editconf(f=filepath, o=temp_pdb)
                    
                    # Parse the converted PDB file
                    structure = parsePDB(temp_pdb)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_pdb):
                        os.remove(temp_pdb)
                    
                    return structure
                except:
                    raise ValueError(f"Unsupported file format: {filepath}. Supported formats: .pdb, .gro")
    except Exception as e:
        raise ValueError(f"Failed to parse structure file {filepath}: {str(e)}")

# Simplified pandas-only EDR processing functions
def parse_edr_file_parallel(edr_file):
    """
    Simple helper function for parallel EDR parsing using pandas only
    """
    try:
        # Check if EDR file exists and is not empty
        if not os.path.exists(edr_file):
            return None
        
        if os.path.getsize(edr_file) == 0:
            return None
        
        # Parse EDR file to DataFrame with suppressed output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = panedr.edr_to_df(edr_file)
        
        if df.empty:
            return None
        
        return df
            
    except Exception as e:
        # Return None on error
        return None

def combine_dataframes_memory_efficient(df_list, outFolder, logger):
    """
    Memory-efficient combination of EDR DataFrames using pandas only
    """
    logger.info('Combining parsed EDR files using memory-efficient pandas approach...')
    
    # Filter out None/empty results
    valid_dfs = [df for df in df_list if df is not None and not df.empty]
    
    if not valid_dfs:
        logger.error("No valid DataFrames to combine!")
        return pd.DataFrame()
    
    logger.info(f'Found {len(valid_dfs)} valid DataFrames to combine')
    
    try:
        # Use the original efficient combination method
        df = valid_dfs[0]
        
        for i, df_pair in enumerate(valid_dfs[1:], 1):
            # Remove already parsed columns to avoid duplication
            df_pair_columns = set(df_pair.columns)
            df_columns = set(df.columns)
            new_columns = list(df_pair_columns - df_columns)
            
            if new_columns:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    df = pd.concat([df, df_pair[new_columns]], axis=1)
            
            logger.info(f'Combined {i + 1} out of {len(valid_dfs)} EDR files...')
            
            # Clean up intermediate DataFrame
            del df_pair
            gc.collect()
        
        logger.info(f'Final combination completed with shape {df.shape}')
        return df
        
    except Exception as e:
        logger.error(f'Error in combine_dataframes_memory_efficient: {str(e)}')
        return pd.DataFrame()

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

def getRibeiroOrtizNetwork(structure_file, df_intEn, includeCovalents=True, intEnCutoff=1, startFrame=0, residue_indices=None):
    sys = parse_structure_file(structure_file)
    if residue_indices is None:
        sys_sel = sys.select("all")
        resIndices = np.unique(sys_sel.getResindices())
    else:
        resIndices = np.array(sorted(residue_indices))
        sys_sel = sys.select(' or '.join([f"resindex {i}" for i in resIndices]))
    numResidues = len(resIndices)
    
    # ✅ FIX: Get residue information correctly by using unique residue indices
    # Create arrays for residue names, numbers, and chains
    resNames = []
    resNums = []
    chains = []
    
    for res_idx in resIndices:
        # Get the first atom of this residue to extract residue information
        atoms_in_residue = sys.select(f"resindex {res_idx}")
        if atoms_in_residue is not None and len(atoms_in_residue) > 0:
            resNames.append(atoms_in_residue.getResnames()[0])
            resNums.append(atoms_in_residue.getResnums()[0])
            chains.append(atoms_in_residue.getChids()[0])
        else:
            # Fallback in case of issues
            resNames.append("UNK")
            resNums.append(0)
            chains.append("A")
    
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
    structure_file, 
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
    sys = parse_structure_file(structure_file)
    source_indices = set(sys.select(source_sel).getResindices())
    target_indices = set(sys.select(target_sel).getResindices())
    residue_indices = sorted(source_indices | target_indices)

    all_bc_results = []
    for include_covalents in include_covalents_options:
        for intEnCutoff in intEnCutoff_values:
            logger and logger.info(f"Creating PENs: include_covalents={include_covalents}, intEnCutoff={intEnCutoff}")
            nx_list = getRibeiroOrtizNetwork(
                structure_file, df_intEn, 
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

def run_gromacs_simulation(structure_filepath, mdp_files_folder, out_folder, ff_folder, nofixpdb, gpu, solvate, npt, logger, nt=1, skip=1, force_field='amber99sb-ildn', water_model='tip3p'):
    """
    Run a GROMACS simulation workflow.

    Parameters:
    - structure_filepath (str): The path to the input structure file (PDB or GRO format).
    - mdp_files_folder (str): The folder containing the MDP files.
    - out_folder (str): The folder where output files will be saved.
    - nofixpdb (bool): Whether to fix the PDB file using pdbfixer (default is True).
    - logger (logging.Logger): The logger object for logging messages.
    - nt (int): Number of threads for GROMACS commands (default is 1).
    - skip (int): Skip every nth frame in trajectory output (default is 1, no skipping).
    - ff_folder (str): The folder containing the force field files (default is None).

    Returns:
    - None
    """

    gromacs.environment.flags['capture_output'] = "file"
    gromacs.environment.flags['capture_output_filename'] = os.path.join(out_folder, "gromacs.log")

    logger.info(f"Running GROMACS simulation for structure file: {structure_filepath}")

    # Determine input file format and prepare for GROMACS
    input_ext = os.path.splitext(structure_filepath)[1].lower()
    
    if nofixpdb or input_ext == '.gro':
        # For GRO files or when PDB fixing is disabled
        if input_ext == '.gro':
            logger.info('Converting GRO file to PDB format for GROMACS processing...')
            structure = parse_structure_file(structure_filepath)
            fixed_pdb_filepath = os.path.join(out_folder, "protein.pdb")
            writePDB(fixed_pdb_filepath, structure.select('protein'))
            logger.info("GRO file converted to PDB format.")
        else:
            # PDB file without fixing
            logger.info('Using PDB file without fixing...')
            structure = parse_structure_file(structure_filepath)
            fixed_pdb_filepath = os.path.join(out_folder, "protein.pdb")
            writePDB(fixed_pdb_filepath, structure.select('protein'))
            logger.info("PDB file processed without fixing.")
    else:
        # Fix PDB file using PDBFixer
        logger.info('Fixing PDB file using PDBFixer...')
        fixer = PDBFixer(filename=structure_filepath)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        pdb_filename = os.path.basename(structure_filepath)
        fixed_pdb_filepath = os.path.join(out_folder, "protein.pdb")
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdb_filepath, 'w'))
        logger.info("PDB file fixed.")
        system = parse_structure_file(fixed_pdb_filepath)
        writePDB(fixed_pdb_filepath, system.select('protein'))

    if ff_folder is not None:
        ff = ff_folder
        logger.info(f"Using custom force field folder: {ff_folder}")
    else:
        ff = force_field
        logger.info(f"Using force field: {force_field}")
    
    logger.info(f"Using water model: {water_model}")

    if gpu:
        gpu="gpu"
    else:
        gpu="cpu"

    # Run GROMACS commands
    try:
        gromacs.pdb2gmx(f=fixed_pdb_filepath, o=os.path.join(out_folder, "protein.pdb"), 
                        p=os.path.join(out_folder, "topol.top"), i=os.path.join(out_folder,"posre.itp"),
                          ff=ff, water=water_model, heavyh=True, ignh=True)
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
            gromacs.trjconv(s=os.path.join(out_folder, 'npt.tpr'), f=os.path.join(out_folder, 'npt.xtc'), o=os.path.join(out_folder, 'traj.xtc'), skip=skip, input=(index_group_name,))
            logger.info(f"trjconv for NPT to XTC conversion command completed (skipping every {skip} frames).")
            next_pdb = "npt.pdb"

        gromacs.trjconv(f=os.path.join(out_folder, next_pdb), o=os.path.join(out_folder, 'system_dry.pdb'), s=os.path.join(out_folder, next_pdb), n=os.path.join(out_folder, 'index.ndx'), input=(index_group_name,))
        logger.info(f"trjconv for {next_pdb} to DRY PDB conversion command completed.")
        
        # Detect and assign chain IDs if missing (after system_dry.pdb is created)
        topology_file = os.path.join(out_folder, 'topol_dry.top')
        try:
            # Pass the original input format to help with chain ID detection
            original_format = 'gro' if input_ext == '.gro' else 'pdb'
            detect_and_assign_chain_ids(os.path.join(out_folder, 'system_dry.pdb'), topology_file, logger, original_format)
        except ValueError as e:
            logger.warning(f"Chain ID assignment failed: {str(e)}")
            logger.warning("Proceeding with existing chain IDs in PDB file. Analysis may be less accurate for multi-chain systems.")
        
        gromacs.trjconv(f=os.path.join(out_folder, 'traj.xtc'), o=os.path.join(out_folder, 'traj_dry.xtc'), s=os.path.join(out_folder, 'system_dry.pdb'), 
                        n=os.path.join(out_folder, 'index.ndx'), skip=skip, input=(index_group_name,))
        logger.info(f"trjconv for traj.xtc to traj_dry.xtc conversion command completed (skipping every {skip} frames).")

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

def detect_and_assign_chain_ids(pdb_file, topology_file=None, logger=None, original_format=None):
    """
    Detect missing chain IDs in PDB file and assign them based on topology or sequence breaks.
    
    Parameters:
    - pdb_file (str): Path to the PDB file to check and potentially modify
    - topology_file (str): Optional path to topology file for chain information
    - logger: Optional logger for output messages
    - original_format (str): Original input file format ('gro', 'pdb', or None for auto-detect)
    
    Returns:
    - bool: True if chain IDs were modified, False if no changes needed
    """
    if logger:
        logger.info("Checking chain ID assignment in PDB file...")
    
    try:
        # Parse the PDB file
        system = parsePDB(pdb_file)
        
        # Check if chain IDs are present and meaningful
        chain_ids = system.getChids()
        unique_chains = np.unique(chain_ids)
        
        # If original input was a GRO file, ignore any chain IDs from editconf conversion
        if original_format and original_format.lower() == 'gro':
            if logger:
                logger.info("Original input was GRO file - ignoring chain IDs from editconf conversion")
                logger.info("GRO files don't contain chain information, so any chain IDs are editconf artifacts")
            chains_missing = True
            chains_likely_corrupted = False
        else:
            # For PDB inputs or unknown formats, check if chains are missing or corrupted
            chains_missing = False
            chains_likely_corrupted = False
            
            # Standard case: missing or blank chain IDs
            if (len(unique_chains) == 1 and (
                unique_chains[0] == '' or 
                unique_chains[0] == ' ' or 
                unique_chains[0] is None
            )):
                chains_missing = True
            
            # GRO conversion issue: single-character chain IDs that are likely residue name spillovers
            elif len(unique_chains) <= 3:  # Few unique chains
                residue_names = system.getResnames()
                unique_residues = np.unique(residue_names)
                
                # Check if any chain IDs match characters from long residue names
                for chain_id in unique_chains:
                    if chain_id and len(chain_id.strip()) == 1:
                        chain_char = chain_id.strip()
                        # Check if this character appears in any residue name at position 3 or 4
                        for res_name in unique_residues:
                            if len(res_name) > 3 and (
                                (len(res_name) >= 4 and res_name[3] == chain_char) or
                                (len(res_name) >= 5 and res_name[4] == chain_char)
                            ):
                                chains_likely_corrupted = True
                                if logger:
                                    logger.warning(f"Detected likely corrupted chain ID '{chain_char}' from residue name '{res_name}'")
                                break
                    if chains_likely_corrupted:
                        break
        
        if not chains_missing and not chains_likely_corrupted:
            if logger:
                logger.info(f"Valid chain IDs detected: {list(unique_chains)}")
            return False
        
        if chains_missing:
            if logger:
                if original_format and original_format.lower() == 'gro':
                    logger.info("Assigning chain IDs for structure converted from GRO format...")
                else:
                    logger.info("No meaningful chain IDs detected. Attempting to assign chain IDs...")
        elif chains_likely_corrupted:
            if logger:
                logger.info("Corrupted chain IDs detected (likely from GRO->PDB conversion). Reassigning chain IDs...")
        # Method 1: Try to extract chain information from topology file
        if topology_file and os.path.exists(topology_file):
            try:
                chain_assignments = extract_chains_from_topology(topology_file, system, logger)
                if chain_assignments:
                    apply_chain_assignments(system, chain_assignments, pdb_file, logger)
                    return True
            except Exception as e:
                if logger:
                    logger.warning(f"Could not extract chain info from topology: {str(e)}")
        
        # Method 2: Assign chains based on sequence breaks and molecule types
        chain_assignments = detect_chains_from_sequence_breaks(system, logger)
        if chain_assignments:
            apply_chain_assignments(system, chain_assignments, pdb_file, logger)
            return True
        
        # No reliable method worked - return error
        error_msg = "Could not reliably assign chain IDs. Please provide a PDB file with proper chain IDs or ensure topology file contains molecule information."
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
        
    except Exception as e:
        if logger:
            logger.error(f"Error in chain ID detection/assignment: {str(e)}")
        return False

def extract_chains_from_topology(topology_file, system, logger=None):
    """
    Extract chain information from GROMACS topology file by analyzing included .itp files
    and mapping residue names to molecule types.
    
    This function:
    1. Parses #include statements to find .itp files
    2. Extracts residue names from each .itp file  
    3. Maps molecules to chain IDs based on the [ molecules ] section
    4. Assigns chain IDs by matching residues to their corresponding molecule types
    
    Parameters:
    - topology_file (str): Path to the GROMACS topology file
    - system (prody.AtomGroup): Parsed PDB/GRO structure
    - logger: Optional logger for output messages
    
    Returns:
    - dict: Mapping of residue indices to chain IDs, or None if extraction fails
    """
    try:
        if logger:
            logger.info("Analyzing topology file to extract chain information...")
        
        # Parse topology file to get included .itp files and molecule information
        topology_info = parse_topology_comprehensive(topology_file, logger)
        if not topology_info:
            if logger:
                logger.warning("Could not parse comprehensive topology information")
            return None
        
        # Get all residues from the system
        all_residues = system.select('all')
        all_resindices = np.unique(all_residues.getResindices())
        sorted_resindices = sorted(all_resindices)
        
        # Map residues to molecule types based on residue names
        residue_to_molecule = map_residues_by_names(
            system, sorted_resindices, topology_info, logger
        )
        
        if not residue_to_molecule:
            if logger:
                logger.warning("Could not map residues to molecules using residue names")
            return None
        
        # Assign chain IDs based on molecule assignments
        chain_assignments = assign_chain_ids_from_molecule_mapping(
            residue_to_molecule, topology_info['molecules'], logger
        )
        
        if logger and chain_assignments:
            unique_chains = list(set(chain_assignments.values()))
            logger.info(f"Successfully assigned {len(unique_chains)} chains from topology: {', '.join(sorted(unique_chains))}")
            
            # Log molecule type distribution
            molecule_counts = {}
            for res_idx, chain_id in chain_assignments.items():
                mol_name = residue_to_molecule.get(res_idx, 'Unknown')
                if mol_name not in molecule_counts:
                    molecule_counts[mol_name] = {'chain': chain_id, 'count': 0}
                molecule_counts[mol_name]['count'] += 1
            
            for mol_name, info in molecule_counts.items():
                logger.info(f"  Molecule '{mol_name}' -> Chain {info['chain']}: {info['count']} residues")
        
        return chain_assignments if chain_assignments else None
        
    except Exception as e:
        if logger:
            logger.warning(f"Topology-based chain assignment failed: {str(e)}")
        return None

def parse_topology_comprehensive(topology_file, logger=None):
    """
    Comprehensively parse topology file to extract:
    1. Included .itp files and their molecule names and residue names
    2. Molecule definitions from [ molecules ] section
    
    Returns:
    - dict: Contains 'itp_residues' (mol_name -> set of residue names) and 'molecules' (ordered list)
    """
    try:
        topology_dir = os.path.dirname(topology_file)
        
        with open(topology_file, 'r') as f:
            top_content = f.read()
        
        # Extract #include statements to find .itp files
        itp_file_paths = extract_itp_includes(top_content, topology_dir, logger)
        
        # Parse each .itp file to get molecule names and their residue names
        itp_residues = {}
        for itp_path in itp_file_paths:
            itp_info = parse_itp_file_comprehensive(itp_path, logger)
            if itp_info:
                mol_name = itp_info['molecule_name']
                residue_names = itp_info['residue_names']
                itp_residues[mol_name] = residue_names
                if logger:
                    logger.debug(f"Molecule '{mol_name}' contains residues: {', '.join(sorted(residue_names))}")
        
        # Parse [ molecules ] section to get molecule order and counts
        molecules = parse_molecules_section(top_content, logger)
        
        if not itp_residues or not molecules:
            if logger:
                logger.warning("Could not extract complete topology information")
            return None
        
        return {
            'itp_residues': itp_residues,  # mol_name -> set of residue names
            'molecules': molecules          # [(mol_name, count), ...]
        }
        
    except Exception as e:
        if logger:
            logger.warning(f"Error parsing topology comprehensively: {str(e)}")
        return None

def extract_itp_includes(top_content, topology_dir, logger=None):
    """
    Extract #include statements from topology file to find .itp files.
    
    Returns:
    - list: List of .itp file paths (without inferring molecule names from filenames)
    """
    itp_files = []
    
    try:
        for line in top_content.split('\n'):
            line = line.strip()
            
            # Look for #include statements with .itp files
            if line.startswith('#include') and '.itp' in line:
                # Extract the include path
                if '"' in line:
                    # Format: #include "path/file.itp"
                    include_path = line.split('"')[1]
                elif '<' in line and '>' in line:
                    # Format: #include <path/file.itp>  
                    include_path = line.split('<')[1].split('>')[0]
                else:
                    continue
                
                # Construct full path
                if os.path.isabs(include_path):
                    full_path = include_path
                else:
                    full_path = os.path.join(topology_dir, include_path)
                
                if os.path.exists(full_path):
                    itp_files.append(full_path)
                    if logger:
                        logger.debug(f"Found .itp file: {include_path}")
                else:
                    if logger:
                        logger.debug(f"Referenced .itp file not found: {include_path}")
        
        return itp_files
        
    except Exception as e:
        if logger:
            logger.warning(f"Error extracting .itp includes: {str(e)}")
        return []

def parse_itp_file_comprehensive(itp_path, logger=None):
    """
    Parse an .itp file to extract both molecule name and residue names.
    
    Parameters:
    - itp_path (str): Path to the .itp file
    - logger: Optional logger for output messages
    
    Returns:
    - dict: Contains 'molecule_name' and 'residue_names' (set), or None if parsing fails
    """
    try:
        with open(itp_path, 'r') as f:
            content = f.read()
        
        molecule_name = None
        residue_names = set()
        in_moleculetype_section = False
        in_atoms_section = False
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check for [ moleculetype ] section
            if line.startswith('[ moleculetype ]'):
                in_moleculetype_section = True
                in_atoms_section = False
                continue
            elif line.startswith('[ atoms ]'):
                in_moleculetype_section = False
                in_atoms_section = True
                continue
            elif line.startswith('['):
                # Entering a different section
                in_moleculetype_section = False
                in_atoms_section = False
                continue
            
            # Parse molecule name from [ moleculetype ] section
            if in_moleculetype_section and line and not line.startswith(';'):
                # Format: molecule_name    nrexcl
                # Example: PROA         3
                parts = line.split()
                if len(parts) >= 1 and not parts[0].startswith(';'):
                    molecule_name = parts[0]
                    if logger:
                        logger.debug(f"Found molecule name '{molecule_name}' in {os.path.basename(itp_path)}")
            
            # Parse residue names from [ atoms ] section
            elif in_atoms_section and line and not line.startswith(';'):
                # Format: nr type resnr residu atom cgnr charge mass
                # Example: 1    NH3    1    LYS    N    1    -0.300000    14.0070
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        residue_name = parts[3]  # 4th column is residue name
                        residue_names.add(residue_name)
                    except (IndexError, ValueError):
                        continue
        
        if molecule_name and residue_names:
            if logger:
                logger.debug(f"Molecule '{molecule_name}' contains {len(residue_names)} unique residue types: {', '.join(sorted(residue_names))}")
            return {
                'molecule_name': molecule_name,
                'residue_names': residue_names
            }
        else:
            if logger:
                logger.debug(f"Could not extract complete information from {itp_path}: molecule_name={molecule_name}, residues={len(residue_names)}")
            return None
        
    except Exception as e:
        if logger:
            logger.debug(f"Error parsing .itp file {itp_path}: {str(e)}")
        return None

def parse_molecules_section(top_content, logger=None):
    """
    Parse the [ molecules ] section to get molecule order and counts.
    
    Returns:
    - list: List of tuples (molecule_name, count) in order
    """
    molecules = []
    
    try:
        in_molecules_section = False
        
        for line in top_content.split('\n'):
            line = line.strip()
            
            if line.startswith('[ molecules ]'):
                in_molecules_section = True
                continue
            elif line.startswith('[') and in_molecules_section:
                # End of molecules section
                break
            elif in_molecules_section and line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mol_name = parts[0]
                        mol_count = int(parts[1])
                        molecules.append((mol_name, mol_count))
                    except (ValueError, IndexError):
                        continue
        
        if logger:
            logger.debug(f"Found {len(molecules)} molecule types in [ molecules ] section")
        
        return molecules
        
    except Exception as e:
        if logger:
            logger.warning(f"Error parsing [ molecules ] section: {str(e)}")
        return []

def map_residues_by_names(system, sorted_resindices, topology_info, logger=None):
    """
    Map residues to molecule types by matching residue names from the structure
    to residue names found in .itp files.
    
    Returns:
    - dict: Mapping of residue index to molecule name
    """
    residue_to_molecule = {}
    itp_residues = topology_info['itp_residues']
    molecules_order = topology_info['molecules']
    
    try:
        # Get residue names from the structure
        structure_residues = {}  # res_idx -> residue_name
        for res_idx in sorted_resindices:
            residue = system.select(f'resindex {res_idx}')
            if residue is not None and len(residue) > 0:
                res_name = residue.getResnames()[0]
                structure_residues[res_idx] = res_name
        
        if logger:
            unique_res_names = set(structure_residues.values())
            logger.debug(f"Structure contains {len(unique_res_names)} unique residue types: {', '.join(sorted(unique_res_names))}")
        
        # Match residues to molecules based on residue names
        # Process molecules in the order they appear in [ molecules ] section
        assigned_residues = set()
        
        for mol_name, mol_count in molecules_order:
            if mol_name not in itp_residues:
                if logger:
                    logger.warning(f"No .itp residue information found for molecule '{mol_name}'")
                continue
            
            mol_residue_names = itp_residues[mol_name]
            
            # Find all structure residues that match this molecule type
            matching_residues = []
            for res_idx, res_name in structure_residues.items():
                if res_idx not in assigned_residues and res_name in mol_residue_names:
                    matching_residues.append(res_idx)
            
            # Sort matching residues to maintain order
            matching_residues.sort()
            
            # Assign the expected number of residues per molecule instance
            residues_per_instance = len(matching_residues) // mol_count if mol_count > 0 else len(matching_residues)
            
            if residues_per_instance * mol_count != len(matching_residues):
                if logger:
                    logger.warning(f"Molecule '{mol_name}': expected {mol_count} instances, but residue count ({len(matching_residues)}) doesn't divide evenly")
                    logger.warning(f"Using {residues_per_instance} residues per instance")
            
            # Assign residues to molecule instances
            for instance in range(mol_count):
                start_idx = instance * residues_per_instance
                end_idx = start_idx + residues_per_instance
                
                for res_idx in matching_residues[start_idx:end_idx]:
                    residue_to_molecule[res_idx] = mol_name
                    assigned_residues.add(res_idx)
            
            if logger:
                assigned_count = min(len(matching_residues), residues_per_instance * mol_count)
                logger.debug(f"Assigned {assigned_count} residues to molecule '{mol_name}' ({mol_count} instances)")
        
        # Handle any unassigned residues
        unassigned = set(sorted_resindices) - assigned_residues
        if unassigned:
            if logger:
                unassigned_names = [structure_residues.get(idx, 'Unknown') for idx in sorted(unassigned)]
                logger.warning(f"Could not assign {len(unassigned)} residues to molecules: {', '.join(set(unassigned_names))}")
            
            # Assign unassigned residues to 'Other'
            for res_idx in unassigned:
                residue_to_molecule[res_idx] = 'Other'
        
        return residue_to_molecule
        
    except Exception as e:
        if logger:
            logger.warning(f"Error mapping residues by names: {str(e)}")
        return {}

def assign_chain_ids_from_molecule_mapping(residue_to_molecule, molecules_order, logger=None):
    """
    Assign chain IDs based on molecule mapping, respecting molecule instance order.
    
    Returns:
    - dict: Mapping of residue index to chain ID
    """
    chain_assignments = {}
    
    try:
        # Create chain ID generator
        chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        chain_idx = 0
        
        # Group residues by molecule type and assign chains per instance
        for mol_name, mol_count in molecules_order:
            # Get all residues for this molecule type
            mol_residues = [res_idx for res_idx, mol in residue_to_molecule.items() if mol == mol_name]
            mol_residues.sort()
            
            if not mol_residues:
                continue
            
            # Calculate residues per instance
            residues_per_instance = len(mol_residues) // mol_count if mol_count > 0 else len(mol_residues)
            
            # Assign chain IDs to each instance
            for instance in range(mol_count):
                if chain_idx < len(chain_letters):
                    chain_id = chain_letters[chain_idx]
                else:
                    # Use double letters for chains beyond Z
                    first_letter_idx = (chain_idx - 26) // 26
                    second_letter_idx = (chain_idx - 26) % 26
                    chain_id = chain_letters[first_letter_idx] + chain_letters[second_letter_idx]
                
                # Assign this chain ID to residues in this instance
                start_idx = instance * residues_per_instance
                end_idx = start_idx + residues_per_instance
                
                for res_idx in mol_residues[start_idx:end_idx]:
                    chain_assignments[res_idx] = chain_id
                
                chain_idx += 1
        
        # Handle 'Other' residues (assign to separate chain)
        other_residues = [res_idx for res_idx, mol in residue_to_molecule.items() if mol == 'Other']
        if other_residues:
            if chain_idx < len(chain_letters):
                other_chain_id = chain_letters[chain_idx]
            else:
                first_letter_idx = (chain_idx - 26) // 26
                second_letter_idx = (chain_idx - 26) % 26
                other_chain_id = chain_letters[first_letter_idx] + chain_letters[second_letter_idx]
            
            for res_idx in other_residues:
                chain_assignments[res_idx] = other_chain_id
        
        return chain_assignments
        
    except Exception as e:
        if logger:
            logger.warning(f"Error assigning chain IDs from molecule mapping: {str(e)}")
        return {}

def parse_topology_molecules(content, logger=None):
    """
    Parse the [molecules] section and gather information about each molecule type.
    
    Returns:
    - list: List of tuples (molecule_name, count, estimated_residues)
    """
    molecules_info = []
    
    # Find molecules section
    molecules_section = False
    for line in content.split('\n'):
        line = line.strip()
        
        if line.startswith('[ molecules ]'):
            molecules_section = True
            continue
        elif line.startswith('[') and molecules_section:
            break
        elif molecules_section and line and not line.startswith(';'):
            parts = line.split()
            if len(parts) >= 2:
                mol_name = parts[0]
                mol_count = int(parts[1])
                
                # Try to estimate number of residues per molecule from topology
                estimated_residues = estimate_molecule_residues(content, mol_name, logger)
                
                molecules_info.append((mol_name, mol_count, estimated_residues))
    
    return molecules_info

def estimate_molecule_residues(content, mol_name, logger=None):
    """
    Estimate the number of residues in a molecule by looking for its definition
    in the topology file.
    
    Returns:
    - int: Estimated number of residues (defaults to 1 for unknown molecules)
    """
    try:
        # Look for molecule definition section
        in_molecule_section = False
        residue_count = 0
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check if we're entering the molecule definition
            if line == f'[ moleculetype ]':
                in_molecule_section = 'search_name'
                continue
            elif in_molecule_section == 'search_name':
                # Next non-comment line should have molecule name
                if not line.startswith(';') and line:
                    parts = line.split()
                    if parts and parts[0] == mol_name:
                        in_molecule_section = 'in_molecule'
                    else:
                        in_molecule_section = False
                continue
            elif in_molecule_section == 'in_molecule':
                # Look for atoms section to count residues
                if line.startswith('[ atoms ]'):
                    in_molecule_section = 'counting_atoms'
                    continue
                elif line.startswith('['):
                    # Entered a different section, stop counting
                    break
            elif in_molecule_section == 'counting_atoms':
                if line.startswith('['):
                    # End of atoms section
                    break
                elif line and not line.startswith(';'):
                    # This is an atom line, extract residue info
                    parts = line.split()
                    if len(parts) >= 4:  # atom_nr, atom_type, residue_nr, residue_name
                        try:
                            res_nr = int(parts[2])
                            residue_count = max(residue_count, res_nr)
                        except (ValueError, IndexError):
                            continue
        
        return max(1, residue_count)  # At least 1 residue
        
    except Exception as e:
        if logger:
            logger.debug(f"Could not estimate residues for molecule {mol_name}: {str(e)}")
        return 1  # Default fallback

def map_residues_to_molecules(system, molecule_info, all_resindices, logger=None):
    """
    Map residues to molecules based on topology order and molecule information.
    
    Returns:
    - dict: Mapping of residue index to molecule name
    """
    residue_to_molecule = {}
    current_residue_idx = 0
    
    try:
        # Sort residue indices to ensure proper ordering
        sorted_resindices = sorted(all_resindices)
        
        for mol_name, mol_count, estimated_residues_per_mol in molecule_info:
            
            for mol_instance in range(mol_count):
                # Assign residues for this molecule instance
                for res_in_mol in range(estimated_residues_per_mol):
                    if current_residue_idx < len(sorted_resindices):
                        res_idx = sorted_resindices[current_residue_idx]
                        residue_to_molecule[res_idx] = mol_name
                        current_residue_idx += 1
                    else:
                        # We've run out of residues - this can happen if our estimation is off
                        if logger:
                            logger.warning(f"Ran out of residues while assigning molecule {mol_name}")
                        break
                
                # If we couldn't assign all estimated residues, continue to next molecule
                if current_residue_idx >= len(sorted_resindices):
                    break
        
        # Handle any remaining residues (assign to last molecule type or create a generic assignment)
        if current_residue_idx < len(sorted_resindices):
            if logger:
                remaining = len(sorted_resindices) - current_residue_idx
                logger.warning(f"Topology mapping left {remaining} residues unassigned - assigning to 'Other'")
            
            for i in range(current_residue_idx, len(sorted_resindices)):
                res_idx = sorted_resindices[i]
                residue_to_molecule[res_idx] = 'Other'
        
        return residue_to_molecule
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to map residues to molecules: {str(e)}")
        return {}


def is_water_molecule(mol_name):
    """Check if molecule name indicates water."""
    water_names = {'sol', 'water', 'wat', 'h2o', 'tip3', 'tip4', 'tip5', 'spc', 'spce'}
    return mol_name.lower() in water_names

def is_ion_molecule(mol_name):
    """Check if molecule name indicates an ion."""
    ion_names = {'na', 'cl', 'k', 'mg', 'ca', 'zn', 'fe', 'na+', 'cl-', 'k+', 'mg2+', 'ca2+', 'zn2+'}
    return mol_name.lower() in ion_names

def detect_chains_from_sequence_breaks(system, logger=None):
    """
    Detect chain breaks based on sequence gaps and molecule types.
    Only returns assignments when confident about chain boundaries.
    
    Returns:
    - dict: Mapping of residue indices to chain IDs, or None if not confident
    """
    try:
        chain_assignments = {}
        
        # Get all residues
        all_residues = system.select('all')
        residue_indices = np.unique(all_residues.getResindices())
        residue_numbers = []
        residue_names = []
        
        for res_idx in residue_indices:
            res_sel = system.select(f'resindex {res_idx}')
            residue_numbers.append(res_sel.getResnums()[0])
            residue_names.append(res_sel.getResnames()[0])
        
        # Count different molecule types to determine if we can confidently assign chains
        protein_count = sum(1 for name in residue_names if is_protein_residue(name))
        nucleic_count = sum(1 for name in residue_names if is_nucleic_residue(name))
        
        # Enhanced water/solvent detection for potentially truncated names from GRO conversion
        water_count = sum(1 for name in residue_names if is_water_residue(name))
        ion_count = sum(1 for name in residue_names if is_ion_residue(name))
        
        other_count = len(residue_names) - protein_count - nucleic_count - water_count - ion_count
        
        # Only proceed if we have clear molecular boundaries or sequence breaks
        has_clear_boundaries = (protein_count > 0 and (nucleic_count > 0 or water_count > 0 or ion_count > 0 or other_count > 0))
        
        # Check for clear sequence breaks in protein/nucleic sequences
        sequence_breaks = []
        current_chain = 'A'
        chain_counter = 0
        
        for i, res_idx in enumerate(residue_indices):
            res_name = residue_names[i]
            res_num = residue_numbers[i]
            
            # Check if this should be a new chain
            should_start_new_chain = False
            
            if i > 0:
                prev_res_num = residue_numbers[i-1]
                prev_res_name = residue_names[i-1]
                
                # Check for significant sequence break (gap > 5 in residue numbering for proteins/nucleics)
                if (is_protein_residue(res_name) or is_nucleic_residue(res_name)) and \
                   (is_protein_residue(prev_res_name) or is_nucleic_residue(prev_res_name)):
                    if res_num - prev_res_num > 5:
                        should_start_new_chain = True
                        sequence_breaks.append(i)
                
                # Check for molecule type change (strong indicator)
                if (is_protein_residue(prev_res_name) != is_protein_residue(res_name) or
                    is_nucleic_residue(prev_res_name) != is_nucleic_residue(res_name)):
                    should_start_new_chain = True
            
            if should_start_new_chain:
                chain_counter += 1
                current_chain = chr(ord('A') + chain_counter % 26)
            
            # Assign chains based on molecule type
            if is_water_residue(res_name):
                chain_assignments[res_idx] = 'W'  # Water chain
            elif is_ion_residue(res_name):
                chain_assignments[res_idx] = 'I'  # Ion chain
            else:
                chain_assignments[res_idx] = current_chain
        
        # Only return assignments if we have confidence
        if has_clear_boundaries or len(sequence_breaks) > 0:
            if logger:
                logger.info(f"Detected chain boundaries: {len(sequence_breaks)} sequence breaks, molecule types: "
                           f"protein={protein_count}, nucleic={nucleic_count}, water={water_count}, ions={ion_count}, other={other_count}")
            return chain_assignments
        else:
            if logger:
                logger.info("No clear chain boundaries detected - cannot confidently assign chains")
            return None
        
    except Exception as e:
        if logger:
            logger.warning(f"Sequence break detection failed: {str(e)}")
        return None

def is_water_residue(res_name):
    """Check if residue is water, including potentially truncated names from GRO conversion."""
    water_names = {
        'SOL', 'WAT', 'H2O',  # Standard water names
        'TIP3', 'TIP4', 'TIP5', 'TIP',  # TIP water models (including truncated)
        'SPC', 'SPCE',  # SPC water models
        'OPC', 'OPC3'   # OPC water models
    }
    return res_name.upper() in water_names

def is_ion_residue(res_name):
    """Check if residue is an ion, including common ion names."""
    ion_names = {
        'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE',  # Common ions
        'NA+', 'CL-', 'K+', 'MG2+', 'CA2+', 'ZN2+',  # With charges
        'SOD', 'CLA', 'POT', 'MAG', 'CAL'  # Alternative names
    }
    return res_name.upper() in ion_names

def is_protein_residue(res_name):
    """Check if residue is a standard protein residue."""
    protein_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP'  # Histidine variants
    }
    return res_name in protein_residues

def is_nucleic_residue(res_name):
    """Check if residue is a nucleic acid residue."""
    nucleic_residues = {
        'A', 'T', 'G', 'C', 'U',  # Single letter
        'DA', 'DT', 'DG', 'DC',   # DNA
        'RA', 'RU', 'RG', 'RC',   # RNA
        'ADE', 'THY', 'GUA', 'CYT', 'URA'  # Full names
    }
    return res_name in nucleic_residues

def apply_chain_assignments(system, chain_assignments, pdb_file, logger=None):
    """
    Apply chain ID assignments to the system and save to file.
    Preserves CRYST1 and other header lines that are crucial for GROMACS.
    
    Parameters:
    - system: ProDy system object
    - chain_assignments: dict mapping residue indices to chain IDs
    - pdb_file: path to save the modified PDB
    - logger: optional logger
    """
    try:
        # Read the original PDB file to preserve header lines (especially CRYST1)
        header_lines = []
        footer_lines = []
        
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
            
        # Extract header lines (everything before first ATOM/HETATM)
        # and footer lines (everything after last ATOM/HETATM)
        atom_start_idx = None
        atom_end_idx = None
        
        for i, line in enumerate(lines):
            if line.startswith(('ATOM', 'HETATM')) and atom_start_idx is None:
                atom_start_idx = i
            if line.startswith(('ATOM', 'HETATM')):
                atom_end_idx = i
        
        if atom_start_idx is not None:
            header_lines = lines[:atom_start_idx]
            if atom_end_idx is not None and atom_end_idx < len(lines) - 1:
                footer_lines = lines[atom_end_idx + 1:]
        
        # Create new chain ID array
        new_chain_ids = system.getChids().copy()
        
        # Apply assignments
        for res_idx, chain_id in chain_assignments.items():
            res_sel = system.select(f'resindex {res_idx}')
            if res_sel:
                atom_indices = res_sel.getIndices()
                for atom_idx in atom_indices:
                    new_chain_ids[atom_idx] = chain_id
        
        # Set new chain IDs
        system.setChids(new_chain_ids)
        
        # Write to temporary file first (use basename without .pdb extension to avoid double extension)
        temp_base = pdb_file.replace('.pdb', '') + '_tmp'
        temp_pdb = temp_base + '.pdb'
        writePDB(temp_base, system)  # writePDB will add .pdb extension
        
        # Now combine header + modified structure + footer
        with open(temp_pdb, 'r') as f:
            structure_lines = f.readlines()
        
        # Write final file with preserved headers/footers
        with open(pdb_file, 'w') as f:
            # Write header lines (including CRYST1)
            f.writelines(header_lines)
            
            # Write structure lines (ATOM/HETATM records with new chain IDs)
            for line in structure_lines:
                if line.startswith(('ATOM', 'HETATM')):
                    f.write(line)
            
            # Write footer lines (END, etc.)
            f.writelines(footer_lines)
        
        # Clean up temporary file (remove the actual file created by writePDB)
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)
        
        # Log results with clear warning
        unique_chains = np.unique(new_chain_ids)
        unique_chains = [c for c in unique_chains if c.strip()]  # Remove empty chains
        
        if logger:
            logger.warning("⚠️  CHAIN ID ASSIGNMENT PERFORMED BY gRINN")
            logger.warning(f"   The input PDB file lacked chain IDs. gRINN has automatically assigned chains.")
            logger.warning(f"   Detected and assigned {len(unique_chains)} chains: {', '.join(unique_chains)}")
            logger.warning(f"   Please verify that chain assignments are correct for your analysis.")
            logger.warning(f"   CRYST1 and other header information have been preserved.")
            
            # Count residues per chain
            for chain_id in unique_chains:
                chain_residues = system.select(f'chain {chain_id}')
                if chain_residues:
                    n_residues = len(np.unique(chain_residues.getResindices()))
                    logger.info(f"  Chain {chain_id}: {n_residues} residues")
        
    except Exception as e:
        if logger:
            logger.error(f"Error applying chain assignments: {str(e)}")
        raise
        
    except Exception as e:
        if logger:
            logger.error(f"Error applying chain assignments: {str(e)}")
        raise

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

    # Validate selections
    if source is None or len(source) == 0:
        logger.error('Source selection "%s" returned no atoms. Please check your selection string.' % source_sel)
        logger.info('Total residues in system: %d' % numResidues)
        raise ValueError('Source selection returned no atoms: %s' % source_sel)
    
    if target is None or len(target) == 0:
        logger.error('Target selection "%s" returned no atoms. Please check your selection string.' % target_sel)
        logger.info('Total residues in system: %d' % numResidues)
        raise ValueError('Target selection returned no atoms: %s' % target_sel)

    sourceResids = np.unique(source.getResindices())
    numSource = len(sourceResids)

    targetResids = np.unique(target.getResindices())
    numTarget = len(targetResids)
    
    logger.info('Source selection: %s (%d residues)' % (source_sel, numSource))
    logger.info('Target selection: %s (%d residues)' % (target_sel, numTarget))

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

    # Check if pairSet is empty - no pairs to filter
    if len(pairSet) == 0:
        logger.warning('No residue pairs found between source and target selections. Check your selections.')
        logger.info('Source selection: %s (found %d residues)' % (source_sel, numSource))
        logger.info('Target selection: %s (found %d residues)' % (target_sel, numTarget))
        initialFilterPickle = os.path.join(os.path.abspath(outFolder), "initialFilter.pkl")
        with open(initialFilterPickle, 'wb') as f:
            pickle.dump(initialFilter, f)
        return initialFilter

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

    # Use the topology file
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
    xtc_file = os.path.join(outFolder, 'traj_dry.xtc')
    top_file = os.path.join(outFolder, 'topol_dry.top')
    
    if not os.path.exists(top_file):
        raise ValueError("Topology file required for detailed interaction energy calculation")

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
                    logger.info('Signal caught. Terminating ongoing tasks.')

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                j = 0
                for future in concurrent.futures.as_completed(futures):
                    edrFile, chunk = future.result()
                    edrFiles.append(edrFile)
                    pairsFilteredChunksProcessed.append(chunk)
                    logger.info('Completed energy calculation for chunk %d' % j)
                    j += 1

            return edrFiles, pairsFilteredChunksProcessed
        
        edrFiles, pairsFilteredChunksProcessed = parallel_process_chunks(pairsFilteredChunks, outFolder, top_file, pdb_file, xtc_file, numCoresIE, logger)
        return edrFiles, pairsFilteredChunksProcessed

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

def find_energy_columns(columns_set, pair):
    """Find energy columns for a pair efficiently"""
    p0, p1 = pair[0], pair[1]
    possible_cols = {
        'LJSR': (f'LJ-SR:res{p0}-res{p1}', f'LJ-SR:res{p1}-res{p0}'),
        'LJ14': (f'LJ-14:res{p0}-res{p1}', f'LJ-14:res{p1}-res{p0}'),
        'CoulSR': (f'Coul-SR:res{p0}-res{p1}', f'Coul-SR:res{p1}-res{p0}'),
        'Coul14': (f'Coul-14:res{p0}-res{p1}', f'Coul-14:res{p1}-res{p0}')
    }
    
    found_cols = {}
    for energy_type, (col1, col2) in possible_cols.items():
        if col1 in columns_set:
            found_cols[energy_type] = col1
        elif col2 in columns_set:
            found_cols[energy_type] = col2
    
    return found_cols if len(found_cols) == 4 else None

def extract_energies_vectorized(df, pair_cols, kj2kcal):
    """Extract energies using vectorized operations"""
    # Get energy arrays
    enLJSR = df[pair_cols['LJSR']].values * kj2kcal
    enLJ14 = df[pair_cols['LJ14']].values * kj2kcal
    enCoulSR = df[pair_cols['CoulSR']].values * kj2kcal
    enCoul14 = df[pair_cols['Coul14']].values * kj2kcal
    
    # Calculate totals vectorized
    enLJ = enLJSR + enLJ14
    enCoul = enCoulSR + enCoul14
    enTotal = enLJ + enCoul
    
    return {
        'VdW': enLJ.tolist(),
        'Elec': enCoul.tolist(),
        'Total': enTotal.tolist()
    }

def collect_energies_from_dataframe(df, pairsFilteredChunks, logger):
    """
    Collect energies from a DataFrame using vectorized operations
    """
    energiesDict = {}
    kj2kcal = 0.239005736
    
    # Flatten all pairs
    all_pairs = []
    for chunk in pairsFilteredChunks:
        all_pairs.extend(chunk)
    
    total_pairs = len(all_pairs)
    logger.info(f'Collecting energies for {total_pairs} residue pairs using vectorized approach...')
    
    # Get available columns once
    columns_set = set(df.columns)
    
    # Process pairs in dynamic batches based on available cores
    available_cores = multiprocessing.cpu_count()
    batch_size = max(20, min(200, int(available_cores * 10)))  # Scale with cores
    processed = 0
    
    for i in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i + batch_size]
        
        # Find valid pairs in this batch
        valid_pairs = []
        for pair in batch_pairs:
            pair_cols = find_energy_columns(columns_set, pair)
            if pair_cols:
                valid_pairs.append((pair, pair_cols))
        
        if not valid_pairs:
            continue
        
        # Process valid pairs in batch
        for pair, pair_cols in valid_pairs:
            try:
                energyDict = extract_energies_vectorized(df, pair_cols, kj2kcal)
                key = f"{pair[0]}-{pair[1]}"
                energiesDict[key] = energyDict
                processed += 1
            except Exception as e:
                logger.warning(f'Error processing pair {pair}: {str(e)}')
                continue
        
        # Progress update
        if (i + batch_size) % (batch_size * 10) == 0:  # Update every 10 batches
            progress_pct = (min(i + batch_size, len(all_pairs)) / len(all_pairs)) * 100
            logger.info(f'Energy collection progress: {progress_pct:.1f}% ({processed} successful extractions)')
    
    success_rate = (processed / total_pairs) * 100 if total_pairs > 0 else 0
    logger.info(f'Vectorized energy collection completed: {processed}/{total_pairs} pairs extracted successfully ({success_rate:.1f}% success rate)')
    
    return energiesDict

def supplement_df_optimized(df, system):
    """Optimized version of supplement_df with caching"""
    # Cache system selections
    selection_cache = {}
    
    def get_cached_selection(resindex):
        if resindex not in selection_cache:
            selection_cache[resindex] = system.select(f'resindex {resindex}')
        return selection_cache[resindex]
    
    # Extract indices
    df['res1_index'] = df['Pair_indices'].str.split('-').str[0].astype(int)
    df['res2_index'] = df['Pair_indices'].str.split('-').str[1].astype(int)
    
    # Get unique residue indices to minimize selections
    unique_indices = pd.concat([df['res1_index'], df['res2_index']]).unique()
    
    # Pre-populate cache
    residue_info = {}
    for idx in unique_indices:
        sel = get_cached_selection(idx)
        residue_info[idx] = {
            'chain': sel.getChids()[0],
            'resnum': sel.getResnums()[0],
            'resname': sel.getResnames()[0]
        }
    
    # Apply cached info vectorized
    df['res1_chain'] = df['res1_index'].map(lambda x: residue_info[x]['chain'])
    df['res2_chain'] = df['res2_index'].map(lambda x: residue_info[x]['chain'])
    df['res1_resnum'] = df['res1_index'].map(lambda x: residue_info[x]['resnum'])
    df['res2_resnum'] = df['res2_index'].map(lambda x: residue_info[x]['resnum'])
    df['res1_resname'] = df['res1_index'].map(lambda x: residue_info[x]['resname'])
    df['res2_resname'] = df['res2_index'].map(lambda x: residue_info[x]['resname'])
    
    # Create composite columns
    df['res1'] = df['res1_resname'] + df['res1_resnum'].astype(str) + '_' + df['res1_chain']
    df['res2'] = df['res2_resname'] + df['res2_resnum'].astype(str) + '_' + df['res2_chain']
    
    return df

def create_result_dataframes_optimized(energiesDict, logger):
    """Create result DataFrames more efficiently"""
    logger.info('Creating result DataFrames...')
    
    # Pre-allocate lists for better performance
    pairs = list(energiesDict.keys())
    n_pairs = len(pairs)
    
    if n_pairs == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Get number of frames from first entry
    n_frames = len(next(iter(energiesDict.values()))['Total'])
    
    # Create DataFrames directly without transpose
    total_data = np.zeros((n_pairs, n_frames))
    elec_data = np.zeros((n_pairs, n_frames))
    vdw_data = np.zeros((n_pairs, n_frames))
    
    for i, (pair, energies) in enumerate(energiesDict.items()):
        total_data[i] = energies['Total']
        elec_data[i] = energies['Elec']
        vdw_data[i] = energies['VdW']
    
    # Create DataFrames with proper column names (using integers like original)
    frame_cols = list(range(n_frames))
    
    df_total = pd.DataFrame(total_data, index=pairs, columns=frame_cols)
    df_elec = pd.DataFrame(elec_data, index=pairs, columns=frame_cols)
    df_vdw = pd.DataFrame(vdw_data, index=pairs, columns=frame_cols)
    
    # Reset index to make pairs a column
    df_total = df_total.reset_index().rename(columns={'index': 'Pair_indices'})
    df_elec = df_elec.reset_index().rename(columns={'index': 'Pair_indices'})
    df_vdw = df_vdw.reset_index().rename(columns={'index': 'Pair_indices'})
    
    return df_total, df_elec, df_vdw

def compute_and_save_average_energies(energiesDict, system, outFolder, logger):
    """
    Compute average interaction energies for each residue pair and energy type,
    and save to a single CSV file.
    
    Parameters:
    - energiesDict (dict): Dictionary containing energy data for residue pairs
    - system: ProDy structure object
    - outFolder (str): Output folder path
    - logger (logging.Logger): Logger object
    """
    logger.info('Creating average energies table...')
    
    # Cache for residue information
    residue_info_cache = {}
    
    def get_residue_info(resindex):
        """Get cached residue information"""
        if resindex not in residue_info_cache:
            sel = system.select(f'resindex {resindex}')
            residue_info_cache[resindex] = {
                'chain': sel.getChids()[0],
                'resnum': sel.getResnums()[0],
                'resname': sel.getResnames()[0]
            }
        return residue_info_cache[resindex]
    
    # Prepare data for the table
    rows = []
    total_pairs = len(energiesDict)
    
    logger.info(f'Processing {total_pairs} residue pairs...')
    
    # Progress tracking with tqdm
    with tqdm.tqdm(total=total_pairs, desc='Computing averages', unit='pair', 
                   ncols=100, disable=False) as pbar:
        
        for i, (pair_key, energies) in enumerate(energiesDict.items()):
            # Parse pair indices
            res1_idx, res2_idx = map(int, pair_key.split('-'))
            
            # Get residue information
            res1_info = get_residue_info(res1_idx)
            res2_info = get_residue_info(res2_idx)
            
            # Create residue identifiers
            res1_id = f"{res1_info['resname']}{res1_info['resnum']}_{res1_info['chain']}"
            res2_id = f"{res2_info['resname']}{res2_info['resnum']}_{res2_info['chain']}"
            
            # Compute averages for each energy type
            avg_total = np.mean(energies['Total'])
            avg_elec = np.mean(energies['Elec'])
            avg_vdw = np.mean(energies['VdW'])
            
            # Create row data
            row = {
                'Residue_1': res1_id,
                'Residue_2': res2_id,
                'Res1_Chain': res1_info['chain'],
                'Res1_ResNum': res1_info['resnum'],
                'Res1_ResName': res1_info['resname'],
                'Res2_Chain': res2_info['chain'],
                'Res2_ResNum': res2_info['resnum'],
                'Res2_ResName': res2_info['resname'],
                'Avg_Total_Energy': avg_total,
                'Avg_Elec_Energy': avg_elec,
                'Avg_VdW_Energy': avg_vdw
            }
            
            rows.append(row)
            pbar.update(1)
            
            # Log progress every 10%
            if (i + 1) % max(1, total_pairs // 10) == 0:
                progress_pct = ((i + 1) / total_pairs) * 100
                logger.info(f'Progress: {progress_pct:.1f}% ({i + 1}/{total_pairs} pairs processed)')
    
    # Create DataFrame
    logger.info('Creating DataFrame from computed averages...')
    df_averages = pd.DataFrame(rows)
    
    # Sort by Residue_1 and Residue_2 for better readability
    df_averages = df_averages.sort_values(['Res1_ResNum', 'Res2_ResNum'])
    
    # Save to CSV
    output_file = os.path.join(outFolder, 'average_interaction_energies.csv')
    logger.info(f'Saving average energies to {output_file}')
    df_averages.to_csv(output_file, index=False, float_format='%.6f')
    
    logger.info(f'Average energies table saved successfully with {len(df_averages)} residue pairs')
    logger.info(f'Columns: {", ".join(df_averages.columns)}')

def parse_interaction_energies(edrFiles, pairsFilteredChunks, outFolder, logger):
    """
    Parse interaction energies from EDR files and save the results.
    Optimized version with parallel processing and vectorized operations.

    Parameters:
    - edrFiles (list): List of paths to the EDR files.
    - pairsFilteredChunks (list): List of filtered residue pair chunks.
    - outFolder (str): The folder where output files will be saved.
    - logger (logging.Logger): The logger object for logging messages.
    """

    system = parsePDB(os.path.join(outFolder, 'system_dry.pdb'))
    
    try:
        # Determine optimal number of workers (80% of available cores)
        available_cores = multiprocessing.cpu_count()
        max_workers = min(len(edrFiles), max(1, int(available_cores * 0.8)))
        
        # Log system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f'System info: {available_cores} cores, {memory_gb:.1f} GB memory')
        logger.info(f'Parsing {len(edrFiles)} EDR files in parallel using {max_workers} workers (80% of {available_cores} cores)...')
        
        # Parse EDR files in parallel with progress tracking
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {executor.submit(parse_edr_file_parallel, edr_file): i for i, edr_file in enumerate(edrFiles)}
            
            # Track progress as jobs complete
            df_list = [None] * len(edrFiles)  # Pre-allocate list to maintain order
            completed = 0
            
            for future in concurrent.futures.as_completed(futures):
                original_index = futures[future]
                df_list[original_index] = future.result()
                completed += 1
                
                # More frequent progress updates - every file for small batches, every 5% for large batches
                should_log = False
                if len(edrFiles) <= 20:  # For small batches, log every file
                    should_log = True
                elif completed % max(1, len(edrFiles) // 20) == 0:  # For large batches, log every 5%
                    should_log = True
                elif completed == len(edrFiles):  # Always log completion
                    should_log = True
                    
                if should_log:
                    progress_pct = (completed / len(edrFiles)) * 100
                    remaining = len(edrFiles) - completed
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = remaining / rate if rate > 0 else 0
                    logger.info(f'EDR parsing progress: {completed}/{len(edrFiles)} files completed ({progress_pct:.1f}%) - {remaining} remaining - ETA: {eta:.1f}s')
            
            parsing_time = time.time() - start_time
            logger.info(f'EDR parsing completed successfully: {len(edrFiles)} files processed in {parsing_time:.1f}s')
        
        # Use memory-efficient combination method
        logger.info('Combining EDR results using memory-efficient pandas processing...')
        df = combine_dataframes_memory_efficient(df_list, outFolder, logger)
        
        # Clear intermediate data and trigger garbage collection
        del df_list
        gc.collect()
        
        # Use direct energy collection from DataFrame
        logger.info('Collecting energy results using vectorized operations...')
        energiesDict = collect_energies_from_dataframe(df, pairsFilteredChunks, logger)
        
        # Clear the large DataFrame
        del df
        gc.collect()
        
        logger.info('Collecting results...')
        
        # Create result DataFrames more efficiently
        df_total, df_elec, df_vdw = create_result_dataframes_optimized(energiesDict, logger)
        
        # Supplement with residue information using optimized version
        df_total = supplement_df_optimized(df_total, system)
        df_elec = supplement_df_optimized(df_elec, system)
        df_vdw = supplement_df_optimized(df_vdw, system)
        
        # Save results (using index=False to avoid extra index column)
        logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnTotal.csv'))
        df_total.to_csv(os.path.join(outFolder, 'energies_intEnTotal.csv'), index=False)
        logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnElec.csv'))
        df_elec.to_csv(os.path.join(outFolder, 'energies_intEnElec.csv'), index=False)
        logger.info('Saving results to ' + os.path.join(outFolder, 'energies_intEnVdW.csv'))
        df_vdw.to_csv(os.path.join(outFolder, 'energies_intEnVdW.csv'), index=False)

        # Compute and save average energies table
        logger.info('Computing average interaction energies for all residue pairs...')
        compute_and_save_average_energies(energiesDict, system, outFolder, logger)

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
        
    except Exception as e:
        logger.error(f"Error in parse_interaction_energies: {str(e)}")
        raise



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

def test_grinn_inputs(structure_file, out_folder, ff_folder=None, init_pair_filter_cutoff=10, 
                     nofixpdb=False, top=None, traj=None, nointeraction=False, 
                     gpu=False, solvate=False, npt=False, source_sel="all", target_sel="all", 
                     nt=1, skip=1, noconsole_handler=False,
                     create_pen=False, pen_cutoffs=[1.0], pen_include_covalents=[True, False],
                     force_field='amber99sb-ildn', water_model='tip3p', recreate_topology=False,
                     ensemble_mode=False, max_frames=None):
    """
    Test and validate inputs for the gRINN workflow.
    
    Supports two input modes:
    1. Trajectory Mode: structure (PDB) + trajectory (XTC) + topology (TOP)
    2. Ensemble Mode: multi-model PDB file (topology generated automatically)
    
    Returns:
    - (bool, list): (is_valid, list_of_errors)
    """
    errors = []
    warnings = []
    
    # Validate structure file is provided and exists
    if not structure_file:
        errors.append("ERROR: Structure file (PDB) is required")
        return False, errors
    
    if not os.path.exists(structure_file):
        errors.append(f"ERROR: Structure file '{structure_file}' does not exist")
        return False, errors
    
    # Validate structure file is PDB format
    if not structure_file.lower().endswith('.pdb'):
        errors.append(f"ERROR: Structure file must be in PDB format, got: {structure_file}")
        errors.append("  Only PDB format is supported")
    
    # Try to parse the structure file
    try:
        structure = parse_structure_file(structure_file)
        if structure is None:
            errors.append(f"ERROR: Could not parse structure file '{structure_file}'")
    except Exception as e:
        errors.append(f"ERROR: Failed to parse structure file '{structure_file}': {str(e)}")
    
    # Validate output folder
    if not out_folder:
        errors.append("ERROR: Output folder path is required")
    
    # Validate input mode
    if ensemble_mode:
        # Ensemble mode validation
        warnings.append("INFO: Conformational Ensemble Mode")
        warnings.append("  Input: Multi-model PDB file")
        warnings.append("  Topology will be generated automatically")
        
        # Check if PDB has multiple models
        try:
            traj_test = md.load(structure_file)
            n_models = traj_test.n_frames
            warnings.append(f"  Found {n_models} models in PDB file")
            
            if n_models < 2:
                warnings.append("  WARNING: PDB contains only 1 model")
                warnings.append("  Consider using trajectory mode instead")
        except Exception as e:
            errors.append(f"ERROR: Could not load PDB as trajectory: {str(e)}")
            errors.append("  Ensure PDB file contains MODEL/ENDMDL entries for ensemble mode")
        
        # In ensemble mode, traj and top should not be provided
        if traj:
            errors.append("ERROR: --traj should not be provided in ensemble mode")
            errors.append("  Trajectory will be generated from PDB models")
        if top:
            errors.append("ERROR: --top should not be provided in ensemble mode")
            errors.append("  Topology will be generated automatically")
    
    else:
        # Trajectory mode validation
        if not traj or not top:
            errors.append("ERROR: Trajectory mode requires both --traj and --top")
            errors.append("  Provide: --traj <file.xtc> --top <file.top>")
            errors.append("  OR use --ensemble_mode for multi-model PDB")
        else:
            warnings.append("INFO: Pre-computed Trajectory Mode")
            warnings.append(f"  Structure: {structure_file}")
            warnings.append(f"  Trajectory: {traj}")
            warnings.append(f"  Topology: {top}")
            
            # Validate trajectory file
            if not os.path.exists(traj):
                errors.append(f"ERROR: Trajectory file '{traj}' does not exist")
            elif not traj.lower().endswith('.xtc'):
                errors.append(f"ERROR: Only XTC trajectory format is supported, got: {traj}")

            # Optional frame-count limit check (best-effort; may be slow for huge trajectories)
            if max_frames is not None:
                try:
                    max_frames_int = int(max_frames)
                    if max_frames_int <= 0:
                        errors.append(f"ERROR: --max_frames must be a positive integer, got: {max_frames}")
                    elif traj and os.path.exists(traj):
                        # Stop counting as soon as we exceed the limit
                        n_frames = get_trajectory_frame_count(traj, stop_after=max_frames_int + 1)
                        if n_frames > max_frames_int:
                            errors.append(
                                f"ERROR: Trajectory has {n_frames} frames which exceeds the configured limit ({max_frames_int})."
                            )
                except Exception as e:
                    errors.append(f"ERROR: Failed to count trajectory frames for --max_frames check: {str(e)}")
            
            # Validate topology file
            if not os.path.exists(top):
                errors.append(f"ERROR: Topology file '{top}' does not exist")
            elif not top.lower().endswith('.top'):
                errors.append(f"ERROR: Only TOP topology format is supported, got: {top}")
    
    # Validate force field folder if provided
    if ff_folder and not os.path.exists(ff_folder):
        errors.append(f"ERROR: Force field folder '{ff_folder}' does not exist")
    
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
    
    try:
        skip_val = int(skip)
        if skip_val <= 0:
            errors.append(f"ERROR: skip must be positive, got {skip_val}")
    except (TypeError, ValueError):
        errors.append(f"ERROR: skip must be an integer, got {skip}")
    
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
    if (source_sel or target_sel) and structure_file and os.path.exists(structure_file):
        try:
            sys = parse_structure_file(structure_file)
            
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
            warnings.append(f"WARNING: Could not validate selections: {str(e)}")
    
    # Validate logical combinations
    if nointeraction and create_pen:
        errors.append("ERROR: Cannot create PEN without calculating interactions (nointeraction=True)")
    
    # Validate force field and water model parameters
    valid_force_fields = [
        'amber99sb-ildn', 'amber99sb', 'amber03', 'amber14sb', 
        'charmm27', 'charmm36', 'oplsaa', 'gromos96', 'gromos54a7'
    ]
    
    valid_water_models = ['tip3p', 'tip4p', 'spc', 'spce', 'tip5p']
    
    if force_field and force_field not in valid_force_fields and not ff_folder:
        warnings.append(f"WARNING: Force field '{force_field}' not in common list: {valid_force_fields}")
        warnings.append("  Consider using --ff_folder for custom force fields")
    
    if water_model and water_model not in valid_water_models:
        warnings.append(f"WARNING: Water model '{water_model}' not in common list: {valid_water_models}")
    
    # Test GROMACS functionality
    print("\nTesting GROMACS functionality...")
    gromacs_errors = test_gromacs_functionality(structure_file, top, traj, ff_folder)
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


def detect_gromacs_version():
    """
    Detect installed GROMACS version and capabilities.
    
    Returns:
    - dict: Information about GROMACS installation, or None if not found
    """
    try:
        # Try to get GROMACS version
        result = subprocess.run(['gmx', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version_info = {}
            
            # Parse version from output
            import re
            version_match = re.search(r'GROMACS version:\s*(\d+\.\d+(?:\.\d+)?)', result.stdout)
            if version_match:
                version_str = version_match.group(1)
                version_info['version_string'] = version_str
                
                # Convert to float for comparisons (use major.minor)
                version_parts = version_str.split('.')
                version_info['version'] = float(f"{version_parts[0]}.{version_parts[1]}")
                
                # Detect capabilities
                version_info['has_gpu'] = 'GPU' in result.stdout or 'CUDA' in result.stdout
                version_info['has_mpi'] = 'MPI' in result.stdout
                version_info['has_double_precision'] = 'double precision' in result.stdout.lower()
                
                # Version-specific features
                version_info['supports_verlet'] = version_info['version'] >= 4.6
                version_info['has_new_mdrun'] = version_info['version'] >= 5.0
                version_info['supports_awh'] = version_info['version'] >= 2016.0
                
                # Get installation path
                try:
                    which_result = subprocess.run(['which', 'gmx'], 
                                                capture_output=True, text=True)
                    if which_result.returncode == 0:
                        version_info['gmx_path'] = which_result.stdout.strip()
                except:
                    pass
                
                return version_info
                
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return None

def get_gromacs_commands(gromacs_info):
    """
    Return version-appropriate GROMACS commands based on detected version.
    
    Parameters:
    - gromacs_info (dict): Information from detect_gromacs_version()
    
    Returns:
    - dict: Command mappings for different GROMACS tools
    """
    if gromacs_info is None:
        # Default to modern GROMACS commands
        return {
            'base_cmd': 'gmx',
            'pdb2gmx': 'gmx pdb2gmx',
            'editconf': 'gmx editconf',
            'solvate': 'gmx solvate',
            'genion': 'gmx genion',
            'grompp': 'gmx grompp',
            'mdrun': 'gmx mdrun',
            'trjconv': 'gmx trjconv',
            'make_ndx': 'gmx make_ndx',
            'check': 'gmx check'
        }
    
    version = gromacs_info['version']
    
    if version >= 5.0:
        # Modern GROMACS (5.0+) - all commands under 'gmx'
        return {
            'base_cmd': 'gmx',
            'pdb2gmx': 'gmx pdb2gmx',
            'editconf': 'gmx editconf',
            'solvate': 'gmx solvate',
            'genion': 'gmx genion',
            'grompp': 'gmx grompp',
            'mdrun': 'gmx mdrun',
            'trjconv': 'gmx trjconv',
            'make_ndx': 'gmx make_ndx',
            'check': 'gmx check'
        }
    else:
        # Legacy GROMACS (4.x) - separate commands
        return {
            'base_cmd': '',
            'pdb2gmx': 'pdb2gmx',
            'editconf': 'editconf',
            'solvate': 'genbox',  # Different name in 4.x
            'genion': 'genion',
            'grompp': 'grompp',
            'mdrun': 'mdrun',
            'trjconv': 'trjconv',
            'make_ndx': 'make_ndx',
            'check': 'g_check'  # Different name in 4.x
        }

def setup_gromacs_environment(logger=None):
    """
    Set up GROMACS environment and detect version compatibility.
    
    Parameters:
    - logger: Optional logger for output messages
    
    Returns:
    - dict: GROMACS information and command mappings
    
    Raises:
    - RuntimeError: If GROMACS is not found or incompatible
    """
    if logger:
        logger.info("Setting up GROMACS environment...")
    
    # Try different GROMACS installation locations
    gromacs_paths = [
        '/opt/gromacs/bin',           # Host-mounted GROMACS
        '/usr/local/gromacs/bin',     # Container GROMACS
        '/usr/bin',                   # System GROMACS
    ]
    
    gromacs_found = False
    for gromacs_path in gromacs_paths:
        gmx_executable = os.path.join(gromacs_path, 'gmx')
        if os.path.exists(gmx_executable):
            # Add to PATH if not already there
            current_path = os.environ.get('PATH', '')
            if gromacs_path not in current_path:
                os.environ['PATH'] = f"{gromacs_path}:{current_path}"
            
            # Try to source GMXRC if it exists
            gmxrc_path = os.path.join(os.path.dirname(gromacs_path), 'bin', 'GMXRC')
            if os.path.exists(gmxrc_path):
                if logger:
                    logger.info(f"Found GMXRC at {gmxrc_path}")
                # Note: In Python, we can't directly source shell scripts
                # The Docker entrypoint should handle GMXRC sourcing
            
            gromacs_found = True
            if logger:
                logger.info(f"Using GROMACS from {gromacs_path}")
            break
    
    if not gromacs_found:
        # Check if gmx is in system PATH
        try:
            subprocess.run(['which', 'gmx'], check=True, capture_output=True)
            gromacs_found = True
            if logger:
                logger.info("Using GROMACS from system PATH")
        except subprocess.CalledProcessError:
            pass
    
    if not gromacs_found:
        error_msg = ("GROMACS not found. Please ensure GROMACS is installed or mount it from the host. "
                    "For Docker: docker run -v /path/to/gromacs:/opt/gromacs ...")
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Detect GROMACS version and capabilities
    gromacs_info = detect_gromacs_version()
    
    if gromacs_info is None:
        error_msg = "GROMACS found but could not determine version. Please check installation."
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Log GROMACS information
    if logger:
        logger.info(f"GROMACS version detected: {gromacs_info['version_string']}")
        if gromacs_info.get('has_gpu'):
            logger.info("GPU support detected")
        if gromacs_info.get('has_mpi'):
            logger.info("MPI support detected")
        if gromacs_info.get('has_double_precision'):
            logger.info("Double precision build detected")
    
    # Get version-appropriate commands
    commands = get_gromacs_commands(gromacs_info)
    
    # Verify critical commands work
    try:
        subprocess.run([commands['base_cmd'] if commands['base_cmd'] else 'gmx', '--version'], 
                      capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        error_msg = f"GROMACS commands not working properly: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Return comprehensive information
    return {
        'info': gromacs_info,
        'commands': commands,
        'version': gromacs_info['version'],
        'version_string': gromacs_info['version_string']
    }

def test_gromacs_functionality(structure_file, top=None, traj=None, ff_folder=None):
    """
    Test if GROMACS can actually process the input files.
    
    Returns:
    - errors (list): List of GROMACS-related errors
    """
    errors = []
    temp_dir = None
    
    try:
        
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp(prefix="grinn_test_")
        
        # Test 1: Check if GROMACS is available and get version info
        try:
            gromacs_env = setup_gromacs_environment()
            gromacs_info = gromacs_env['info']
            commands = gromacs_env['commands']
            
            print(f"✓ GROMACS found: version {gromacs_info['version_string']}")
            
            # Print capabilities
            capabilities = []
            if gromacs_info.get('has_gpu'):
                capabilities.append("GPU")
            if gromacs_info.get('has_mpi'):
                capabilities.append("MPI")
            if gromacs_info.get('has_double_precision'):
                capabilities.append("Double precision")
            
            if capabilities:
                print(f"  Capabilities: {', '.join(capabilities)}")
                
        except Exception as e:
            errors.append(f"ERROR: GROMACS not found or not working: {str(e)}")
            return errors
        
        # Test 2: Test structure file with gmx editconf (quick structure check)
        # NOTE: Do not require `-princ` here; it can fail for some valid structures
        # (e.g., when element/mass inference is ambiguous), even though GROMACS can
        # still read/convert the structure.
        if structure_file and os.path.exists(structure_file):
            try:
                # Basic sanity checks to provide clearer errors than a generic GROMACS failure.
                try:
                    size_bytes = os.path.getsize(structure_file)
                except OSError:
                    size_bytes = None

                if size_bytes == 0:
                    errors.append("ERROR: Structure file is empty")
                else:
                    with open(structure_file, 'rb') as fh:
                        head = fh.read(4096)
                    if b'\x00' in head:
                        errors.append("ERROR: Structure file appears to be binary (contains NUL bytes)")
                    else:
                        test_out = os.path.join(temp_dir, "test.pdb")
                        try:
                            # First try with -princ (kept for backwards compatibility / richer checks)
                            gromacs.editconf(f=structure_file, o=test_out, princ=True)
                            print(f"✓ Structure file is readable by GROMACS")
                        except Exception as e_princ:
                            # Fallback: just test read/convert without -princ.
                            # This avoids false negatives when -princ fails due to mass inference.
                            try:
                                gromacs.editconf(f=structure_file, o=test_out)
                                print("✓ Structure file is readable by GROMACS (skipping -princ check)")
                            except Exception as e:
                                errors.append(
                                    "ERROR: GROMACS cannot read structure file via gmx editconf. "
                                    f"Failure with -princ: {e_princ}. "
                                    f"Failure without -princ: {e}"
                                )
            except Exception as e:
                errors.append(f"ERROR: Unexpected error while validating structure file: {str(e)}")
        
        # Test 3: If topology provided, test with gmx grompp
        if top and os.path.exists(top) and structure_file and os.path.exists(structure_file):
            try:
                # Create a minimal mdp file for testing
                test_mdp = os.path.join(temp_dir, "test.mdp")
                with open(test_mdp, 'w') as f:
                    f.write("integrator = md\n")
                    f.write("nsteps = 0\n")
                    f.write("cutoff-scheme = Verlet\n")
                
                test_tpr = os.path.join(temp_dir, "test.tpr")
                # Copy structure file to temp dir to ensure paths work
                temp_pdb = os.path.join(temp_dir, "system.pdb")
                shutil.copy(structure_file, temp_pdb)
                
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
            shutil.rmtree(temp_dir)
    
    return errors


def run_grinn_workflow(structure_file, out_folder, ff_folder, init_pair_filter_cutoff, nofixpdb=False, top=False, 
                       traj=False, nointeraction=False, gpu=False, solvate=False, npt=False, source_sel="all", target_sel="all", 
                       nt=1, skip=1, noconsole_handler=False, create_pen=False, pen_cutoffs=[1.0], 
                       pen_include_covalents=[True, False], test_only=False, force_field='amber99sb-ildn', water_model='tip3p',
                       recreate_topology=False, ensemble_mode=False, max_frames=None):
    
    # If test_only flag is set, just validate inputs and exit
    if test_only:
        is_valid, errors = test_grinn_inputs(
            structure_file, out_folder, ff_folder, init_pair_filter_cutoff, nofixpdb, top, 
            traj, nointeraction, gpu, solvate, npt, source_sel, target_sel, nt, skip,
            noconsole_handler, create_pen, pen_cutoffs, pen_include_covalents,
            force_field, water_model, recreate_topology, ensemble_mode, max_frames
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
    
    # Set up and validate GROMACS environment
    try:
        gromacs_env = setup_gromacs_environment(logger)
        logger.info(f"GROMACS environment ready: version {gromacs_env['version_string']}")
    except RuntimeError as e:
        logger.error(f"GROMACS setup failed: {str(e)}")
        logger.error("Please check GROMACS installation or mount GROMACS from host")
        sys.exit(1)
    
    # Log system resources and check disk space
    try:
        
        # System resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        logger.info(f'System resources: {cpu_count} CPU cores, {memory_gb:.1f} GB memory')
        
        # Check disk space
        total, used, free = shutil.disk_usage(out_folder)
        free_gb = free / (1024**3)
        logger.info(f'Disk space available: {free_gb:.1f} GB')
        
        # Log frame skipping configuration
        if skip > 1:
            logger.info(f'Frame skipping enabled: analyzing every {skip} frames (skipping {skip-1} frames between analyses)')
        else:
            logger.info('Frame skipping disabled: analyzing all frames')
        
        # Warn if disk space is low
        if free_gb < 10:
            logger.warning(f'Warning: Low disk space ({free_gb:.1f} GB). Consider freeing up space.')
        
    except ImportError:
        logger.info('psutil not available, skipping system resource logging')
    except Exception as e:
        logger.warning(f'Could not check system resources: {str(e)}')
    
    # Print the command-line used to call this workflow to the log file
    logger.info('gRINN workflow was called as follows: ')
    logger.info(' '.join(sys.argv))

    # Validate that we have a structure file
    if not structure_file:
        logger.error('No structure file provided')
        logger.error('Please provide a PDB structure file')
        raise ValueError("No structure file available")
    
    # Validate structure file exists
    if not os.path.exists(structure_file):
        logger.error(f'Structure file not found: {structure_file}')
        raise ValueError(f"Structure file not found: {structure_file}")
    
    # Check if structure file is PDB format (required)
    if not structure_file.lower().endswith('.pdb'):
        logger.error('Structure file must be in PDB format')
        logger.error(f'Got: {structure_file}')
        raise ValueError("Only PDB format is supported for structure files")
    
    # Handle ensemble mode: multi-model PDB to XTC conversion
    if ensemble_mode:
        logger.info('=' * 60)
        logger.info('CONFORMATIONAL ENSEMBLE MODE')
        logger.info('=' * 60)
        logger.info('Input: Multi-model PDB file')
        logger.info('Creating XTC trajectory from PDB models...')
        
        # Convert multi-model PDB to XTC trajectory
        ensemble_xtc = os.path.join(out_folder, 'ensemble_trajectory.xtc')
        ensemble_pdb = os.path.join(out_folder, 'ensemble_reference.pdb')
        
        try:
            # Load the multi-model PDB
            trajectory = md.load(structure_file)
            n_models = trajectory.n_frames
            
            logger.info(f'Found {n_models} models in PDB file')

            # Enforce frame limit for ensemble input (models == frames)
            if max_frames is not None:
                try:
                    max_frames_int = int(max_frames)
                except Exception:
                    raise ValueError(f"--max_frames must be an integer, got: {max_frames}")
                if max_frames_int <= 0:
                    raise ValueError(f"--max_frames must be a positive integer, got: {max_frames}")
                if n_models > max_frames_int:
                    logger.error(
                        f"Input ensemble contains {n_models} models/frames, which exceeds --max_frames={max_frames_int}."
                    )
                    raise ValueError(
                        f"Trajectory frame limit exceeded: {n_models} > {max_frames_int} (ensemble models)"
                    )
            
            if n_models < 2:
                logger.warning('PDB file contains only 1 model - ensemble mode may not be appropriate')
                logger.warning('Consider using trajectory mode instead')
            
            # Save the first frame as reference structure
            trajectory[0].save_pdb(ensemble_pdb)
            logger.info(f'Saved reference structure: {ensemble_pdb}')
            
            # Save all frames as XTC trajectory
            trajectory.save_xtc(ensemble_xtc)
            logger.info(f'Created XTC trajectory with {n_models} frames: {ensemble_xtc}')
            
            # Update structure_file and traj for the workflow
            structure_file = ensemble_pdb
            traj = ensemble_xtc
            
            logger.info('Ensemble mode setup complete')
            logger.info(f'  Reference structure: {structure_file}')
            logger.info(f'  Trajectory: {traj}')
            
        except Exception as e:
            logger.error(f'Failed to process ensemble PDB file: {str(e)}')
            logger.error('Please ensure the PDB file contains multiple MODEL entries')
            raise ValueError(f"Ensemble PDB processing failed: {str(e)}")
    
    # Validate input mode consistency
    logger.info('=' * 60)
    logger.info('INPUT MODE VALIDATION')
    logger.info('=' * 60)
    
    if ensemble_mode:
        logger.info('Mode: Conformational Ensemble')
        logger.info(f'  Structure: {structure_file}')
        logger.info(f'  Trajectory: {traj} (generated from ensemble)')
        logger.info('  Topology: Will be generated')
    elif traj and top:
        logger.info('Mode: Pre-computed Trajectory')
        logger.info(f'  Structure: {structure_file}')
        logger.info(f'  Trajectory: {traj}')

        # Enforce frame limit for input trajectory (count frames with early-exit)
        if max_frames is not None:
            try:
                max_frames_int = int(max_frames)
            except Exception:
                raise ValueError(f"--max_frames must be an integer, got: {max_frames}")
            if max_frames_int <= 0:
                raise ValueError(f"--max_frames must be a positive integer, got: {max_frames}")

            try:
                n_frames = get_trajectory_frame_count(traj, stop_after=max_frames_int + 1)
            except Exception as e:
                logger.error(f"Failed to count frames in trajectory '{traj}': {str(e)}")
                raise

            logger.info(f"Trajectory frame count: {n_frames}")
            if n_frames > max_frames_int:
                logger.error(
                    f"Input trajectory has {n_frames} frames, which exceeds --max_frames={max_frames_int}."
                )
                raise ValueError(f"Trajectory frame limit exceeded: {n_frames} > {max_frames_int}")
        logger.info(f'  Topology: {top}')
        
        # Validate trajectory file
        if not os.path.exists(traj):
            logger.error(f'Trajectory file not found: {traj}')
            raise ValueError(f"Trajectory file not found: {traj}")
        
        if not traj.lower().endswith('.xtc'):
            logger.error('Only XTC trajectory format is supported')
            logger.error(f'Got: {traj}')
            raise ValueError("Only XTC format is supported for trajectory files")
        
        # Validate topology file
        if not os.path.exists(top):
            logger.error(f'Topology file not found: {top}')
            raise ValueError(f"Topology file not found: {top}")
        
        if not top.lower().endswith('.top'):
            logger.error('Only TOP topology format is supported')
            logger.error(f'Got: {top}')
            raise ValueError("Only TOP format is supported for topology files")
    else:
        logger.error('Invalid input combination')
        logger.error('Please use one of:')
        logger.error('  1. Trajectory mode: --traj <file.xtc> --top <file.top>')
        logger.error('  2. Ensemble mode: --ensemble_mode (with multi-model PDB)')
        raise ValueError("Invalid input mode - must provide either trajectory+topology or use ensemble mode")

    # If a force field folder is provided
    if ff_folder:
        logger.info('Force field folder provided. Using provided force field folder.')
        logger.info('Copying force field folder to output folder...')
        # Normalize path to handle trailing slashes properly
        ff_folder_normalized = os.path.normpath(ff_folder)
        ff_folder_basename = os.path.basename(ff_folder_normalized)
        # Ensure we have a valid basename (fallback if somehow still empty)
        if not ff_folder_basename:
            ff_folder_basename = "force_field"
        shutil.copytree(ff_folder, os.path.join(out_folder, ff_folder_basename), dirs_exist_ok=True)

    # Handle topology file with flexible approach for missing topology scenarios
    # In ensemble mode or trajectory mode without topology, create topology
    if ensemble_mode or (not top and traj):
        topology_result = handle_missing_topology_flexible(
            structure_file=structure_file,
            out_folder=out_folder,
            traj=traj,
            top=top,
            force_field=force_field,
            water_model=water_model,
            ff_folder=ff_folder,
            recreate_topology=recreate_topology,
            logger=logger
        )
    else:
        # Topology provided
        topology_result = {'topology_available': False, 'topology_created': False, 'method_used': None}
    
    # Handle existing topology file case
    if top and not recreate_topology:
        logger.info('Topology file provided. Using provided topology file.')
        logger.info('Copying topology file to output folder...')
        shutil.copy(top, os.path.join(out_folder, 'topol_dry.top'))
        topology_result['topology_available'] = True
        topology_result['method_used'] = 'provided'

        # Automatically detect and copy topology dependencies (include files and toppar)
        try:
            auto_detect_and_copy_topology_dependencies(top, out_folder, gromacs_dir=None, logger=logger)
        except RuntimeError as e:
            logger.error(f"Failed to copy topology dependencies: {str(e)}")
            logger.error("This may cause issues during GROMACS execution")
            logger.error("Please ensure all required files are accessible in the container/environment")
            sys.exit(1)
    
    # Check the result of topology handling
    if not topology_result['topology_available']:
        if traj and not topology_result['topology_created']:
            logger.warning('Trajectory provided but topology could not be created/found')
            logger.warning('Will attempt to proceed with simulation to generate topology')
        elif not traj:
            logger.info('No topology file available. Will generate topology during simulation.')
        else:
            logger.error('Could not create or find topology file for trajectory analysis')
            logger.error('Please provide one of:')
            logger.error('  1. A topology file with --top')
            logger.error('  2. Use --ensemble_mode for multi-model PDB')
            logger.error('  3. Ensure structure file is compatible with pdb2gmx')
            logger.error('  4. Check force field and water model parameters')
    else:
        logger.info(f'Topology handling completed via method: {topology_result["method_used"]}')
        if topology_result['topology_created']:
            logger.info('Topology file successfully created/recreated')

    if traj:
        logger.info('Copying input structure_file to output_folder as "system_dry.pdb"...')
        
        # Convert to PDB format if input is GRO
        input_ext = os.path.splitext(structure_file)[1].lower()
        if input_ext == '.gro':
            logger.info('Input is GRO format, converting to PDB format with box information...')
            # Use gmx editconf directly to preserve CRYST1 record with box information
            gromacs.editconf(f=structure_file, o=os.path.join(out_folder, 'system_dry.pdb'))
        else:
            # For PDB files, just copy
            shutil.copy(structure_file, os.path.join(out_folder, 'system_dry.pdb'))

        # Detect and assign chain IDs if missing
        topology_file = os.path.join(out_folder, 'topol_dry.top') if top else None
        try:
            # Pass the original input format to help with chain ID detection
            original_format = 'gro' if input_ext == '.gro' else 'pdb'
            detect_and_assign_chain_ids(os.path.join(out_folder, 'system_dry.pdb'), topology_file, logger, original_format)
        except ValueError as e:
            logger.warning(f"Chain ID assignment failed: {str(e)}")
            logger.warning("Proceeding with existing chain IDs in PDB file. Analysis may be less accurate for multi-chain systems.")

        # Check whether also a trajectory file is provided
        logger.info('Trajectory file provided. Processing trajectory file with frame skipping...')
        logger.info(f'Applying frame skipping (every {skip} frames) to trajectory...')
        # Use trjconv to apply frame skipping instead of just copying
        gromacs.trjconv(f=traj, o=os.path.join(out_folder, 'traj_dry.xtc'), s=os.path.join(out_folder, 'system_dry.pdb'), skip=skip, input=('0',))
        logger.info(f'Trajectory processing completed (skipped every {skip} frames).')
    else:
        if not top:
            # Only run simulation if we don't have topology and trajectory
            run_gromacs_simulation(structure_file, mdp_files_folder, out_folder, ff_folder, nofixpdb, gpu, solvate, npt, logger, nt, skip, force_field, water_model)
        else:
            logger.info('Generating traj.xtc file from input structure_file...')
            
            # Convert to PDB format if input is GRO
            input_ext = os.path.splitext(structure_file)[1].lower()
            if input_ext == '.gro':
                logger.info('Input is GRO format, converting to PDB format with box information...')
                # Use gmx editconf directly to preserve CRYST1 record with box information
                gromacs.editconf(f=structure_file, o=os.path.join(out_folder, 'system_dry.pdb'))
            else:
                # For PDB files, just copy
                shutil.copy(structure_file, os.path.join(out_folder, 'system_dry.pdb'))
            
            # Detect and assign chain IDs if missing
            topology_file = os.path.join(out_folder, 'topol_dry.top')
            try:
                # Pass the original input format to help with chain ID detection
                original_format = 'gro' if input_ext == '.gro' else 'pdb'
                detect_and_assign_chain_ids(os.path.join(out_folder, 'system_dry.pdb'), topology_file, logger, original_format)
            except ValueError as e:
                logger.warning(f"Chain ID assignment failed: {str(e)}")
                logger.warning("Proceeding with existing chain IDs in PDB file. Analysis may be less accurate for multi-chain systems.")
            
            gromacs.trjconv(f=os.path.join(out_folder, 'system_dry.pdb'), o=os.path.join(out_folder, 'traj_dry.xtc'))

    if nointeraction:
        logger.info('Not calculating interaction energies as per user request.')
    else:
        # Check if we have topology file for interaction analysis
        topology_file = os.path.join(out_folder, 'topol_dry.top')
        
        if os.path.exists(topology_file):
            logger.info('Using topology file for detailed residue-residue interaction analysis')
            initialFilter = perform_initial_filtering(out_folder, source_sel, target_sel, init_pair_filter_cutoff, 4, logger)
            edrFiles, pairsFilteredChunks = calculate_interaction_energies(out_folder, initialFilter, nt, logger)
            parse_interaction_energies(edrFiles, pairsFilteredChunks, out_folder, logger)
        else:
            logger.error('No topology (.top) file available for interaction energy analysis')
            raise ValueError("Topology file required for interaction energy calculation")

    # --- PEN analysis ---
    if create_pen:
        logger.info('Starting PEN (Protein Energy Network) analysis...')
        pen_csv = os.path.join(out_folder, 'energies_intEnTotal.csv')
        pdb_path = os.path.join(out_folder, 'system_dry.pdb')
        if os.path.exists(pen_csv) and os.path.exists(pdb_path):
            compute_pen_and_bc(
                structure_file=pdb_path,
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
    
    # Generate workflow summary report
    try:
        workflow_params = {
            'structure_file': structure_file,
            'out_folder': out_folder,
            'ff_folder': ff_folder,
            'init_pair_filter_cutoff': init_pair_filter_cutoff,
            'nofixpdb': nofixpdb,
            'top': top,
            'traj': traj,
            'nointeraction': nointeraction,
            'gpu': gpu,
            'solvate': solvate,
            'npt': npt,
            'source_sel': source_sel,
            'target_sel': target_sel,
            'nt': nt,
            'create_pen': create_pen,
            'pen_cutoffs': pen_cutoffs,
            'pen_include_covalents': pen_include_covalents
        }
        generate_workflow_summary_report(out_folder, logger, workflow_params)
    except Exception as e:
        logger.warning(f'Could not generate workflow summary report: {str(e)}')
    
    elapsed_time = time.time() - start_time  # Calculate the elapsed time    
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    logger.info('Elapsed time: {:.2f} seconds'.format(elapsed_time))
    logger.info('### gRINN workflow completed successfully ###')
    # Clear handlers to avoid memory leak
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gRINN workflow",
        epilog="""
Input Modes:
  1. Trajectory Mode: Provide structure (PDB), trajectory (XTC), and topology (TOP) files
     Example: grinn_workflow.py structure.pdb output/ --traj trajectory.xtc --top topology.top
  
  2. Conformational Ensemble Mode: Provide multi-model PDB file (topology will be generated)
     Example: grinn_workflow.py ensemble.pdb output/ --ensemble_mode
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("structure_file", type=str, help="Input structure file (PDB format). Can be single structure or multi-model ensemble.")
    parser.add_argument("out_folder", type=str, help="Output folder")
    parser.add_argument("--nofixpdb", action="store_true", help="Skip PDB fixing with pdbfixer")
    parser.add_argument("--initpairfiltercutoff", type=float, default=10, help="Initial pair filter cutoff (default is 10)")
    parser.add_argument("--nointeraction", action="store_true", help="Do not calculate interaction energies")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for non-bonded interactions in GROMACS commands")
    parser.add_argument("--solvate", action="store_true", help="Run solvation")
    parser.add_argument("--npt", action="store_true", help="Run NPT equilibration")
    parser.add_argument("--source_sel", nargs="+", type=str, help="Source selection")
    parser.add_argument("--target_sel", nargs="+", type=str, help="Target selection")
    parser.add_argument("--nt", type=int, default=1, help="Number of threads for GROMACS commands (default is 1)")
    parser.add_argument("--skip", type=int, default=1, help="Skip every nth frame in trajectory analysis (default is 1, no skipping)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum allowed number of frames in the input trajectory. If the trajectory contains more frames, the workflow errors.")
    parser.add_argument("--noconsole_handler", action="store_true", help="Do not add console handler to the logger")
    parser.add_argument("--ff_folder", type=str, help="Folder containing the force field files")
    
    # Input mode arguments
    parser.add_argument('--ensemble_mode', action='store_true',
                       help='Conformational ensemble mode: input PDB contains multiple models. XTC trajectory will be generated from models.')
    parser.add_argument('--top', type=str, help='Topology file (.top) - REQUIRED for trajectory mode')
    parser.add_argument('--traj', type=str, help='Trajectory file (.xtc) - REQUIRED for trajectory mode')
    
    # Topology recreation arguments
    parser.add_argument('--force_field', type=str, default='amber99sb-ildn', 
                       help='Force field to use for topology recreation (default: amber99sb-ildn). Common options: amber99sb-ildn, charmm27, oplsaa, gromos96')
    parser.add_argument('--water_model', type=str, default='tip3p', 
                       help='Water model to use for topology recreation (default: tip3p). Common options: tip3p, tip4p, spc, spce')
    parser.add_argument('--recreate_topology', action='store_true',
                       help='Force recreation of topology file even if one exists')
    
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

def generate_workflow_summary_report(out_folder, logger, workflow_params=None):
    """
    Generate a comprehensive summary report of the gRINN workflow results.
    
    Parameters:
    - out_folder (str): The output folder containing workflow results
    - logger (logging.Logger): The logger object
    - workflow_params (dict): Optional dictionary containing workflow parameters
    
    Returns:
    - None (writes report to file)
    """
    try:
        
        report_file = os.path.join(out_folder, 'grinn_workflow_summary.json')
        logger.info(f'Generating workflow summary report: {report_file}')
        
        report = {
            'workflow_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'grinn_version': 'gRINN v2.0 (optimized)',
                'output_folder': out_folder
            },
            'input_parameters': workflow_params or {},
            'output_files': {},
            'analysis_results': {},
            'system_info': {}
        }
        
        # System info
        try:
            report['system_info'] = {
                'platform': platform.system(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                'python_version': platform.python_version()
            }
        except ImportError:
            report['system_info'] = {'note': 'psutil not available for system info'}
        
        # Check for output files
        output_files = {
            'energy_files': {
                'total_energy_csv': os.path.join(out_folder, 'energies_intEnTotal.csv'),
                'elec_energy_csv': os.path.join(out_folder, 'energies_intEnElec.csv'),
                'vdw_energy_csv': os.path.join(out_folder, 'energies_intEnVdW.csv')
            },
            'structure_files': {
                'system_pdb': os.path.join(out_folder, 'system_dry.pdb'),
                'trajectory': os.path.join(out_folder, 'traj_dry.xtc'),
                'topology': os.path.join(out_folder, 'topol_dry.top')
            },
            'pen_files': {
                'betweenness_centralities': os.path.join(out_folder, 'pen_betweenness_centralities.csv')
            },
            'pen_networks': glob.glob(os.path.join(out_folder, 'pen_*.gml')),
            'log_files': {
                'main_log': os.path.join(out_folder, 'calc.log'),
                'gromacs_log': os.path.join(out_folder, 'gromacs.log')
            }
        }
        
        # Check file existence and get sizes
        for category, files in output_files.items():
            report['output_files'][category] = {}
            if isinstance(files, dict):
                for file_type, file_path in files.items():
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                       
                        report['output_files'][category][file_type] = {
                            'path': file_path,
                            'size_mb': round(file_size / (1024**2), 2),
                            'exists': True
                        }
                    else:
                        report['output_files'][category][file_type] = {
                            'path': file_path,
                            'exists': False
                        }
            elif isinstance(files, list):
                report['output_files'][category] = {
                    'count': len(files),
                    'files': files
                }
        
        # Analyze energy results if available
        energy_csv = os.path.join(out_folder, 'energies_intEnTotal.csv')
        if os.path.exists(energy_csv):
            try:
                df = pd.read_csv(energy_csv)
                
                # Get frame columns (numeric columns)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                report['analysis_results']['energy_analysis'] = {
                    'total_residue_pairs': len(df),
                    'trajectory_frames': len(numeric_cols),
                    'energy_statistics': {
                        'mean_energy': float(df[numeric_cols].mean().mean()),
                        'std_energy': float(df[numeric_cols].std().mean()),
                        'min_energy': float(df[numeric_cols].min().min()),
                        'max_energy': float(df[numeric_cols].max().max())
                    }
                }
            except Exception as e:
                report['analysis_results']['energy_analysis'] = {
                    'error': f'Could not analyze energy CSV: {str(e)}'
                }
        
        # Analyze PEN results if available
        pen_csv = os.path.join(out_folder, 'pen_betweenness_centralities.csv')
        if os.path.exists(pen_csv):
            try:
                df_pen = pd.read_csv(pen_csv)
                
                report['analysis_results']['pen_analysis'] = {
                    'unique_residues': len(df_pen['Residue'].unique()),
                    'pen_conditions': len(df_pen.groupby(['include_covalents', 'intEnCutoff'])),
                    'frames_analyzed': len(df_pen['Frame'].unique()),
                    'bc_statistics': {
                        'mean_bc': float(df_pen['BC'].mean()),
                        'max_bc': float(df_pen['BC'].max()),
                        'min_bc': float(df_pen['BC'].min())
                    }
                }
            except Exception as e:
                report['analysis_results']['pen_analysis'] = {
                    'error': f'Could not analyze PEN CSV: {str(e)}'
                }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f'Workflow summary report generated: {report_file}')
        
        # Also create a human-readable text summary
        text_report = os.path.join(out_folder, 'grinn_workflow_summary.txt');
        with open(text_report, 'w') as f:
            f.write("gRINN Workflow Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {report['workflow_info']['timestamp']}\n")
            f.write(f"Output Folder: {out_folder}\n\n")
            
            # System info
            if 'cpu_count' in report['system_info']:
                f.write("System Information:\n")
                f.write(f"  Platform: {report['system_info']['platform']}\n")
                f.write(f"  CPU Cores: {report['system_info']['cpu_count']}\n")
                f.write(f"  Memory: {report['system_info']['memory_gb']} GB\n\n")
            
            # Energy analysis
            if 'energy_analysis' in report['analysis_results']:
                ea = report['analysis_results']['energy_analysis']
                f.write("Energy Analysis:\n")
                f.write(f"  Total Residue Pairs: {ea.get('total_residue_pairs', 'N/A')}\n")
                f.write(f"  Trajectory Frames: {ea.get('trajectory_frames', 'N/A')}\n")
                if 'energy_statistics' in ea:
                    stats = ea['energy_statistics']
                    f.write(f"  Mean Energy: {stats['mean_energy']:.2f} kcal/mol\n")
                    f.write(f"  Energy Range: {stats['min_energy']:.2f} to {stats['max_energy']:.2f} kcal/mol\n")
                f.write("\n")
            
            # PEN analysis
            if 'pen_analysis' in report['analysis_results']:
                pa = report['analysis_results']['pen_analysis']
                f.write("PEN Analysis:\n")
                f.write(f"  Unique Residues: {pa.get('unique_residues', 'N/A')}\n")
                f.write(f"  PEN Conditions: {pa.get('pen_conditions', 'N/A')}\n")
                f.write(f"  Frames Analyzed: {pa.get('frames_analyzed', 'N/A')}\n")
                if 'bc_statistics' in pa:
                    stats = pa['bc_statistics']
                    f.write(f"  Mean Betweenness Centrality: {stats['mean_bc']:.4f}\n")
                    f.write(f"  Max Betweenness Centrality: {stats['max_bc']:.4f}\n")
                f.write("\n")
            
            # Output files
            f.write("Output Files:\n")
            for category, files in report['output_files'].items():
                f.write(f"  {category.replace('_', ' ').title()}:\n")
                if isinstance(files, dict):
                    for file_type, file_info in files.items():
                        if file_info.get('exists', False):
                            f.write(f"    ✓ {file_type}: {file_info['size_mb']} MB\n")
                        else:
                            f.write(f"    ✗ {file_type}: Not found\n")
                f.write("\n")
        
        logger.info(f'Human-readable summary generated: {text_report}')
        
    except Exception as e:
        logger.error(f'Error generating workflow summary report: {str(e)}')

def main():
    args = parse_args()
    run_grinn_workflow(
        args.structure_file, args.out_folder, args.ff_folder, args.initpairfiltercutoff, 
        args.nofixpdb, args.top, args.traj, args.nointeraction, 
        args.gpu, args.solvate, args.npt, args.source_sel, args.target_sel, 
        args.nt, args.skip, args.noconsole_handler,
        create_pen=args.create_pen,
        pen_cutoffs=args.pen_cutoffs,
        pen_include_covalents=args.pen_include_covalents,
        test_only=getattr(args, 'test_only', False),
        force_field=args.force_field,
        water_model=args.water_model,
        recreate_topology=args.recreate_topology,
        ensemble_mode=args.ensemble_mode,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    def global_signal_handler(sig, frame):
            print('Signal caught in main. Exiting...')
            sys.exit(0)

    signal.signal(signal.SIGINT, global_signal_handler)
    signal.signal(signal.SIGTERM, global_signal_handler)
    main()
