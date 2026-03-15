import os
import re
import sys
import argparse
import csv
import json
from functools import lru_cache
from flask import request as flask_request
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx, ALL
import dash
from dash_chat import ChatComponent
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import dash_molstar
from dash_molstar.utils import molstar_helper
from dash_molstar.utils.representations import Representation
import numpy as np
import networkx as nx
from prody import parsePDB
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='gRINN Dashboard - Interactive visualization of protein interaction energies')
    parser.add_argument('results_folder', 
                       help='Path to the results folder containing gRINN output files')
    parser.add_argument('--job-id', dest='job_id', default=None,
                       help='Job identifier (for grinn-web integration). If provided, shown instead of folder path.')
    return parser.parse_args()

def setup_data_paths(results_folder, job_id=None):
    """Setup data paths based on the results folder argument"""
    data_dir = os.path.abspath(results_folder)
    
    # Determine display name for the folder
    # Priority: job_id > real path (if different from data_dir) > data_dir
    if job_id:
        display_name = f"Job: {job_id}"
    elif os.path.realpath(data_dir) != data_dir:
        # Mounted path - show real path
        display_name = os.path.realpath(data_dir)
    else:
        display_name = data_dir
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist!")
        sys.exit(1)
    
    # Define expected file paths
    pdb_path = os.path.join(data_dir, 'system_dry.pdb')
    total_csv = os.path.join(data_dir, 'energies_intEnTotal.csv')
    vdw_csv = os.path.join(data_dir, 'energies_intEnVdW.csv')
    elec_csv = os.path.join(data_dir, 'energies_intEnElec.csv')
    avg_csv = os.path.join(data_dir, 'average_interaction_energies.csv')
    traj_xtc = os.path.join(data_dir, 'traj_dry.xtc')
    
    # Check if required files exist
    required_files = [pdb_path, total_csv, vdw_csv, elec_csv]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files in '{data_dir}':")
        for file in missing_files:
            print(f"  - {os.path.basename(file)}")
        print("\nRequired files for gRINN dashboard:")
        print("  - system_dry.pdb")
        print("  - energies_intEnTotal.csv")
        print("  - energies_intEnVdW.csv")
        print("  - energies_intEnElec.csv")
        print("  - average_interaction_energies.csv (optional)")
        print("  - traj_dry.xtc (optional, for trajectory visualization)")
        sys.exit(1)
    
    # Check if optional files exist
    if not os.path.exists(traj_xtc):
        traj_xtc = None
    if not os.path.exists(avg_csv):
        avg_csv = None
    
    return data_dir, display_name, pdb_path, total_csv, vdw_csv, elec_csv, avg_csv, traj_xtc

def load_average_energies(avg_csv):
    """
    Load pre-computed average energies from CSV file.
    
    Parameters:
    - avg_csv (str): Path to average_interaction_energies.csv
    
    Returns:
    - dict: Nested dictionary with structure {res1: {res2: {'Total': val, 'VdW': val, 'Electrostatic': val}}}
    """
    print("\n[4/5] Loading pre-computed average energies...", flush=True)
    
    try:
        df = pd.read_csv(avg_csv)
        print(f"      ✓ Loaded {len(df)} residue pairs from {os.path.basename(avg_csv)}", flush=True)
    except Exception as e:
        print(f"      ✗ Error loading average energies: {e}", flush=True)
        sys.exit(1)
    
    # Build nested dictionary structure for fast lookups
    _pairwise_avg_energies = {}
    
    for _, row in df.iterrows():
        res1 = row['Residue_1']
        res2 = row['Residue_2']
        
        # Initialize nested dicts if needed
        if res1 not in _pairwise_avg_energies:
            _pairwise_avg_energies[res1] = {}
        if res2 not in _pairwise_avg_energies:
            _pairwise_avg_energies[res2] = {}
        
        # Store both directions for symmetric lookup
        _pairwise_avg_energies[res1][res2] = {
            'Total': round(row['Avg_Total_Energy'], 3),
            'VdW': round(row['Avg_VdW_Energy'], 3),
            'Electrostatic': round(row['Avg_Elec_Energy'], 3)
        }
        _pairwise_avg_energies[res2][res1] = {
            'Total': round(row['Avg_Total_Energy'], 3),
            'VdW': round(row['Avg_VdW_Energy'], 3),
            'Electrostatic': round(row['Avg_Elec_Energy'], 3)
        }
    
    print(f"      ✓ Average energies loaded for {len(_pairwise_avg_energies)} residues", flush=True)
    return _pairwise_avg_energies

def main():
    """Main function to setup and run the dashboard"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup data paths
    data_dir, display_name, pdb_path, total_csv, vdw_csv, elec_csv, avg_csv, traj_xtc = setup_data_paths(args.results_folder, args.job_id)
    
    # Load and transform interaction energy data
    print("\n" + "="*60, flush=True)
    print("🍀 gRINN Dashboard Initialization", flush=True)
    print("="*60, flush=True)
    try:
        print("\n[1/5] Loading energy data files...", flush=True)
        total_df = pd.read_csv(total_csv)
        vdw_df = pd.read_csv(vdw_csv)
        elec_df = pd.read_csv(elec_csv)
        print(f"      ✓ Loaded data from {data_dir}", flush=True)
    except Exception as e:
        print(f"      ✗ Error loading energy data: {e}", flush=True)
        sys.exit(1)

    # Keep original wide-format data for proper network construction
    total_df_wide = total_df.copy()
    vdw_df_wide = vdw_df.copy()
    elec_df_wide = elec_df.copy()

    # Create a combined dataframe with all energy types
    energy_dfs = {
        'Total': total_df,
        'VdW': vdw_df,
        'Electrostatic': elec_df
    }

    print("\n[2/5] Processing energy data...", flush=True)
    # PERFORMANCE OPTIMIZATION: Create Pair column efficiently
    total_df['Pair'] = total_df['res1'] + '-' + total_df['res2']
    vdw_df['Pair'] = vdw_df['res1'] + '-' + vdw_df['res2']
    elec_df['Pair'] = elec_df['res1'] + '-' + elec_df['res2']

    cols2drop = [
        'Unnamed: 0','res1_index','res2_index','res1_chain','res2_chain',
        'res1_resnum','res2_resnum','res1_resname','res2_resname'
    ]

    # Process all energy types
    energy_long = {}
    for energy_type, df in energy_dfs.items():
        # Only drop columns that actually exist in the DataFrame
        existing_cols_to_drop = [col for col in cols2drop if col in df.columns]
        other_cols_to_drop = [col for col in ['res1', 'res2'] if col in df.columns]
        
        # Create a copy to avoid modifying original data
        df_copy = df.copy()
        
        long_df = (
            df_copy
            .drop(columns=existing_cols_to_drop + other_cols_to_drop, errors='ignore')
            .melt(id_vars=['Pair'], var_name='Frame', value_name='Energy')
        )
        long_df['Energy'] = pd.to_numeric(long_df['Energy'], errors='coerce')
        long_df = long_df[long_df['Energy'].notna()].copy()
        
        # Ensure we have data before continuing
        if long_df.empty:
            print(f"Warning: No valid energy data for {energy_type}")
            continue
            
        long_df['EnergyType'] = energy_type
        energy_long[energy_type] = long_df

    # Verify we have at least Total energy data
    if 'Total' not in energy_long or energy_long['Total'].empty:
        print("      ✗ Error: No valid Total energy data found!")
        sys.exit(1)
    
    print("      ✓ Processed energy data successfully", flush=True)

    # Keep the original for compatibility
    total_long = energy_long['Total']
    
    print("\n[3/5] Optimizing data structures for fast access...", flush=True)
    # PERFORMANCE OPTIMIZATION: Convert Frame columns to string once
    for energy_type in energy_long.keys():
        energy_long[energy_type]['Frame'] = energy_long[energy_type]['Frame'].astype(str)
    
    # Update total_long reference
    total_long = energy_long['Total']
    
    # AGGRESSIVE PERFORMANCE OPTIMIZATION: Pre-index data by frame for instant lookups
    frame_indexed_data = {}
    print("      Indexing frames for instant lookup...", flush=True)
    for energy_type in tqdm(energy_long.keys(), desc="      Progress", ncols=70):
        frame_indexed_data[energy_type] = {}
        for frame_str in energy_long[energy_type]['Frame'].unique():
            frame_data = energy_long[energy_type][energy_long[energy_type]['Frame'] == frame_str]
            # Store as dict for O(1) lookups
            frame_indexed_data[energy_type][frame_str] = {
                'pairs': frame_data['Pair'].values,
                'energies': frame_data['Energy'].values
            }
    print("      ✓ Frame indexing complete", flush=True)

    # Determine frame range
    try:
        df_frames = pd.to_numeric(total_long['Frame'], errors='coerce').dropna().astype(int)
        if df_frames.empty:
            print("Error: No valid frame numbers found in data!", flush=True)
            sys.exit(1)
        frame_min, frame_max = int(df_frames.min()), int(df_frames.max())
        print(f"      Frame range: {frame_min} to {frame_max}", flush=True)
        print(f"Frame range: {frame_min} to {frame_max}")
    except Exception as e:
        print(f"Error determining frame range: {e}")
        sys.exit(1)

    # Residue list - sort by chain first, then residue number to maintain protein sequence order
    # PERFORMANCE OPTIMIZATION: Cache residue sort key extraction
    _residue_sort_key_cache = {}  # Fresh cache for new sorting approach
    
    def extract_residue_sort_key(res_name):
        """Extract (chain, residue_number) tuple for sorting residues like 'ALA390_X'"""
        # Don't use cache during initial setup to ensure fresh parsing
        
        try:
            # Extract from residue name like 'ALA390_X' -> chain='X', resnum=390
            parts = res_name.split('_')
            if len(parts) >= 2:
                resnum_part = parts[0]  # e.g., 'ALA390'
                chain = parts[1]  # e.g., 'X'
                
                # Extract residue number
                numbers = re.findall(r'\d+', resnum_part)
                if numbers:
                    resnum = int(numbers[0])
                    result = (chain, resnum)  # Sort by chain first, then number
                    return result
                else:
                    # Debug: print why parsing failed
                    print(f"[DEBUG] No number found in resnum_part: '{resnum_part}' from '{res_name}'")
            else:
                # Debug: print malformed name
                print(f"[DEBUG] Malformed residue name (no underscore): '{res_name}'")
            
            # Fallback for malformed names
            return ('ZZZ', 99999)  # Sort malformed names to the end
        except Exception as e:
            print(f"[DEBUG] Exception parsing '{res_name}': {e}")
            import traceback
            traceback.print_exc()
            return ('ZZZ', 99999)
    
    def sort_residues_by_sequence(residues):
        """Sort residues by chain first, then by residue number within each chain"""
        return sorted(residues, key=extract_residue_sort_key)

    # Get all unique residues from both res1 and res2 columns
    all_residues = set(total_df['res1'].unique()).union(set(total_df['res2'].unique()))
    first_res_list = sort_residues_by_sequence(all_residues)
    
    # Debug: Print first few residues to verify sorting
    print(f"\n[DEBUG] First 20 residues in sorted order:")
    for i, res in enumerate(first_res_list[:20]):
        sort_key = extract_residue_sort_key(res)
        print(f"  {i+1}. {res} -> chain='{sort_key[0]}', resnum={sort_key[1]}")
    if len(first_res_list) > 20:
        print(f"  ... and {len(first_res_list) - 20} more residues")
    
    # PERFORMANCE OPTIMIZATION: Cache sorted residue list for frequent lookups
    _sorted_residues_cache = {}
    
    # Load pre-computed average energies from CSV (if available)
    if avg_csv is not None:
        _pairwise_avg_energies = load_average_energies(avg_csv)
    else:
        print("[4/5] Skipping average energies (file not found)", flush=True)
        _pairwise_avg_energies = {}

    # Simple network cache for faster UI responsiveness

    # Molecular visualization setup
    print("\n[5/5] Setting up molecular viewer...", flush=True)
    try:
        # Detect protein vs ligand chains using ProDy
        structure = parsePDB(pdb_path)
        
        # Get protein chains
        protein_atoms = structure.select('protein')
        protein_chains = sorted(set(protein_atoms.getChids())) if protein_atoms else []
        
        # Get ligand/hetero chains (non-water, non-protein)
        ligand_atoms = structure.select('hetero and not water')
        ligand_chains = sorted(set(ligand_atoms.getChids())) if ligand_atoms else []
        
        print(f"      Detected protein chains: {protein_chains}", flush=True)
        print(f"      Detected ligand chains: {ligand_chains}", flush=True)
        
        # Build components list
        components = []
        
        # Create protein component with cartoon representation
        if protein_chains:
            cartoon = Representation(type='cartoon', color='uniform')
            cartoon.set_color_params({'value': 0xD3D3D3})
            protein_targets = [molstar_helper.get_targets(chain=ch) for ch in protein_chains]
            protein_component = molstar_helper.create_component(
                label='Protein', targets=protein_targets, representation=cartoon
            )
            components.append(protein_component)
        
        # Create ligand component with ball-and-stick representation (if ligands exist)
        if ligand_chains:
            ball_stick = Representation(type='ball-and-stick', color='element-symbol')
            ligand_targets = [molstar_helper.get_targets(chain=ch) for ch in ligand_chains]
            ligand_component = molstar_helper.create_component(
                label='Ligand', targets=ligand_targets, representation=ball_stick
            )
            components.append(ligand_component)
            print(f"      ✓ Added ball-and-stick representation for ligand(s)", flush=True)
        
        # Parse molecule with custom components (disable default representation)
        if components:
            topo = molstar_helper.parse_molecule(pdb_path, component=components, preset={'kind': 'empty'})
        else:
            # Fallback to default if no chains detected
            topo = molstar_helper.parse_molecule(pdb_path)
        
        # Handle trajectory loading
        if traj_xtc:
            coords = molstar_helper.parse_coordinate(traj_xtc)
            def get_full_trajectory():
                return molstar_helper.get_trajectory(topo, coords)
            initial_traj = get_full_trajectory()
            print(f"      ✓ Loaded trajectory from {traj_xtc}", flush=True)
        else:
            # Use static structure only
            initial_traj = topo
            print("      ✓ Using static structure (no trajectory file)", flush=True)
    except Exception as e:
        print(f"      ⚠ Warning: Error setting up molecular viewer: {e}", flush=True)
        print("      Dashboard will continue without 3D viewer functionality", flush=True)
        # Create a minimal fallback
        topo = molstar_helper.parse_molecule(pdb_path)
        initial_traj = topo

    # Load trajectory coordinates for 3D network visualization
    print("\n[6/6] Loading trajectory coordinates for 3D network visualization...", flush=True)
    trajectory_coords = {}
    try:
        if traj_xtc:
            # Use MDTraj to load trajectory (supports XTC format)
            import mdtraj as md
            
            print(f"      Loading trajectory with MDTraj...", flush=True)
            traj = md.load(traj_xtc, top=pdb_path)
            n_frames = traj.n_frames
            
            print(f"      Loaded {n_frames} frames from trajectory", flush=True)
            
            # Create mapping of residue names to MDTraj residue objects
            residue_map = {}
            for res_name in first_res_list:
                # Parse residue name format like 'GLY290_A' -> resnum=290, chain='A'
                parts = res_name.split('_')
                if len(parts) >= 2:
                    resnum_part = parts[0]
                    chain_id = parts[1] if len(parts) > 1 else 'A'
                    
                    # Extract residue number
                    resnum_match = re.findall(r'\d+', resnum_part)
                    if resnum_match:
                        resnum = int(resnum_match[0])
                        
                        # Find the residue in the topology
                        # MDTraj uses 0-based indexing, but PDB uses 1-based
                        for residue in traj.topology.residues:
                            if residue.resSeq == resnum and residue.chain.chain_id == chain_id:
                                residue_map[res_name] = residue
                                break
            
            print(f"      Mapped {len(residue_map)} residues to topology", flush=True)
            
            # Extract coordinates for each frame
            print(f"      Calculating center of mass for each frame...", flush=True)
            for frame_idx in tqdm(range(frame_min, min(frame_max + 1, n_frames)), 
                                 desc="      Progress", ncols=70):
                frame_coords = {}
                
                for res_name, residue in residue_map.items():
                    # Get atom indices for this residue
                    atom_indices = [atom.index for atom in residue.atoms]
                    
                    if len(atom_indices) > 0:
                        # Get coordinates for this frame (in nanometers, convert to Angstroms)
                        coords_nm = traj.xyz[frame_idx, atom_indices, :]  # Shape: (n_atoms, 3)
                        coords_angstrom = coords_nm * 10.0  # Convert nm to Angstrom
                        
                        # Get masses for the atoms
                        masses = np.array([atom.element.mass for atom in residue.atoms])
                        
                        # Calculate center of mass
                        total_mass = np.sum(masses)
                        if total_mass > 0:
                            com = np.sum(coords_angstrom * masses[:, np.newaxis], axis=0) / total_mass
                            frame_coords[res_name] = com.tolist()
                        else:
                            # Fallback to geometric center
                            com = np.mean(coords_angstrom, axis=0)
                            frame_coords[res_name] = com.tolist()
                
                trajectory_coords[frame_idx] = frame_coords
            
            # Supplement trajectory_coords with nodes from pen_precomputed/nodes.csv
            # that are absent from first_res_list (e.g. HSP residues with no energy CSV entry)
            pen_nodes_csv = os.path.join(data_dir, 'pen_precomputed', 'nodes.csv')
            if os.path.exists(pen_nodes_csv):
                try:
                    nodes_df = pd.read_csv(pen_nodes_csv)
                    sample_frame = next(iter(trajectory_coords)) if trajectory_coords else None
                    already_mapped = set(trajectory_coords[sample_frame].keys()) if sample_frame is not None else set()
                    supplemented = 0
                    for _, row in nodes_df.iterrows():
                        res_name = str(row['residue'])
                        if res_name in already_mapped:
                            continue
                        resnum = int(row['resnum'])
                        chain_id = str(row['chain'])
                        for residue in traj.topology.residues:
                            if residue.resSeq == resnum and residue.chain.chain_id == chain_id:
                                atom_indices = [atom.index for atom in residue.atoms]
                                if atom_indices:
                                    masses = np.array([atom.element.mass for atom in residue.atoms])
                                    total_mass = np.sum(masses)
                                    for frame_idx in list(trajectory_coords.keys()):
                                        coords_nm = traj.xyz[frame_idx, atom_indices, :]
                                        coords_ang = coords_nm * 10.0
                                        if total_mass > 0:
                                            com = np.sum(coords_ang * masses[:, np.newaxis], axis=0) / total_mass
                                        else:
                                            com = np.mean(coords_ang, axis=0)
                                        trajectory_coords[frame_idx][res_name] = com.tolist()
                                    supplemented += 1
                                break
                    if supplemented > 0:
                        print(f"      ✓ Supplemented {supplemented} PEN node(s) missing from energy CSV", flush=True)
                except Exception as sup_e:
                    print(f"      ⚠ Warning: Could not supplement PEN node coordinates: {sup_e}", flush=True)

            print(f"      ✓ Loaded center of mass coordinates for {len(trajectory_coords)} frames", flush=True)
        else:
            # Use static structure coordinates (no trajectory available)
            print("      Using static structure coordinates from PDB", flush=True)
            
            # Load PDB as a single-frame trajectory
            import mdtraj as md
            pdb_traj = md.load(pdb_path)
            
            # Create mapping of residue names to MDTraj residue objects
            residue_map = {}
            for res_name in first_res_list:
                parts = res_name.split('_')
                if len(parts) >= 2:
                    resnum_part = parts[0]
                    chain_id = parts[1] if len(parts) > 1 else 'A'
                    
                    resnum_match = re.findall(r'\d+', resnum_part)
                    if resnum_match:
                        resnum = int(resnum_match[0])
                        
                        for residue in pdb_traj.topology.residues:
                            if residue.resSeq == resnum and residue.chain.chain_id == chain_id:
                                residue_map[res_name] = residue
                                break
            
            # Calculate center of mass for each residue in static structure
            static_coords = {}
            for res_name, residue in residue_map.items():
                atom_indices = [atom.index for atom in residue.atoms]
                
                if len(atom_indices) > 0:
                    # Get coordinates (convert nm to Angstrom)
                    coords_nm = pdb_traj.xyz[0, atom_indices, :]
                    coords_angstrom = coords_nm * 10.0
                    
                    # Get masses
                    masses = np.array([atom.element.mass for atom in residue.atoms])
                    
                    # Calculate center of mass
                    total_mass = np.sum(masses)
                    if total_mass > 0:
                        com = np.sum(coords_angstrom * masses[:, np.newaxis], axis=0) / total_mass
                        static_coords[res_name] = com.tolist()
                    else:
                        com = np.mean(coords_angstrom, axis=0)
                        static_coords[res_name] = com.tolist()
            
            # Use same coordinates for all frames
            for frame_idx in range(frame_min, frame_max + 1):
                trajectory_coords[frame_idx] = static_coords.copy()
            
            # Supplement static_coords with nodes from pen_precomputed/nodes.csv
            # that are absent from first_res_list (e.g. HSP residues with no energy CSV entry)
            pen_nodes_csv = os.path.join(data_dir, 'pen_precomputed', 'nodes.csv')
            if os.path.exists(pen_nodes_csv):
                try:
                    nodes_df = pd.read_csv(pen_nodes_csv)
                    already_mapped = set(static_coords.keys())
                    supplemented = 0
                    for _, row in nodes_df.iterrows():
                        res_name = str(row['residue'])
                        if res_name in already_mapped:
                            continue
                        resnum = int(row['resnum'])
                        chain_id = str(row['chain'])
                        for residue in pdb_traj.topology.residues:
                            if residue.resSeq == resnum and residue.chain.chain_id == chain_id:
                                atom_indices = [atom.index for atom in residue.atoms]
                                if atom_indices:
                                    masses = np.array([atom.element.mass for atom in residue.atoms])
                                    total_mass = np.sum(masses)
                                    coords_nm = pdb_traj.xyz[0, atom_indices, :]
                                    coords_ang = coords_nm * 10.0
                                    if total_mass > 0:
                                        com = np.sum(coords_ang * masses[:, np.newaxis], axis=0) / total_mass
                                    else:
                                        com = np.mean(coords_ang, axis=0)
                                    static_coords[res_name] = com.tolist()
                                    supplemented += 1
                                break
                    if supplemented > 0:
                        # Propagate newly added coords to all already-written frame dicts
                        for frame_idx in list(trajectory_coords.keys()):
                            for res_name, coords in static_coords.items():
                                if res_name not in trajectory_coords[frame_idx]:
                                    trajectory_coords[frame_idx][res_name] = coords
                        print(f"      ✓ Supplemented {supplemented} PEN node(s) missing from energy CSV", flush=True)
                except Exception as sup_e:
                    print(f"      ⚠ Warning: Could not supplement PEN node coordinates: {sup_e}", flush=True)

            print(f"      ✓ Loaded static center of mass coordinates for {len(static_coords)} residues", flush=True)
    except Exception as e:
        print(f"      ⚠ Warning: Could not load trajectory coordinates: {e}", flush=True)
        print(f"      Error details: {type(e).__name__}", flush=True)
        import traceback
        traceback.print_exc()
        print("      3D network visualization will use default positions", flush=True)
        trajectory_coords = {}

    # --- PEN precomputed outputs (workflow-owned; dashboard load-only) ---
    pen_root = os.path.join(data_dir, 'pen_precomputed')
    pen_manifest = None
    if os.path.exists(os.path.join(pen_root, 'manifest.json')):
        try:
            with open(os.path.join(pen_root, 'manifest.json'), 'r') as f:
                pen_manifest = json.load(f)
            print(f"      ✓ Loaded PEN manifest from {pen_root}", flush=True)
        except Exception as e:
            print(f"      ⚠ Warning: Could not read PEN manifest: {e}", flush=True)

    pen_cutoffs = [1.0]
    if isinstance(pen_manifest, dict) and isinstance(pen_manifest.get('cutoffs'), list) and pen_manifest['cutoffs']:
        pen_cutoffs = [float(x) for x in pen_manifest['cutoffs']]

    def _pen_cov_flag(include_cov_value):
        return 1 if include_cov_value and 'include' in str(include_cov_value) else 0

    def _pen_energy_key(energy_type_selector_value: str) -> str:
        if energy_type_selector_value in ('Total', 'VdW', 'Electrostatic'):
            return energy_type_selector_value
        if energy_type_selector_value == 'Elec':
            return 'Electrostatic'
        return 'Total'

    def _pen_metrics_path(energy_key: str, cov_flag: int, cutoff: float) -> str:
        return os.path.join(pen_root, f"metrics_{energy_key}_cov{cov_flag}_cutoff{float(cutoff)}.csv")

    def _pen_edges_path(energy_key: str, cov_flag: int, cutoff: float, frame: int) -> str:
        return os.path.join(pen_root, f"edges_{energy_key}_cov{cov_flag}_cutoff{float(cutoff)}_frame{int(frame)}.csv")

    def _pen_paths_path(energy_key: str, cov_flag: int, cutoff: float, frame: int) -> str:
        return os.path.join(pen_root, f"paths_{energy_key}_cov{cov_flag}_cutoff{float(cutoff)}_frame{int(frame)}.csv")

    @lru_cache(maxsize=16)
    def _load_metrics_df(path: str):
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    # --- Helper functions for Network Metrics visualization ---
    def _sort_residues_by_metric(df: pd.DataFrame, metric: str, residues: list, order: str) -> list:
        """Sort residues based on mean metric value across all frames.
        
        Args:
            df: Metrics DataFrame with columns [frame, residue, degree, betweenness, closeness]
            metric: One of 'degree', 'betweenness', 'closeness'
            residues: List of residue names to sort
            order: 'sequence' (original), 'ascending', or 'descending'
        
        Returns:
            Sorted list of residue names
        """
        if order == 'sequence' or df is None or df.empty:
            return residues
        
        # Compute mean metric per residue
        filtered = df[df['residue'].isin(residues)]
        if filtered.empty:
            return residues
        
        mean_values = filtered.groupby('residue')[metric].mean()
        
        # Sort residues by mean value
        sorted_residues = []
        for res in residues:
            if res in mean_values.index:
                sorted_residues.append((res, mean_values[res]))
            else:
                sorted_residues.append((res, 0.0))
        
        ascending = (order == 'ascending')
        sorted_residues.sort(key=lambda x: x[1], reverse=not ascending)
        
        return [r[0] for r in sorted_residues]

    def _filter_metrics_by_cutoff(df: pd.DataFrame, metric: str, lower: float, upper: float) -> tuple:
        """Filter residues based on mean metric value cutoffs.
        
        Args:
            df: Metrics DataFrame with columns [frame, residue, degree, betweenness, closeness]
            metric: One of 'degree', 'betweenness', 'closeness'
            lower: Lower cutoff (None = no lower bound)
            upper: Upper cutoff (None = no upper bound)
        
        Returns:
            Tuple of (filtered_df, filtered_residues)
        """
        if df is None or df.empty:
            return df, []
        
        # Compute mean metric per residue
        mean_values = df.groupby('residue')[metric].mean()
        
        # Apply cutoffs
        valid_residues = set(mean_values.index)
        if lower is not None:
            valid_residues = {r for r in valid_residues if mean_values[r] >= lower}
        if upper is not None:
            valid_residues = {r for r in valid_residues if mean_values[r] <= upper}
        
        filtered_df = df[df['residue'].isin(valid_residues)]
        return filtered_df, list(valid_residues)

    def _create_metrics_violin_figure(df: pd.DataFrame, metric: str, current_frame: int, 
                                       residues: list, frame_min: int, frame_max: int,
                                       soft_palette: dict) -> go.Figure:
        """Create violin plot showing metric distribution per residue.
        
        Args:
            df: Metrics DataFrame with columns [frame, residue, degree, betweenness, closeness]
            metric: One of 'degree', 'betweenness', 'closeness'
            current_frame: Current frame number to highlight
            residues: List of residue names to display
            frame_min: Minimum frame number
            frame_max: Maximum frame number
            soft_palette: Color palette dictionary
        
        Returns:
            Plotly Figure with violin plots (horizontally scrollable for many residues)
        """
        metric_titles = {
            'degree': 'Degree Centrality',
            'betweenness': 'Betweenness Centrality',
            'closeness': 'Closeness Centrality'
        }
        
        fig = go.Figure()
        
        if df is None or df.empty:
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=soft_palette['text'])
            )
            return fig
        
        # Filter data for selected residues (no limit - scrolling handles large numbers)
        plot_df = df[df['residue'].isin(residues)].copy()
        
        if plot_df.empty:
            fig.add_annotation(
                text="No data for selected residues",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=soft_palette['text'])
            )
            return fig
        
        # Get current frame values for markers
        current_frame_data = plot_df[plot_df['frame'] == current_frame]
        current_values = {row['residue']: row[metric] for _, row in current_frame_data.iterrows()}
        
        # Create violin traces using categorical x-axis with residue names
        # This ensures proper alignment of violins and markers
        for res in residues:
            res_data = plot_df[plot_df['residue'] == res][metric].values
            if len(res_data) == 0:
                continue
            
            fig.add_trace(go.Violin(
                x=[res] * len(res_data),  # Use residue name for x-axis (categorical)
                y=res_data,
                name=res,
                box_visible=True,
                meanline_visible=True,
                fillcolor='rgba(70, 130, 180, 0.5)',
                line_color='rgb(70, 130, 180)',
                opacity=0.7,
                showlegend=False,
                hovertemplate=f'<b>{res}</b><br>{metric_titles[metric]}: %{{y:.4f}}<extra></extra>'
            ))
        
        # Add scatter markers for current frame values - use residue names (categorical x)
        current_residues = []
        current_y = []
        for res in residues:
            if res in current_values:
                current_residues.append(res)
                current_y.append(current_values[res])
        
        if current_residues:
            fig.add_trace(go.Scatter(
                x=current_residues,  # Use residue names to align with violins
                y=current_y,
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(width=1, color='darkred')
                ),
                name=f'Frame {current_frame}',
                hovertemplate='<b>%{x}</b><br>Frame ' + str(current_frame) + ': %{y:.4f}<extra></extra>',
                showlegend=True
            ))
        
        title_text = f'{metric_titles[metric]} Distribution per Residue'
        if len(residues) > 200:
            title_text += f' ({len(residues)} residues - scroll horizontally)'
        
        # Calculate dynamic width based on number of residues
        # Minimum 800px, add ~25px per residue for comfortable viewing
        dynamic_width = max(800, len(residues) * 25)
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=14, color=soft_palette['primary'], family='Roboto, sans-serif')
            ),
            xaxis=dict(
                title='Residue',
                type='category',  # Categorical axis for residue names
                categoryorder='array',
                categoryarray=residues,  # Maintain residue order
                tickangle=45,
                tickfont=dict(size=12),
                showgrid=False
            ),
            yaxis=dict(
                title=metric_titles[metric],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=12)
            ),
            plot_bgcolor='white',
            paper_bgcolor='rgba(255,255,255,0.9)',
            margin=dict(l=80, r=40, t=50, b=100),
            font=dict(family='Roboto, sans-serif', color='#4A5A4A', size=12),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            autosize=True  # Fit to container width
        )
        
        return fig

    def _load_edges_for_frame(path: str):
        if not os.path.exists(path):
            return []
        out = []
        with open(path, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                out.append({
                    'source': row.get('source'),
                    'target': row.get('target'),
                    'weight': float(row.get('weight', 0.0)),
                    'distance': float(row.get('distance', 1.0)),
                })
        return out

    def _compute_shortest_paths_for_pair_from_edges(edges: list, source: str, target: str):
        """Compute all shortest paths between source and target using the same method as grinn_workflow.py.

        - Builds an undirected NetworkX graph from edges.csv rows with 'distance' edge attribute.
        - Uses nx.dijkstra_predecessor_and_distance(..., weight='distance')
        - Enumerates all shortest paths via predecessor lists.
        """
        if not edges:
            return []
        if not source or not target or source == target:
            return []

        G = nx.Graph()
        for e in edges:
            u = e.get('source')
            v = e.get('target')
            if not u or not v:
                continue
            try:
                weight = float(e.get('weight', 0.0))
            except Exception:
                weight = 0.0
            try:
                distance = float(e.get('distance', 1.0))
            except Exception:
                distance = 1.0
            G.add_edge(u, v, weight=weight, distance=distance)

        if not G.has_node(source) or not G.has_node(target):
            return []

        try:
            pred, dist = nx.dijkstra_predecessor_and_distance(G, source, weight='distance')
        except Exception:
            return []

        if target not in dist:
            return []

        rows = []
        # Enumerate all shortest paths from source to target using predecessor lists.
        stack = [(target, [target])]
        while stack:
            node, path_rev = stack.pop()
            if node == source:
                path = list(reversed(path_rev))
                total_distance = 0.0
                for uu, vv in zip(path[:-1], path[1:]):
                    ed = G.get_edge_data(uu, vv) or {}
                    total_distance += float(ed.get('distance', 1.0))
                hops = len(path) - 1
                path_str = ' --> '.join(path)
                rows.append({
                    'path': path_str,
                    'length': f"{total_distance:.4f}",
                    'hops': hops,
                })
                continue

            for p in pred.get(node, []):
                stack.append((p, path_rev + [p]))

        try:
            rows.sort(key=lambda x: float(x['length']))
        except Exception:
            pass
        return rows

    def _load_shortest_paths_for_pair(paths_csv_path: str, edges_csv_path: str, source: str, target: str):
        """Load shortest paths from precomputed paths CSV if present; otherwise compute from edges CSV."""
        if os.path.exists(paths_csv_path):
            rows = []
            with open(paths_csv_path, 'r', newline='') as f:
                r = csv.DictReader(f)
                for row in r:
                    if row.get('source') == source and row.get('target') == target:
                        rows.append({
                            'path': row.get('path', ''),
                            'length': row.get('length', ''),
                            'hops': int(row.get('hops', 0)) if str(row.get('hops', '')).isdigit() else row.get('hops', ''),
                        })
            try:
                rows.sort(key=lambda x: float(x['length']))
            except Exception:
                pass
            return rows

        # Fallback: compute on-the-fly using edges for this frame.
        edges = _load_edges_for_frame(edges_csv_path)
        return _compute_shortest_paths_for_pair_from_edges(edges, source, target)

    print("      ✓ PEN dashboard mode: load-only (no NetworkX computations)", flush=True)

    # App layout
    # AGGRESSIVE OPTIMIZATION: Configure app for maximum performance
    # Get URL base pathname from environment (for grinn-web proxy integration)
    url_base_pathname = os.getenv('DASH_URL_BASE_PATHNAME', '/')
    if url_base_pathname != '/':
        print(f"      ✓ Using URL base pathname: {url_base_pathname}", flush=True)
    
    app = Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,  # Faster callback registration
        compress=True,  # Enable compression for faster data transfer
        url_base_pathname=url_base_pathname  # Support reverse proxy routing
    )

    # Add Flask error handler to capture and log 500 errors
    @app.server.errorhandler(500)
    def handle_500_error(e):
        import traceback
        print("\n" + "="*60, flush=True)
        print("🔴 FLASK 500 ERROR CAUGHT:", flush=True)
        print("="*60, flush=True)
        print(f"Exception: {e}", flush=True)
        print("Traceback:", flush=True)
        traceback.print_exc()
        print("="*60 + "\n", flush=True)
        # Re-raise to let Flask handle the response
        raise e

    # Soft, harmonious color palette with low contrast and close color relationships
    soft_palette = {
        'primary': '#7C9885',          # Soft sage green
        'secondary': '#A8C4A2',        # Light sage
        'tertiary': '#B8D4B8',         # Very light sage
        'accent': '#9AB3A8',           # Soft mint
        'light_accent': '#C5D5C5',     # Very light mint
        'background': '#F5F7F5',       # Off-white with hint of green
        'white': '#FFFFFF',            # Pure white
        'surface': '#E8EDE8',          # Light surface
        'muted': '#D0D8D0',           # Muted green-gray
        'text': '#4A5A4A',            # Soft dark green for text
        'border': '#B5C5B5',         # Soft border color
        'hover': '#95A895',           # Slightly darker for hover states
        'active': '#6B806B',          # Active state color
        'subtle': '#F0F3F0',          # Very subtle background
        'light_blue': '#B8D4E6',      # Light blue for St. Patrick's theme
        'pale_blue': '#E6F2F8'        # Very pale blue
    }

    # CSS styling is now loaded from external file in assets/styles.css

    # Chatbot state stores
    session_store = dcc.Store(id='chat-session-id', storage_type='session')
    chatbot_visible_store = dcc.Store(id='chatbot-visible', data=False, storage_type='session')
    chatbot_expanded_store = dcc.Store(id='chatbot-expanded', data=False, storage_type='session')
    cleanup_store = dcc.Store(id='chat-cleanup', storage_type='memory')

    def _maybe_load_env_file() -> None:
        """Best-effort load of KEY=VALUE pairs from a local env file.

        This lets users set PANDASAI_MODELS (and API keys) in a file when the
        dashboard is launched without those vars exported in the shell.

                Precedence:
                - Existing process env vars are never overwritten (except empty values).
                - File candidates (first found wins):
                    1) $PANDASAI_ENV_FILE / $DOTENV_FILE
                    2) ./.env then ./.venv (if .venv is a file, not a directory)
                    3) <this_script_dir>/.env
                    4) <repo_root>/grinn-web/.env (common sibling checkout)
        """

        def _strip_quotes(v: str) -> str:
            v = v.strip()
            if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                return v[1:-1]
            return v

        def _load_file(path: str) -> bool:
            try:
                if not path or not os.path.exists(path) or not os.path.isfile(path):
                    return False
                with open(path, 'r', encoding='utf-8') as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith('#'):
                            continue
                        if line.lower().startswith('export '):
                            line = line[7:].lstrip()
                        if '=' not in line:
                            continue
                        k, v = line.split('=', 1)
                        k = k.strip()
                        if not k:
                            continue
                        # Don't override existing env vars, unless they are empty.
                        if k in os.environ and str(os.environ.get(k, '')).strip() != '':
                            continue
                        os.environ[k] = _strip_quotes(v)
                return True
            except Exception:
                return False

        candidates: list[str] = []
        for env_key in ('PANDASAI_ENV_FILE', 'DOTENV_FILE'):
            p = (os.getenv(env_key) or '').strip()
            if p:
                candidates.append(p)
        cwd = os.getcwd()
        candidates.append(os.path.join(cwd, '.env'))
        candidates.append(os.path.join(cwd, '.venv'))
        try:
            candidates.append(os.path.join(os.path.dirname(__file__), '.env'))
        except Exception:
            pass

        # If grinn-web is checked out next to grinn/, load its .env as well.
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            sibling_env = os.path.abspath(os.path.join(repo_root, '..', 'grinn-web', '.env'))
            candidates.append(sibling_env)
        except Exception:
            pass

        # Also try to find grinn-web/.env up a few levels from CWD.
        try:
            for i in range(0, 4):
                base = os.path.abspath(os.path.join(cwd, *(['..'] * i)))
                candidates.append(os.path.join(base, 'grinn-web', '.env'))
        except Exception:
            pass

        for p in candidates:
            # If .venv is a directory (common), skip it.
            if os.path.basename(p) == '.venv' and os.path.isdir(p):
                continue
            if _load_file(p):
                break

    # Load env file before reading PANDASAI_MODELS/default model.
    _maybe_load_env_file()

    MAX_CHAT_VALUES: int = int(os.getenv('GRINN_CHAT_MAX_VALUES', '5000'))

    def _parse_models_env(raw: str) -> list[str]:
        import json

        s = (raw or '').strip()
        if not s:
            return []

        # JSON array format: ["model1","model2"]
        if s.startswith('['):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    out: list[str] = []
                    for item in parsed:
                        if isinstance(item, str) and item.strip():
                            out.append(item.strip())
                    # de-dupe while preserving order
                    seen = set()
                    uniq: list[str] = []
                    for m in out:
                        if m not in seen:
                            seen.add(m)
                            uniq.append(m)
                    return uniq
            except Exception:
                pass

        # Comma/newline separated
        parts = [p.strip() for p in s.replace('\n', ',').split(',')]
        out = [p for p in parts if p]
        seen = set()
        uniq: list[str] = []
        for m in out:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq

    def _get_available_models() -> tuple[list[str], str]:
        models = _parse_models_env(os.getenv('PANDASAI_MODELS', ''))
        fallback = 'gemini/gemini-pro-latest'
        default_model = (os.getenv('PANDASAI_DEFAULT_MODEL') or os.getenv('PANDASAI_MODEL') or (models[0] if models else fallback)).strip()
        if not models:
            models = [default_model or fallback]
        elif default_model and default_model not in models:
            models = [default_model] + models
        return models, (default_model or models[0] or fallback)

    AVAILABLE_MODELS, DEFAULT_MODEL = _get_available_models()

    # Token limit from environment (0 or empty = unlimited)
    _token_limit_raw = os.getenv('PANDASAI_TOKEN_LIMIT', '').strip()
    PANDASAI_TOKEN_LIMIT = int(_token_limit_raw) if _token_limit_raw.isdigit() else 0

    # --- Build DataFrame registry for chatbot context selection ---
    # This registry allows users to select which DataFrames to include in LLM queries.
    # Categories: Pairwise Interaction Energies, Network Metrics, Network Edges
    import glob
    import re as _re

    def _pen_folder_from_dashboard() -> str:
        return os.path.join(data_dir, 'pen_precomputed')

    def _build_chatbot_dataframe_registry():
        """Build the registry of available DataFrames for chatbot queries."""
        registry_items = {}

        # 1. Pairwise Interaction Energy DataFrames (wide format - original CSVs)
        for energy_type, df in [('Total', total_df_wide), ('VdW', vdw_df_wide), ('Electrostatic', elec_df_wide)]:
            key = f"IE_{energy_type}"
            registry_items[key] = {
                'df': df,
                'category': 'Pairwise Energies',
                'label': f"IE: {energy_type}",
                'description': f"Pairwise {energy_type} interaction energies (all frames)",
                'rows': len(df),
                'cols': len(df.columns),
            }

        # 2. Network Metrics DataFrames (from pen_precomputed/metrics_*.csv)
        pen = _pen_folder_from_dashboard()
        metrics_files = sorted(glob.glob(os.path.join(pen, 'metrics_*.csv')))
        for path in metrics_files:
            basename = os.path.basename(path)
            m = _re.match(r"metrics_(?P<energy>[^_]+)_cov(?P<cov>[01])_cutoff(?P<cutoff>-?[0-9.]+)\.csv", basename)
            if not m:
                continue
            energy = m.group('energy')
            cov = 'Cov' if m.group('cov') == '1' else 'NoCov'
            cutoff = m.group('cutoff')
            key = f"Metrics_{energy}_{cov}_Cut{cutoff}"
            try:
                df = pd.read_csv(path)
                registry_items[key] = {
                    'df': df,
                    'category': 'Network Metrics',
                    'label': f"Metrics: {energy} {cov} Cut{cutoff}",
                    'description': f"Network metrics ({energy}, covalents={'included' if cov == 'Cov' else 'excluded'}, cutoff={cutoff})",
                    'rows': len(df),
                    'cols': len(df.columns),
                }
            except Exception:
                continue

        # NOTE: Network Edges DataFrames are excluded from chatbot to reduce token usage.
        # Edge files are per-frame and can be very large; use the Network tab for edge analysis.

        return registry_items

    _CHATBOT_DATAFRAMES = _build_chatbot_dataframe_registry()

    # Build dropdown options grouped by category
    def _get_dataframe_dropdown_options():
        options = []
        categories = {}
        for key, info in _CHATBOT_DATAFRAMES.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({'label': info['label'], 'value': key})
        # Flatten with category headers
        for cat in ['Pairwise Energies', 'Network Metrics', 'Network Edges']:
            if cat in categories:
                for opt in categories[cat]:
                    options.append(opt)
        return options

    DATAFRAME_OPTIONS = _get_dataframe_dropdown_options()
    DEFAULT_DATAFRAMES = ['IE_Total']  # Default selection
    if 'Metrics_Total_Cov_Cut1.0' in _CHATBOT_DATAFRAMES:
        DEFAULT_DATAFRAMES.append('Metrics_Total_Cov_Cut1.0')
    elif _CHATBOT_DATAFRAMES:
        # Pick first metrics if available
        for k in _CHATBOT_DATAFRAMES:
            if k.startswith('Metrics_'):
                DEFAULT_DATAFRAMES.append(k)
                break

    def _resolve_model_selection(selected: str | None) -> str:
        if isinstance(selected, str) and selected.strip() and selected.strip() in AVAILABLE_MODELS:
            return selected.strip()
        return DEFAULT_MODEL

    app.layout = dbc.Container([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div(style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'space-between',
                        'margin': '4px 0 6px 0'
                    }, children=[
                        html.H1("gRINN Workflow Results",
                                className="main-title",
                                style={
                                    'color': soft_palette['primary'],
                                    'fontFamily': 'Roboto, sans-serif',
                                    'fontWeight': '700',
                                    'fontSize': '1.6rem',
                                    'margin': '0',
                                    'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                                    'letterSpacing': '1px',
                                    'flex': '0 0 auto'
                                }),
                        html.P(f"📁 {display_name}",
                               style={
                                   'textAlign': 'center',
                                   'color': soft_palette['text'],
                                   'fontFamily': 'Roboto, sans-serif',
                                   'fontSize': '0.9rem',
                                   'margin': '0',
                                   'flex': '1',
                                   'paddingLeft': '12px',
                                   'paddingRight': '12px',
                                   'wordBreak': 'break-all'
                               }),
                        html.Button('💬 gRINN Chatbot', id='toggle-chatbot', n_clicks=0,
                                    style={
                                        'backgroundColor': soft_palette['primary'],
                                        'color': 'white',
                                        'border': 'none',
                                        'borderRadius': '8px',
                                        'padding': '6px 10px',
                                        'fontWeight': 'bold',
                                        'fontSize': '12px',
                                        'cursor': 'pointer',
                                        'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
                                    })
                    ])
                ], width=12)
            ]),
            # Store components for chatbot state
            session_store,
            chatbot_visible_store,
            chatbot_expanded_store,
            cleanup_store,
            dcc.Interval(id='chat-cleanup-tick', interval=60_000, n_intervals=0),
        ]),
        html.Div([
            dbc.Row([
            # Left Panel: Tabs
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Tabs(id='main-tabs', value='tab-pairwise', 
                                 style={
                                     'fontFamily': 'Roboto, sans-serif',
                                     'fontWeight': '500',
                                     'height': '32px',
                                     'fontSize': '12px'
                                 },
                                 colors={
                                     'border': soft_palette['border'],
                                     'primary': soft_palette['primary'],
                                     'background': soft_palette['surface']
                                 }, children=[
                    # Pairwise Energies Tab
                    dcc.Tab(label='🔗 Pairwise Energies', value='tab-pairwise', children=[
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.H6("🎯 Select First Residue", className="text-white text-center mb-0", style={'fontSize': '13px'})
                                        ], style={'backgroundColor': soft_palette["light_blue"], 'padding': '6px'}),
                                        dbc.CardBody([
                                            dash_table.DataTable(
                                                id='first_residue_table',
                                                columns=[{'name': 'Residue', 'id': 'Residue'}],
                                                data=[{'Residue': r} for r in first_res_list],
                                                row_selectable='single',
                                                style_table={
                                                    'height': 'calc(100vh - 360px)',
                                                    'overflowY': 'scroll',
                                                    'borderRadius': '8px',
                                                    'border': f'2px solid {soft_palette["accent"]}',
                                                    'backgroundColor': 'rgba(250,255,250,0.4)'
                                                },
                                                style_header={
                                                    'backgroundColor': soft_palette['accent'],
                                                    'color': 'white',
                                                    'fontWeight': 'bold',
                                                    'textAlign': 'center'
                                                },
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'fontFamily': 'Roboto, sans-serif',
                                                    'fontSize': '12px',
                                                    'backgroundColor': 'rgba(250,255,250,0.4)',
                                                    'border': '1px solid #C0E0C0',
                                                    'padding': '6px'
                                                },
                                                style_data_conditional=[
                                                    {
                                                        'if': {'state': 'selected'},
                                                        'backgroundColor': soft_palette['light_accent'],
                                                        'border': f'2px solid {soft_palette["primary"]}'
                                                    }
                                                ]
                                            )
                                        ], style={'padding': '5px'})
                                    ])
                                ], width=2),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.H6("🎯 Select Second Residue", className="text-white text-center mb-0", style={'fontSize': '13px'})
                                        ], style={'backgroundColor': soft_palette["light_blue"], 'padding': '6px'}),
                                        dbc.CardBody([
                                            dash_table.DataTable(
                                                id='second_residue_table',
                                                columns=[{'name': 'Residue', 'id': 'Residue'}],
                                                data=[],
                                                row_selectable='single',
                                                style_table={
                                                    'height': 'calc(100vh - 360px)',
                                                    'overflowY': 'scroll',
                                                    'borderRadius': '8px',
                                                    'border': f'2px solid {soft_palette["accent"]}',
                                                    'backgroundColor': 'rgba(250,255,250,0.4)'
                                                },
                                                style_header={
                                                    'backgroundColor': soft_palette['accent'],
                                                    'color': 'white',
                                                    'fontWeight': 'bold',
                                                    'textAlign': 'center'
                                                },
                                                style_cell={
                                                    'textAlign': 'center',
                                                    'fontFamily': 'Roboto, sans-serif',
                                                    'fontSize': '12px',
                                                    'backgroundColor': 'rgba(250,255,250,0.4)',
                                                    'border': '1px solid #C0E0C0',
                                                    'padding': '6px'
                                                },
                                                style_data_conditional=[
                                                    {
                                                        'if': {'state': 'selected'},
                                                        'backgroundColor': soft_palette['light_accent'],
                                                        'border': f'2px solid {soft_palette["primary"]}'
                                                    }
                                                ]
                                            )
                                        ], style={'padding': '5px'})
                                    ])
                                ], width=2),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.H6("📊 Average Energies", className="text-white text-center mb-0", style={'fontSize': '13px'})
                                        ], style={'backgroundColor': soft_palette["light_blue"], 'padding': '6px'}),
                                        dbc.CardBody([
                                            dcc.Graph(
                                                id='energy_bar_chart',
                                                style={'height': 'calc(100vh - 360px)'},
                                                config={'displayModeBar': False}
                                            )
                                        ], style={'padding': '5px'})
                                    ])
                                ], width=2),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            dcc.Graph(id='total_energy_graph', style={'height': 'calc((100vh - 420px) / 3)'}),
                                            dcc.Graph(id='vdw_energy_graph', style={'height': 'calc((100vh - 420px) / 3)'}),
                                            dcc.Graph(id='elec_energy_graph', style={'height': 'calc((100vh - 420px) / 3)'})
                                        ])
                                    ])
                                ], width=6)
                            ])
                        ], fluid=True)
                    ]),
                    # Interaction Energy Matrix Tab
                    dcc.Tab(label='🔥 Interaction Energy Matrix', value='tab-matrix', children=[
                        dbc.Container([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.Div([
                                                # Energy Type Selector (Left side)
                                                html.Div([
                                                    html.Label("🎯 Type:", className="text-white mb-0", style={'fontSize': '12px', 'marginRight': '10px'}),
                                                    dcc.RadioItems(
                                                        id='energy_type_selector',
                                                        options=[
                                                            {'label': '🔥 Total', 'value': 'Total'},
                                                            {'label': '⚡ Elec', 'value': 'Electrostatic'},
                                                            {'label': '🌊 VdW', 'value': 'VdW'}
                                                        ],
                                                        value='Total',
                                                        inline=True,
                                                        className="text-white",
                                                        style={'fontSize': '12px'}
                                                    )
                                                ], style={'display': 'flex', 'alignItems': 'center', 'flexShrink': 0}),
                                                
                                                # Range Controls (Right side)
                                                html.Div([
                                                    html.Label("📊 Range:", className="text-white mb-0", style={'fontSize': '12px', 'marginRight': '10px'}),
                                                    html.Div([
                                                        dcc.Slider(
                                                            id='heatmap_range_slider',
                                                            min=1,
                                                            max=20,
                                                            step=0.5,
                                                            value=10,
                                                            marks={i: {'label': f'±{i}', 'style': {'color': 'white', 'fontSize': '9px', 'fontWeight': 'bold'}} for i in range(1, 21, 5)},
                                                            tooltip={'placement': 'bottom', 'always_visible': True},
                                                            updatemode='mouseup'
                                                        )
                                                    ], style={'width': '180px', 'marginRight': '15px'}),
                                                    html.Label("Manual:", className="text-white mb-0", style={'fontSize': '12px', 'marginRight': '5px'}),
                                                    dcc.Input(
                                                        id='manual_range_input',
                                                        type='number',
                                                        placeholder='±value',
                                                        min=0.1,
                                                        step=0.1,
                                                        style={
                                                            'width': '60px',
                                                            'borderRadius': '5px',
                                                            'border': '2px solid #9AB3A8',
                                                            'padding': '3px',
                                                            'textAlign': 'center',
                                                            'fontSize': '11px'
                                                        }
                                                    )
                                                ], style={'display': 'flex', 'alignItems': 'center', 'marginLeft': 'auto'})
                                            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'width': '100%'})
                                        ], style={'backgroundColor': soft_palette["light_blue"], 'padding': '10px'}),
                                        dbc.CardBody([
                                            dcc.Graph(id='matrix_heatmap', style={'height': 'calc(100vh - 360px)'})
                                        ])
                                    ])
                                ], width=12)
                            ])
                        ], fluid=True)
                    ]),
                    # Network Analysis Tab
                    dcc.Tab(label='🕸️ Network Analysis', value='tab-network', children=[
                        html.Div(id='network-analysis-tab', className="tab-content", style={
                            'background': f'rgba(248,255,248,0.6)',
                            'borderRadius': '10px',
                            'padding': '8px',
                            'margin': '6px',
                            'border': f'2px solid {soft_palette["accent"]}'
                        }, children=[
                            # General Network Settings (moved from Network Metrics tab)
                            html.Div(style={
                                'display': 'flex', 
                                'alignItems': 'center', 
                                'paddingBottom': '15px',
                                'background': soft_palette["light_blue"],
                                'borderRadius': '10px',
                                'padding': '10px',
                                'marginBottom': '10px',
                                'boxShadow': '0 4px 16px rgba(0,0,0,0.1)'
                            }, children=[
                                dcc.Checklist(
                                    id='include_covalent_edges',
                                    options=[{'label': '🔗 Include covalent bonds as edges', 'value': 'include'}],
                                    value=['include'],
                                    style={
                                        'marginRight': '20px',
                                        'color': 'white',
                                        'fontFamily': 'Roboto, sans-serif',
                                        'fontWeight': '500',
                                        'fontSize': '12px'
                                    }
                                ),
                                html.Label("🔥 Energy type:", style={
                                    'color': 'white',
                                    'fontFamily': 'Roboto, sans-serif',
                                    'fontWeight': '500',
                                    'fontSize': '12px',
                                    'marginRight': '10px'
                                }),
                                dcc.RadioItems(
                                    id='network_energy_type',
                                    options=[
                                        {'label': '🔥 Total', 'value': 'Total'},
                                        {'label': '⚡ Elec', 'value': 'Electrostatic'},
                                        {'label': '🌊 VdW', 'value': 'VdW'},
                                    ],
                                    value='Total',
                                    inline=True,
                                    className='text-white',
                                    style={
                                        'fontSize': '12px',
                                        'marginRight': '20px'
                                    }
                                ),
                                html.Label("⚡ Edge addition energy cutoff (kcal/mol): ", style={
                                    'color': 'white',
                                    'fontFamily': 'Roboto, sans-serif',
                                    'fontWeight': '500',
                                    'fontSize': '12px'
                                }),
                                dcc.Dropdown(
                                    id='energy_cutoff',
                                    options=[{'label': str(float(c)), 'value': float(c)} for c in pen_cutoffs],
                                    value=float(pen_cutoffs[0]) if pen_cutoffs else 1.0,
                                    clearable=False,
                                    style={
                                        'width': '120px',
                                        'marginRight': '20px',
                                    }
                                ),
                                html.Button('🔄 Update Network', 
                                           id='update_network_btn', 
                                           n_clicks=0,
                                           style={
                                               'backgroundColor': soft_palette['primary'],
                                               'color': 'white',
                                               'border': 'none',
                                               'borderRadius': '8px',
                                               'padding': '10px 20px',
                                               'fontWeight': 'bold',
                                               'fontSize': '12px',
                                               'cursor': 'pointer',
                                               'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
                                           })
                            ]),
                            
                            # Sub-tabs for Network Metrics and Shortest Path Analysis
                            dcc.Tabs(id='network-sub-tabs', value='tab-network-metrics', 
                                     style={'height': '32px', 'fontSize': '12px'},
                                     children=[
                                # Network Metrics Sub-tab
                                dcc.Tab(label='📊 Network Metrics', value='tab-network-metrics', children=[
                                    html.Div(style={'padding': '10px'}, children=[
                                        # Metric selector
                                        html.Div(style={'marginBottom': '8px'}, children=[
                                            html.Label("Select Metric:", style={
                                                'fontWeight': 'bold',
                                                'color': soft_palette['primary'],
                                                'marginBottom': '4px',
                                                'display': 'block',
                                                'fontSize': '12px'
                                            }),
                                            dcc.RadioItems(
                                                id='metric_selector',
                                                options=[
                                                    {'label': ' Degree Centrality', 'value': 'degree'},
                                                    {'label': ' Betweenness Centrality', 'value': 'betweenness'},
                                                    {'label': ' Closeness Centrality', 'value': 'closeness'}
                                                ],
                                                value='degree',
                                                inline=True,
                                                style={
                                                    'fontSize': '12px',
                                                    'display': 'flex',
                                                    'gap': '20px'
                                                },
                                                labelStyle={
                                                    'display': 'flex',
                                                    'alignItems': 'center',
                                                    'cursor': 'pointer'
                                                }
                                            )
                                        ]),
                                        
                                        # Visualization type tabs + Sort/Filter controls row
                                        html.Div(style={
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'gap': '20px',
                                            'marginBottom': '10px',
                                            'flexWrap': 'wrap'
                                        }, children=[
                                            # Visualization type toggle
                                            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}, children=[
                                                html.Label("View:", style={
                                                    'fontWeight': 'bold',
                                                    'color': soft_palette['primary'],
                                                    'fontSize': '12px'
                                                }),
                                                dcc.Tabs(
                                                    id='metrics-viz-tabs',
                                                    value='heatmap',
                                                    style={'height': '32px'},
                                                    children=[
                                                        dcc.Tab(label='📊 Heatmap', value='heatmap', 
                                                                style={'padding': '4px 14px', 'fontSize': '12px', 'minWidth': '90px'},
                                                                selected_style={'padding': '4px 14px', 'fontSize': '12px', 'minWidth': '90px', 'fontWeight': 'bold'}),
                                                        dcc.Tab(label='🎻 Violin', value='violin',
                                                                style={'padding': '4px 14px', 'fontSize': '12px', 'minWidth': '90px'},
                                                                selected_style={'padding': '4px 14px', 'fontSize': '12px', 'minWidth': '90px', 'fontWeight': 'bold'})
                                                    ]
                                                )
                                            ]),
                                            # Sort order dropdown
                                            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}, children=[
                                                html.Label("Sort:", style={
                                                    'fontWeight': 'bold',
                                                    'color': soft_palette['primary'],
                                                    'fontSize': '12px'
                                                }),
                                                dcc.Dropdown(
                                                    id='metrics-sort-order',
                                                    options=[
                                                        {'label': 'Sequence Order', 'value': 'sequence'},
                                                        {'label': 'Ascending', 'value': 'ascending'},
                                                        {'label': 'Descending', 'value': 'descending'}
                                                    ],
                                                    value='sequence',
                                                    clearable=False,
                                                    style={'width': '140px', 'fontSize': '12px'}
                                                )
                                            ]),
                                            # Metric cutoff filters
                                            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                                                html.Label("Range:", style={
                                                    'fontWeight': 'bold',
                                                    'color': soft_palette['primary'],
                                                    'fontSize': '12px'
                                                }),
                                                dcc.Input(
                                                    id='metrics-lower-cutoff',
                                                    type='number',
                                                    placeholder='Min',
                                                    debounce=True,
                                                    style={
                                                        'width': '70px',
                                                        'fontSize': '12px',
                                                        'padding': '4px 8px',
                                                        'borderRadius': '4px',
                                                        'border': f'1px solid {soft_palette["border"]}'
                                                    }
                                                ),
                                                html.Span("–", style={'color': soft_palette['text']}),
                                                dcc.Input(
                                                    id='metrics-upper-cutoff',
                                                    type='number',
                                                    placeholder='Max',
                                                    debounce=True,
                                                    style={
                                                        'width': '70px',
                                                        'fontSize': '12px',
                                                        'padding': '4px 8px',
                                                        'borderRadius': '4px',
                                                        'border': f'1px solid {soft_palette["border"]}'
                                                    }
                                                )
                                            ])
                                        ]),
                                        
                                        # Residue filter selector
                                        html.Div(style={'marginBottom': '8px'}, children=[
                                            html.Label("Filter Residues (leave empty for all):", style={
                                                'fontWeight': 'bold',
                                                'color': soft_palette['primary'],
                                                'marginBottom': '4px',
                                                'display': 'block',
                                                'fontSize': '12px'
                                            }),
                                            html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center'}, children=[
                                                dcc.Dropdown(
                                                    id='selected_residues_dropdown',
                                                    options=[{'label': res, 'value': res} for res in first_res_list],
                                                    value=[],  # Empty = show all
                                                    multi=True,
                                                    placeholder="Search and select residues to analyze (multi-select)",
                                                    searchable=True,
                                                    style={'flex': '1', 'fontSize': '12px'}
                                                ),
                                                html.Button('Reset to All', 
                                                    id='reset_residues_btn',
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': soft_palette['accent'],
                                                        'color': 'white',
                                                        'border': 'none',
                                                        'padding': '6px 12px',
                                                        'borderRadius': '6px',
                                                        'cursor': 'pointer',
                                                        'fontWeight': 'bold',
                                                        'fontSize': '12px',
                                                        'whiteSpace': 'nowrap'
                                                    })
                                            ])
                                        ]),
                                        
                                        # Visualization container (heatmap or violin) - fits to available space
                                        html.Div(
                                            id='metrics-viz-container',
                                            style={'width': '100%'},
                                            children=[
                                                dcc.Graph(
                                                    id='network_metrics_heatmap',
                                                    style={'height': 'calc(100vh - 480px)'}
                                                )
                                            ]
                                        )
                                    ])
                                ]),
                                
                                # Shortest Path Analysis Sub-tab (renamed from Network Visualization)
                                dcc.Tab(label='🛤️ Shortest Path Analysis', value='tab-shortest-path', children=[
                                    html.Div(style={'padding': '10px'}, children=[
                                        # Source and Target Selection
                                        html.Div(style={
                                            'display': 'flex',
                                            'gap': '20px',
                                            'marginBottom': '10px',
                                            'alignItems': 'flex-end'
                                        }, children=[
                                            html.Div(style={'flex': '1'}, children=[
                                                html.Label("🎯 Source Residue:", style={
                                                    'fontWeight': 'bold',
                                                    'color': soft_palette['primary'],
                                                    'marginBottom': '5px',
                                                    'display': 'block',
                                                    'fontSize': '12px'
                                                }),
                                                dcc.Dropdown(
                                                    id='source_residue_dropdown',
                                                    options=[{'label': res, 'value': res} for res in first_res_list],
                                                    value=first_res_list[0] if first_res_list else None,
                                                    placeholder="Select source residue",
                                                    searchable=True,
                                                    style={'width': '100%'}
                                                )
                                            ]),
                                            html.Div(style={'flex': '1'}, children=[
                                                html.Label("🎯 Target Residue:", style={
                                                    'fontWeight': 'bold',
                                                    'color': soft_palette['primary'],
                                                    'marginBottom': '5px',
                                                    'display': 'block',
                                                    'fontSize': '12px'
                                                }),
                                                dcc.Dropdown(
                                                    id='target_residue_dropdown',
                                                    options=[{'label': res, 'value': res} for res in first_res_list],
                                                    value=first_res_list[min(10, len(first_res_list)-1)] if len(first_res_list) > 1 else (first_res_list[0] if first_res_list else None),
                                                    placeholder="Select target residue",
                                                    searchable=True,
                                                    style={'width': '100%'}
                                                )
                                            ]),
                                            html.Div(children=[
                                                html.Button('🔍 Find Shortest Paths', 
                                                    id='find_paths_btn',
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': soft_palette['primary'],
                                                        'color': 'white',
                                                        'border': 'none',
                                                        'padding': '10px 20px',
                                                        'borderRadius': '8px',
                                                        'cursor': 'pointer',
                                                        'fontWeight': 'bold',
                                                        'fontSize': '12px',
                                                        'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
                                                    })
                                            ])
                                        ]),
                                        
                                        # Status message
                                        html.Div(id='path_status_message', style={
                                            'marginBottom': '8px',
                                            'padding': '8px',
                                            'borderRadius': '5px',
                                            'fontWeight': 'bold'
                                        }),
                                        
                                        # Results table
                                        html.Div(children=[
                                            html.Label("📊 Shortest Paths (Ctrl+Click for multi-select):", style={
                                                'fontWeight': 'bold',
                                                'color': soft_palette['primary'],
                                                'marginBottom': '5px',
                                                'display': 'block',
                                                'fontSize': '12px'
                                            }),
                                            dash_table.DataTable(
                                                id='shortest_paths_table',
                                                columns=[
                                                    {'name': 'Path', 'id': 'path'},
                                                    {'name': 'Length (Distance)', 'id': 'length'},
                                                    {'name': 'Hops', 'id': 'hops'}
                                                ],
                                                data=[],
                                                row_selectable='multi',
                                                selected_rows=[],
                                                style_table={
                                                    'overflowY': 'auto',
                                                    'maxHeight': '400px',
                                                    'border': f'2px solid {soft_palette["border"]}',
                                                    'borderRadius': '10px'
                                                },
                                                style_cell={
                                                    'textAlign': 'left',
                                                    'padding': '10px',
                                                    'fontFamily': 'Roboto, sans-serif',
                                                    'fontSize': '12px',
                                                    'backgroundColor': 'rgba(248,255,248,0.6)',
                                                    'cursor': 'pointer'
                                                },
                                                style_data={
                                                    'cursor': 'pointer'
                                                },
                                                style_header={
                                                    'backgroundColor': soft_palette['primary'],
                                                    'color': 'white',
                                                    'fontWeight': 'bold',
                                                    'border': 'none'
                                                },
                                                style_data_conditional=[
                                                    {
                                                        'if': {'state': 'selected'},
                                                        'backgroundColor': 'rgba(124, 152, 133, 0.3)',
                                                        'border': f'1px solid {soft_palette["primary"]}'
                                                    },
                                                    {
                                                        'if': {'state': 'active'},
                                                        'backgroundColor': 'rgba(124, 152, 133, 0.15)',
                                                        'border': f'1px solid {soft_palette["primary"]}'
                                                    }
                                                ]
                                            )
                                        ])
                                    ])
                                ])

                            ])
                        ])
                    ])
                ])
            ])
        ], style={'padding': '12px', 'backgroundColor': 'rgba(255,255,255,0.95)', 'borderRadius': '15px', 'border': f'3px solid {soft_palette["border"]}', 'boxShadow': '0 8px 32px rgba(0,0,0,0.1)', 'maxHeight': 'calc(100vh - 80px)', 'overflowY': 'auto'})
        ], width=8, id='left-panel'),
        # Middle Panel: 3D Viewer
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H3("🧬 3D Viewer", className="text-center mb-0", style={'color': soft_palette['primary']})
                ]),
                dbc.CardBody([
                    # Tabbed selector for 3D views
                    dcc.Tabs(id='viewer-tabs', value='tab-structure-viewer', style={'fontSize': '12px', 'height': '32px'}, children=[
                        # 3D Structure Viewer Tab
                        dcc.Tab(label='🧬 Structure Viewer', value='tab-structure-viewer', children=[
                            dbc.Card([
                                dbc.CardBody([
                                    dash_molstar.MolstarViewer(
                                        id='viewer', 
                                        data=initial_traj, 
                                        layout={'modelIndex': frame_min}, 
                                        style={'width': '100%','height':'calc(100vh - 520px)'}
                                    )
                                ])
                            ], style={'border': f'3px solid {soft_palette["border"]}', 'borderRadius': '10px', 'backgroundColor': 'rgba(250,255,250,0.4)', 'marginTop': '10px'})
                        ]),
                        
                        # 3D Network Visualization Tab
                        dcc.Tab(label='🌐 Network Visualization', value='tab-network-viewer', children=[
                            html.Div(style={'padding': '10px'}, children=[
                                # 3D Force Graph container
                                html.Div(id='network-3d-container', style={
                                    'width': '100%',
                                    'height': 'calc(100vh - 520px)',
                                    'border': f'2px solid {soft_palette["border"]}',
                                    'borderRadius': '10px',
                                    'backgroundColor': 'rgba(255,255,255,0.9)',
                                    'position': 'relative',
                                    'marginTop': '10px'
                                })
                            ])
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader([
                            html.Label("🎬 Frame:", className="text-white mb-0", style={'fontSize': '12px'})
                        ], style={'backgroundColor': soft_palette["light_blue"]}),
                        dbc.CardBody([
                            dcc.Slider(
                                id='frame_slider', 
                                min=frame_min, 
                                max=frame_max, 
                                step=1, 
                                value=frame_min,
                                marks={i: {
                                    'label': str(i),
                                    'style': {'color': soft_palette['text'], 'fontWeight': 'bold'}
                                } for i in range(frame_min, frame_max+1, max(1,(frame_max-frame_min)//10))},
                                tooltip={'always_visible':True,'placement':'top'},
                                updatemode='mouseup'
                            )
                        ])
                    ], className="mt-3")
                ])
            ])
        ], width=4, id='viewer-panel'),
        # Right Panel: Chatbot (shown/hidden via toggle)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div(style={
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'alignItems': 'center'
                    }, children=[
                        html.H5('💬 gRINN Chatbot', className='mb-0', style={'color': 'white'}),
                        html.Div([
                            html.Button('⤢', id='expand-chatbot', n_clicks=0, title='Expand', style={
                                'backgroundColor': 'transparent',
                                'border': 'none',
                                'color': 'white',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'padding': '0',
                                'width': '30px',
                                'height': '30px',
                            }),
                            html.Button('✕', id='close-chatbot', n_clicks=0, style={
                                'backgroundColor': 'transparent',
                                'border': 'none',
                                'color': 'white',
                                'fontSize': '16px',
                                'cursor': 'pointer',
                                'padding': '0',
                                'width': '30px',
                                'height': '30px'
                            }),
                        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'})
                    ])
                ], style={'backgroundColor': soft_palette['primary']}),
                dbc.CardBody([
                    html.Div(
                        [
                            # Token usage display (right-aligned in header area) - always visible
                            html.Div([
                                html.Span(id='chat-size-display', style={'fontSize': '11px', 'color': '#888', 'marginRight': '8px'}),
                                html.Span(
                                    "Tokens: 0 / ∞" if PANDASAI_TOKEN_LIMIT <= 0 else f"Tokens: 0 / {PANDASAI_TOKEN_LIMIT:,}",
                                    id='token-usage-display',
                                    style={'fontSize': '11px', 'color': soft_palette['text'], 'fontFamily': 'monospace'}
                                ),
                            ], style={'textAlign': 'right', 'marginBottom': '6px'}),
                            # Collapsible settings toggle
                            html.Div([
                                html.A(
                                    [html.Span('⚙️ Settings ', style={'fontSize': '11px'}), html.Span('▼', id='chat-settings-icon', style={'fontSize': '10px'})],
                                    id='chat-settings-toggle',
                                    style={'cursor': 'pointer', 'color': soft_palette['muted'], 'textDecoration': 'none', 'fontSize': '11px'},
                                    n_clicks=0
                                ),
                            ], style={'marginBottom': '6px'}),
                            # Collapsible settings container
                            dbc.Collapse(
                                html.Div([
                                    # Global settings (always visible)
                                    html.Div([
                                        html.Label('Model:', style={'fontSize': '11px', 'color': '#aaa', 'marginBottom': '2px'}),
                                        dcc.Dropdown(
                                            id='llm-model',
                                            options=[{'label': m, 'value': m} for m in AVAILABLE_MODELS],
                                            value=DEFAULT_MODEL,
                                            clearable=False,
                                            searchable=True,
                                            persistence=True,
                                            persistence_type='session',
                                            style={'fontSize': '12px'}
                                        ),
                                        dbc.Tooltip("AI model used for all chatbot queries.", target='llm-model', placement='right'),
                                        html.Label('DataFrames:', style={'fontSize': '11px', 'color': '#aaa', 'marginBottom': '2px', 'marginTop': '6px'}),
                                        dcc.Dropdown(
                                            id='chat-dataframe-selector',
                                            options=DATAFRAME_OPTIONS,
                                            value=DEFAULT_DATAFRAMES[:4],
                                            multi=True,
                                            clearable=False,
                                            searchable=True,
                                            persistence=True,
                                            persistence_type='session',
                                            style={'fontSize': '11px'},
                                            maxHeight=200
                                        ),
                                        dbc.Tooltip("Energy types passed to the AI (max 4 for timeseries). Fewer = faster.", target='chat-dataframe-selector', placement='right'),
                                        html.Label('Residue filter:', style={'fontSize': '11px', 'color': '#aaa', 'marginBottom': '2px', 'marginTop': '6px'}),
                                        dcc.Dropdown(
                                            id='chat-residue-filter',
                                            options=[{'label': r, 'value': r} for r in first_res_list],
                                            value=[],
                                            multi=True,
                                            clearable=True,
                                            searchable=True,
                                            placeholder='Filter by residue(s)...',
                                            persistence=True,
                                            persistence_type='session',
                                            style={'fontSize': '11px'},
                                            maxHeight=200
                                        ),
                                        dbc.Tooltip("Restrict analysis to specific residues. Leave empty for all.", target='chat-residue-filter', placement='right'),
                                        html.Label('\U0001f52c PubMed search:', style={'fontSize': '11px', 'color': '#aaa', 'marginBottom': '2px', 'marginTop': '6px'}),
                                        dbc.Checklist(
                                            id='chat-search-literature',
                                            options=[{'label': ' Search PubMed on query', 'value': 'pubmed'}],
                                            value=[],
                                            persistence=True,
                                            persistence_type='session',
                                            inputStyle={'marginRight': '4px'},
                                            labelStyle={'fontSize': '11px', 'color': soft_palette['text']}
                                        ),
                                        dbc.Tooltip("Append relevant PubMed citations to the AI response.", target='chat-search-literature', placement='right'),
                                    ]),
                                    html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)', 'margin': '8px 0'}),
                                    # Mode radio
                                    html.Div([
                                        html.Label('Data mode:', style={'fontSize': '11px', 'color': '#444', 'marginRight': '8px'}),
                                        dcc.RadioItems(
                                            id='chat-mode',
                                            options=[
                                                {'label': ' Summary', 'value': 'summary'},
                                                {'label': ' Timeseries', 'value': 'timeseries'},
                                                {'label': ' Snapshot', 'value': 'snapshot'},
                                            ],
                                            value='summary',
                                            inline=True,
                                            persistence=True,
                                            persistence_type='session',
                                            style={'display': 'inline-flex', 'gap': '12px'},
                                            labelStyle={'fontSize': '12px', 'color': '#444'},
                                            inputStyle={'marginRight': '4px'},
                                        ),
                                    ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'}),
                                    html.Div([
                                        # Shared frame range (shown for summary and timeseries, hidden for snapshot)
                                        html.Div([
                                            html.Div([
                                                html.Span('Frame range:', className='chat-form-label'),
                                                dcc.Input(
                                                    id='chat-frame-min', type='number',
                                                    value=frame_min, min=frame_min, max=frame_max,
                                                    placeholder='Min', debounce=True,
                                                    persistence=True, persistence_type='session',
                                                    style={'width': '70px', 'fontSize': '11px'},
                                                ),
                                                html.Span('–', className='chat-form-unit'),
                                                dcc.Input(
                                                    id='chat-frame-max', type='number',
                                                    value=frame_max, min=frame_min, max=frame_max,
                                                    placeholder='Max', debounce=True,
                                                    persistence=True, persistence_type='session',
                                                    style={'width': '70px', 'fontSize': '11px'},
                                                ),
                                            ], className='chat-form-row'),
                                            dbc.Tooltip("Trajectory frame range to include.", target='chat-frame-min', placement='right'),
                                        ], id='chat-frame-range-section', style={'marginBottom': '8px'}),
                                        # Summary mode settings
                                        html.Div([
                                            html.Span("Mean / std / min / max computed over all pairs in the frame range.", className='chat-hint'),
                                        ], id='chat-summary-settings', style={'marginBottom': '8px'}),
                                        # Timeseries mode settings
                                        html.Div([
                                            html.Div([
                                                html.Div([
                                                    html.Span('Energy threshold: ', style={'fontSize': '11px', 'color': '#444'}),
                                                    html.Span('0.5', id='chat-et-val',
                                                              style={'fontSize': '11px', 'color': '#333', 'fontFamily': 'monospace'}),
                                                    html.Span(' kcal/mol', style={'fontSize': '11px', 'color': '#666'}),
                                                ], style={'marginBottom': '3px'}),
                                                dcc.Slider(
                                                    id='chat-energy-threshold',
                                                    min=0.0, max=5.0, step=0.1, value=0.5,
                                                    marks=None,
                                                    tooltip={'always_visible': False},
                                                    updatemode='mouseup',
                                                    className='chat-slider',
                                                    persistence=True, persistence_type='session',
                                                ),
                                                dbc.Tooltip("Pairs with |mean energy| below this threshold are excluded.", target='chat-energy-threshold', placement='right'),
                                            ], style={'marginBottom': '8px'}),
                                            html.Div([
                                                html.Div([
                                                    html.Span('Stride:', className='chat-form-label'),
                                                    dbc.Switch(
                                                        id='chat-stride-mode',
                                                        label='Auto',
                                                        value=True,
                                                        persistence=True, persistence_type='session',
                                                        style={'display': 'inline-block', 'marginRight': '8px', 'fontSize': '11px'},
                                                        label_style={'fontSize': '11px', 'color': '#444'},
                                                    ),
                                                    html.Div(
                                                        dcc.Input(
                                                            id='chat-stride-manual',
                                                            type='number', min=1, step=1, value=1,
                                                            placeholder='e.g. 5',
                                                            style={'width': '70px', 'fontSize': '11px'},
                                                            persistence=True, persistence_type='session',
                                                            debounce=True,
                                                        ),
                                                        id='chat-stride-manual-div',
                                                        style={'display': 'none'},
                                                    ),
                                                ], className='chat-form-row'),
                                                dbc.Tooltip(f"Auto: stride computed to stay within {MAX_CHAT_VALUES:,} values. Manual: set stride explicitly.", target='chat-stride-mode', placement='right'),
                                                dbc.Tooltip("Every Nth frame is kept. Stride of 1 = all frames.", target='chat-stride-manual', placement='right'),
                                            ], style={'marginBottom': '5px'}),
                                            html.Span(id='chat-stride-display', style={'fontSize': '11px', 'color': '#666', 'display': 'block', 'marginTop': '2px'}),
                                            html.Span(id='chat-size-badge-ts', style={'fontSize': '11px', 'display': 'none'}),
                                            html.Span(
                                                f"Auto stride targets \u2264 {MAX_CHAT_VALUES:,} values (pairs \u00d7 frames). Filter residues to get more frames with large datasets.",
                                                className='chat-hint',
                                            ),
                                        ], id='chat-timeseries-settings', style={'display': 'none'}),
                                        # Snapshot mode settings
                                        html.Div([
                                            html.Div([
                                                html.Div([
                                                    html.Span('Frame #:', className='chat-form-label'),
                                                    dcc.Input(
                                                        id='chat-snapshot-frame', type='number',
                                                        value=frame_min, min=frame_min, max=frame_max,
                                                        placeholder='Frame #', debounce=True,
                                                        persistence=True, persistence_type='session',
                                                        style={'width': '80px', 'fontSize': '11px'},
                                                    ),
                                                ], className='chat-form-row'),
                                                dbc.Tooltip("Frame number to use for snapshot mode.", target='chat-snapshot-frame', placement='right'),
                                            ], style={'marginBottom': '5px'}),
                                            html.Div([
                                                html.Div([
                                                    html.Span('Energy threshold: ', style={'fontSize': '11px', 'color': '#444'}),
                                                    html.Span('0.5', id='chat-et-snapshot-val',
                                                              style={'fontSize': '11px', 'color': '#333', 'fontFamily': 'monospace'}),
                                                    html.Span(' kcal/mol', style={'fontSize': '11px', 'color': '#666'}),
                                                ], style={'marginBottom': '3px'}),
                                                dcc.Slider(
                                                    id='chat-energy-threshold-snapshot',
                                                    min=0.0, max=5.0, step=0.1, value=0.5,
                                                    marks=None,
                                                    tooltip={'always_visible': False},
                                                    updatemode='mouseup',
                                                    className='chat-slider',
                                                    persistence=True, persistence_type='session',
                                                ),
                                                dbc.Tooltip("Pairs with |energy| below this threshold are excluded.", target='chat-energy-threshold-snapshot', placement='right'),
                                            ], style={'marginBottom': '5px'}),
                                            html.Span(
                                                "Single frame sent to the AI. No stride or size limit applies.",
                                                className='chat-hint',
                                            ),
                                        ], id='chat-snapshot-settings', style={'display': 'none'}),
                                    ], className='chat-mode-box'),
                                ]),
                                id='chat-settings-collapse',
                                is_open=True
                            ),
                            # Store for token usage tracking per session
                            dcc.Store(id='chat-token-usage', data={'used': 0, 'limit': PANDASAI_TOKEN_LIMIT}, storage_type='session'),
                            dcc.Store(id='chat-pairs-store', storage_type='memory', data={'total': 0}),
                            dcc.Store(id='chat-max-values-store', data=MAX_CHAT_VALUES),
                            # Watchdog interval — ticks every 3 s to detect hung callbacks
                            dcc.Interval(id='chat-watchdog', interval=3000, disabled=False),
                            # Error banner shown when a request has been pending for too long
                            html.Div([
                                html.Span('⚠️ The request is taking too long — this is likely a network issue (e.g. 504 Gateway Timeout). '
                                          'You can try sending your message again.'),
                                html.Span(' ✕', id='chat-timeout-dismiss',
                                          style={'cursor': 'pointer', 'marginLeft': '8px', 'fontWeight': 'bold'}),
                            ], id='chat-timeout-error', style={
                                'display': 'none',
                                'backgroundColor': '#fff3cd', 'color': '#856404',
                                'border': '1px solid #ffc107', 'borderRadius': '4px',
                                'padding': '6px 10px', 'fontSize': '11px',
                                'flex': '0 0 auto', 'marginBottom': '4px',
                            }),
                            html.Div(
                                ChatComponent(
                                    id='chat',
                                    messages=[],
                                    assistant_bubble_style={
                                        'backgroundColor': soft_palette['surface'],
                                        'color': soft_palette['text'],
                                        'marginRight': 'auto',
                                        'textAlign': 'left',
                                        'maxWidth': '100%',
                                        'width': '100%',
                                        'overflow': 'hidden',
                                        'boxSizing': 'border-box',
                                        'position': 'relative',
                                        'zIndex': 'auto',
                                        'fontSize': '12px'
                                    },
                                    user_bubble_style={
                                        'fontSize': '12px'
                                    },
                                ),
                                id='chat-component-wrap',
                                style={'flex': '1 1 0', 'minHeight': 0, 'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column'}
                            ),
                            # "Explain biologically" button
                            dbc.Button(
                                '✨ Explain biologically',
                                id='explain-bio-btn',
                                size='sm',
                                color='success',
                                outline=True,
                                style={'display': 'none', 'marginTop': '4px', 'fontSize': '11px', 'flex': '0 0 auto'}
                            ),
                            # Chart gallery: shows buttons for all generated charts
                            html.Div([
                                html.Div('Charts:', style={'fontSize': '10px', 'color': soft_palette['muted'], 'marginBottom': '4px'}),
                                html.Div(id='chat-chart-gallery', children=[], style={
                                    'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px', 'maxHeight': '60px', 'overflowY': 'auto'
                                })
                            ], id='chat-chart-gallery-container', style={'display': 'none', 'marginTop': '8px', 'padding': '6px', 'backgroundColor': soft_palette['surface'], 'borderRadius': '4px', 'flex': '0 0 auto'}),
                            # Store for chart figures (list of dicts)
                            dcc.Store(id='chat-charts-store', data=[], storage_type='session'),
                            # Modal for full-resolution chart display
                            dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle('Chart View', style={'fontSize': '14px'}), close_button=True),
                                dbc.ModalBody([
                                    html.Img(id='chart-modal-img', style={
                                        'maxWidth': '100%',
                                        'maxHeight': '70vh',
                                        'objectFit': 'contain',
                                        'display': 'block',
                                        'margin': '0 auto'
                                    })
                                ], style={'textAlign': 'center', 'overflow': 'auto'})
                            ], id='chart-view-modal', size='xl', centered=True, is_open=False),
                            # Store for which chart index to display in modal
                            dcc.Store(id='chart-modal-index', data=None),
                            # DataFrame gallery: shows buttons for all returned DataFrames
                            html.Div([
                                html.Div('Tables:', style={'fontSize': '10px', 'color': soft_palette['muted'], 'marginBottom': '4px'}),
                                html.Div(id='chat-df-gallery', children=[], style={
                                    'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px', 'maxHeight': '60px', 'overflowY': 'auto'
                                })
                            ], id='chat-df-gallery-container', style={
                                'display': 'none', 'marginTop': '8px', 'padding': '6px',
                                'backgroundColor': soft_palette['surface'], 'borderRadius': '4px', 'flex': '0 0 auto'
                            }),
                            # Store for DataFrame gallery (list of {'columns': [...], 'data': [...], 'label': '...'})
                            dcc.Store(id='chat-dataframes-store', data=[], storage_type='session'),
                            # Store for which DataFrame index is currently open in the modal
                            dcc.Store(id='df-modal-index', data=None),
                            # Modal for full DataFrame display with download
                            dbc.Modal([
                                dbc.ModalHeader(
                                    dbc.ModalTitle('Table View', style={'fontSize': '14px'}),
                                    close_button=True
                                ),
                                dbc.ModalBody([
                                    dash_table.DataTable(
                                        id='df-modal-table',
                                        page_size=20,
                                        sort_action='native',
                                        filter_action='native',
                                        style_table={'overflowX': 'auto'},
                                        style_cell={'fontSize': '11px', 'padding': '4px 8px', 'textAlign': 'left'},
                                        style_header={'fontWeight': 'bold', 'fontSize': '11px'},
                                    ),
                                    html.Div(
                                        dbc.Button('⬇ Download CSV', id='df-download-btn', size='sm',
                                                   color='secondary', outline=True,
                                                   style={'marginTop': '10px', 'fontSize': '11px'}),
                                        style={'textAlign': 'right'}
                                    ),
                                    dcc.Download(id='df-download'),
                                ]),
                            ], id='df-view-modal', size='xl', centered=True, is_open=False),
                        ],
                        id='chat-body-scroller',
                        style={'display': 'flex', 'flexDirection': 'column', 'height': 'calc(100vh - 280px)', 'minHeight': 0, 'overflow': 'hidden'}
                    )
                ], style={'padding': '10px', 'overflow': 'hidden', 'height': '100%', 'display': 'flex', 'flexDirection': 'column'})
            ], style={'height': '100%', 'borderRadius': '10px', 'border': f'3px solid {soft_palette["border"]}', 'maxHeight': 'calc(100vh - 80px)', 'overflow': 'hidden'})
        ], width=3, id='chat-panel', style={'display': 'none', 'overflow': 'hidden'})
        ], className="mt-1")
        ])
    ], fluid=True, style={'background': soft_palette["background"], 'padding': '10px'})

    # --- Chatbot backend helpers ---
    import uuid
    import time
    from dataclasses import dataclass
    from typing import Optional
    @dataclass
    class _SessionEntry:
        value: object
        last_access_epoch_s: float

    class _TTLRegistry:
        def __init__(self, ttl_seconds: int):
            self._ttl = ttl_seconds
            self._items: dict[str, _SessionEntry] = {}
        def get_or_create(self, sid: str, factory):
            now = time.time()
            entry = self._items.get(sid)
            if entry is None:
                val = factory()
                self._items[sid] = _SessionEntry(value=val, last_access_epoch_s=now)
                return val
            entry.last_access_epoch_s = now
            return entry.value
        def reset(self, sid: str) -> bool:
            """Remove an existing session and best-effort stop its sandbox."""
            entry = self._items.pop(sid, None)
            if entry is None:
                return False
            try:
                if hasattr(entry.value, 'sandbox'):
                    _stop_session_context(entry.value)
            except Exception:
                pass
            return True
        def cleanup(self):
            now = time.time()
            remove = []
            for sid, entry in list(self._items.items()):
                if now - entry.last_access_epoch_s > self._ttl:
                    remove.append(sid)
            for sid in remove:
                entry = self._items.pop(sid, None)
                if entry is None:
                    continue
                try:
                    # Best-effort: stop Docker sandbox to avoid leaks.
                    if hasattr(entry.value, 'sandbox'):
                        _stop_session_context(entry.value)
                except Exception:
                    pass
            return len(remove)

    registry = _TTLRegistry(ttl_seconds=int(os.getenv('CHAT_SESSION_TTL_SECONDS', '3600')))

    # Load PEN master tables once from this dashboard's folder
    def _read_csvs(files: list[str], kind: str) -> pd.DataFrame:
        rows = []
        for path in files:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            basename = os.path.basename(path)
            m = _re.match(rf"{kind}_(?P<energy>[^_]+)_cov(?P<cov>[01])_cutoff(?P<cutoff>-?[0-9.]+)(?:_frame(?P<frame>\d+))?\.csv", basename)
            meta = {
                'source_file': path,
                'energy_type': m.group('energy') if m else pd.NA,
                'include_covalents': (bool(int(m.group('cov'))) if m else pd.NA),
                'cutoff': (float(m.group('cutoff')) if m else pd.NA),
                'frame': (int(m.group('frame')) if (m and m.group('frame')) else pd.NA),
                'kind': kind,
            }
            df = df.copy()
            for k, v in meta.items():
                df[k] = v
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _pen_cached: Optional[dict] = None
    _pen_sig: Optional[tuple] = None
    def _get_pen_tables():
        nonlocal _pen_cached, _pen_sig

        def _files_signature(paths: list[str]) -> tuple:
            sig = []
            for p in paths:
                try:
                    st = os.stat(p)
                    sig.append((p, st.st_mtime, st.st_size))
                except Exception:
                    sig.append((p, None, None))
            return tuple(sig)

        pen = _pen_folder_from_dashboard()
        metrics_files = sorted(glob.glob(os.path.join(pen, 'metrics_*.csv')))
        edges_files = sorted(glob.glob(os.path.join(pen, 'edges_*.csv')))

        sig = _files_signature(metrics_files) + _files_signature(edges_files)
        if _pen_cached is not None and _pen_sig == sig:
            return _pen_cached

        _pen_sig = sig
        _pen_cached = {
            'pen_folder': pen,
            'sig': str(abs(hash(sig))),
            'metrics_master': _read_csvs(metrics_files, 'metrics'),
            'edges_master': _read_csvs(edges_files, 'edges'),
        }
        return _pen_cached

    # Build PandasAI session context (lazy imports)
    @dataclass
    class _SessionCtx:
        sandbox: object
        agent: object
    def _build_llm(model: Optional[str] = None):
        print(f"[DEBUG _build_llm] Starting with model={model}", flush=True)
        try:
            from pandasai_litellm.litellm import LiteLLM
            print(f"[DEBUG _build_llm] LiteLLM imported successfully", flush=True)
        except Exception as import_err:
            print(f"[DEBUG _build_llm] Failed to import LiteLLM: {import_err}", flush=True)
            raise
        chosen_model = _resolve_model_selection(model)
        print(f"[DEBUG _build_llm] Resolved model: {chosen_model}", flush=True)
        m = (chosen_model or '').strip().lower()
        is_claude = ('claude' in m) or m.startswith('anthropic/')
        if is_claude:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise RuntimeError('Missing API key: set ANTHROPIC_API_KEY for Claude models')
            # LiteLLM requires anthropic/ prefix for Claude models
            if not chosen_model.startswith('anthropic/'):
                chosen_model = f'anthropic/{chosen_model}'
            print(f"[DEBUG _build_llm] Using Claude model: {chosen_model}, API key present: {bool(api_key)}", flush=True)
        else:
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise RuntimeError('Missing API key: set GEMINI_API_KEY (or GOOGLE_API_KEY)')
            print(f"[DEBUG _build_llm] Using Gemini model: {chosen_model}, API key present: {bool(api_key)}", flush=True)
        try:
            llm = LiteLLM(model=chosen_model, api_key=api_key, timeout=120)
            print(f"[DEBUG _build_llm] LiteLLM instance created successfully", flush=True)
            return llm
        except Exception as llm_err:
            print(f"[DEBUG _build_llm] Failed to create LiteLLM: {llm_err}", flush=True)
            raise
    def _wrap_df(df: pd.DataFrame, name: str):
        import pandasai as pai
        try:
            return pai.DataFrame(df, name=name)
        except TypeError:
            w = pai.DataFrame(df)
            try:
                setattr(w, 'name', name)
            except Exception:
                pass
            return w

    def _filter_dataframe_for_chat(df: pd.DataFrame, df_key: str, residue_filter: list[str],
                                    frame_min_val: int, frame_max_val: int, stride: int,
                                    mode: str = 'timeseries', snapshot_frame: int = None,
                                    min_abs_energy: float = 0.0) -> pd.DataFrame:
        """
        Filter a DataFrame for chatbot queries based on residue filter and frame range.

        Args:
            df: The original DataFrame to filter
            df_key: The key identifying the DataFrame type (e.g., 'IE_Total', 'Metrics_...')
            residue_filter: List of residues to filter by (empty = no filter)
            frame_min_val: Minimum frame number (inclusive)
            frame_max_val: Maximum frame number (inclusive)
            stride: Frame stride (e.g., 2 = every 2nd frame) — used only for timeseries mode
            mode: 'timeseries', 'snapshot', or 'summary'
            snapshot_frame: Frame number to use for snapshot mode (None = middle frame)
            min_abs_energy: Minimum absolute IE value to keep rows (0.0 = no filter)

        Returns:
            Filtered DataFrame
        """
        import math
        result = df.copy()

        # Determine column names based on DataFrame type
        is_ie = df_key.startswith('IE_')
        is_metrics = df_key.startswith('Metrics_')

        # Apply residue filter
        if residue_filter:
            if is_ie:
                # IE DataFrames have 'res1' and 'res2' columns - inclusive OR filter
                if 'res1' in result.columns and 'res2' in result.columns:
                    mask = result['res1'].isin(residue_filter) | result['res2'].isin(residue_filter)
                    result = result[mask]
            elif is_metrics:
                # Metrics DataFrames have 'residue' column
                if 'residue' in result.columns:
                    result = result[result['residue'].isin(residue_filter)]

        # Apply frame filter
        # For IE DataFrames, frame data is in columns (wide format)
        # For Metrics DataFrames, frame is a column
        if is_ie:
            # Wide format: columns are frame numbers after res1, res2
            non_frame_cols = ['res1', 'res2', 'Pair', 'Unnamed: 0', 'res1_index', 'res2_index',
                              'res1_chain', 'res2_chain', 'res1_resnum', 'res2_resnum',
                              'res1_resname', 'res2_resname']
            frame_cols = [c for c in result.columns if c not in non_frame_cols]

            try:
                frame_nums = []
                for c in frame_cols:
                    try:
                        frame_nums.append((int(c), c))
                    except ValueError:
                        pass

                frame_nums_sorted = sorted([fn for fn, _ in frame_nums])
                frame_nums_in_range = [fn for fn in frame_nums_sorted if frame_min_val <= fn <= frame_max_val]

                if mode == 'snapshot':
                    # Pick single frame closest to snapshot_frame
                    if not frame_nums_in_range:
                        pass  # keep all if no frames in range
                    else:
                        if snapshot_frame is None:
                            target = frame_nums_in_range[len(frame_nums_in_range) // 2]
                        else:
                            target = min(frame_nums_in_range, key=lambda fn: abs(fn - snapshot_frame))
                        keep_frame_cols = [str(target)]
                        keep_cols = [c for c in result.columns if c in non_frame_cols or c in keep_frame_cols]
                        result = result[keep_cols]
                elif mode == 'summary':
                    # Compute mean/std/min/max across frame columns in range
                    range_frame_cols = [str(fn) for fn in frame_nums_in_range if str(fn) in result.columns]
                    if range_frame_cols:
                        numeric_data = result[range_frame_cols].apply(pd.to_numeric, errors='coerce')
                        summary_df = pd.DataFrame()
                        # Keep non-frame identifier columns that exist
                        for col in ['res1', 'res2', 'Pair']:
                            if col in result.columns:
                                summary_df[col] = result[col].values
                        summary_df['mean_ie'] = numeric_data.mean(axis=1).values
                        summary_df['std_ie'] = numeric_data.std(axis=1).values
                        summary_df['min_ie'] = numeric_data.min(axis=1).values
                        summary_df['max_ie'] = numeric_data.max(axis=1).values
                        result = summary_df
                    else:
                        # No frame cols in range — return identifier cols with NaN stats
                        summary_df = pd.DataFrame()
                        for col in ['res1', 'res2', 'Pair']:
                            if col in result.columns:
                                summary_df[col] = result[col].values
                        for stat_col in ['mean_ie', 'std_ie', 'min_ie', 'max_ie']:
                            summary_df[stat_col] = float('nan')
                        result = summary_df
                else:
                    # timeseries: existing stride logic
                    filtered_frames = []
                    for i, fn in enumerate(frame_nums_in_range):
                        if i % stride == 0:
                            filtered_frames.append(str(fn))
                    keep_cols = [c for c in result.columns if c in non_frame_cols or c in filtered_frames]
                    result = result[keep_cols]
            except Exception:
                pass  # If frame parsing fails, keep all columns

            # Apply energy threshold filter for IE DataFrames
            if min_abs_energy > 0.0 and len(result) > 0:
                try:
                    if mode == 'summary' and 'mean_ie' in result.columns:
                        mask = result['mean_ie'].abs() >= min_abs_energy
                        result = result[mask]
                    else:
                        # For timeseries/snapshot: compute row-wise mean across frame columns
                        frame_cols_present = [c for c in result.columns if c not in non_frame_cols and c not in ['res1', 'res2', 'Pair', 'mean_ie', 'std_ie', 'min_ie', 'max_ie']]
                        if frame_cols_present:
                            row_means = result[frame_cols_present].apply(pd.to_numeric, errors='coerce').mean(axis=1)
                            mask = row_means.abs() >= min_abs_energy
                            result = result[mask]
                except Exception:
                    pass

        elif is_metrics:
            # Long format: 'frame' column — all modes behave the same
            if 'frame' in result.columns:
                try:
                    # Filter by frame range
                    result = result[(result['frame'] >= frame_min_val) & (result['frame'] <= frame_max_val)]

                    # Apply stride (only for timeseries mode)
                    if mode == 'timeseries' and stride > 1:
                        all_frames = sorted(result['frame'].unique())
                        keep_frames = [f for i, f in enumerate(all_frames) if i % stride == 0]
                        result = result[result['frame'].isin(keep_frames)]
                except Exception:
                    pass  # If filtering fails, keep original

        return result

    def _compute_chat_stride(frame_min_val: int, frame_max_val: int,
                             n_pairs: int = 1, max_values: int = 5000) -> int:
        """Compute stride so that (n_pairs × frames_kept) ≤ max_values."""
        import math
        total_frames = max(1, frame_max_val - frame_min_val + 1)
        if n_pairs <= 0:
            return 1
        # How many frames can we afford?
        max_frames = max(1, max_values // n_pairs)
        if total_frames <= max_frames:
            return 1
        return math.ceil(total_frames / max_frames)

    def _estimate_filtered_size(df_key: str, residue_filter: list[str],
                                 frame_min_val: int, frame_max_val: int,
                                 mode: str = 'timeseries',
                                 min_abs_energy: float = 0.0) -> tuple[int, int, int]:
        """Estimate filtered DataFrame size without copying data.

        Returns:
            (est_rows, est_cols, est_tokens)
        """
        import math
        info = _CHATBOT_DATAFRAMES.get(df_key)
        if info is None:
            return (0, 0, 1)

        df = info['df']
        orig_rows, orig_cols = info['rows'], info['cols']
        is_ie = df_key.startswith('IE_')
        is_metrics = df_key.startswith('Metrics_')

        # Estimate row count after residue filter
        if residue_filter:
            if is_ie and 'res1' in df.columns and 'res2' in df.columns:
                mask = df['res1'].isin(residue_filter) | df['res2'].isin(residue_filter)
                est_rows = int(mask.sum())
            elif is_metrics and 'residue' in df.columns:
                est_rows = int(df['residue'].isin(residue_filter).sum())
            else:
                est_rows = orig_rows
        else:
            est_rows = orig_rows

        # Apply energy threshold filter for IE DataFrames
        if min_abs_energy > 0.0 and is_ie:
            non_frame_cols_set = {
                'res1', 'res2', 'Pair', 'Unnamed: 0', 'res1_index', 'res2_index',
                'res1_chain', 'res2_chain', 'res1_resnum', 'res2_resnum',
                'res1_resname', 'res2_resname',
            }
            frame_cols = [c for c in df.columns if c not in non_frame_cols_set]
            if frame_cols:
                try:
                    if residue_filter and 'res1' in df.columns and 'res2' in df.columns:
                        row_mask = df['res1'].isin(residue_filter) | df['res2'].isin(residue_filter)
                        row_means = df.loc[row_mask, frame_cols].apply(
                            pd.to_numeric, errors='coerce').mean(axis=1).abs()
                    else:
                        row_means = df[frame_cols].apply(
                            pd.to_numeric, errors='coerce').mean(axis=1).abs()
                    est_rows = int((row_means >= min_abs_energy).sum())
                except Exception:
                    pass  # leave est_rows unchanged if something goes wrong

        # Estimate column count after frame filter (for IE wide format)
        stride = _compute_chat_stride(frame_min_val, frame_max_val, n_pairs=max(1, est_rows), max_values=MAX_CHAT_VALUES)
        total_frames_in_range = frame_max_val - frame_min_val + 1
        frames_after_stride = math.ceil(total_frames_in_range / stride)

        if is_ie:
            if mode == 'summary':
                est_cols = 7  # res1, res2, Pair, mean_ie, std_ie, min_ie, max_ie
            elif mode == 'snapshot':
                est_cols = 4  # res1, res2, Pair, one frame col
            else:
                # timeseries: non-frame cols + filtered frame cols
                non_frame_cols_count = 3  # res1, res2, Pair typically
                est_cols = non_frame_cols_count + frames_after_stride
        elif is_metrics:
            # Long format: same columns, but rows are reduced by frame filter
            est_cols = orig_cols
            # Rows also reduced by frame range
            if 'frame' in df.columns:
                frames_in_df = df['frame'].nunique()
                if frames_in_df > 0:
                    rows_per_frame = est_rows / frames_in_df
                    est_rows = int(rows_per_frame * frames_after_stride)
        else:
            est_cols = orig_cols

        est_rows = max(0, est_rows)
        est_cols = max(1, est_cols)
        est_tokens = max(1, (est_rows * est_cols * 6) // 4)
        return (est_rows, est_cols, est_tokens)

    def _build_session_context_for_dfs(selected_df_keys: list[str], model: Optional[str] = None,
                                        residue_filter: list[str] = None, frame_min_val: int = None,
                                        frame_max_val: int = None, mode: str = 'timeseries',
                                        snapshot_frame: int = None,
                                        min_abs_energy: float = 0.0,
                                        stride: Optional[int] = None) -> _SessionCtx:
        """Build session context with selected DataFrames from registry."""
        print(f"[DEBUG _build_session_context_for_dfs] Starting with keys={selected_df_keys}, model={model}", flush=True)
        
        try:
            from pandasai import Agent
            print("[DEBUG _build_session_context_for_dfs] pandasai.Agent imported", flush=True)
        except Exception as e:
            print(f"[DEBUG _build_session_context_for_dfs] Failed to import pandasai.Agent: {e}", flush=True)
            raise
        
        # Check if Docker sandbox should be used (default: true for standalone, false for containers)
        use_sandbox = os.getenv('PANDASAI_USE_DOCKER_SANDBOX', 'true').lower() in ('true', '1', 'yes')
        sandbox = None
        
        if use_sandbox:
            try:
                from pandasai_docker import DockerSandbox
                print("[DEBUG _build_session_context_for_dfs] DockerSandbox imported", flush=True)
                print("[DEBUG _build_session_context_for_dfs] Creating DockerSandbox...", flush=True)
                sandbox = DockerSandbox()
                print("[DEBUG _build_session_context_for_dfs] Starting sandbox...", flush=True)
                sandbox.start()
                print("[DEBUG _build_session_context_for_dfs] Sandbox started successfully", flush=True)
            except Exception as e:
                print(f"[WARNING] Docker sandbox unavailable: {e}", flush=True)
                print("[WARNING] Running WITHOUT sandbox - code will execute locally", flush=True)
                sandbox = None
        else:
            print("[INFO] Docker sandbox disabled via PANDASAI_USE_DOCKER_SANDBOX=false", flush=True)
        
        # Build LLM with validated configuration
        try:
            llm = _build_llm(model=model)
        except Exception as e:
            # Clean up sandbox if LLM fails
            try:
                if sandbox is not None:
                    sandbox.stop()
            except Exception:
                pass
            raise
        
        # Compute stride if frame range is specified (only used for timeseries mode)
        if stride is None:
            stride = 1
            if mode == 'timeseries' and frame_min_val is not None and frame_max_val is not None:
                stride = _compute_chat_stride(frame_min_val, frame_max_val, max_values=MAX_CHAT_VALUES)
        else:
            stride = max(1, int(stride))

        # Collect selected DataFrames with optional filtering
        pai_dfs = []
        df_names = []
        for key in selected_df_keys[:4]:  # max 4
            info = _CHATBOT_DATAFRAMES.get(key)
            if info is None:
                continue
            df = info['df']

            # Apply filtering if residue filter, frame range, or energy threshold is specified
            if residue_filter or (frame_min_val is not None and frame_max_val is not None) or min_abs_energy > 0.0:
                df = _filter_dataframe_for_chat(
                    df, key,
                    residue_filter=residue_filter or [],
                    frame_min_val=frame_min_val if frame_min_val is not None else 0,
                    frame_max_val=frame_max_val if frame_max_val is not None else 999999,
                    stride=stride,
                    mode=mode,
                    snapshot_frame=snapshot_frame,
                    min_abs_energy=min_abs_energy,
                )
            
            df_name = key.replace('-', '_').replace(' ', '_')
            df_names.append(df_name)
            pai_dfs.append(_wrap_df(df, df_name))
        
        if not pai_dfs:
            # Fallback to default if nothing valid selected
            if _CHATBOT_DATAFRAMES:
                fallback_key = DEFAULT_DATAFRAMES[0] if DEFAULT_DATAFRAMES else list(_CHATBOT_DATAFRAMES.keys())[0]
                info = _CHATBOT_DATAFRAMES[fallback_key]
                df_name = fallback_key.replace('-', '_').replace(' ', '_')
                df_names.append(df_name)
                pai_dfs.append(_wrap_df(info['df'], df_name))
        
        # Build custom instructions listing available DataFrames
        df_list_str = ', '.join(df_names)
        custom_instructions = (
            f'You have exactly {len(pai_dfs)} pandas DataFrame(s) available: {df_list_str}. '
            'Use ONLY Python + pandas operations on these DataFrames. '
            'NEVER generate SQL queries. Do NOT use SELECT, FROM, WHERE, or any SQL syntax. '
            'Do NOT use execute_sql_query() or reference internal table names like table_*. '
            'You MAY create intermediate results as Python variables (e.g., df_tmp = ...), but never as new tables. '
            'CRITICAL: Your Python code MUST assign the final answer to a variable named result (e.g., result = ...). '
            'Set result to a string for narrative answers, a pandas DataFrame for tabular outputs, or a matplotlib chart (when asked to plot). '
            'When returning tables/lists: return a concise result (prefer a pandas DataFrame or list of dicts), '
            'limit to relevant columns and at most ~50 rows, and include a short textual summary when appropriate. '
            'When plotting charts, use matplotlib: plt.figure(figsize=(6, 4), dpi=300) and plt.tight_layout(). '
            '\n\n--- Scientific Context ---\n'
            'These DataFrames contain Residue Interaction Energies (IEs) computed from molecular dynamics (MD) simulations. '
            'IE values are in kcal/mol. Negative values indicate attractive interactions; positive values indicate repulsion. '
            'Typical biologically significant IE values are stronger than -1 to -2 kcal/mol. '
            'IE columns include: IE_Total (total interaction energy), IE_VdW (van der Waals component), IE_Elec (electrostatic component). '
            'VdW interactions reflect steric/hydrophobic contacts; Electrostatic interactions reflect charge-charge and hydrogen bond effects. '
            'For IE DataFrames in summary mode: mean_ie is the time-averaged IE, std_ie reflects fluctuations, min_ie/max_ie show extremes. '
            'For Metrics DataFrames: these contain Protein Energy Network (PEN) node metrics including betweenness_centrality (how often a residue is on shortest paths), '
            'closeness_centrality (how close a residue is to all others), and degree (number of significant interaction partners). '
            'High betweenness/closeness centrality residues are often functionally important (e.g., allosteric hubs). '
            'When interpreting results, mention units (kcal/mol), note whether interactions are attractive or repulsive, '
            'and flag residues/interactions that may be biologically significant.'
        )
        
        agent = Agent(pai_dfs, config={
            'llm': llm,
            'custom_instructions': custom_instructions,
        }, sandbox=sandbox)
        return _SessionCtx(sandbox=sandbox, agent=agent)
    
    # Legacy wrapper for backward compatibility
    def _build_session_context(metrics_master: pd.DataFrame, edges_master: pd.DataFrame, model: Optional[str] = None) -> _SessionCtx:
        from pandasai import Agent
        
        # Check if Docker sandbox should be used
        use_sandbox = os.getenv('PANDASAI_USE_DOCKER_SANDBOX', 'true').lower() in ('true', '1', 'yes')
        sandbox = None
        
        if use_sandbox:
            try:
                from pandasai_docker import DockerSandbox
                sandbox = DockerSandbox()
                sandbox.start()
            except Exception as e:
                print(f"[WARNING] Docker sandbox unavailable: {e}", flush=True)
                sandbox = None
        else:
            print("[INFO] Docker sandbox disabled via PANDASAI_USE_DOCKER_SANDBOX=false", flush=True)
        llm = _build_llm(model=model)
        pai_metrics = _wrap_df(metrics_master, 'metrics_master')
        pai_edges = _wrap_df(edges_master, 'edges_master')
        agent = Agent([pai_metrics, pai_edges], config={
            'llm': llm,
            'custom_instructions': (
                'You have exactly two pandas DataFrames available: metrics_master and edges_master. '
                'Prefer metrics_master for per-residue/aggregate metrics, edges_master for interaction edges. '
                'Use ONLY Python + pandas operations on these DataFrames; do NOT write SQL and do NOT reference internal table names like table_*. '
                'You MAY create intermediate results as Python variables (e.g., df_tmp = ...), but never as new tables. '
                'CRITICAL: Your Python code MUST assign the final answer to a variable named result (e.g., result = ...). '
                'Set result to a string for narrative answers, a pandas DataFrame for tabular outputs, or a matplotlib chart (when asked to plot). '
                'When returning tables/lists: return a concise result (prefer a pandas DataFrame or list of dicts), '
                'limit to relevant columns and at most ~50 rows, and include a short textual summary when appropriate. '
                'When plotting charts, use matplotlib: plt.figure(figsize=(6, 4), dpi=300) and plt.tight_layout().'
            ),
        }, sandbox=sandbox)
        return _SessionCtx(sandbox=sandbox, agent=agent)
    def _stop_session_context(ctxobj: _SessionCtx):
        try:
            if ctxobj.sandbox is not None:
                ctxobj.sandbox.stop()
        except Exception:
            pass
    def _chart_to_plotly(resp):
        import base64
        from PIL import Image as PILImage
        from io import BytesIO
        data_uri = getattr(resp, 'get_base64_image', lambda: None)()
        if not data_uri:
            return None
        b64_str = data_uri.replace('data:image/png;base64,', '')
        img_bytes = base64.b64decode(b64_str)
        img = PILImage.open(BytesIO(img_bytes))
        fig = go.Figure()
        fig.add_trace(go.Image(z=img))
        # Use autosize for responsive scaling within chat panel; cap height only
        fig.update_layout(
            xaxis_visible=False, yaxis_visible=False, hovermode=False, showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=min(img.size[1], 350),
            autosize=True
        )
        return fig

    def _downgrade_md_headers(text: str) -> str:
        """Replace markdown headers with bold text to avoid disproportionate font sizes in chat."""
        import re
        if not isinstance(text, str):
            return text
        return re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)

    def _sanitize_user_text(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        # Drop control characters (except common whitespace)
        s = ''.join(ch for ch in s if (ch >= ' ' or ch in '\n\t\r'))
        return s.strip()

    def _limit_df(df: pd.DataFrame, max_rows: int = 30, max_cols: int = 6) -> pd.DataFrame:
        """Limit DataFrame size for display in narrow chat panel."""
        try:
            if df.shape[1] > max_cols:
                df = df.iloc[:, :max_cols]
            if df.shape[0] > max_rows:
                df = df.head(max_rows)
            return df
        except Exception:
            return df

    def _format_cell_value(val) -> str:
        """Format cell values for compact display."""
        try:
            if pd.isna(val):
                return ''
            if isinstance(val, float):
                # Use scientific notation for very large/small numbers
                if abs(val) > 1e6 or (abs(val) < 1e-3 and val != 0):
                    return f'{val:.2e}'
                return f'{val:.4f}'.rstrip('0').rstrip('.')
            return str(val)[:20]  # Truncate long strings
        except Exception:
            return str(val)[:20]

    def _df_to_markdown(df: pd.DataFrame) -> str:
        """Convert DataFrame to compact markdown table string."""
        dff = _limit_df(df)
        if dff.empty:
            return "(Empty table)"
        
        # Format all values
        formatted = dff.copy()
        for col in formatted.columns:
            formatted[col] = formatted[col].apply(_format_cell_value)
        
        # Build markdown table
        cols = list(formatted.columns)
        # Truncate column names if too long
        cols_display = [c[:12] if len(str(c)) > 12 else str(c) for c in cols]
        
        lines = []
        lines.append('| ' + ' | '.join(cols_display) + ' |')
        lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
        
        for _, row in formatted.iterrows():
            row_vals = [str(row[c])[:15] for c in cols]  # Truncate cell values
            lines.append('| ' + ' | '.join(row_vals) + ' |')
        
        result = '\n'.join(lines)
        if len(dff) < len(df):
            result += f'\n\n*(Showing {len(dff)} of {len(df)} rows)*'
        if dff.shape[1] < df.shape[1]:
            result += f'\n*(Showing {dff.shape[1]} of {df.shape[1]} columns)*'
        return result

    def _plotly_table_figure(df: pd.DataFrame) -> go.Figure:
        """Create a Plotly table figure - kept for backwards compatibility but less preferred."""
        dff = _limit_df(df)
        cols = list(dff.columns)
        # Format and truncate values for compact display
        cell_values = []
        for c in cols:
            formatted_col = [_format_cell_value(v) for v in dff[c]]
            cell_values.append(formatted_col)
        
        # Truncate column headers
        cols_display = [str(c)[:10] for c in cols]
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=cols_display, align='left', font=dict(size=11)),
            cells=dict(values=cell_values, align='left', font=dict(size=10)),
            columnwidth=[1] * len(cols)  # Equal column widths
        )])
        # Dynamic height based on row count
        row_height = 24
        header_height = 28
        dynamic_height = min(300, header_height + len(dff) * row_height)
        fig.update_layout(
            margin=dict(l=2, r=2, t=2, b=2),
            height=dynamic_height,
            autosize=True
        )
        return fig

    def _strip_ansi(text) -> str:
        """Remove ANSI escape codes from text for clean display."""
        import re
        if text is None:
            return ''
        text_str = str(text)
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text_str)

    def _coerce_tabular(val) -> Optional[pd.DataFrame]:
        try:
            if isinstance(val, pd.DataFrame):
                return val
            if isinstance(val, list) and val:
                # list[dict] or list[list]
                if all(isinstance(x, dict) for x in val):
                    return pd.DataFrame(val)
                if all(isinstance(x, (list, tuple)) for x in val):
                    return pd.DataFrame(val)
            if isinstance(val, dict) and val:
                # dict -> single-row table for readability
                return pd.DataFrame([val])
        except Exception:
            return None
        return None

    def _is_missing_result_error(msg: str) -> bool:
        try:
            s = (msg or '').lower()
            return ('nameerror' in s) and ('result' in s) and ('not defined' in s)
        except Exception:
            return False

    def _is_sql_parsing_error(msg: str) -> bool:
        """Detect when PandasAI's sqlglot parser failed on non-SQL content."""
        try:
            s = (msg or '').lower()
            return (
                ('invalid expression' in s and 'unexpected token' in s)
                or ('sqlglot' in s)
                or ('parseerror' in s and ('line' in s or 'col' in s))
                or ('syntax error' in s and 'line 1' in s)
            )
        except Exception:
            return False

    def _df_to_store_entry(df: pd.DataFrame) -> dict:
        """Serialize a DataFrame to a dict suitable for dcc.Store and DataTable."""
        nrows, ncols = df.shape
        cols = [{'name': str(c), 'id': str(c)} for c in df.columns]
        data = df.to_dict('records')
        return {
            'columns': cols,
            'data': data,
            'label': f'{nrows}r\u00d7{ncols}c',
            'shape': [nrows, ncols],
        }

    def _normalize_response(resp):
        """Normalize PandasAI response. Returns (message_dict, chart_base64_or_None, df_entry_or_None)."""
        rtype = getattr(resp, 'type', None)
        if rtype == 'chart':
            # Extract base64 data URI directly from PandasAI response
            data_uri = getattr(resp, 'get_base64_image', lambda: None)()
            if data_uri:
                # Ensure proper data URI format
                if not data_uri.startswith('data:'):
                    data_uri = f'data:image/png;base64,{data_uri}'
                # Return a placeholder message and the data URI for storage
                return (
                    {'role': 'assistant', 'content': {'type': 'text', 'text': '📊 Chart generated! Click the button below to view.'}},
                    data_uri,
                    None
                )
            return ({'role': 'assistant', 'content': {'type': 'text', 'text': '(Failed to render chart)'}}, None, None)
        if rtype == 'dataframe':
            val = getattr(resp, 'value', None)
            df = _coerce_tabular(val)
            if df is not None and not df.empty:
                nrows, ncols = df.shape
                entry = _df_to_store_entry(df)
                placeholder = f'📋 Table returned ({nrows} rows \u00d7 {ncols} cols). Click the button below to view and download the full data.'
                return ({'role': 'assistant', 'content': {'type': 'text', 'text': placeholder}}, None, entry)
            return ({'role': 'assistant', 'content': {'type': 'text', 'text': str(val)}}, None, None)
        if rtype == 'error':
            err = getattr(resp, 'error', None)
            return ({'role': 'assistant', 'content': {'type': 'text', 'text': f'Error: {_strip_ansi(err or resp)}'}}, None, None)
        val = getattr(resp, 'value', None)
        df = _coerce_tabular(val)
        if df is not None and not df.empty:
            nrows, ncols = df.shape
            entry = _df_to_store_entry(df)
            placeholder = f'📋 Table returned ({nrows} rows \u00d7 {ncols} cols). Click the button below to view and download the full data.'
            return ({'role': 'assistant', 'content': {'type': 'text', 'text': placeholder}}, None, entry)
        return ({'role': 'assistant', 'content': {'type': 'text', 'text': _downgrade_md_headers(_strip_ansi(val if val is not None else resp))}}, None, None)

    # --- Chatbot callbacks ---
    @app.callback(
        Output('chat-session-id', 'data'),
        Input('url', 'pathname'),
        State('chat-session-id', 'data')
    )
    def _ensure_chat_session_id(_pathname, sid):
        if sid:
            return sid
        return str(uuid.uuid4())

    @app.callback(
        Output('chat-panel', 'style'),
        Output('left-panel', 'width'),
        Output('viewer-panel', 'width'),
        Output('chatbot-visible', 'data'),
        Output('chatbot-expanded', 'data', allow_duplicate=True),
        Input('toggle-chatbot', 'n_clicks'),
        Input('close-chatbot', 'n_clicks'),
        State('chatbot-visible', 'data'),
        prevent_initial_call=True
    )
    def _toggle_chat(toggle_clicks, close_clicks, visible):
        if not toggle_clicks and not close_clicks and not visible:
            raise PreventUpdate
        # Close button always hides; toggle button toggles state
        if ctx.triggered and 'close-chatbot' in ctx.triggered[0]['prop_id']:
            new_vis = False
        else:
            new_vis = not bool(visible)

        # When chatbot is visible, allocate space for it on the right.
        if new_vis:
            chat_style = {'display': 'block'}
            left_width = 6
            viewer_width = 3
        else:
            chat_style = {'display': 'none'}
            left_width = 8
            viewer_width = 4

        # Always reset expanded state on open/close so panel starts collapsed
        return chat_style, left_width, viewer_width, new_vis, False

    # Clientside callback: expand/collapse toggle + drag wiring
    app.clientside_callback(
        """
        function(n_clicks, isExpanded) {
            if (!n_clicks) { return [false, '\u2922']; }

            var newExpanded = !isExpanded;
            var panel = document.getElementById('chat-panel');
            if (!panel) { return [newExpanded, newExpanded ? '\u2921' : '\u2922']; }

            if (newExpanded) {
                /* ── EXPAND: make panel a fixed floating overlay ── */
                var expandW = Math.max(500, Math.min(1100, Math.round(window.innerWidth  * 0.75)));
                var expandH = Math.max(400, Math.min(900,  Math.round(window.innerHeight * 0.80)));
                var rect = panel.getBoundingClientRect();
                panel.style.position    = 'fixed';
                panel.style.top         = '80px';
                panel.style.left        = '50%';
                panel.style.transform   = 'translateX(-50%)';
                panel.style.width       = expandW + 'px';
                panel.style.height      = expandH + 'px';
                panel.style.zIndex      = '9999';
                panel.style.boxShadow   = '0 16px 48px rgba(0,0,0,0.35)';
                panel.style.borderRadius = '12px';
                panel.style.overflow    = 'hidden';
                panel.style.display     = 'block';
                panel.style.flex        = 'none';

                /* Adjust inner body scroller height */
                var scroller = document.getElementById('chat-body-scroller');
                if (scroller) {
                    scroller._origHeight = scroller.style.height;
                    scroller.style.height = (expandH - 180) + 'px';
                }

                /* ── Inject resize handles ── */
                function makeResizer(resizeW, resizeH) {
                    return function(e) {
                        e.preventDefault(); e.stopPropagation();
                        var startX = e.clientX, startY = e.clientY;
                        var startW = panel.offsetWidth, startH = panel.offsetHeight;
                        function onMove(ev) {
                            if (resizeW) {
                                panel.style.width = Math.max(400, startW + (ev.clientX - startX)) + 'px';
                            }
                            if (resizeH) {
                                var newH = Math.max(300, startH + (ev.clientY - startY));
                                panel.style.height = newH + 'px';
                                var sc = document.getElementById('chat-body-scroller');
                                if (sc) sc.style.height = (newH - 180) + 'px';
                            }
                        }
                        function onUp() {
                            document.removeEventListener('mousemove', onMove);
                            document.removeEventListener('mouseup', onUp);
                        }
                        document.addEventListener('mousemove', onMove);
                        document.addEventListener('mouseup', onUp);
                    };
                }
                var rightHandle  = document.createElement('div');
                rightHandle.id   = 'chat-resize-e';
                rightHandle.style.cssText = 'position:absolute;top:0;right:0;width:8px;height:100%;cursor:ew-resize;z-index:10001;background:transparent;';
                var bottomHandle  = document.createElement('div');
                bottomHandle.id   = 'chat-resize-s';
                bottomHandle.style.cssText = 'position:absolute;bottom:0;left:0;width:100%;height:8px;cursor:ns-resize;z-index:10001;background:transparent;';
                var cornerHandle  = document.createElement('div');
                cornerHandle.id   = 'chat-resize-se';
                cornerHandle.style.cssText = 'position:absolute;bottom:0;right:0;width:18px;height:18px;cursor:nwse-resize;z-index:10002;background:transparent;';
                rightHandle.addEventListener('mousedown',  makeResizer(true,  false));
                bottomHandle.addEventListener('mousedown', makeResizer(false, true));
                cornerHandle.addEventListener('mousedown', makeResizer(true,  true));
                panel.appendChild(rightHandle);
                panel.appendChild(bottomHandle);
                panel.appendChild(cornerHandle);

                /* ── Attach drag handler to card header ── */
                var header = panel.querySelector('.card-header');
                if (header && !header._dragHandler) {
                    header.style.cursor = 'move';
                    header._dragHandler = function(e) {
                        if (e.target.tagName === 'BUTTON') return;
                        e.preventDefault();
                        /* Resolve absolute position once, removing centering transform */
                        var r = panel.getBoundingClientRect();
                        panel.style.left      = r.left + 'px';
                        panel.style.top       = r.top  + 'px';
                        panel.style.transform = 'none';
                        var startX = e.clientX;
                        var startY = e.clientY;
                        function onMove(ev) {
                            var dx = ev.clientX - startX;
                            var dy = ev.clientY - startY;
                            startX = ev.clientX;
                            startY = ev.clientY;
                            panel.style.left = (parseFloat(panel.style.left) + dx) + 'px';
                            panel.style.top  = (parseFloat(panel.style.top)  + dy) + 'px';
                        }
                        function onUp() {
                            document.removeEventListener('mousemove', onMove);
                            document.removeEventListener('mouseup',   onUp);
                        }
                        document.addEventListener('mousemove', onMove);
                        document.addEventListener('mouseup',   onUp);
                    };
                    header.addEventListener('mousedown', header._dragHandler);
                }

            } else {
                /* ── COLLAPSE: restore panel to normal Bootstrap flow ── */
                panel.style.position    = '';
                panel.style.top         = '';
                panel.style.left        = '';
                panel.style.transform   = '';
                panel.style.width       = '';
                panel.style.maxWidth    = '';
                panel.style.height      = '';
                panel.style.maxHeight   = '';
                panel.style.zIndex      = '';
                panel.style.boxShadow   = '';
                panel.style.borderRadius = '';
                panel.style.flex        = '';

                /* Restore inner body scroller height */
                var scroller = document.getElementById('chat-body-scroller');
                if (scroller && scroller._origHeight !== undefined) {
                    scroller.style.height = scroller._origHeight;
                }

                /* ── Remove resize handles ── */
                ['chat-resize-e', 'chat-resize-s', 'chat-resize-se'].forEach(function(id) {
                    var el = document.getElementById(id);
                    if (el && el.parentNode) el.parentNode.removeChild(el);
                });

                /* ── Remove drag handler ── */
                var header = panel.querySelector('.card-header');
                if (header) {
                    header.style.cursor = '';
                    if (header._dragHandler) {
                        header.removeEventListener('mousedown', header._dragHandler);
                        header._dragHandler = null;
                    }
                }
            }

            return [newExpanded, newExpanded ? '\u2921' : '\u2922'];
        }
        """,
        Output('chatbot-expanded', 'data'),
        Output('expand-chatbot', 'children'),
        Input('expand-chatbot', 'n_clicks'),
        State('chatbot-expanded', 'data'),
        prevent_initial_call=True
    )

    # Clientside callback: reset panel geometry + icon when Python closes/opens the panel
    app.clientside_callback(
        """
        function(isExpanded) {
            if (isExpanded) { return window.dash_clientside.no_update; }
            var panel = document.getElementById('chat-panel');
            if (panel) {
                panel.style.position    = '';
                panel.style.top         = '';
                panel.style.left        = '';
                panel.style.transform   = '';
                panel.style.width       = '';
                panel.style.maxWidth    = '';
                panel.style.height      = '';
                panel.style.maxHeight   = '';
                panel.style.zIndex      = '';
                panel.style.boxShadow   = '';
                panel.style.borderRadius = '';
                panel.style.flex        = '';
                var header = panel.querySelector('.card-header');
                if (header) {
                    header.style.cursor = '';
                    if (header._dragHandler) {
                        header.removeEventListener('mousedown', header._dragHandler);
                        header._dragHandler = null;
                    }
                }
                var scroller = document.getElementById('chat-body-scroller');
                if (scroller && scroller._origHeight !== undefined) {
                    scroller.style.height = scroller._origHeight;
                }
            }
            return '\u2922';
        }
        """,
        Output('expand-chatbot', 'children', allow_duplicate=True),
        Input('chatbot-expanded', 'data'),
        prevent_initial_call=True
    )

    # Clientside callback: convert "Explain biologically" button click into a
    # synthetic new_message so ChatComponent shows its typing indicator.
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return window.dash_clientside.no_update;
            return {role: 'user', content: '__explain_biologically__', _ts: Date.now()};
        }
        """,
        Output('chat', 'new_message'),
        Input('explain-bio-btn', 'n_clicks'),
        prevent_initial_call=True
    )

    @app.callback(
        Output('chat-cleanup', 'data'),
        Input('chat-cleanup-tick', 'n_intervals')
    )
    def _chat_cleanup(_n):
        ev = registry.cleanup()
        return {'evicted': ev}

    # Callback to toggle chat settings collapse
    @app.callback(
        Output('chat-settings-collapse', 'is_open'),
        Output('chat-settings-icon', 'children'),
        Input('chat-settings-toggle', 'n_clicks'),
        State('chat-settings-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def _toggle_chat_settings(n_clicks, is_open):
        if n_clicks:
            new_state = not is_open
            icon = '▲' if new_state else '▼'
            return new_state, icon
        return is_open, '▼' if is_open else '▲'

    # Callback to toggle mode-specific settings visibility
    @app.callback(
        Output('chat-frame-range-section', 'style'),
        Output('chat-summary-settings', 'style'),
        Output('chat-timeseries-settings', 'style'),
        Output('chat-snapshot-settings', 'style'),
        Input('chat-mode', 'value'),
    )
    def _toggle_mode_settings_visibility(mode):
        show = {'marginBottom': '8px'}
        hide = {'display': 'none'}
        frame_range_style = hide if mode == 'snapshot' else show
        return (
            frame_range_style,
            show if mode == 'summary' else hide,
            show if mode == 'timeseries' else hide,
            show if mode == 'snapshot' else hide,
        )

    # Callback to limit DataFrame selection to max 4 for timeseries
    @app.callback(
        Output('chat-dataframe-selector', 'value'),
        Input('chat-dataframe-selector', 'value'),
        State('chat-mode', 'value'),
        prevent_initial_call=True
    )
    def _limit_dataframe_selection(selected, mode):
        max_dfs = 4 if (mode == 'timeseries') else len(DATAFRAME_OPTIONS)
        return (selected or [])[:max_dfs]

    # Clientside callback to update stride display and size badge
    app.clientside_callback(
        """
        function(fmin, fmax, mode, autoStride, manualStride, pairsStore, maxValues) {
            const MAX_VALUES = maxValues || 5000;
            const NO_UPDATE = window.dash_clientside.no_update;
            if (!pairsStore) return [NO_UPDATE, NO_UPDATE, NO_UPDATE];

            const totalPairs = pairsStore.total || 1;
            const frameRange = Math.max(1, (fmax || 0) - (fmin || 0) + 1);

            if (mode !== 'timeseries') {
                return ['', {display:'none'}, {display:'none'}];
            }

            let stride;
            if (autoStride) {
                const maxFrames = Math.max(1, Math.floor(MAX_VALUES / totalPairs));
                stride = Math.max(1, Math.ceil(frameRange / maxFrames));
            } else {
                stride = Math.max(1, manualStride || 1);
            }

            const framesKept = Math.ceil(frameRange / stride);
            const totalValues = totalPairs * framesKept;
            const overLimit = totalValues > MAX_VALUES;
            const modeLabel = autoStride ? ' (auto)' : ' (manual)';
            let strideText = `Stride ${stride}${modeLabel} \u2192 ${framesKept} frame${framesKept===1?'':'s'}`;
            if (autoStride && framesKept === 1 && overLimit) {
                strideText += ' \u2014 filter residues for more frames';
            }
            const badgeText = `~${totalValues.toLocaleString()} / ${MAX_VALUES.toLocaleString()} values`
                              + (overLimit ? ' \u26a0\ufe0f' : ' \u2713');
            const badgeStyle = {fontSize:'11px', color: overLimit ? '#d9534f':'#5cb85c', display:'inline'};
            return [strideText, badgeStyle, badgeText];
        }
        """,
        Output('chat-stride-display', 'children'),
        Output('chat-size-badge-ts', 'style'),
        Output('chat-size-badge-ts', 'children'),
        Input('chat-frame-min', 'value'),
        Input('chat-frame-max', 'value'),
        Input('chat-mode', 'value'),
        Input('chat-stride-mode', 'value'),
        Input('chat-stride-manual', 'value'),
        Input('chat-pairs-store', 'data'),
        Input('chat-max-values-store', 'data'),
    )

    app.clientside_callback(
        """
        function(v1, v2) {
            function fmt(v) { return v != null ? parseFloat(v).toFixed(1) : '0.5'; }
            return [fmt(v1), fmt(v2)];
        }
        """,
        Output('chat-et-val', 'children'),
        Output('chat-et-snapshot-val', 'children'),
        Input('chat-energy-threshold', 'value'),
        Input('chat-energy-threshold-snapshot', 'value'),
    )

    # Clientside callback: watchdog to detect network-level timeouts (e.g. 504)
    app.clientside_callback(
        """
        function(new_message, messages, n_intervals, dismiss_clicks) {
            const ctx = window.dash_clientside.callback_context;
            const triggered = ctx.triggered[0] ? ctx.triggered[0].prop_id : '';

            // User dismissed the banner manually
            if (triggered === 'chat-timeout-dismiss.n_clicks' && dismiss_clicks) {
                window._grinn_pending_ts = null;
                return {'display': 'none'};
            }

            // New message sent — record timestamp, hide any existing banner
            if (triggered === 'chat.new_message' && new_message) {
                window._grinn_pending_ts = Date.now();
                return {'display': 'none'};
            }

            // Response received — clear pending state, hide banner
            if (triggered === 'chat.messages') {
                window._grinn_pending_ts = null;
                return {'display': 'none'};
            }

            // Watchdog tick — check if pending for > 90 s
            if (window._grinn_pending_ts && (Date.now() - window._grinn_pending_ts > 90000)) {
                window._grinn_pending_ts = null;
                return {'display': 'flex', 'backgroundColor': '#fff3cd', 'color': '#856404',
                        'border': '1px solid #ffc107', 'borderRadius': '4px',
                        'padding': '6px 10px', 'fontSize': '11px',
                        'flex': '0 0 auto', 'marginBottom': '4px'};
            }

            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-timeout-error', 'style'),
        Input('chat', 'new_message'),
        Input('chat', 'messages'),
        Input('chat-watchdog', 'n_intervals'),
        Input('chat-timeout-dismiss', 'n_clicks'),
        prevent_initial_call=True,
    )

    # Callback to populate chat-pairs-store
    @app.callback(
        Output('chat-pairs-store', 'data'),
        Input('chat-residue-filter', 'value'),
        Input('chat-energy-threshold', 'value'),
        Input('chat-energy-threshold-snapshot', 'value'),
        Input('chat-dataframe-selector', 'value'),
        Input('chat-frame-min', 'value'),
        Input('chat-frame-max', 'value'),
        Input('chat-mode', 'value'),
        prevent_initial_call=False,
    )
    def _update_size_store(residue_filter, energy_threshold, energy_threshold_snapshot, selected_dfs, fmin_val, fmax_val, mode):
        residue_filter = residue_filter or []
        mode = mode or 'summary'
        if mode == 'snapshot':
            min_abs = float(energy_threshold_snapshot or 0.0)
        else:
            min_abs = float(energy_threshold or 0.0)
        fmin_val = fmin_val if fmin_val is not None else frame_min
        fmax_val = fmax_val if fmax_val is not None else frame_max
        keys = selected_dfs or DEFAULT_DATAFRAMES[:4]
        total_pairs = 0
        for key in keys[:4]:
            info = _CHATBOT_DATAFRAMES.get(key, {})
            if info.get('category') == 'Network Metrics':
                # Metrics DataFrames count nodes, not pairs
                est_rows, _, _ = _estimate_filtered_size(
                    key, residue_filter, fmin_val, fmax_val,
                    mode='summary', min_abs_energy=0.0
                )
            else:
                est_rows, _, _ = _estimate_filtered_size(
                    key, residue_filter, fmin_val, fmax_val,
                    mode='timeseries', min_abs_energy=min_abs
                )
            total_pairs += est_rows
        return {'total': max(1, total_pairs)}

    # Callback to update size display badge
    @app.callback(
        Output('chat-size-display', 'children'),
        Input('chat-pairs-store', 'data'),
        Input('chat-mode', 'value'),
        Input('chat-frame-min', 'value'),
        Input('chat-frame-max', 'value'),
        Input('chat-stride-mode', 'value'),
        Input('chat-stride-manual', 'value'),
    )
    def _update_size_display(size_store, mode, fmin_v, fmax_v, auto_stride, manual_stride):
        import math
        n_rows = (size_store or {}).get('total', 0)
        if n_rows <= 0:
            return ''
        mode = mode or 'summary'
        if mode == 'timeseries':
            fmin_v = fmin_v if fmin_v is not None else frame_min
            fmax_v = fmax_v if fmax_v is not None else frame_max
            auto = auto_stride if auto_stride is not None else True
            stride = _compute_chat_stride(fmin_v, fmax_v, n_pairs=n_rows, max_values=MAX_CHAT_VALUES) \
                     if auto else max(1, int(manual_stride or 1))
            total_frames = max(1, fmax_v - fmin_v + 1)
            frames_kept = math.ceil(total_frames / stride)
            total_values = n_rows * frames_kept
            return f'~{total_values:,} data points ({n_rows:,} pairs \u00d7 {frames_kept:,} frames)'
        else:
            return f'~{n_rows:,} data points'

    # Callback to show/hide manual stride input
    @app.callback(
        Output('chat-stride-manual-div', 'style'),
        Input('chat-stride-mode', 'value'),
    )
    def _toggle_manual_stride_visibility(auto):
        return {'display': 'none'} if auto else {'display': 'inline-block'}

    def _search_pubmed(query: str, max_results: int = 3) -> list:
        import urllib.request
        import urllib.parse
        import json
        try:
            # Search for PMIDs
            search_url = (
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'
                + urllib.parse.urlencode({'db': 'pubmed', 'term': query, 'retmax': max_results, 'retmode': 'json'})
            )
            with urllib.request.urlopen(search_url, timeout=5) as resp:
                search_data = json.loads(resp.read().decode())
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            if not pmids:
                return []

            # Fetch abstracts
            fetch_url = (
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'
                + urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml', 'rettype': 'abstract'})
            )
            with urllib.request.urlopen(fetch_url, timeout=5) as resp:
                xml_data = resp.read().decode()

            # Parse XML for titles and abstracts
            import re as _re2
            results = []
            articles = _re2.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_data, _re2.DOTALL)
            for i, art in enumerate(articles[:max_results]):
                pmid_match = _re2.search(r'<PMID[^>]*>(\d+)</PMID>', art)
                title_match = _re2.search(r'<ArticleTitle>(.*?)</ArticleTitle>', art, _re2.DOTALL)
                abstract_match = _re2.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', art, _re2.DOTALL)
                pmid_val = pmid_match.group(1) if pmid_match else pmids[i] if i < len(pmids) else '?'
                title_val = _re2.sub(r'<[^>]+>', '', title_match.group(1)) if title_match else 'No title'
                abstract_val = _re2.sub(r'<[^>]+>', '', abstract_match.group(1)) if abstract_match else 'No abstract'
                results.append({'pmid': pmid_val, 'title': title_val.strip(), 'abstract': abstract_val.strip()[:500]})
            return results
        except Exception:
            return []

    # Helper to extract token usage from PandasAI response
    def _extract_token_usage(resp) -> int:
        """Try to extract token usage from response. Returns 0 if not available."""
        try:
            # Try various attributes that might hold usage info
            usage = getattr(resp, 'usage', None)
            if usage is not None:
                if isinstance(usage, dict):
                    return usage.get('total_tokens', 0) or usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
                elif hasattr(usage, 'total_tokens'):
                    return getattr(usage, 'total_tokens', 0) or 0
            # Try metadata
            metadata = getattr(resp, 'metadata', None)
            if metadata and isinstance(metadata, dict):
                if 'usage' in metadata:
                    u = metadata['usage']
                    return u.get('total_tokens', 0) if isinstance(u, dict) else 0
            # Estimate from response size as fallback (rough: ~4 chars per token)
            val = getattr(resp, 'value', None)
            if val is not None:
                est = len(str(val)) // 4
                return min(est, 1000)  # Cap estimate at 1000
        except Exception:
            pass
        return 500  # Conservative default estimate per query

    def _build_token_display(used, limit, palette):
        if limit <= 0:
            return f"Tokens: {used:,} / \u221e", {'fontSize': '11px', 'color': palette['text'], 'fontFamily': 'monospace'}
        pct = (used / limit) * 100
        if pct >= 100:
            return f"Tokens: {used:,} / {limit:,} (EXCEEDED)", {'fontSize': '11px', 'color': '#dc3545', 'fontFamily': 'monospace', 'fontWeight': 'bold'}
        elif pct >= 80:
            return f"Tokens: {used:,} / {limit:,} ({pct:.0f}%)", {'fontSize': '11px', 'color': '#ffc107', 'fontFamily': 'monospace'}
        else:
            return f"Tokens: {used:,} / {limit:,}", {'fontSize': '11px', 'color': palette['text'], 'fontFamily': 'monospace'}

    @app.callback(
        Output('chat', 'messages'),
        Output('chat-token-usage', 'data'),
        Output('token-usage-display', 'children'),
        Output('token-usage-display', 'style'),
        Output('chat-charts-store', 'data'),
        Output('chat-chart-gallery', 'children'),
        Output('chat-chart-gallery-container', 'style'),
        Output('chat-dataframes-store', 'data'),
        Output('chat-df-gallery', 'children'),
        Output('chat-df-gallery-container', 'style'),
        Output('explain-bio-btn', 'style'),
        Input('chat', 'new_message'),
        State('chat', 'messages'),
        State('chat-session-id', 'data'),
        State('llm-model', 'value'),
        State('chat-dataframe-selector', 'value'),
        State('chat-token-usage', 'data'),
        State('chat-residue-filter', 'value'),
        State('chat-frame-min', 'value'),
        State('chat-frame-max', 'value'),
        State('chat-charts-store', 'data'),
        State('chat-dataframes-store', 'data'),
        State('chat-mode', 'value'),
        State('chat-energy-threshold', 'value'),
        State('chat-snapshot-frame', 'value'),
        State('chat-search-literature', 'value'),
        State('chat-stride-mode', 'value'),
        State('chat-stride-manual', 'value'),
        State('chat-pairs-store', 'data'),
        State('chat-energy-threshold-snapshot', 'value'),
        prevent_initial_call=True
    )
    def _on_chat_msg(new_message, messages, sid, selected_model, selected_dfs, token_data,
                     residue_filter, chat_frame_min, chat_frame_max, charts_store, dfs_store,
                     chat_mode, energy_threshold, snapshot_frame, search_literature,
                     stride_mode_auto, stride_manual, pairs_store, energy_threshold_snapshot):
        if not new_message:
            return (no_update,) * 11

        messages = list(messages or [])
        charts_store = list(charts_store or [])
        dfs_store = dfs_store or []
        token_data = dict(token_data) if token_data else {'used': 0, 'limit': PANDASAI_TOKEN_LIMIT}
        used = token_data.get('used', 0)
        limit = token_data.get('limit', PANDASAI_TOKEN_LIMIT)

        EXPLAIN_SENTINEL = '__explain_biologically__'
        is_explain = (
            isinstance(new_message, dict) and
            new_message.get('content', '') == EXPLAIN_SENTINEL
        )

        # --- Explain biologically branch ---
        if is_explain:
            last_user = ''
            last_assistant = ''
            for msg in reversed(messages):
                role = msg.get('role', '')
                content = msg.get('content', '')
                if isinstance(content, dict):
                    content = content.get('text', str(content))
                if role == 'assistant' and not last_assistant:
                    last_assistant = str(content)
                elif role == 'user' and not last_user:
                    last_user = str(content)
                if last_user and last_assistant:
                    break
            if not last_assistant:
                return (no_update,) * 11
            CHART_PLACEHOLDER = '📊 Chart generated!'
            is_chart_response = last_assistant.startswith(CHART_PLACEHOLDER)
            chart_uri = (charts_store or [None])[-1]
            try:
                import litellm
                chosen_model = _resolve_model_selection(selected_model)
                m = (chosen_model or '').strip().lower()
                is_claude = ('claude' in m) or m.startswith('anthropic/')
                if is_claude:
                    api_key = os.getenv('ANTHROPIC_API_KEY')
                    if not chosen_model.startswith('anthropic/'):
                        chosen_model = f'anthropic/{chosen_model}'
                    litellm_kwargs = {'api_key': api_key}
                else:
                    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                    litellm_kwargs = {'api_key': api_key}
                system_prompt = (
                    "You are an expert biophysicist and structural biologist specializing in protein-protein and protein-ligand interactions. "
                    "The user is analyzing molecular dynamics simulation data using gRINN (get Residue Interaction Energies and Networks). "
                    "Provide a concise, scientifically grounded biological interpretation of the data result shown. "
                    "Discuss what the interaction energies (in kcal/mol) suggest about the protein's structure/function, "
                    "mention relevant biological implications (e.g., stability, allosteric communication, binding interfaces), "
                    "and note any caveats. Be specific and cite units. Keep your response under 200 words."
                )
                if is_chart_response and chart_uri:
                    b64_data = chart_uri.split(',', 1)[1] if ',' in chart_uri else chart_uri
                    user_content = [
                        {
                            'type': 'text',
                            'text': (
                                f"Original question: {last_user}\n\n"
                                "A chart was generated from molecular dynamics data. "
                                "Please provide a biological interpretation of what it shows."
                            )
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/png;base64,{b64_data}'}
                        }
                    ]
                elif is_chart_response:
                    user_content = (
                        f"Original question: {last_user}\n\n"
                        "A chart was generated from molecular dynamics data, but the image is unavailable. "
                        "Please provide a general biological interpretation of what such a chart might show."
                    )
                else:
                    user_content = (
                        f"Original question: {last_user}\n\n"
                        f"Data result:\n{last_assistant}\n\n"
                        "Please provide a biological interpretation of this result."
                    )
                response = litellm.completion(
                    model=chosen_model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_content},
                    ],
                    max_tokens=400,
                    timeout=120,
                    **litellm_kwargs
                )
                explanation = response.choices[0].message.content
                tokens_used = getattr(response.usage, 'total_tokens', 300) if hasattr(response, 'usage') else 300
                messages.append({'role': 'assistant', 'content': {'type': 'text', 'text': f'**Biological interpretation:**\n\n{_downgrade_md_headers(explanation)}'}})
                new_used = used + tokens_used
                token_data['used'] = new_used
                display, disp_style = _build_token_display(new_used, limit, soft_palette)
                return (messages, token_data, display, disp_style,
                        charts_store, no_update, no_update, dfs_store, no_update, no_update, {'display': 'none'})
            except Exception as e:
                err_str = str(e).lower()
                if any(k in err_str for k in ('504', 'gateway timeout', 'timed out', 'timeout', 'read timeout')):
                    explain_err = '⚠️ Request timed out — the AI service took too long to respond. Please try again in a moment.'
                elif any(k in err_str for k in ('429', 'rate limit', 'rate_limit', 'too many requests')):
                    explain_err = '⚠️ Rate limit reached. Please wait a moment before sending another message.'
                elif any(k in err_str for k in ('503', 'service unavailable')):
                    explain_err = '⚠️ AI service temporarily unavailable (503). Please try again shortly.'
                elif any(k in err_str for k in ('connectionerror', 'connection refused', 'network')):
                    explain_err = '⚠️ Something went wrong while contacting the AI service. This is likely a temporary issue — please try again in a moment.'
                else:
                    explain_err = f'⚠️ Explanation failed: {str(e)}'
                messages.append({'role': 'assistant', 'content': {'type': 'text', 'text': explain_err}})
                return (messages, token_data, no_update, no_update,
                        charts_store, no_update, no_update, dfs_store, no_update, no_update, {'display': 'none'})

        try:
            # Normalize filter values
            residue_filter = residue_filter or []
            fmin = chat_frame_min if chat_frame_min is not None else frame_min
            fmax = chat_frame_max if chat_frame_max is not None else frame_max
            # Ensure valid range
            fmin = max(frame_min, min(fmin, frame_max))
            fmax = max(frame_min, min(fmax, frame_max))
            if fmin > fmax:
                fmin, fmax = fmax, fmin
            
            # Token budget check
            if limit > 0 and used >= limit:
                messages.append({'role': 'user', 'content': new_message if isinstance(new_message, str) else new_message.get('content', str(new_message))})
                messages.append({'role': 'assistant', 'content': {'type': 'text', 'text': '⚠️ Token budget exhausted for this session. Please refresh the page to start a new session.'}})
                style = {'fontSize': '11px', 'color': '#dc3545', 'fontFamily': 'monospace'}
                display = f"Tokens: {used:,} / {limit:,} (EXCEEDED)"
                return messages, token_data, display, style, charts_store, no_update, no_update, dfs_store, no_update, no_update, {'display': 'none'}

            # normalize user message
            if isinstance(new_message, str):
                user_msg = {'role': 'user', 'content': new_message}
                user_text = new_message
            else:
                user_msg = new_message
                user_text = new_message.get('content') if isinstance(new_message, dict) else str(new_message)
            messages.append(user_msg)
            user_text = _sanitize_user_text(user_text)
            
            # Determine DataFrames to use
            selected_df_keys = selected_dfs if selected_dfs else DEFAULT_DATAFRAMES[:4]
            mode = chat_mode or 'summary'
            if mode == 'snapshot':
                min_abs_energy = float(energy_threshold_snapshot or 0.0)
            else:
                min_abs_energy = float(energy_threshold if energy_threshold is not None else 0.0)
            snap_frame = snapshot_frame

            # Compute stride based on auto/manual mode
            auto_stride = stride_mode_auto if stride_mode_auto is not None else True
            if auto_stride or mode != 'timeseries':
                n_pairs = (pairs_store or {}).get('total', 1)
                chat_stride = _compute_chat_stride(fmin, fmax, n_pairs=n_pairs, max_values=MAX_CHAT_VALUES)
            else:
                chat_stride = max(1, int(stride_manual or 1))

            # Block over-budget manual timeseries queries
            if mode == 'timeseries' and not auto_stride:
                import math as _math
                n_pairs = (pairs_store or {}).get('total', 1)
                total_frames = max(1, fmax - fmin + 1)
                frames_kept = _math.ceil(total_frames / chat_stride)
                total_values = n_pairs * frames_kept
                if total_values > MAX_CHAT_VALUES:
                    err = (f"\u26a0\ufe0f Manual stride would send ~{total_values:,} values "
                           f"(limit: {MAX_CHAT_VALUES:,}). Increase stride or switch to Auto mode.")
                    new_messages = list(messages or []) + [{'role': 'assistant', 'content': err}]
                    return new_messages, token_data, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

            # create/reuse session - include selected_dfs AND filter settings in session key
            chosen_model = _resolve_model_selection(selected_model)
            dfs_sig = '|'.join(sorted(selected_df_keys[:4]))
            filter_sig = f"res{'_'.join(sorted(residue_filter)) if residue_filter else 'ALL'}_f{fmin}-{fmax}"
            mode_sig = f"mode{mode}_thr{min_abs_energy}_s{chat_stride}"
            pen_folder = _pen_folder_from_dashboard()
            skey = f"{sid}|{pen_folder}|{dfs_sig}|{filter_sig}|{mode_sig}|{chosen_model}"

            tokens_used_this_query = 0
            ctxobj = registry.get_or_create(skey, lambda: _build_session_context_for_dfs(
                selected_df_keys, model=chosen_model,
                residue_filter=residue_filter, frame_min_val=fmin, frame_max_val=fmax,
                mode=mode, snapshot_frame=snap_frame, min_abs_energy=min_abs_energy,
                stride=chat_stride
            ))
            try:
                resp = ctxobj.agent.chat(user_text)
                tokens_used_this_query += _extract_token_usage(resp)
                # Some backends return "error" responses instead of raising.
                if getattr(resp, 'type', None) == 'error':
                    err = str(getattr(resp, 'error', '') or resp)
                    if _is_missing_result_error(err):
                        registry.reset(skey)
                        ctxobj = registry.get_or_create(skey, lambda: _build_session_context_for_dfs(
                            selected_df_keys, model=chosen_model,
                            residue_filter=residue_filter, frame_min_val=fmin, frame_max_val=fmax,
                            mode=mode, snapshot_frame=snap_frame, min_abs_energy=min_abs_energy,
                            stride=chat_stride
                        ))
                        retry_prompt = (
                            user_text
                            + "\n\nIMPORTANT: Your Python code must end by assigning the final output to a variable named result (e.g., result = ...)."
                        )
                        resp = ctxobj.agent.chat(retry_prompt)
                        tokens_used_this_query += _extract_token_usage(resp)
            except Exception as e:
                msg = str(e)
                df_names = [k.replace('-', '_').replace(' ', '_') for k in selected_df_keys[:4]]
                df_list_str = ', '.join(df_names)
                retryable = (
                    ('unauthorized' in msg.lower() and 'table' in msg.lower())
                    or ('unauthorized table' in msg.lower())
                    or ('table_' in msg.lower() and ('unauthorized' in msg.lower() or 'not allowed' in msg.lower() or 'forbidden' in msg.lower()))
                )
                if _is_missing_result_error(msg):
                    # Reset session/sandbox and retry once with an explicit contract.
                    registry.reset(skey)
                    ctxobj = registry.get_or_create(skey, lambda: _build_session_context_for_dfs(
                        selected_df_keys, model=chosen_model,
                        residue_filter=residue_filter, frame_min_val=fmin, frame_max_val=fmax,
                        mode=mode, snapshot_frame=snap_frame, min_abs_energy=min_abs_energy
                    ))
                    retry_prompt = (
                        user_text
                        + "\n\nIMPORTANT: Your Python code must end by assigning the final output to a variable named result (e.g., result = ...)."
                    )
                    resp = ctxobj.agent.chat(retry_prompt)
                    tokens_used_this_query += _extract_token_usage(resp)
                elif retryable:
                    retry_prompt = (
                        user_text
                        + f"\n\nIMPORTANT: Use Python/pandas only on {df_list_str}. "
                          "Do not use SQL and do not reference any table_* identifiers. "
                          "Intermediate results must be stored in Python variables."
                    )
                    resp = ctxobj.agent.chat(retry_prompt)
                    tokens_used_this_query += _extract_token_usage(resp)
                elif _is_sql_parsing_error(msg):
                    # SQL parsing error - retry with explicit pandas-only instruction
                    registry.reset(skey)
                    ctxobj = registry.get_or_create(skey, lambda: _build_session_context_for_dfs(
                        selected_df_keys, model=chosen_model,
                        residue_filter=residue_filter, frame_min_val=fmin, frame_max_val=fmax,
                        mode=mode, snapshot_frame=snap_frame, min_abs_energy=min_abs_energy
                    ))
                    retry_prompt = (
                        user_text
                        + "\n\nCRITICAL: Use ONLY Python pandas code. Do NOT use SQL syntax. "
                          "Access data with df['column'], df.loc[], df.groupby(), df.query() etc. "
                          "Assign your final answer to the variable 'result'."
                    )
                    resp = ctxobj.agent.chat(retry_prompt)
                    tokens_used_this_query += _extract_token_usage(resp)
                else:
                    raise
            assistant_msg, chart_fig, df_result = _normalize_response(resp)
            messages.append(assistant_msg)
            # If a chart was generated, store it and update the message
            if chart_fig is not None:
                chart_idx = len(charts_store)
                charts_store.append(chart_fig)
            if df_result is not None:
                dfs_store.append(df_result)

            # PubMed literature search (if enabled)
            if search_literature and 'pubmed' in (search_literature or []):
                try:
                    pubmed_results = _search_pubmed(f"{user_text} protein interaction energy MD simulation", max_results=3)
                    if pubmed_results:
                        lit_lines = ['**\U0001f4da Related literature:**\n']
                        for art in pubmed_results:
                            lit_lines.append(f"- **PMID {art['pmid']}**: {art['title']}\n  _{art['abstract'][:300]}..._")
                        lit_text = '\n'.join(lit_lines)
                        messages.append({'role': 'assistant', 'content': {'type': 'text', 'text': lit_text}})
                except Exception:
                    pass  # Silently ignore PubMed errors

            explain_btn_style = {'display': 'inline-block', 'marginTop': '4px', 'fontSize': '11px'}
        except Exception as e:
            err_str = str(e).lower()
            # Detect specific HTTP/network errors first, in priority order
            if any(k in err_str for k in ('504', 'gateway timeout', 'timed out', 'timeout', 'read timeout')):
                assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                    '⚠️ Request timed out — the AI service took too long to respond. '
                    'Please try again in a moment.'
                }}
            elif any(k in err_str for k in ('429', 'rate limit', 'rate_limit', 'too many requests')):
                assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                    '⚠️ Rate limit reached. Please wait a moment before sending another message.'
                }}
            elif any(k in err_str for k in ('503', 'service unavailable')):
                assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                    '⚠️ AI service temporarily unavailable (503). Please try again shortly.'
                }}
            elif any(k in err_str for k in ('connectionerror', 'connection refused', 'network')):
                assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                    '⚠️ Something went wrong while contacting the AI service. '
                    'This is likely a temporary issue — please try again in a moment.'
                }}
            # Detect HTML error responses (typically from proxy/server failures)
            elif '<!doctype html>' in err_str or '<html' in err_str:
                if '500' in err_str or 'internal server error' in err_str.lower():
                    assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                        '⚠️ Server error: The AI service returned an internal error. This may be due to:\n'
                        '• Docker sandbox not running or inaccessible\n'
                        '• Invalid API key or model configuration\n'
                        '• Temporary service unavailability\n\n'
                        'Please check Docker is running and try again.'
                    }}
                else:
                    assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                        f'⚠️ Unexpected server response. Please verify your configuration and try again.'
                    }}
            elif 'docker' in err_str.lower() or 'sandbox' in err_str.lower():
                assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                    f'⚠️ Docker sandbox error: {_strip_ansi(e)}\n\nEnsure Docker is running and accessible.'
                }}
            elif 'api' in err_str.lower() and ('key' in err_str.lower() or 'auth' in err_str.lower()):
                assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text':
                    f'⚠️ API authentication error: {_strip_ansi(e)}\n\nPlease check your API key configuration.'
                }}
            else:
                assistant_msg = {'role': 'assistant', 'content': {'type': 'text', 'text': f'Error: {_strip_ansi(e)}'}}
            messages.append(assistant_msg)
            tokens_used_this_query = 0  # Reset on error
            explain_btn_style = {'display': 'none'}

        # Update token usage
        new_used = used + tokens_used_this_query
        token_data['used'] = new_used

        # Build chart gallery buttons
        gallery_buttons = []
        for i in range(len(charts_store)):
            gallery_buttons.append(
                dbc.Button(
                    f"📊 {i+1}",
                    id={'type': 'view-chart-btn', 'index': i},
                    size='sm',
                    color='info',
                    outline=True,
                    style={'fontSize': '10px', 'padding': '2px 6px'}
                )
            )
        # Show/hide gallery container based on whether there are charts
        gallery_style = {
            'display': 'block' if charts_store else 'none',
            'marginTop': '8px',
            'padding': '6px',
            'backgroundColor': soft_palette['surface'],
            'borderRadius': '4px',
            'flex': '0 0 auto'
        }

        # Build DataFrame gallery buttons
        df_gallery_buttons = []
        for i, entry in enumerate(dfs_store):
            label = entry.get('label', str(i + 1))
            df_gallery_buttons.append(
                dbc.Button(
                    f'📋 {label}',
                    id={'type': 'view-df-btn', 'index': i},
                    size='sm',
                    color='warning',
                    outline=True,
                    style={'fontSize': '10px', 'padding': '2px 6px'}
                )
            )
        df_gallery_style = {
            'display': 'block' if dfs_store else 'none',
            'marginTop': '8px',
            'padding': '6px',
            'backgroundColor': soft_palette['surface'],
            'borderRadius': '4px',
            'flex': '0 0 auto',
        }

        display, style = _build_token_display(new_used, limit, soft_palette)

        return (messages, token_data, display, style, charts_store, gallery_buttons, gallery_style,
                dfs_store, df_gallery_buttons, df_gallery_style, explain_btn_style)

    # Callback to open chart modal when a chart button is clicked
    @app.callback(
        Output('chart-view-modal', 'is_open'),
        Output('chart-modal-img', 'src'),
        Input({'type': 'view-chart-btn', 'index': ALL}, 'n_clicks'),
        State('chat-charts-store', 'data'),
        prevent_initial_call=True
    )
    def _open_chart_modal(n_clicks_list, charts_store):
        from dash import ctx
        if not n_clicks_list or not any(n_clicks_list):
            return False, ''
        
        # Find which button was clicked
        triggered = ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            return False, ''
        
        chart_idx = triggered.get('index')
        if chart_idx is None or charts_store is None or chart_idx >= len(charts_store):
            return False, ''
        
        # Get the chart base64 data URI and return it
        chart_src = charts_store[chart_idx]
        return True, chart_src

    @app.callback(
        Output('df-view-modal', 'is_open'),
        Output('df-modal-table', 'data'),
        Output('df-modal-table', 'columns'),
        Output('df-modal-index', 'data'),
        Input({'type': 'view-df-btn', 'index': ALL}, 'n_clicks'),
        State('chat-dataframes-store', 'data'),
        prevent_initial_call=True
    )
    def _open_df_modal(n_clicks_list, dfs_store):
        from dash import ctx
        if not n_clicks_list or not any(n_clicks_list):
            return False, [], [], None
        triggered = ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            return False, [], [], None
        df_idx = triggered.get('index')
        if df_idx is None or dfs_store is None or df_idx >= len(dfs_store):
            return False, [], [], None
        entry = dfs_store[df_idx]
        return True, entry['data'], entry['columns'], df_idx

    @app.callback(
        Output('df-download', 'data'),
        Input('df-download-btn', 'n_clicks'),
        State('df-modal-index', 'data'),
        State('chat-dataframes-store', 'data'),
        prevent_initial_call=True
    )
    def _download_df(n_clicks, df_idx, dfs_store):
        if not n_clicks or df_idx is None or not dfs_store:
            return no_update
        entry = dfs_store[df_idx]
        df = pd.DataFrame(entry['data'])
        return dcc.send_data_frame(df.to_csv, 'grinn_table.csv', index=False)

    # Pairwise & Viewer - Split into separate callbacks for better performance
    @app.callback(
        Output('total_energy_graph','figure'),
        Output('vdw_energy_graph','figure'),
        Output('elec_energy_graph','figure'),
        Output('energy_bar_chart','figure'),
        Output('second_residue_table','data'),
        Output('second_residue_table','selected_rows'),
        Input('first_residue_table','selected_rows'),
        Input('second_residue_table','selected_rows'),
        Input('frame_slider','value'),
        State('second_residue_table','data')
    )
    def update_pairwise_graphs(sel1, sel2, selected_frame, second_data):
        # Initialize empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor='rgba(250,255,250,0.4)',
            paper_bgcolor='rgba(250,255,250,0.4)',
            font=dict(family='Roboto, sans-serif', size=12, color='#4A5A4A')
        )
        
        total_fig = empty_fig
        vdw_fig = empty_fig
        elec_fig = empty_fig
        bar_fig = go.Figure()
        
        # First selection
        if not sel1:
            return total_fig, vdw_fig, elec_fig, bar_fig, [], []
        
        first = first_res_list[sel1[0]]
        
        # PERFORMANCE OPTIMIZATION: Cache filtering results
        cache_key = f"pairwise_{first}"
        if cache_key not in _sorted_residues_cache:
            # Always update second table when first residue is selected
            filt = total_df[(total_df['res1']==first)|(total_df['res2']==first)]
            others = [r for r in pd.concat([filt['res1'],filt['res2']]).unique() if r!=first]
            # Sort the interacting residues by sequence order
            others_sorted = sort_residues_by_sequence(others)
            _sorted_residues_cache[cache_key] = others_sorted
        else:
            others_sorted = _sorted_residues_cache[cache_key]
        
        table = [{'Residue': r} for r in others_sorted]
        
        # Create bar chart for average energies
        if others_sorted:
            bar_data = {'Residue': [], 'Total': [], 'VdW': [], 'Electrostatic': []}
            # AGGRESSIVE OPTIMIZATION: Use pre-computed average energies
            for r in others_sorted:
                bar_data['Residue'].append(r)
                
                for energy_type in ['Total', 'VdW', 'Electrostatic']:
                    ie = _pairwise_avg_energies.get(first, {}).get(r, {}).get(energy_type, 0)
                    bar_data[energy_type].append(ie)
            
            bar_fig = go.Figure()
            colors = {'Total': '#7C9885', 'VdW': '#A8C4A2', 'Electrostatic': '#9AB3A8'}
            for energy_type in ['Total', 'VdW', 'Electrostatic']:
                bar_fig.add_trace(go.Bar(
                    x=bar_data[energy_type],
                    y=bar_data['Residue'],
                    name=energy_type,
                    orientation='h',
                    marker_color=colors[energy_type],
                    opacity=0.8
                ))
            
            bar_fig.update_layout(
                title="Average Interaction Energies",
                xaxis_title='Energy (kcal/mol)',
                yaxis_title='Residue',
                barmode='group',
                height=400,
                plot_bgcolor='rgba(250,255,250,0.4)',
                paper_bgcolor='rgba(250,255,250,0.4)',
                font=dict(family='Roboto, sans-serif', size=10, color='#4A5A4A'),
                title_font=dict(family='Roboto, sans-serif', size=12, color='#4A5A4A'),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#B5C5B5',
                    borderwidth=1
                ),
                margin=dict(l=80, r=20, t=40, b=80),
                # AGGRESSIVE OPTIMIZATION: Disable animations
                transition={'duration': 0}
            )
        
        # Check if first residue table was the trigger - if so, clear second table selection
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'first_residue_table.selected_rows':
            return total_fig, vdw_fig, elec_fig, bar_fig, table, []
        
        # If no second residue is selected, return with updated table and cleared selection
        if not sel2:
            return total_fig, vdw_fig, elec_fig, bar_fig, table, []
        
        # Validate that the selected second residue exists in the updated table
        if sel2[0] >= len(table):
            return total_fig, vdw_fig, elec_fig, bar_fig, table, []
        
        # Energy graphs and points
        second = table[sel2[0]]['Residue']
        p1, p2 = f"{first}-{second}", f"{second}-{first}"
        
        # Create graphs for each energy type
        energy_configs = {
            'Total': {'color': '#7C9885', 'title': '🔥 Total Interaction Energy'},
            'VdW': {'color': '#A8C4A2', 'title': '🌊 van der Waals Energy'},
            'Electrostatic': {'color': '#9AB3A8', 'title': '⚡ Electrostatic Energy'}
        }
        
        figures = {}
        for energy_type, config in energy_configs.items():
            fig = go.Figure()
            
            # Use DataFrame directly - filtering is fast enough with indexed data
            df_line = energy_long[energy_type][(energy_long[energy_type]['Pair']==p1)|(energy_long[energy_type]['Pair']==p2)]
            
            if not df_line.empty:
                # Convert to numpy for faster plotting
                frames = df_line['Frame'].values
                energies = df_line['Energy'].values
                
                fig.add_trace(go.Scatter(
                    x=frames,
                    y=energies,
                    mode='lines+markers',
                    marker=dict(size=4, opacity=0.7, color=config['color']),
                    line=dict(color=config['color'], width=2),
                    name=energy_type
                ))
                
                # Find current frame value (frames are strings)
                frame_str = str(selected_frame)
                if frame_str in frames:
                    idx = np.where(frames == frame_str)[0][0]
                    e0 = energies[idx]
                    fig.add_trace(go.Scatter(
                        x=[selected_frame],
                        y=[e0],
                        mode='markers',
                        marker=dict(color='#FF1493', size=8, symbol='diamond', line=dict(color='#FF69B4', width=2)),
                        name='Current Frame',
                        showlegend=False
                    ))
            
            fig.update_layout(
                hovermode='x unified',
                title=f"{config['title']} for {first}-{second}",
                xaxis_title='Frame',
                yaxis_title='Energy (kcal/mol)',
                plot_bgcolor='rgba(250,255,250,0.4)',
                paper_bgcolor='rgba(250,255,250,0.4)',
                font=dict(family='Roboto, sans-serif', size=10, color='#4A5A4A'),
                title_font=dict(family='Roboto, sans-serif', size=12, color='#4A5A4A'),
                margin=dict(l=60, r=10, t=40, b=40),
                showlegend=False,
                # AGGRESSIVE OPTIMIZATION: Disable animations for instant updates
                transition={'duration': 0}
            )
            figures[energy_type] = fig
        
        return figures['Total'], figures['VdW'], figures['Electrostatic'], bar_fig, table, sel2

    # Separate callback for molecular viewer
    @app.callback(
        Output('viewer','frame'),
        Input('frame_slider','value')
    )
    def update_viewer_frame(selected_frame):
        return selected_frame

    # Drag handler for molecular viewer
    @app.callback(
        Output('viewer', 'drag'),
        Input('viewer', 'drag'),
        prevent_initial_call=True
    )
    def update_viewer_drag(drag_value):
        if drag_value is not None:
            return drag_value
        return no_update

    # Molecular selection based on residue tables (only for pairwise tab)
    @app.callback(
        Output('viewer','selection'),
        Output('viewer','focus'),
        Input('first_residue_table','selected_rows'),
        Input('second_residue_table','selected_rows'),
        Input('main-tabs','value'),
        State('second_residue_table','data'),
        prevent_initial_call=True
    )
    def update_molecular_selection(sel1, sel2, active_tab, second_data):
        seldata = no_update
        focusdata = no_update
        
        # Only apply molecular selection if we're on the pairwise energies tab
        if active_tab != 'tab-pairwise':
            return seldata, focusdata
        
        if not sel1 or not sel2 or not second_data:
            return seldata, focusdata
        
        if sel2[0] >= len(second_data):
            return seldata, focusdata
        
        try:
            first = first_res_list[sel1[0]]
            second = second_data[sel2[0]]['Residue']
            
            # Parse residue names (e.g., "GLY290_A" -> chain "A", resnum "290")
            first_parts = first.split('_')
            second_parts = second.split('_')
            
            if len(first_parts) < 2 or len(second_parts) < 2:
                return seldata, focusdata
            
            # Extract residue number from format like "GLY290"
            r1_match = re.findall(r'\d+', first_parts[0])
            r2_match = re.findall(r'\d+', second_parts[0])
            
            if not r1_match or not r2_match:
                return seldata, focusdata
            
            r1, c1 = r1_match[0], first_parts[1]
            r2, c2 = r2_match[0], second_parts[1]
            
            t1 = molstar_helper.get_targets(c1, r1)
            t2 = molstar_helper.get_targets(c2, r2)
            seldata = molstar_helper.get_selection([t1, t2], select=True, add=False)
            focusdata = molstar_helper.get_focus([t1, t2], analyse=True)
        except Exception as e:
            print(f"Error in molecular selection: {e}")
            pass
        
        return seldata, focusdata

    # Molecular selection based on heatmap clicks
    @app.callback(
        Output('viewer','selection', allow_duplicate=True),
        Output('viewer','focus', allow_duplicate=True),
        Input('matrix_heatmap','clickData'),
        State('main-tabs','value'),
        prevent_initial_call=True
    )
    def update_molecular_selection_from_heatmap(clickData, active_tab):
        """Update molecular viewer when user clicks on heatmap"""
        seldata = no_update
        focusdata = no_update
        
        # Only apply molecular selection if we're on the matrix tab and have click data
        if active_tab != 'tab-matrix' or not clickData:
            return seldata, focusdata
        
        try:
            # Extract clicked point data
            point = clickData['points'][0]
            x_residue = point['x']  # Column residue
            y_residue = point['y']  # Row residue
            
            print(f"🖱️ Heatmap clicked: {x_residue} ↔ {y_residue}")
            
            # Parse residue names to extract chain and residue number
            def parse_residue(res_name):
                """Parse residue name to extract chain and residue number"""
                try:
                    if '_' in res_name:
                        # Format: RES123_A or similar
                        parts = res_name.split('_')
                        res_part = parts[0]  # e.g., "ALA123"
                        chain = parts[1] if len(parts) > 1 else 'A'
                        
                        # Extract residue number from the end of res_part
                        match = re.search(r'(\d+)$', res_part)
                        if match:
                            res_num = match.group(1)
                        else:
                            # If no digits found, try to extract any digits
                            res_num = ''.join(filter(str.isdigit, res_part))
                            if not res_num:
                                res_num = '1'  # fallback
                        
                        return chain, res_num
                    else:
                        # Try to parse without underscore - assume format like "ALA123"
                        match = re.search(r'([A-Za-z]+)(\d+)', res_name)
                        if match:
                            res_num = match.group(2)
                            return 'A', res_num  # Default to chain A
                        else:
                            # Last resort - extract any digits
                            res_num = ''.join(filter(str.isdigit, res_name))
                            return 'A', res_num if res_num else '1'
                            
                except Exception as e:
                    print(f"Error parsing residue {res_name}: {e}")
                    return 'A', '1'  # Safe fallback
            
            # Parse both residues
            chain1, res_num1 = parse_residue(x_residue)
            chain2, res_num2 = parse_residue(y_residue)
            
            print(f"🎯 Targeting: Chain {chain1} Res {res_num1} ↔ Chain {chain2} Res {res_num2}")
            
            # Create molecular targets for both residues
            target1 = molstar_helper.get_targets(chain1, res_num1)
            target2 = molstar_helper.get_targets(chain2, res_num2)
            
            # Create selection and focus data
            seldata = molstar_helper.get_selection([target1, target2], select=True, add=False)
            focusdata = molstar_helper.get_focus([target1, target2], analyse=True)
            
            print(f"✅ Molecular viewer updated for residue pair")
            
        except Exception as e:
            print(f"❌ Error processing heatmap click: {e}")
            print(f"Click data: {clickData}")
            # Return no_update to avoid errors
            return no_update, no_update
        
        return seldata, focusdata

    # Clear molecular selection when switching away from pairwise tab
    @app.callback(
        Output('viewer','selection', allow_duplicate=True),
        Output('viewer','focus', allow_duplicate=True),
        Input('main-tabs','value'),
        prevent_initial_call=True
    )
    def clear_molecular_selection_on_tab_change(active_tab):
        # Clear selection and focus when not on pairwise tab
        if active_tab != 'tab-pairwise':
            # Return empty objects instead of None to avoid JS errors
            return {}, {}
        return no_update, no_update

    # Auto-switch viewer tabs based on active main tab
    @app.callback(
        Output('viewer-tabs', 'value'),
        Input('main-tabs', 'value'),
        prevent_initial_call=True
    )
    def switch_viewer_tab_on_main_tab_change(main_tab):
        """Switch to Network Visualization when Network Analysis tab is active, otherwise Structure Viewer."""
        if main_tab == 'tab-network':
            return 'tab-network-viewer'
        else:
            return 'tab-structure-viewer'

    # Range Slider Update
    @app.callback(
        Output('heatmap_range_slider', 'min'),
        Output('heatmap_range_slider', 'max'),
        Output('heatmap_range_slider', 'value'),
        Output('heatmap_range_slider', 'marks'),
        Output('manual_range_input', 'value'),
        Input('energy_type_selector', 'value')
    )
    def update_range_slider(energy_type):
        # Get the selected energy dataframe
        selected_df = energy_dfs[energy_type]
        
        # Get all energy columns (excluding metadata columns) and ensure they are numeric
        excluded_cols = ['res1', 'res2', 'Pair', 'Unnamed: 0','res1_index','res2_index','res1_chain','res2_chain',
                         'res1_resnum','res2_resnum','res1_resname','res2_resname']
        
        # Filter for numeric columns only
        energy_cols = []
        for col in selected_df.columns:
            if col not in excluded_cols:
                try:
                    # Try to convert the column to numeric to ensure it's a frame column
                    pd.to_numeric(selected_df[col], errors='raise')
                    energy_cols.append(col)
                except (ValueError, TypeError):
                    # Skip non-numeric columns
                    continue
        
        if not energy_cols:
            # Fallback if no numeric columns found
            return 0.5, 10, 5, {i: {'label': f'±{i}', 'style': {'color': 'white', 'fontSize': '11px', 'fontWeight': 'bold'}} for i in range(1, 11, 2)}, 5
        
        # Calculate min and max across all frames for this energy type
        energy_values = selected_df[energy_cols].values.flatten()
        energy_values = pd.to_numeric(energy_values, errors='coerce')  # Convert to numeric, NaN for non-numeric
        energy_values = energy_values[~pd.isna(energy_values)]  # Remove NaN values
        
        if len(energy_values) > 0:
            data_min = float(energy_values.min())
            data_max = float(energy_values.max())
            
            # Create symmetric range around zero for proper color mapping
            abs_max = max(abs(data_min), abs(data_max))
            
            # Add some padding (20% of the maximum absolute value)
            padding = abs_max * 0.2
            range_limit = abs_max + padding
            
            # Round to reasonable precision for nice numbers
            if range_limit <= 2:
                range_limit = round(range_limit * 2) / 2  # Round to nearest 0.5
            elif range_limit <= 10:
                range_limit = round(range_limit)  # Round to nearest integer
            else:
                range_limit = round(range_limit / 5) * 5  # Round to nearest 5
            
            # Ensure minimum range of 1.0
            range_limit = max(1.0, range_limit)
            
            # Set slider range from 0.5 to range_limit
            slider_min = 0.5
            slider_max = range_limit
            
            # Set initial value to 20% of the maximum range
            initial_value = max(slider_min, range_limit * 0.2)
            
            # Create exactly 5 marks evenly distributed across the slider range
            mark_values = np.linspace(slider_min, slider_max, 5)
            
            marks = {}
            for value in mark_values:
                # Round to appropriate decimal places for clean display
                if value < 1:
                    rounded_value = round(value, 2)
                elif value < 10:
                    rounded_value = round(value, 1)
                else:
                    rounded_value = round(value)
                
                marks[value] = {
                    'label': f'±{rounded_value:g}',
                    'style': {'color': 'white', 'fontSize': '11px', 'fontWeight': 'bold'}
                }
        else:
            # Fallback values if no data
            slider_min, slider_max = 0.5, 10
            initial_value = 5
            marks = {i: {'label': f'±{i}', 'style': {'color': 'white', 'fontSize': '11px', 'fontWeight': 'bold'}} for i in range(1, 11, 2)}
        
        return slider_min, slider_max, initial_value, marks, initial_value

    # Synchronize slider and manual input
    @app.callback(
        Output('heatmap_range_slider', 'value', allow_duplicate=True),
        Output('manual_range_input', 'value', allow_duplicate=True),
        Input('heatmap_range_slider', 'value'),
        Input('manual_range_input', 'value'),
        prevent_initial_call=True
    )
    def sync_range_inputs(slider_value, manual_value):
        # Determine which input triggered the callback
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'heatmap_range_slider' and slider_value is not None:
                # Slider was moved, update manual input
                return slider_value, slider_value
            elif trigger_id == 'manual_range_input' and manual_value is not None:
                # Manual input was changed, update slider
                return manual_value, manual_value
        
        # Return no update if no clear trigger
        return no_update, no_update

    # Interaction Matrix
    @app.callback(
        Output('matrix_heatmap','figure'),
        Input('frame_slider','value'),
        Input('energy_type_selector','value'),
        Input('heatmap_range_slider','value'),
        Input('manual_range_input','value')
    )
    def update_energy_matrix(frame_value, energy_type, slider_range_value, manual_range_value):
        # Use manual input if provided, otherwise use slider value
        range_value = manual_range_value if manual_range_value is not None else slider_range_value
        
        frame_col = str(frame_value)
        
        # Select the appropriate dataframe
        selected_df = energy_dfs[energy_type]
        
        if frame_col not in selected_df.columns:
            return go.Figure()
        
        # Get data for the specific frame
        df = selected_df[['res1','res2',frame_col]].copy()
        df.columns = ['res1','res2','energy']
        
        # Remove any NaN values
        df = df.dropna()
        
        if df.empty:
            return go.Figure()
        
        # Get residues that actually appear in this frame's data and sort them
        all_residues_in_frame = set(df['res1']).union(df['res2'])
        residues = sort_residues_by_sequence(all_residues_in_frame)
        
        # AGGRESSIVE OPTIMIZATION: Use numpy array instead of DataFrame for faster operations
        n = len(residues)
        res_to_idx = {res: i for i, res in enumerate(residues)}
        matrix = np.zeros((n, n), dtype=np.float32)
        
        # Fill matrix using numpy indexing (much faster than DataFrame.loc)
        for res1, res2, energy_val in zip(df['res1'], df['res2'], df['energy']):
            if res1 in res_to_idx and res2 in res_to_idx:
                i, j = res_to_idx[res1], res_to_idx[res2]
                matrix[i, j] = energy_val
                matrix[j, i] = energy_val  # Symmetric
        
        # Use symmetric range based on slider value
        zmin, zmax = -range_value, range_value
        
        # Create the heatmap using numpy array directly
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=residues,
            y=residues,
            colorscale='RdBu_r',
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=f'{energy_type} Energy (kcal/mol)',
                    font=dict(color='#4A5A4A', family='Roboto, sans-serif')
                ),
                tickfont=dict(color='#4A5A4A', family='Roboto, sans-serif'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#B5C5B5',
                borderwidth=2
            )
        ))
        
        fig.update_layout(
            title=f'{energy_type} Interaction Energy Matrix (Frame {frame_value})<br><sub>💡 Click on any cell to zoom into the residue pair in the molecular viewer</sub>',
            xaxis_title='🧬 Residue',
            yaxis_title='🧬 Residue',
            xaxis={
                'tickangle': 45, 
                'automargin': True,
                'categoryorder': 'array',
                'categoryarray': residues
            },
            yaxis={
                'automargin': True,
                'categoryorder': 'array',
                'categoryarray': residues
            },
            margin=dict(l=80, r=50, t=100, b=100),
            font=dict(size=10, family='Roboto, sans-serif', color='#4A5A4A'),
            title_font=dict(size=16, family='Roboto, sans-serif', color='#4A5A4A'),
            plot_bgcolor='rgba(250,255,250,0.4)',
            paper_bgcolor='rgba(250,255,250,0.4)',
            height=600,
            # AGGRESSIVE OPTIMIZATION: Disable animations for instant updates
            transition={'duration': 0}
        )
        return fig

    # Reset residues selection callback
    @app.callback(
        Output('selected_residues_dropdown', 'value'),
        [Input('reset_residues_btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_residue_selection(n_clicks):
        """Reset the residue selection to show all residues."""
        return []

    # Network Metrics - trigger on button click OR on initial load
    # Network Metrics Visualization - responsive to metric selector, frame slider, network settings, residue selection, viz type, sort order, and cutoffs
    @app.callback(
        Output('network_metrics_heatmap', 'figure'),
        [Input('metric_selector', 'value'),
         Input('frame_slider', 'value'),
         Input('update_network_btn', 'n_clicks'),
         Input('selected_residues_dropdown', 'value'),
         Input('network_energy_type', 'value'),
         Input('metrics-viz-tabs', 'value'),
         Input('metrics-sort-order', 'value'),
         Input('metrics-lower-cutoff', 'value'),
         Input('metrics-upper-cutoff', 'value')],
        [State('include_covalent_edges', 'value'),
         State('energy_cutoff', 'value')],
        prevent_initial_call=False
    )
    def update_network_metrics_visualization(metric, current_frame, n_clicks, selected_residues, energy_type,
                                              viz_type, sort_order, lower_cutoff, upper_cutoff, include_cov, cutoff):
        """Generate heatmap or violin plot showing network metrics across all frames and residues."""
        print(f"[CALLBACK TRIGGERED] metric={metric}, current_frame={current_frame}, viz_type={viz_type}, sort={sort_order}", flush=True)
        
        # Use default values if not provided
        if current_frame is None:
            current_frame = frame_min
        if cutoff is None:
            cutoff = 1.0
        if include_cov is None:
            include_cov = ['include']
        if metric is None:
            metric = 'degree'
        if viz_type is None:
            viz_type = 'heatmap'
        if sort_order is None:
            sort_order = 'sequence'
        if selected_residues is None or len(selected_residues) == 0:
            selected_residues = first_res_list  # Show all if none selected

        energy_key = _pen_energy_key(energy_type)
        cov_flag = _pen_cov_flag(include_cov)
        metrics_path = _pen_metrics_path(energy_key, cov_flag, float(cutoff))
        dfm = _load_metrics_df(metrics_path)
        if dfm is None or dfm.empty:
            return go.Figure()

        # Apply metric cutoff filtering
        if lower_cutoff is not None or upper_cutoff is not None:
            dfm, valid_residues = _filter_metrics_by_cutoff(dfm, metric, lower_cutoff, upper_cutoff)
            # Intersect with selected residues
            selected_residues = [r for r in selected_residues if r in valid_residues]
            if not selected_residues:
                fig = go.Figure()
                fig.add_annotation(
                    text="No residues match the cutoff criteria",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color=soft_palette['text'])
                )
                return fig

        # Apply sorting
        residues = _sort_residues_by_metric(dfm, metric, list(selected_residues), sort_order)

        # Generate violin plot or heatmap based on viz_type
        if viz_type == 'violin':
            return _create_metrics_violin_figure(
                dfm, metric, current_frame, residues,
                frame_min, frame_max, soft_palette
            )

        # Heatmap visualization (default)
        frames = list(range(frame_min, frame_max + 1))
        idx_res = {r: i for i, r in enumerate(residues)}
        idx_frame = {f: j for j, f in enumerate(frames)}
        heatmap_data = np.zeros((len(residues), len(frames)), dtype=float)

        try:
            dfm_filtered = dfm[dfm['residue'].isin(residues)]
        except Exception:
            return go.Figure()

        for _, row in dfm_filtered.iterrows():
            try:
                i = idx_res.get(row['residue'])
                j = idx_frame.get(int(row['frame']))
                if i is None or j is None:
                    continue
                heatmap_data[i, j] = float(row[metric])
            except Exception:
                continue
        
        print(f"[Network Metrics Heatmap] Data shape: {heatmap_data.shape}", flush=True)
        print(f"[Network Metrics Heatmap] Data range: [{np.min(heatmap_data):.4f}, {np.max(heatmap_data):.4f}]", flush=True)
        print(f"[Network Metrics Heatmap] Non-zero values: {np.count_nonzero(heatmap_data)}", flush=True)
        
        # Create frame labels
        frame_labels = [str(f) for f in range(frame_min, frame_max + 1)]
        
        # Metric titles
        metric_titles = {
            'degree': 'Degree Centrality',
            'betweenness': 'Betweenness Centrality',
            'closeness': 'Closeness Centrality'
        }
        
        # Create heatmap with white-to-blue colorscale
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=frame_labels,
            y=residues,
            colorscale=[
                [0, 'white'],
                [0.5, 'rgb(173, 216, 230)'],  # Light blue
                [1, 'rgb(70, 130, 180)']      # Steel blue (matches theme)
            ],
            colorbar=dict(
                title=dict(
                    text=metric_titles[metric],
                    side='right',
                    font=dict(size=12)
                ),
                thickness=15,
                len=0.7
            ),
            hovertemplate='<b>%{{y}}</b><br>Frame: %{{x}}<br>{}: %{{z:.4f}}<extra></extra>'.format(metric_titles[metric])
        ))
        
        # Add vertical line to highlight current frame
        frame_idx = current_frame - frame_min
        fig.add_shape(
            type='rect',
            x0=frame_idx - 0.5,
            x1=frame_idx + 0.5,
            y0=-0.5,
            y1=len(residues) - 0.5,
            line=dict(color='red', width=2),
            fillcolor='rgba(0,0,0,0)'
        )
        
        # Build title with sort info
        title_text = f'{metric_titles[metric]} Across Frames'
        if sort_order != 'sequence':
            title_text += f' (sorted {sort_order})'
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=14, color=soft_palette['primary'], family='Roboto, sans-serif')
            ),
            xaxis=dict(
                title='Frame',
                tickfont=dict(size=12),
                side='bottom',
                showgrid=False
            ),
            yaxis=dict(
                title='Residue',
                tickfont=dict(size=12),
                showgrid=False,
                autorange=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='rgba(255,255,255,0.9)',
            margin=dict(l=80, r=60, t=50, b=50),
            font=dict(family='Roboto, sans-serif', color='#4A5A4A', size=12),
            autosize=True  # Fit to container
        )
        
        return fig

    # 3D Network Visualization callback
    @app.callback(
        Output('network-3d-container', 'children'),
        [Input('frame_slider', 'value'),
         Input('update_network_btn', 'n_clicks'),
         Input('shortest_paths_table', 'selected_rows'),
         Input('metric_selector', 'value'),
         Input('selected_residues_dropdown', 'value'),
         Input('network_energy_type', 'value')],
        [State('include_covalent_edges', 'value'),
         State('energy_cutoff', 'value'),
         State('shortest_paths_table', 'data')],
        prevent_initial_call=False
    )
    def update_3d_network(frame, n_clicks, selected_path_rows, metric, selected_residues, energy_type, include_cov, cutoff, path_table_data):
        """Update 3D force graph network visualization based on frame and network parameters."""
        print(f"\n{'='*60}", flush=True)
        print(f"[3D NETWORK CALLBACK] FIRED!", flush=True)
        print(f"  frame={frame}, n_clicks={n_clicks}, metric={metric}", flush=True)
        print(f"  selected_residues={len(selected_residues) if selected_residues else 0}", flush=True)
        print(f"  include_cov={include_cov}, cutoff={cutoff}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        try:
            # Use default values if not provided
            if frame is None:
                frame = frame_min
            if cutoff is None:
                cutoff = 1.0
            if include_cov is None:
                include_cov = ['include']
            if metric is None:
                metric = 'degree'
            if selected_residues is None or len(selected_residues) == 0:
                selected_residues = []  # Empty means no special highlighting
            
            print(f"[3D Network] Updating for frame {frame}, cutoff {cutoff}, metric {metric}", flush=True)
            
            # Convert selected residues to set for fast lookup
            selected_set = set(selected_residues) if selected_residues else set()

            energy_key = _pen_energy_key(energy_type)
            cov_flag = _pen_cov_flag(include_cov)

            metric_data = {}
            metrics_path = _pen_metrics_path(energy_key, cov_flag, float(cutoff))
            dfm = _load_metrics_df(metrics_path)
            if dfm is not None and not dfm.empty:
                try:
                    dff = dfm[dfm['frame'] == int(frame)]
                    metric_data = {r: float(v) for r, v in zip(dff['residue'], dff[metric])}
                except Exception:
                    metric_data = {}

            edges_path = _pen_edges_path(energy_key, cov_flag, float(cutoff), int(frame))
            edges = _load_edges_for_frame(edges_path)
            nodes_in_edges = set()
            for e in edges:
                if e.get('source'):
                    nodes_in_edges.add(e['source'])
                if e.get('target'):
                    nodes_in_edges.add(e['target'])

            nodes_for_frame = sorted(nodes_in_edges) if nodes_in_edges else list(first_res_list)

            print(f"[3D Network] Loaded {len(nodes_for_frame)} nodes, {len(edges)} edges", flush=True)
            print(f"[3D Network] trajectory_coords has {len(trajectory_coords)} frames", flush=True)
            if frame in trajectory_coords:
                print(f"[3D Network] Frame {frame} has coords for {len(trajectory_coords[frame])} residues", flush=True)
            else:
                print(f"[3D Network] WARNING: No coordinates for frame {frame}", flush=True)
            
            # Prepare nodes data
            nodes = []
            nodes_with_coords = 0
            
            # Get metric values and calculate scaling
            metric_values = [metric_data.get(node, 0.0) for node in nodes_for_frame]
            max_metric = max(metric_values) if metric_values else 1.0
            
            # Use aggressive scaling to maximize visual differences
            if metric_values:
                print(f"[3D Network] Metric range: [{min(metric_values):.4f}, {max_metric:.4f}]", flush=True)
            else:
                print(f"[3D Network] Metric range: [0.0000, {max_metric:.4f}]", flush=True)
            
            for node in nodes_for_frame:
                raw_value = metric_data.get(node, 0.0)
                
                # Aggressive scaling strategies to emphasize differences
                if metric == 'degree':
                    # Degree: linear scaling with large multiplier
                    # Even small degree differences become very visible
                    node_size = 0.2 + raw_value * 0.5
                elif metric == 'betweenness':
                    # Betweenness: square root scaling with huge multiplier
                    # Central nodes (high betweenness) will be dramatically larger
                    node_size = 0.2 + (raw_value ** 0.5) * 50.0
                else:  # closeness
                    # Closeness: power scaling with large multiplier
                    # More central nodes have much higher closeness
                    node_size = 0.2 + (raw_value ** 1.5) * 12.0
                
                node_data = {
                    'id': node,
                    'name': node,
                    'val': node_size,
                    'metricValue': raw_value,  # Store actual metric value for tooltip
                    'isSelected': node in selected_set  # Mark if this node is in selected residues
                }
                
                # Add coordinates if available
                if frame in trajectory_coords and node in trajectory_coords[frame]:
                    coords = trajectory_coords[frame][node]
                    node_data['x'] = coords[0]
                    node_data['y'] = coords[1]
                    node_data['z'] = coords[2]
                    nodes_with_coords += 1
                
                nodes.append(node_data)
            
            print(f"[3D Network] {nodes_with_coords}/{len(nodes)} nodes have coordinates", flush=True)
            
            # Calculate coordinate ranges for debugging
            if nodes_with_coords > 0:
                x_coords = [n['x'] for n in nodes if 'x' in n]
                y_coords = [n['y'] for n in nodes if 'y' in n]
                z_coords = [n['z'] for n in nodes if 'z' in n]
                print(f"[3D Network] Coordinate ranges: X=[{min(x_coords):.1f}, {max(x_coords):.1f}], Y=[{min(y_coords):.1f}, {max(y_coords):.1f}], Z=[{min(z_coords):.1f}, {max(z_coords):.1f}]", flush=True)
            
            # Process selected paths for highlighting
            path_edges = set()  # Set of (source, target) tuples in selected paths
            path_colors = {}    # Maps edge tuple to color
            
            # Define dark colors for path highlighting (black as default, then distinct colors)
            highlight_colors = [
                'rgba(0, 0, 0, 0.95)',        # Black (first path)
                'rgba(139, 0, 0, 0.95)',      # Dark red
                'rgba(0, 100, 0, 0.95)',      # Dark green
                'rgba(0, 0, 139, 0.95)',      # Dark blue
                'rgba(128, 0, 128, 0.95)',    # Purple
                'rgba(139, 69, 19, 0.95)',    # Saddle brown
                'rgba(0, 139, 139, 0.95)',    # Dark cyan
                'rgba(184, 134, 11, 0.95)',   # Dark goldenrod
            ]
            
            if selected_path_rows and path_table_data:
                print(f"[3D Network] Highlighting {len(selected_path_rows)} selected path(s)", flush=True)
                for idx, row_idx in enumerate(selected_path_rows):
                    if row_idx < len(path_table_data):
                        path_str = path_table_data[row_idx]['path']
                        # Parse the path string (format: "RES1 --> RES2 --> RES3")
                        residues = [r.strip() for r in path_str.split('-->')]
                        
                        # Get color for this path
                        color = highlight_colors[idx % len(highlight_colors)]
                        
                        # Mark edges in this path
                        for i in range(len(residues) - 1):
                            edge = (residues[i], residues[i+1])
                            edge_rev = (residues[i+1], residues[i])  # Check both directions
                            path_edges.add(edge)
                            path_edges.add(edge_rev)
                            # Last selected path wins if multiple paths share an edge
                            path_colors[edge] = color
                            path_colors[edge_rev] = color
                        
                        print(f"[3D Network]   Path {idx+1}: {len(residues)} nodes, color {color}", flush=True)
            
            # Prepare edges data with path highlighting
            links = []
            for e in edges:
                source = e.get('source')
                target = e.get('target')
                edge = (source, target)
                is_in_path = edge in path_edges
                
                link_data = {
                    'source': source,
                    'target': target,
                    'value': e.get('weight', 1.0),
                    'isPathEdge': is_in_path
                }
                
                # Add color if this edge is in a selected path
                if is_in_path and edge in path_colors:
                    link_data['color'] = path_colors[edge]
                
                links.append(link_data)
            
            print(f"[3D Network] Created {len(links)} links", flush=True)
            
            # Convert nodes and links to JSON strings for embedding
            import json
            nodes_json = json.dumps(nodes)
            links_json = json.dumps(links)
            
            print(f"[3D Network] JSON data prepared, creating HTML...", flush=True)
            
            # Create the 3D force graph HTML with embedded JavaScript
            graph_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ margin: 0; padding: 0; overflow: hidden; }}
                    #3d-graph {{ width: 100%; height: 100vh; }}
                </style>
                <script src="https://unpkg.com/three@0.159.0/build/three.min.js"></script>
                <script src="https://unpkg.com/three-spritetext@1.8.2/dist/three-spritetext.min.js"></script>
                <script src="https://unpkg.com/3d-force-graph@1.73.3/dist/3d-force-graph.min.js"></script>
            </head>
            <body>
                <div id="3d-graph"></div>
                <script>
                    console.log('Initializing 3D force graph...');
                    
                    const graphData = {{
                        nodes: {nodes_json},
                        links: {links_json}
                    }};
                    
                    console.log('Graph data loaded:', graphData.nodes.length, 'nodes,', graphData.links.length, 'links');
                    
                    // Check what's available in global scope
                    console.log('ForceGraph3D available:', typeof ForceGraph3D);
                    console.log('SpriteText available:', typeof SpriteText);
                    console.log('THREE available:', typeof THREE);
                    
                    // Log first few nodes to verify data
                    console.log('First 3 nodes:', graphData.nodes.slice(0, 3));
                    console.log('First 3 links:', graphData.links.slice(0, 3));
                    
                    // Get the container element
                    const container = document.getElementById('3d-graph');
                    console.log('Container dimensions:', container.offsetWidth, 'x', container.offsetHeight);
                    
                    // Use new constructor syntax - start simple without custom node objects
                    try {{
                    console.log('Creating ForceGraph3D instance...');
                    const Graph = ForceGraph3D()(container);
                    console.log('ForceGraph3D instance created:', Graph);
                    
                    // Set graph data with optimized settings for protein structure
                    Graph.graphData(graphData)
                        .nodeLabel(node => {{
                            // Create tooltip with dark background for better readability
                            const metricName = '{metric}';
                            const metricValue = node.metricValue !== undefined ? node.metricValue.toFixed(3) : 'N/A';
                            const selectedStatus = node.isSelected ? '<br/><span style="color: #FFD700;">⭐ Selected</span>' : '';
                            return `<div style="background-color: rgba(40, 40, 40, 0.95); color: white; padding: 8px 12px; border-radius: 4px; font-family: monospace; font-size: 12px;">
                                <strong style="color: #7FD8BE;">${{node.name}}</strong>${{selectedStatus}}<br/>
                                ${{metricName}}: ${{metricValue}}
                            </div>`;
                        }})
                        .nodeColor(node => {{
                            // Highlight selected residues with gold color
                            if (node.isSelected) {{
                                return '#FFD700';  // Gold for selected residues
                            }} else {{
                                return '#7C9885';  // Sage green for all other nodes
                            }}
                        }})
                        .nodeRelSize(0.5)  // Base size multiplier
                        .nodeVal(node => node.val)  // Use metric-based size from node data
                        .linkDirectionalParticles(0)  // No moving particles
                        .linkWidth(0.3)  // Very thin, thread-like edges for all
                        .linkColor(link => {{
                            if (link.isPathEdge) {{
                                // Use specific colors for paths, default to black
                                return link.color || 'rgba(0, 0, 0, 0.95)';
                            }} else {{
                                // Light gray for regular edges
                                return 'rgba(220, 220, 220, 0.6)';
                            }}
                        }})
                        .linkOpacity(1.0)  // Full opacity for all edges
                        .backgroundColor('#f8fff8')
                        .width(container.offsetWidth)
                        .height(container.offsetHeight)
                        .enableNodeDrag(false);  // Disable node dragging
                    
                    console.log('Graph initialized (basic mode)');
                    
                    // If coordinates are provided, fix node positions and disable animation
                    const nodesWithCoords = graphData.nodes.filter(
                        n => n.x !== undefined && n.y !== undefined && n.z !== undefined
                    );
                    if (nodesWithCoords.length > 0) {{
                        console.log('Using provided coordinates - fixing positions');

                        // Calculate bounding box for proper camera positioning
                        const xs = nodesWithCoords.map(n => n.x);
                        const ys = nodesWithCoords.map(n => n.y);
                        const zs = nodesWithCoords.map(n => n.z);

                        const minX = Math.min(...xs), maxX = Math.max(...xs);
                        const minY = Math.min(...ys), maxY = Math.max(...ys);
                        const minZ = Math.min(...zs), maxZ = Math.max(...zs);
                        
                        const centerX = (minX + maxX) / 2;
                        const centerY = (minY + maxY) / 2;
                        const centerZ = (minZ + maxZ) / 2;
                        
                        const sizeX = maxX - minX;
                        const sizeY = maxY - minY;
                        const sizeZ = maxZ - minZ;
                        const maxSize = Math.max(sizeX, sizeY, sizeZ);
                        
                        console.log('Bounding box center:', centerX, centerY, centerZ);
                        console.log('Bounding box size:', sizeX, sizeY, sizeZ, 'max:', maxSize);
                        
                        // Stop the simulation immediately and disable all forces
                        Graph.numDimensions(3)
                            .cooldownTicks(0)
                            .d3AlphaDecay(1)
                            .d3VelocityDecay(1)
                            .warmupTicks(0);
                        
                        // Fix all node positions to prevent movement (only for nodes with coordinates)
                        graphData.nodes.forEach(node => {{
                            if (node.x !== undefined) {{
                                node.fx = node.x;
                                node.fy = node.y;
                                node.fz = node.z;
                            }}
                        }});
                        
                        // Update with fixed positions
                        Graph.graphData(graphData);
                        
                        // Position camera to look at center from appropriate distance
                        // Camera distance = 1.5 * maxSize to fit everything in view
                        const cameraDistance = maxSize * 1.5;
                        Graph.cameraPosition(
                            {{ x: centerX, y: centerY, z: centerZ + cameraDistance }},  // Camera position
                            {{ x: centerX, y: centerY, z: centerZ }},  // Look at center
                            0  // No animation
                        );
                        
                        console.log('Camera positioned at distance:', cameraDistance);
                    }} else {{
                        console.log('No coordinates provided, using force-directed layout');
                    }}
                    }} catch (e) {{
                        container.innerHTML = `<div style="padding:20px;color:red;font-family:monospace;">
                            <b>3D Network Error:</b> ${{e.message}}<br/>
                            <small>Check browser console for details.</small>
                        </div>`;
                        console.error('3D Network initialization failed:', e);
                    }}
                </script>
            </body>
            </html>
            """
            
            print(f"[3D Network] Returning HTML iframe", flush=True)
            
            return html.Iframe(
                srcDoc=graph_html,
                style={
                    'width': '100%', 
                    'height': '100%',
                    'border': 'none',
                    'display': 'block'
                }
            )
        
        except Exception as e:
            print(f"[3D Network] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Return error message
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head><style>body {{ font-family: Arial; padding: 20px; }}</style></head>
            <body>
                <h2>Error generating 3D network visualization</h2>
                <p><b>Error:</b> {str(e)}</p>
                <p>Check the dashboard logs for more details.</p>
            </body>
            </html>
            """
            return html.Iframe(
                srcDoc=error_html,
                style={'width': '100%', 'height': '100%', 'border': 'none'}
            )


    # Shortest Path Analysis Callback
    @app.callback(
        [Output('shortest_paths_table', 'data'),
         Output('path_status_message', 'children'),
         Output('path_status_message', 'style')],
        [Input('find_paths_btn', 'n_clicks'),
         Input('frame_slider', 'value'),
         Input('network_energy_type', 'value')],
        [State('source_residue_dropdown', 'value'),
         State('target_residue_dropdown', 'value'),
         State('include_covalent_edges', 'value'),
         State('energy_cutoff', 'value')],
        prevent_initial_call=True
    )
    def find_shortest_paths(n_clicks, frame, energy_type, source, target, include_cov, cutoff):
        """Find all shortest paths between source and target residues."""
        # Get callback context
        callback_ctx = ctx
        
        # Check which input triggered the callback
        if not callback_ctx.triggered:
            return [], "", {'display': 'none'}
        
        trigger_id = callback_ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Only compute if button was clicked or frame changed (and we have previous results)
        if trigger_id == 'frame_slider' and n_clicks == 0:
            return [], "", {'display': 'none'}
        
        # Validate inputs
        if not source or not target:
            return [], "⚠️ Please select both source and target residues", {
                'backgroundColor': '#FFF3CD',
                'color': '#856404',
                'border': '1px solid #FFE69C',
                'marginBottom': '15px',
                'padding': '10px',
                'borderRadius': '5px',
                'fontWeight': 'bold'
            }
        
        if source == target:
            return [], "⚠️ Source and target must be different residues", {
                'backgroundColor': '#FFF3CD',
                'color': '#856404',
                'border': '1px solid #FFE69C',
                'marginBottom': '15px',
                'padding': '10px',
                'borderRadius': '5px',
                'fontWeight': 'bold'
            }
        
        try:
            if frame is None:
                frame = frame_min
            if cutoff is None:
                cutoff = 1.0
            if include_cov is None:
                include_cov = ['include']

            energy_key = _pen_energy_key(energy_type)
            cov_flag = _pen_cov_flag(include_cov)
            paths_path = _pen_paths_path(energy_key, cov_flag, float(cutoff), int(frame))
            edges_path = _pen_edges_path(energy_key, cov_flag, float(cutoff), int(frame))
            path_data = _load_shortest_paths_for_pair(paths_path, edges_path, source, target)

            if not path_data:
                return [], f"⚠️ No shortest paths found for {source} → {target} (Frame {frame})", {
                    'backgroundColor': '#FFF3CD',
                    'color': '#856404',
                    'border': '1px solid #FFE69C',
                    'marginBottom': '15px',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'fontWeight': 'bold'
                }

            success_msg = f"✅ Loaded {len(path_data)} shortest path(s) between {source} and {target} (Frame {frame})"
            success_style = {
                'backgroundColor': '#D4EDDA',
                'color': '#155724',
                'border': '1px solid #C3E6CB',
                'marginBottom': '15px',
                'padding': '10px',
                'borderRadius': '5px',
                'fontWeight': 'bold'
            }
            return path_data, success_msg, success_style

        except Exception as e:
            print(f"[Shortest Path] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return [], f"❌ Error finding paths: {str(e)}", {
                'backgroundColor': '#F8D7DA',
                'color': '#721C24',
                'border': '1px solid #F5C6CB',
                'marginBottom': '15px',
                'padding': '10px',
                'borderRadius': '5px',
                'fontWeight': 'bold'
            }

    # Get port from environment or use default
    port = int(os.getenv('DASHBOARD_PORT', '8060'))

    print(f"📊 Data: {data_dir} | Frames: {frame_min}-{frame_max} | Residues: {len(first_res_list)}", flush=True)
    print(f"🌐 Dashboard: http://0.0.0.0:{port}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("✓ Initialization complete!", flush=True)
    print("="*60, flush=True)
    print("\n🚀 Starting dashboard server...", flush=True)
    print(f"   Open your browser to: http://localhost:{port}", flush=True)
    print("   Press Ctrl+C to stop the server\n", flush=True)

    app.run(debug=False, host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
