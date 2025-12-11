import os
import re
import sys
import argparse
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import dash_molstar
from dash_molstar.utils import molstar_helper
from dash_molstar.utils.representations import Representation
import networkx as nx
import numpy as np
from prody import parsePDB
from tqdm import tqdm

# Import the sophisticated network construction function from the main workflow
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from grinn_workflow import getRibeiroOrtizNetwork

# Global cache variables for network data
last_network_params = None
last_network_data = None

# Global cache for pre-computed network metrics
# Structure: {(include_cov, cutoff): {'degree': {frame: {res: val}}, 'betweenness': {...}, 'closeness': {...}}}
precomputed_metrics_cache = None
default_network_params = (['include'], 1.0)  # Default: include covalent, cutoff=1.0

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='gRINN Dashboard - Interactive visualization of protein interaction energies')
    parser.add_argument('results_folder', 
                       help='Path to the results folder containing gRINN output files, or "test" to use test data')
    return parser.parse_args()

def setup_data_paths(results_folder):
    """Setup data paths based on the results folder argument"""
    if results_folder == "test":
        # Use hardcoded test data directory
        data_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'prot_lig_1')
    else:
        # Use provided results folder
        data_dir = os.path.abspath(results_folder)
    
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
    
    return data_dir, pdb_path, total_csv, vdw_csv, elec_csv, avg_csv, traj_xtc

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
    data_dir, pdb_path, total_csv, vdw_csv, elec_csv, avg_csv, traj_xtc = setup_data_paths(args.results_folder)
    
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
        cartoon = Representation(type='cartoon', color='uniform')
        cartoon.set_color_params({'value': 0xD3D3D3})
        chainA = molstar_helper.get_targets(chain='A')
        component = molstar_helper.create_component(label='Protein', targets=[chainA], representation=cartoon)
        topo = molstar_helper.parse_molecule(pdb_path, component=component)
        
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
            
            print(f"      ✓ Loaded static center of mass coordinates for {len(static_coords)} residues", flush=True)
    except Exception as e:
        print(f"      ⚠ Warning: Could not load trajectory coordinates: {e}", flush=True)
        print(f"      Error details: {type(e).__name__}", flush=True)
        import traceback
        traceback.print_exc()
        print("      3D network visualization will use default positions", flush=True)
        trajectory_coords = {}

    def precompute_network_metrics(include_cov, cutoff):
        """
        Pre-compute network metrics for all frames with given network parameters.
        This dramatically speeds up the dashboard by computing once at startup.
        """
        global precomputed_metrics_cache
        
        print(f"\n[5/5] Pre-computing network metrics (include_cov={include_cov}, cutoff={cutoff})...", flush=True)
        
        cache_key = (str(include_cov), cutoff)
        metrics = {
            'degree': {},
            'betweenness': {},
            'closeness': {}
        }
        
        for frame in tqdm(range(frame_min, frame_max + 1), desc="      Computing metrics", ncols=70):
            G = build_graph(frame, include_cov, cutoff)
            
            if G.number_of_nodes() == 0:
                # Empty graph
                metrics['degree'][frame] = {}
                metrics['betweenness'][frame] = {}
                metrics['closeness'][frame] = {}
            else:
                # Degree centrality
                metrics['degree'][frame] = dict(G.degree())
                
                # Betweenness and closeness
                if G.number_of_edges() == 0:
                    metrics['betweenness'][frame] = {node: 0.0 for node in G.nodes()}
                    metrics['closeness'][frame] = {node: 0.0 for node in G.nodes()}
                else:
                    try:
                        num_nodes = G.number_of_nodes()
                        if num_nodes > 100:
                            # Use approximate betweenness for large graphs
                            k = min(50, num_nodes // 2)
                            metrics['betweenness'][frame] = nx.betweenness_centrality(G, k=k, normalized=True)
                        else:
                            metrics['betweenness'][frame] = nx.betweenness_centrality(G)
                        
                        metrics['closeness'][frame] = nx.closeness_centrality(G)
                    except Exception as e:
                        print(f"\n      Warning: Error computing centrality for frame {frame}: {e}", flush=True)
                        metrics['betweenness'][frame] = {node: 0.0 for node in G.nodes()}
                        metrics['closeness'][frame] = {node: 0.0 for node in G.nodes()}
        
        precomputed_metrics_cache = {cache_key: metrics}
        print(f"      ✓ Pre-computed metrics for {frame_max - frame_min + 1} frames", flush=True)
        print("="*60, flush=True)
        print("✓ Dashboard initialization complete!", flush=True)
        print("="*60 + "\n", flush=True)

    # Build graph helper - optimized for speed
    def build_graph(frame, include_cov, cutoff):
        """
        Build a protein energy network using Ribeiro-Ortiz methodology.
        Matches getRibeiroOrtizNetwork from grinn_workflow.py
        """
        
        # AGGRESSIVE OPTIMIZATION: Use pre-indexed data instead of DataFrame filtering
        frame_str = str(frame)
        G = nx.Graph()
        
        # Add all residues as nodes
        G.add_nodes_from(first_res_list)
        
        # Get pre-indexed data for this frame (O(1) lookup)
        if frame_str in frame_indexed_data['Total']:
            pairs = frame_indexed_data['Total'][frame_str]['pairs']
            energies = frame_indexed_data['Total'][frame_str]['energies']
            
            # Following Ribeiro-Ortiz methodology:
            # 1. Only use negative (attractive) energies
            # 2. Normalize by maximum absolute value
            # 3. Clip to [0, 0.99]
            # 4. Set weight=normalized_energy, distance=1-weight
            
            # First pass: collect all negative energies for normalization
            neg_energies = []
            valid_pairs = []
            for pair, energy in zip(pairs, energies):
                if energy < 0 and abs(energy) >= cutoff:
                    neg_energies.append(abs(energy))
                    valid_pairs.append((pair, energy))
            
            # Normalize by maximum absolute value
            if neg_energies:
                max_abs = max(neg_energies)
                edges = []
                for pair, energy in valid_pairs:
                    r1, r2 = pair.split('-', 1)
                    # Normalize: weight = abs(energy) / max_abs, clipped to [0, 0.99]
                    normalized_weight = min(abs(energy) / max_abs, 0.99)
                    # Distance = 1 - weight (as per Ribeiro-Ortiz)
                    distance = 1.0 - normalized_weight
                    edges.append((r1, r2, {'weight': normalized_weight, 'distance': distance}))
                
                if edges:
                    G.add_edges_from(edges)
        
        # Add covalent bonds if requested
        if include_cov and 'include' in str(include_cov):
            # PERFORMANCE OPTIMIZATION: Create edges in batch
            covalent_edges = []
            for i in range(len(first_res_list) - 1):
                if not G.has_edge(first_res_list[i], first_res_list[i+1]):
                    # Covalent bonds get neutral weight (0.0) and distance (1.0)
                    covalent_edges.append((first_res_list[i], first_res_list[i+1], {'weight': 0.0, 'distance': 1.0}))
            if covalent_edges:
                G.add_edges_from(covalent_edges)
        
        return G

    def get_cached_network_data(frame, include_cov, cutoff):
        """Get network data with caching and fast approximate centrality for large graphs."""
        global last_network_params, last_network_data
        
        # Create cache key
        cache_key = (frame, str(include_cov), cutoff)
        
        # Check if we have cached data for these exact parameters
        if last_network_params == cache_key and last_network_data is not None:
            return last_network_data['deg'], last_network_data['btw'], last_network_data['clo']
        
        # Compute new network data
        try:
            G = build_graph(frame, include_cov, cutoff)
            
            # Check if graph has nodes
            if G.number_of_nodes() == 0:
                print(f"Warning: Empty graph for frame {frame}")
                deg = {}
                btw = {}
                clo = {}
            else:
                deg = dict(G.degree())
                
                # Betweenness and closeness require connected components
                if G.number_of_edges() == 0:
                    # No edges - all centralities are 0
                    btw = {node: 0.0 for node in G.nodes()}
                    clo = {node: 0.0 for node in G.nodes()}
                else:
                    try:
                        # AGGRESSIVE OPTIMIZATION: Use approximate algorithms for large graphs
                        num_nodes = G.number_of_nodes()
                        if num_nodes > 100:
                            # Use approximate betweenness with sampling for faster computation
                            k = min(50, num_nodes // 2)  # Sample subset of nodes
                            btw = nx.betweenness_centrality(G, k=k, normalized=True)
                        else:
                            btw = nx.betweenness_centrality(G)
                        
                        # Closeness is fast, compute normally
                        clo = nx.closeness_centrality(G)
                    except Exception as e:
                        print(f"Error computing centrality: {e}")
                        btw = {node: 0.0 for node in G.nodes()}
                        clo = {node: 0.0 for node in G.nodes()}
            
            # Cache the results
            last_network_params = cache_key
            last_network_data = {'deg': deg, 'btw': btw, 'clo': clo}
            
            return deg, btw, clo
            
        except Exception as e:
            print(f"Error in get_cached_network_data: {e}")
            # Return empty dicts to avoid crashing
            return {}, {}, {}

    # Pre-compute network metrics with default parameters for fast dashboard startup
    precompute_network_metrics(default_network_params[0], default_network_params[1])

    # App layout
    # AGGRESSIVE OPTIMIZATION: Configure app for maximum performance
    app = Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,  # Faster callback registration
        compress=True  # Enable compression for faster data transfer
    )

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

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div(style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'space-between',
                    'margin': '10px 0 15px 0'
                }, children=[
                    html.H1("gRINN Workflow Results",
                            className="main-title",
                            style={
                                'color': soft_palette['primary'],
                                'fontFamily': 'Roboto, sans-serif',
                                'fontWeight': '700',
                                'fontSize': '2rem',
                                'margin': '0',
                                'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                                'letterSpacing': '1px',
                                'flex': '0 0 auto'
                            }),
                    html.P(f"📁 {data_dir}",
                           style={
                               'textAlign': 'center',
                               'color': soft_palette['text'],
                               'fontFamily': 'Roboto, sans-serif',
                               'fontSize': '0.9rem',
                               'margin': '0',
                               'flex': '1',
                               'paddingLeft': '20px',
                               'paddingRight': '20px'
                           })
                ])
            ], width=12)
        ]),
        dbc.Row([
            # Left Panel: Tabs
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Tabs(id='main-tabs', value='tab-pairwise', 
                                 style={
                                     'fontFamily': 'Roboto, sans-serif',
                                     'fontWeight': '500',
                                     'height': '35px'
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
                                                    'height': 'calc(100vh - 250px)', 
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
                                                    'height': 'calc(100vh - 250px)', 
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
                                                style={'height': 'calc(100vh - 250px)'},
                                                config={'displayModeBar': False}
                                            )
                                        ], style={'padding': '5px'})
                                    ])
                                ], width=2),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            dcc.Graph(id='total_energy_graph', style={'height': 'calc((100vh - 200px) / 3)'}),
                                            dcc.Graph(id='vdw_energy_graph', style={'height': 'calc((100vh - 200px) / 3)'}),
                                            dcc.Graph(id='elec_energy_graph', style={'height': 'calc((100vh - 200px) / 3)'})
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
                                            dcc.Graph(id='matrix_heatmap', style={'height': 'calc(100vh - 240px)'})
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
                            'padding': '15px',
                            'margin': '10px',
                            'border': f'2px solid {soft_palette["accent"]}'
                        }, children=[
                            # General Network Settings (moved from Network Metrics tab)
                            html.Div(style={
                                'display': 'flex', 
                                'alignItems': 'center', 
                                'paddingBottom': '15px',
                                'background': soft_palette["light_blue"],
                                'borderRadius': '10px',
                                'padding': '15px',
                                'marginBottom': '15px',
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
                                html.Label("⚡ Edge addition energy cutoff (kcal/mol): ", style={
                                    'color': 'white',
                                    'fontFamily': 'Roboto, sans-serif',
                                    'fontWeight': '500',
                                    'fontSize': '12px'
                                }),
                                dcc.Input(
                                    id='energy_cutoff', 
                                    type='number', 
                                    value=1.0, 
                                    step=0.1, 
                                    style={
                                        'width': '80px', 
                                        'marginRight': '20px',
                                        'borderRadius': '5px',
                                        'border': f'2px solid {soft_palette["accent"]}',
                                        'padding': '5px'
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
                                     style={'height': '35px'},
                                     children=[
                                # Network Metrics Sub-tab
                                dcc.Tab(label='📊 Network Metrics', value='tab-network-metrics', children=[
                                    html.Div(style={'padding': '10px'}, children=[
                                        # Metric selector
                                        html.Div(style={'marginBottom': '10px'}, children=[
                                            html.Label("Select Metric:", style={
                                                'fontWeight': 'bold',
                                                'color': soft_palette['primary'],
                                                'marginBottom': '5px',
                                                'display': 'block',
                                                'fontSize': '14px'
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
                                                    'fontSize': '14px',
                                                    'display': 'flex',
                                                    'gap': '30px'
                                                },
                                                labelStyle={
                                                    'display': 'flex',
                                                    'alignItems': 'center',
                                                    'cursor': 'pointer'
                                                }
                                            )
                                        ]),
                                        
                                        # Residue filter selector
                                        html.Div(style={'marginBottom': '10px'}, children=[
                                            html.Label("Filter Residues (leave empty for all):", style={
                                                'fontWeight': 'bold',
                                                'color': soft_palette['primary'],
                                                'marginBottom': '5px',
                                                'display': 'block',
                                                'fontSize': '14px'
                                            }),
                                            html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center'}, children=[
                                                dcc.Dropdown(
                                                    id='selected_residues_dropdown',
                                                    options=[{'label': res, 'value': res} for res in first_res_list],
                                                    value=[],  # Empty = show all
                                                    multi=True,
                                                    placeholder="Search and select residues to analyze (multi-select)",
                                                    searchable=True,
                                                    style={'flex': '1'}
                                                ),
                                                html.Button('Reset to All', 
                                                    id='reset_residues_btn',
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': soft_palette['accent'],
                                                        'color': 'white',
                                                        'border': 'none',
                                                        'padding': '8px 16px',
                                                        'borderRadius': '8px',
                                                        'cursor': 'pointer',
                                                        'fontWeight': 'bold',
                                                        'fontSize': '13px',
                                                        'whiteSpace': 'nowrap'
                                                    })
                                            ])
                                        ]),
                                        
                                        # Heatmap
                                        html.Div(children=[
                                            dcc.Graph(
                                                id='network_metrics_heatmap',
                                                config={'displayModeBar': True, 'displaylogo': False},
                                                style={'width': '100%', 'height': '70vh'}
                                            )
                                        ])
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
                                                    'display': 'block'
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
                                                    'display': 'block'
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
                                                        'fontSize': '14px',
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
                                                'display': 'block'
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
                                                    'fontSize': '13px',
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
        ], style={'padding': '20px', 'backgroundColor': 'rgba(255,255,255,0.95)', 'borderRadius': '15px', 'border': f'3px solid {soft_palette["border"]}', 'boxShadow': '0 8px 32px rgba(0,0,0,0.1)'})
        ], width=8),
        # Right Panel: 3D Viewer with tabs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H3("🧬 3D Viewer", className="text-center mb-0", style={'color': soft_palette['primary']})
                ]),
                dbc.CardBody([
                    # Tabbed selector for 3D views
                    dcc.Tabs(id='viewer-tabs', value='tab-structure-viewer', children=[
                        # 3D Structure Viewer Tab
                        dcc.Tab(label='🧬 Structure Viewer', value='tab-structure-viewer', children=[
                            dbc.Card([
                                dbc.CardBody([
                                    dash_molstar.MolstarViewer(
                                        id='viewer', 
                                        data=initial_traj, 
                                        layout={'modelIndex': frame_min}, 
                                        style={'width': '100%','height':'65vh'}
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
                                    'height': '65vh',
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
                            html.Label("🎬 Frame:", className="text-white mb-0", style={'fontSize': '16px'})
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
        ], width=4)
    ], className="mt-3")
    ], fluid=True, style={'background': soft_palette["background"], 'minHeight': '100vh', 'padding': '20px'})

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
    # Network Metrics Heatmap - responsive to metric selector, frame slider, network settings, and residue selection
    @app.callback(
        Output('network_metrics_heatmap', 'figure'),
        [Input('metric_selector', 'value'),
         Input('frame_slider', 'value'),
         Input('update_network_btn', 'n_clicks'),
         Input('selected_residues_dropdown', 'value')],
        [State('include_covalent_edges', 'value'),
         State('energy_cutoff', 'value')],
        prevent_initial_call=False
    )
    def update_network_metrics_heatmap(metric, current_frame, n_clicks, selected_residues, include_cov, cutoff):
        """Generate heatmap showing network metrics across all frames and residues."""
        global precomputed_metrics_cache
        
        print(f"[CALLBACK TRIGGERED] metric={metric}, current_frame={current_frame}, n_clicks={n_clicks}, selected_residues={len(selected_residues) if selected_residues else 0}", flush=True)
        
        # Use default values if not provided
        if current_frame is None:
            current_frame = frame_min
        if cutoff is None:
            cutoff = 1.0
        if include_cov is None:
            include_cov = ['include']
        if metric is None:
            metric = 'degree'
        if selected_residues is None or len(selected_residues) == 0:
            selected_residues = first_res_list  # Show all if none selected
        
        # Check if we need to recompute (network settings changed)
        cache_key = (str(include_cov), cutoff)
        
        # Check if Update Network button was clicked (triggers recomputation)
        triggered_id = ctx.triggered_id if ctx.triggered_id else None
        
        if (precomputed_metrics_cache is None or 
            cache_key not in precomputed_metrics_cache or 
            triggered_id == 'update_network_btn'):
            # Recompute metrics with new network settings
            print(f"[Network Metrics Heatmap] Recomputing metrics with new network settings (include_cov={include_cov}, cutoff={cutoff})", flush=True)
            precompute_network_metrics(include_cov, cutoff)
        
        # Use pre-computed data
        print(f"[Network Metrics Heatmap] Using pre-computed {metric} data for {len(selected_residues)} residues", flush=True)
        metrics_data = precomputed_metrics_cache[cache_key][metric]
        
        # Build heatmap data from pre-computed metrics, filtered by selected residues
        all_frames_data = []
        for frame in range(frame_min, frame_max + 1):
            data = metrics_data.get(frame, {})
            frame_values = [data.get(res, 0.0) for res in selected_residues]
            all_frames_data.append(frame_values)
        
        # Convert to numpy array (residues x frames)
        heatmap_data = np.array(all_frames_data).T  # Transpose to get residues as rows
        heatmap_data = np.nan_to_num(heatmap_data, nan=0.0)
        
        print(f"[Network Metrics Heatmap] Data shape: {heatmap_data.shape}", flush=True)
        print(f"[Network Metrics Heatmap] Data range: [{np.min(heatmap_data):.4f}, {np.max(heatmap_data):.4f}]", flush=True)
        print(f"[Network Metrics Heatmap] Non-zero values: {np.count_nonzero(heatmap_data)}", flush=True)
        
        # Create frame labels
        frame_labels = [str(f) for f in range(frame_min, frame_max + 1)]
        
        # Residue labels (filtered)
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
            y=selected_residues,
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
            y1=len(selected_residues) - 0.5,
            line=dict(color='red', width=2),
            fillcolor='rgba(0,0,0,0)'
        )
        
        fig.update_layout(
            title=dict(
                text=f'{metric_titles[metric]} Across Frames',
                font=dict(size=16, color=soft_palette['primary'], family='Roboto, sans-serif')
            ),
            xaxis=dict(
                title='Frame',
                tickfont=dict(size=10),
                side='bottom',
                showgrid=False
            ),
            yaxis=dict(
                title='Residue',
                tickfont=dict(size=8),
                showgrid=False,
                autorange=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='rgba(255,255,255,0.9)',
            margin=dict(l=100, r=100, t=60, b=60),
            font=dict(family='Roboto, sans-serif', color='#4A5A4A')
        )
        
        return fig


    # Shortest Paths
    @app.callback(
        Output('paths_table','data'),
        Input('find_paths_btn','n_clicks'),
        State('frame_slider','value'),
        State('include_covalent_edges','value'),
        State('energy_cutoff','value'),
        State('source_residue','value'),
        State('target_residue','value'),
        prevent_initial_call=True
    )
    def find_paths(n_clicks, frame, include_cov, cutoff, source, target):
        if n_clicks is None or n_clicks < 1 or not source or not target or source == target:
            return []
        
        # Use default values if not provided
        if frame is None:
            frame = frame_min
        if cutoff is None:
            cutoff = 1.0
        if include_cov is None:
            include_cov = ['include']
            
        G = build_graph(frame, include_cov, cutoff)
        
        try:
            paths_gen = nx.shortest_simple_paths(G, source, target, weight='weight')
            out = []
            for i, path in enumerate(paths_gen):
                if i >= 10: 
                    break
                length = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                out.append({'Path': '-'.join(path), 'Length': round(length, 6)})
            return out
        except nx.NetworkXNoPath:
            print(f"No path found between {source} and {target}")
            return []
        except nx.NodeNotFound as e:
            print(f"Node not found in graph: {e}")
            return []
        except Exception as e:
            print(f"Error finding paths: {e}")
            return []

    # 3D Network Visualization callback
    @app.callback(
        Output('network-3d-container', 'children'),
        [Input('frame_slider', 'value'),
         Input('update_network_btn', 'n_clicks'),
         Input('shortest_paths_table', 'selected_rows'),
         Input('metric_selector', 'value'),
         Input('selected_residues_dropdown', 'value')],
        [State('include_covalent_edges', 'value'),
         State('energy_cutoff', 'value'),
         State('shortest_paths_table', 'data')],
        prevent_initial_call=False
    )
    def update_3d_network(frame, n_clicks, selected_path_rows, metric, selected_residues, include_cov, cutoff, path_table_data):
        """Update 3D force graph network visualization based on frame and network parameters."""
        global precomputed_metrics_cache
        
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
            
            # Get metric data for node sizing
            cache_key = (str(include_cov), cutoff)
            if precomputed_metrics_cache and cache_key in precomputed_metrics_cache:
                metric_data = precomputed_metrics_cache[cache_key][metric].get(frame, {})
            else:
                # Fallback: compute on the fly
                deg, btw, clo = get_cached_network_data(frame, include_cov, cutoff)
                metric_data = {'degree': deg, 'betweenness': btw, 'closeness': clo}[metric]
            
            print(f"[3D Network] Got metric data for {len(metric_data)} nodes", flush=True)
            
            # Build network graph for current frame
            G = build_graph(frame, include_cov, cutoff)
            
            print(f"[3D Network] Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
            print(f"[3D Network] trajectory_coords has {len(trajectory_coords)} frames", flush=True)
            if frame in trajectory_coords:
                print(f"[3D Network] Frame {frame} has coords for {len(trajectory_coords[frame])} residues", flush=True)
            else:
                print(f"[3D Network] WARNING: No coordinates for frame {frame}", flush=True)
            
            # Prepare nodes data
            nodes = []
            nodes_with_coords = 0
            
            # Get metric values and calculate scaling
            metric_values = [metric_data.get(node, 0.0) for node in G.nodes()]
            max_metric = max(metric_values) if metric_values else 1.0
            
            # Use aggressive scaling to maximize visual differences
            print(f"[3D Network] Metric range: [{min(metric_values):.4f}, {max_metric:.4f}]", flush=True)
            
            for node in G.nodes():
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
            for source, target, data in G.edges(data=True):
                edge = (source, target)
                is_in_path = edge in path_edges
                
                link_data = {
                    'source': source,
                    'target': target,
                    'value': data.get('weight', 1.0),
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
                    if (graphData.nodes.length > 0 && graphData.nodes[0].x !== undefined) {{
                        console.log('Using provided coordinates - fixing positions');
                        
                        // Calculate bounding box for proper camera positioning
                        const xs = graphData.nodes.map(n => n.x);
                        const ys = graphData.nodes.map(n => n.y);
                        const zs = graphData.nodes.map(n => n.z);
                        
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
                        
                        // Fix all node positions to prevent movement
                        graphData.nodes.forEach(node => {{
                            node.fx = node.x;
                            node.fy = node.y;
                            node.fz = node.z;
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
                </script>
            </body>
            </html>
            """
            
            print(f"[3D Network] Returning HTML iframe", flush=True)
            
            return html.Iframe(
                srcDoc=graph_html,
                style={
                    'width': '100%', 
                    'height': '70vh',
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
         Input('frame_slider', 'value')],
        [State('source_residue_dropdown', 'value'),
         State('target_residue_dropdown', 'value'),
         State('include_covalent_edges', 'value'),
         State('energy_cutoff', 'value')],
        prevent_initial_call=True
    )
    def find_shortest_paths(n_clicks, frame, source, target, include_cov, cutoff):
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
            # Build graph with current parameters
            if frame is None:
                frame = frame_min
            if cutoff is None:
                cutoff = 1.0
            if include_cov is None:
                include_cov = ['include']
            
            G = build_graph(frame, include_cov, cutoff)
            
            # Check if both residues are in the graph
            if source not in G.nodes() or target not in G.nodes():
                return [], f"⚠️ One or both residues not found in network for frame {frame}", {
                    'backgroundColor': '#FFF3CD',
                    'color': '#856404',
                    'border': '1px solid #FFE69C',
                    'marginBottom': '15px',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'fontWeight': 'bold'
                }
            
            # Check if there's a path between source and target
            if not nx.has_path(G, source, target):
                return [], f"⚠️ No path exists between {source} and {target} in frame {frame}", {
                    'backgroundColor': '#FFF3CD',
                    'color': '#856404',
                    'border': '1px solid #FFE69C',
                    'marginBottom': '15px',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'fontWeight': 'bold'
                }
            
            # Find ALL shortest paths
            # The graph already has 'distance' attributes from Ribeiro-Ortiz methodology
            # where distance = 1 - normalized_weight
            # Lower distance = stronger interaction (higher weight)
            # NetworkX's all_shortest_paths will use 'distance' as the weight parameter
            
            # Verify all edges have distance attribute (they should from build_graph)
            for u, v, data in G.edges(data=True):
                if 'distance' not in data:
                    # Fallback: if somehow missing, use neutral distance
                    data['distance'] = 1.0
            
            # Find all shortest paths using distance as the weight
            all_paths = list(nx.all_shortest_paths(G, source, target, weight='distance'))

            
            if not all_paths:
                return [], f"⚠️ No paths found between {source} and {target}", {
                    'backgroundColor': '#FFF3CD',
                    'color': '#856404',
                    'border': '1px solid #FFE69C',
                    'marginBottom': '15px',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'fontWeight': 'bold'
                }
            
            # Calculate path lengths (sum of distances along the path)
            path_data = []
            for path in all_paths:
                # Calculate total distance along this path
                total_distance = 0.0
                total_weight = 0.0  # Sum of normalized weights for reference
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i+1])
                    if edge_data:
                        if 'distance' in edge_data:
                            total_distance += edge_data['distance']
                        if 'weight' in edge_data:
                            total_weight += edge_data['weight']
                
                path_data.append({
                    'path': ' --> '.join(path),
                    'length': f"{total_distance:.4f}",
                    'hops': len(path) - 1
                })
            
            # Sort by length (distance)
            path_data.sort(key=lambda x: float(x['length']))
            
            # Success message
            success_msg = f"✅ Found {len(all_paths)} shortest path(s) between {source} and {target} (Frame {frame})"
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


        print(f"🍀 gRINN Dashboard starting...", flush=True)
    print(f"📊 Data: {data_dir} | Frames: {frame_min}-{frame_max} | Residues: {len(first_res_list)}", flush=True)
    print(f"🌐 Dashboard: http://0.0.0.0:8050", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("✓ Initialization complete!", flush=True)
    print("="*60, flush=True)
    print("\n🚀 Starting dashboard server...", flush=True)
    print("   Open your browser to: http://localhost:8050", flush=True)
    print("   Press Ctrl+C to stop the server\n", flush=True)
    
    app.run(debug=False, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()
