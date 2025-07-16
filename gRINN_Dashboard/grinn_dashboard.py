import os
import re
import sys
import argparse
import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import dash_molstar
from dash_molstar.utils import molstar_helper
from dash_molstar.utils.representations import Representation
import networkx as nx
import numpy as np

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
        print(f"Using test data directory: {data_dir}")
    else:
        # Use provided results folder
        data_dir = os.path.abspath(results_folder)
        print(f"Using results directory: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist!")
        sys.exit(1)
    
    # Define expected file paths
    pdb_path = os.path.join(data_dir, 'system_dry.pdb')
    total_csv = os.path.join(data_dir, 'energies_intEnTotal.csv')
    vdw_csv = os.path.join(data_dir, 'energies_intEnVdW.csv')
    elec_csv = os.path.join(data_dir, 'energies_intEnElec.csv')
    traj_xtc = os.path.join(data_dir, 'traj_superposed.xtc')
    
    # Check if required files exist
    required_files = [pdb_path, total_csv, vdw_csv, elec_csv]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files in '{data_dir}':")
        for file in missing_files:
            print(f"  - {file}")
        print("\nRequired files for gRINN dashboard:")
        print("  - system_dry.pdb")
        print("  - energies_intEnTotal.csv")
        print("  - energies_intEnVdW.csv")
        print("  - energies_intEnElec.csv")
        print("  - traj_superposed.xtc (optional, for trajectory visualization)")
        sys.exit(1)
    
    # Check if trajectory file exists (optional)
    if not os.path.exists(traj_xtc):
        print(f"Warning: Trajectory file '{traj_xtc}' not found. Using static structure only.")
        traj_xtc = None
    
    return data_dir, pdb_path, total_csv, vdw_csv, elec_csv, traj_xtc

def main():
    """Main function to setup and run the dashboard"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup data paths
    data_dir, pdb_path, total_csv, vdw_csv, elec_csv, traj_xtc = setup_data_paths(args.results_folder)
    
    # Load and transform interaction energy data
    try:
        total_df = pd.read_csv(total_csv)
        vdw_df = pd.read_csv(vdw_csv)
        elec_df = pd.read_csv(elec_csv)
        print(f"Successfully loaded energy data from {data_dir}")
    except Exception as e:
        print(f"Error loading energy data: {e}")
        sys.exit(1)

    # Create a combined dataframe with all energy types
    energy_dfs = {
        'Total': total_df,
        'VdW': vdw_df,
        'Electrostatic': elec_df
    }

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
        long_df = (
            df
            .drop(columns=cols2drop + ['res1', 'res2'])
            .melt(id_vars=['Pair'], var_name='Frame', value_name='Energy')
        )
        long_df['Energy'] = pd.to_numeric(long_df['Energy'], errors='coerce')
        long_df = long_df[long_df['Energy'].notna()].copy()
        long_df['EnergyType'] = energy_type
        energy_long[energy_type] = long_df

    # Keep the original for compatibility
    total_long = energy_long['Total']

    # Determine frame range
    df_frames = pd.to_numeric(total_long['Frame'], errors='coerce').dropna().astype(int)
    frame_min, frame_max = int(df_frames.min()), int(df_frames.max())

    # Residue list - sort by residue number to maintain protein sequence order
    def sort_residues_by_sequence(residues):
        """Sort residues by their sequence number extracted from residue names like GLY290_A"""
        def extract_residue_number(res_name):
            try:
                # Extract number from residue name like 'GLY290_A'
                parts = res_name.split('_')
                if len(parts) >= 2:
                    # Get the number part from the first part (e.g., '290' from 'GLY290')
                    number = re.findall(r'\d+', parts[0])
                    if number:
                        return int(number[0])
                return 0
            except:
                return 0
        
        return sorted(residues, key=extract_residue_number)

    first_res_list = sort_residues_by_sequence(total_df['res1'].unique())

    # Molecular visualization setup
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
    else:
        # Use static structure only
        initial_traj = topo

    # Build graph helper
    def build_graph(frame, include_cov, cutoff):
        df_f = total_long[total_long['Frame'].astype(int) == frame]
        G = nx.Graph()
        for res in first_res_list:
            G.add_node(res)
        for _, row in df_f.iterrows():
            r1, r2 = row['Pair'].split('-')
            e = row['Energy']
            if abs(e) >= cutoff:
                G.add_edge(r1, r2, weight=abs(e))
        if 'include' in include_cov:
            for i in range(len(first_res_list) - 1):
                G.add_edge(first_res_list[i], first_res_list[i+1], weight=0.0)
        return G

    # App layout
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                html.H1("🍀 gRINN Workflow Results 🍀",
                        className="main-title",
                        style={
                            'textAlign': 'center',
                            'color': soft_palette['primary'],
                            'fontFamily': 'Roboto, sans-serif',
                            'fontWeight': '700',
                            'fontSize': '2.5rem',
                            'margin': '20px 0',
                            'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                            'letterSpacing': '1px'
                        }),
                html.P(f"📁 Data source: {data_dir}",
                       style={
                           'textAlign': 'center',
                           'color': soft_palette['text'],
                           'fontFamily': 'Roboto, sans-serif',
                           'fontSize': '1rem',
                           'margin': '0 0 20px 0'
                       })
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
                                     'fontWeight': '500'
                                 },
                                 colors={
                                     'border': soft_palette['border'],
                                     'primary': soft_palette['primary'],
                                     'background': soft_palette['surface']
                                 }, children=[
                    # Pairwise Energies Tab
                    dcc.Tab(label='🔗 Pairwise Energies', value='tab-pairwise', children=[
                        html.Div(id='pairwise-tab-content', children=[
                            html.H4("Select two residues to analyze their interaction energy", style={'textAlign': 'center', 'color': soft_palette['text']})
                        ])
                    ]),
                    # Interaction Energy Matrix Tab
                    dcc.Tab(label='🔥 Interaction Energy Matrix', value='tab-matrix', children=[
                        html.Div(id='matrix-tab-content', children=[
                            html.H4("Interaction Energy Matrix", style={'textAlign': 'center', 'color': soft_palette['text']})
                        ])
                    ]),
                    # Network Analysis Tab
                    dcc.Tab(label='🕸️ Network Analysis', value='tab-network', children=[
                        html.Div(id='network-tab-content', children=[
                            html.H4("Network Analysis", style={'textAlign': 'center', 'color': soft_palette['text']})
                        ])
                    ])
                ])
            ])
        ], style={'padding': '20px', 'backgroundColor': 'rgba(255,255,255,0.95)', 'borderRadius': '15px', 'border': f'3px solid {soft_palette["border"]}', 'boxShadow': '0 8px 32px rgba(0,0,0,0.1)'})
        ], width=8),
        # Right Panel: 3D Viewer
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H3("🧬 3D Molecular Viewer", className="text-center mb-0", style={'color': soft_palette['primary']})
                ]),
                dbc.CardBody([
                    dbc.Card([
                        dbc.CardBody([
                            dash_molstar.MolstarViewer(
                                id='viewer', 
                                data=initial_traj, 
                                layout={'modelIndex': frame_min}, 
                                style={'width': '100%','height':'65vh'}
                            )
                        ])
                    ], style={'border': f'3px solid {soft_palette["border"]}', 'borderRadius': '10px', 'backgroundColor': 'rgba(250,255,250,0.4)'}),
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

    # Basic callback for frame slider
    @app.callback(
        Output('viewer','frame'),
        Input('frame_slider','value')
    )
    def update_viewer_frame(selected_frame):
        return selected_frame

    print(f"Starting gRINN Dashboard...")
    print(f"Data directory: {data_dir}")
    print(f"Frame range: {frame_min} - {frame_max}")
    print(f"Number of residues: {len(first_res_list)}")
    print(f"Dashboard will be available at: http://0.0.0.0:8051")
    
    app.run(debug=False, host='0.0.0.0', port=8051)

if __name__ == '__main__':
    main()
