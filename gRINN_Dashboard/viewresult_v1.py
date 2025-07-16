import os
import re
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

# File paths
data_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'prot_lig_1')
pdb_path = os.path.join(data_dir, 'system_dry.pdb')
total_csv = os.path.join(data_dir, 'energies_intEnTotal.csv')
vdw_csv = os.path.join(data_dir, 'energies_intEnVdW.csv')
elec_csv = os.path.join(data_dir, 'energies_intEnElec.csv')
traj_xtc = os.path.join(data_dir, 'traj_superposed.xtc')

# Load and transform interaction energy data
total_df = pd.read_csv(total_csv)
vdw_df = pd.read_csv(vdw_csv)
elec_df = pd.read_csv(elec_csv)

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
                import re
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
coords = molstar_helper.parse_coordinate(traj_xtc)

def get_full_trajectory():
    return molstar_helper.get_trajectory(topo, coords)

initial_traj = get_full_trajectory()

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
                # Pairwise Energies
                dcc.Tab(label='🔗 Pairwise Energies', value='tab-pairwise', children=[
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader([
                                        html.H6("🍀 Select First Residue", className="text-white text-center mb-0", style={'fontSize': '14px'})
                                    ], style={'backgroundColor': soft_palette["light_blue"], 'padding': '8px'}),
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
                                        html.H6("🎯 Select Second Residue", className="text-white text-center mb-0", style={'fontSize': '14px'})
                                    ], style={'backgroundColor': soft_palette["light_blue"], 'padding': '8px'}),
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
                                        html.H6("📊 Average Energies", className="text-white text-center mb-0", style={'fontSize': '14px'})
                                    ], style={'backgroundColor': soft_palette["light_blue"], 'padding': '8px'}),
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
                # Interaction Energy Matrix
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
                # Network Analysis
                dcc.Tab(label='🕸️ Network Analysis', value='tab-network', children=[
                    html.Div(id='network-analysis-tab', className="tab-content", style={
                        'background': f'rgba(248,255,248,0.6)',
                        'borderRadius': '10px',
                        'padding': '15px',
                        'margin': '10px',
                        'border': f'2px solid {soft_palette["accent"]}'
                    }, children=[
                        # Controls
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
                                       }),
                            html.Button('💾 Export Network to File...', 
                                       id='export_network_btn', 
                                       n_clicks=0, 
                                       style={
                                           'marginLeft': '10px',
                                           'backgroundColor': soft_palette['secondary'],
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
                        # Subtabs
                        dcc.Tabs(id='network-tabs', value='residue-metrics', 
                                colors={
                                    'border': soft_palette['border'],
                                    'primary': soft_palette['primary'],
                                    'background': soft_palette['surface']
                                }, children=[
                            dcc.Tab(label='📊 Residue Metrics', value='residue-metrics', children=[
                                html.Div(style={
                                    'display': 'flex', 
                                    'gap': '5px',  # Reduced gap between plots
                                    'padding': '10px',
                                    'width': '100%',
                                    'overflowX': 'hidden',  # Prevent horizontal scrolling on parent
                                    'overflowY': 'hidden',  # Parent doesn't need vertical scroll
                                    'height': '75vh'  # Fixed height for the container
                                }, children=[
                                    # Degree Centrality - with scrolling container
                                    html.Div(className='network-plot-wrapper', children=[
                                        dcc.Graph(
                                            id='degree_centrality', 
                                            config={
                                                'displayModeBar': False, 
                                                'responsive': False,  # Disable responsive to maintain fixed height
                                                'scrollZoom': False
                                            },
                                            style={
                                                'width': '100%',
                                                'minHeight': '100%',  # Force minimum height
                                                'position': 'relative'
                                            }
                                        )
                                    ]),
                                    # Betweenness Centrality - with scrolling container
                                    html.Div(className='network-plot-wrapper', children=[
                                        dcc.Graph(
                                            id='betweenness_centrality', 
                                            config={
                                                'displayModeBar': False, 
                                                'responsive': False,  # Disable responsive to maintain fixed height
                                                'scrollZoom': False
                                            },
                                            style={
                                                'width': '100%',
                                                'minHeight': '100%',  # Force minimum height
                                                'position': 'relative'
                                            }
                                        )
                                    ]),
                                    # Closeness Centrality - with scrolling container
                                    html.Div(className='network-plot-wrapper', children=[
                                        dcc.Graph(
                                            id='closeness_centrality', 
                                            config={
                                                'displayModeBar': False, 
                                                'responsive': False,  # Disable responsive to maintain fixed height
                                                'scrollZoom': False
                                            },
                                            style={
                                                'width': '100%',
                                                'minHeight': '100%',  # Force minimum height
                                                'position': 'relative'
                                            }
                                        )
                                    ])
                                ])
                            ]),
                            dcc.Tab(label='🛤️ Shortest Paths', value='shortest-paths', children=[
                                html.Div(style={
                                    'display':'flex',
                                    'alignItems':'center',
                                    'gap':'15px',
                                    'padding':'15px',
                                    'background': soft_palette["light_blue"],
                                    'borderRadius': '10px',
                                    'marginBottom': '15px'
                                }, children=[
                                    html.Label("🎯 Source Residue:", style={
                                        'color': 'white',
                                        'fontFamily': 'Roboto, sans-serif',
                                        'fontWeight': '500',
                                        'fontSize': '12px'
                                    }),
                                    dcc.Dropdown(
                                        id='source_residue', 
                                        options=[{'label': r,'value': r} for r in first_res_list],
                                        value=None,
                                        style={'width':'150px'}
                                    ),
                                    html.Label("🏁 Target Residue:", style={
                                        'color': 'white',
                                        'fontFamily': 'Roboto, sans-serif',
                                        'fontWeight': '500',
                                        'fontSize': '12px'
                                    }),
                                    dcc.Dropdown(
                                        id='target_residue', 
                                        options=[{'label': r,'value': r} for r in first_res_list],
                                        value=None,
                                        style={'width':'150px'}
                                    ),
                                    html.Button('🔍 Find', 
                                               id='find_paths_btn', 
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
                                dash_table.DataTable(
                                    id='paths_table',
                                    columns=[{'name':'Path','id':'Path'},{'name':'Length','id':'Length'}],
                                    data=[],
                                    style_table={
                                        'height':'50vh',
                                        'overflowY':'auto',
                                        'borderRadius': '8px',
                                        'border': f'2px solid {soft_palette["accent"]}'
                                    },
                                    style_header={
                                        'backgroundColor': soft_palette['accent'],
                                        'color': 'white',
                                        'fontWeight': 'bold',
                                        'textAlign': 'center'
                                    },
                                    style_cell={
                                        'textAlign':'left',
                                        'whiteSpace':'normal',
                                        'height':'auto',
                                        'fontFamily': 'Roboto, sans-serif',
                                        'fontSize': '12px'
                                    }
                                )
                            ])
                        ])
                    ])
                ])
            ])
        ])
        ], style={'padding': '20px', 'backgroundColor': 'rgba(255,255,255,0.95)', 'borderRadius': '15px', 'border': f'3px solid {soft_palette["border"]}', 'boxShadow': '0 8px 32px rgba(0,0,0,0.1)'})
        ], width=8),
        # Right Panel
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

# Callbacks
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
    Input('frame_slider','value'),  # Use 'value' instead of 'drag_value' for less frequent updates
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
    
    # Always update second table when first residue is selected
    filt = total_df[(total_df['res1']==first)|(total_df['res2']==first)]
    others = [r for r in pd.concat([filt['res1'],filt['res2']]).unique() if r!=first]
    # Sort the interacting residues by sequence order
    others_sorted = sort_residues_by_sequence(others)
    table = [{'Residue': r} for r in others_sorted]
    
    # Create bar chart for average energies
    if others_sorted:
        bar_data = {'Residue': [], 'Total': [], 'VdW': [], 'Electrostatic': []}
        for r in others_sorted:
            p1, p2 = f"{first}-{r}", f"{r}-{first}"
            bar_data['Residue'].append(r)
            
            for energy_type in ['Total', 'VdW', 'Electrostatic']:
                vals = energy_long[energy_type][(energy_long[energy_type]['Pair']==p1)|(energy_long[energy_type]['Pair']==p2)]['Energy']
                ie = round(vals.mean(), 3) if not vals.empty else 0
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
            title="🍀 Average Interaction Energies",
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
            margin=dict(l=80, r=20, t=40, b=80)
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
        df_line = energy_long[energy_type][(energy_long[energy_type]['Pair']==p1)|(energy_long[energy_type]['Pair']==p2)]
        
        if not df_line.empty:
            fig.add_trace(go.Scatter(
                x=df_line['Frame'],
                y=df_line['Energy'],
                mode='lines+markers',
                marker=dict(size=4, opacity=0.7, color=config['color']),
                line=dict(color=config['color'], width=2),
                name=energy_type
            ))
            
            if selected_frame in df_line['Frame'].astype(int).values:
                e0 = df_line[df_line['Frame'].astype(int)==selected_frame]['Energy'].values[0]
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
            showlegend=False
        )
        figures[energy_type] = fig
    
    return figures['Total'], figures['VdW'], figures['Electrostatic'], bar_fig, table, sel2

# Separate callback for molecular viewer (lighter weight)
@app.callback(
    Output('viewer','frame'),
    Input('frame_slider','value')
)
def update_viewer_frame(selected_frame):
    return selected_frame

# Lightweight callback for drag updates - only updates molecular viewer during dragging
@app.callback(
    Output('viewer','frame', allow_duplicate=True),
    Input('frame_slider','drag_value'),
    prevent_initial_call=True
)
def update_viewer_drag(drag_value):
    if drag_value is not None:
        return drag_value
    return no_update

# Separate callback for molecular selection (only when residues change AND on pairwise tab)
@app.callback(
    Output('viewer','selection'),
    Output('viewer','focus'),
    Input('first_residue_table','selected_rows'),
    Input('second_residue_table','selected_rows'),
    Input('main-tabs','value'),  # Add tab state as input
    State('second_residue_table','data')
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
        
        r1,c1=first.split('_')[0][3:],first.split('_')[1]
        r2,c2=second.split('_')[0][3:],second.split('_')[1]
        t1=molstar_helper.get_targets(c1,r1); t2=molstar_helper.get_targets(c2,r2)
        seldata=molstar_helper.get_selection([t1,t2],select=True,add=False)
        focusdata=molstar_helper.get_focus([t1,t2],analyse=True)
    except:
        pass
    
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
        return 0.5, 10, 5, {i: {'label': f'±{i}', 'style': {'color': 'white', 'fontSize': '11px', 'fontWeight': 'bold'}} for i in range(1, 11, 2)}
    
    # Calculate min and max across all frames for this energy type
    energy_values = selected_df[energy_cols].values.flatten()
    energy_values = pd.to_numeric(energy_values, errors='coerce')  # Convert to numeric, NaN for non-numeric
    energy_values = energy_values[~pd.isna(energy_values)]  # Remove NaN values
    
    if len(energy_values) > 0:
        data_min = float(energy_values.min())
        data_max = float(energy_values.max())
        
        # Create symmetric range around zero for proper color mapping
        # This ensures zero always corresponds to white in the diverging colormap
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
        
        # Set slider range from 0.5 to range_limit (since we'll use ±value)
        slider_min = 0.5
        slider_max = range_limit
        
        # Set initial value to 20% of the maximum range to focus on common values
        # This helps avoid outliers dominating the color scale
        initial_value = max(slider_min, range_limit * 0.2)
        
        # Create exactly 5 marks evenly distributed across the slider range
        import numpy as np
        
        # Generate exactly 5 evenly spaced values across the slider range
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
            
            # Use the actual value as the key for correct positioning
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
    
    # Get all residues and sort them by sequence order
    all_residues = set(df['res1']).union(df['res2'])
    
    # Sort residues by sequence number for proper protein order
    def extract_residue_number(res_name):
        try:
            # Extract number from residue name like 'GLY290_A' 
            import re
            number = re.findall(r'\d+', res_name)
            if number:
                return int(number[0])
            return 0
        except:
            return 0
    
    residues = sorted(all_residues, key=extract_residue_number)
    
    # Create matrix with proper indexing - initialize with NaN to distinguish from zero
    matrix_df = pd.DataFrame(float('nan'), index=residues, columns=residues)
    
    # Fill the matrix with energy values
    for _, row in df.iterrows():
        if row['res1'] in residues and row['res2'] in residues:
            energy_val = float(row['energy'])
            matrix_df.loc[row['res1'], row['res2']] = energy_val
            matrix_df.loc[row['res2'], row['res1']] = energy_val
    
    # Set diagonal to 0 (self-interactions)
    for res in residues:
        matrix_df.loc[res, res] = 0.0
    
    # Fill remaining NaN values with 0
    matrix_df = matrix_df.fillna(0.0)
    
    # Use symmetric range based on slider value
    zmin, zmax = -range_value, range_value
    
    # Define consistent diverging color palette for all energy types
    # All energy types use the same RdBu_r colorscale with white at zero
    energy_configs = {
        'Total': {'colorscale': 'RdBu_r', 'symbol': '🔥'},
        'VdW': {'colorscale': 'RdBu_r', 'symbol': '🌊'},
        'Electrostatic': {'colorscale': 'RdBu_r', 'symbol': '⚡'}
    }
    
    config = energy_configs[energy_type]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns.tolist(),
        y=matrix_df.index.tolist(),
        colorscale=config['colorscale'],
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
        title=f'{config["symbol"]} {energy_type} Interaction Energy Matrix (Frame {frame_value})',
        xaxis_title='🧬 Residue',
        yaxis_title='🧬 Residue',
        xaxis={'tickangle': 45, 'automargin': True},
        yaxis={'automargin': True},
        margin=dict(l=80, r=50, t=80, b=100),
        font=dict(size=10, family='Roboto, sans-serif', color='#4A5A4A'),
        title_font=dict(size=16, family='Roboto, sans-serif', color='#4A5A4A'),
        plot_bgcolor='rgba(250,255,250,0.4)',
        paper_bgcolor='rgba(250,255,250,0.4)',
        height=600
    )
    return fig

# Network Metrics
@app.callback(
    Output('degree_centrality','figure'),Output('betweenness_centrality','figure'),Output('closeness_centrality','figure'),
    Input('update_network_btn','n_clicks'),Input('frame_slider','value'),Input('include_covalent_edges','value'),Input('energy_cutoff','value')
)
def update_network(n_clicks,frame,include_cov,cutoff):
    G=build_graph(frame,include_cov,cutoff)
    deg=dict(G.degree()); btw=nx.betweenness_centrality(G); clo=nx.closeness_centrality(G)
    
    def mk(data,title): 
        # Get all residues and sort them by sequence order (same as in heatmap and tables)
        # Use the global residue list to ensure consistency
        residues_sorted = sort_residues_by_sequence(first_res_list)
        
        # Create values list in the same order as sorted residues, ensuring ALL residues are included
        vals = [data.get(res, 0.0) for res in residues_sorted]
        
        # Ensure we have valid numeric values (no NaN or None)
        vals = [float(v) if v is not None and not pd.isna(v) else 0.0 for v in vals]
        
        fig=go.Figure(go.Bar(
            x=vals,
            y=residues_sorted,
            orientation='h',
            marker=dict(
                color=vals,
                colorscale='Greens',
                line=dict(color='#B5C5B5', width=1)
            ),
            # Add explicit name to prevent axis confusion
            name=title
        ))
        
        # Calculate height based on number of residues with fixed height per residue
        num_residues = len(residues_sorted)
        fixed_height_per_residue = 24
        
        fig.update_layout(
            title=f'🍀 {title}',
            margin=dict(l=100,r=10,t=40,b=20),  # Very compact margins
            font=dict(family='Roboto, sans-serif', size=8, color='#4A5A4A'),  # Smaller font
            title_font=dict(size=12, family='Roboto, sans-serif', color='#4A5A4A'),
            plot_bgcolor='rgba(240,255,240,0.3)',
            paper_bgcolor='rgba(255,255,255,0.9)',
            # NEW APPROACH: Calculate height based on actual residue count for natural scrolling
            height=num_residues * fixed_height_per_residue + 80,  # 80px for margins and title
            # CRITICAL: Set a narrower fixed width to fit 3 plots horizontally
            width=220,  # Reduced width for each bar plot (3 * 220 = 660px + gaps should fit)
            autosize=False,  # Disable autosize to maintain fixed dimensions
            xaxis=dict(
                title='',  # Remove x-axis title to save space
                title_font=dict(color='#4A5A4A', size=9),
                tickfont=dict(color='#4A5A4A', size=8),
                gridcolor='rgba(180, 180, 180, 0.3)',
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(0, 0, 0, 0.3)',
                fixedrange=True,  # Prevent zooming on x-axis
                autorange=True,  # Let x-axis auto-range to fit the data properly
                type='linear',
                side='bottom',
                showline=True,
                linecolor='rgba(0, 0, 0, 0.3)',
                mirror=True
            ),
            yaxis=dict(
                title='',  # Remove y-axis title to save space
                title_font=dict(color='#4A5A4A', size=9),
                tickfont=dict(color='#4A5A4A', size=8),  # Very small font for compact layout
                # CRITICAL: Force category order and prevent reordering
                categoryorder='array',
                categoryarray=residues_sorted,
                # Ensure all residues are shown with proper spacing
                dtick=1,
                tickmode='array',
                tickvals=residues_sorted,
                ticktext=residues_sorted,
                # Make sure the y-axis shows all residues properly
                range=[-0.5, len(residues_sorted) - 0.5],
                fixedrange=True,  # Prevent zooming on y-axis
                autorange=False,  # CRITICAL: Prevent automatic range adjustment
                # Reduce spacing between tick labels
                ticklen=3,
                tickwidth=1,
                type='category',  # Force category type to maintain order
                # Ensure proper axis configuration
                side='left',
                showline=True,
                linecolor='rgba(0, 0, 0, 0.3)',
                mirror=True
            ),
            # Ensure bars have consistent width and very compact spacing
            bargap=0.02,  # Minimal gap between bars for very compact layout
            bargroupgap=0.01,
            # Disable unnecessary features for cleaner look
            showlegend=False,
            # Ensure plot is responsive but maintains fixed height per residue
            hovermode='closest',
            # Lock the aspect ratio to prevent distortion
            dragmode=False,
            # Remove toolbar for cleaner look
            modebar=dict(remove=['zoom', 'pan', 'select', 'lasso', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']),
            # CRITICAL: Force specific layout settings that prevent reset
            uirevision='static',  # Keep UI state constant
            template=None  # Don't apply any template that might override settings
        )
        return fig
    return mk(deg,'Degree'),mk(btw,'Betweenness centrality'),mk(clo,'Closeness centrality')

# Shortest Paths
@app.callback(
    Output('paths_table','data'),
    Input('find_paths_btn','n_clicks'),Input('frame_slider','value'),Input('include_covalent_edges','value'),Input('energy_cutoff','value'),
    State('source_residue','value'),State('target_residue','value')
)
def find_paths(n_clicks,frame,include_cov,cutoff,source,target):
    if n_clicks<1 or not source or not target or source==target:
        return []
    G=build_graph(frame,include_cov,cutoff)
    try:
        paths_gen=nx.shortest_simple_paths(G,source,target,weight='weight')
        out=[]
        for i,path in enumerate(paths_gen):
            if i>=10: break
            length=sum(G[u][v]['weight'] for u,v in zip(path[:-1],path[1:]))
            out.append({'Path':'-'.join(path),'Length':round(length,6)})
        return out
    except:
        return []

if __name__ == '__main__':
    app.run(debug=True, port=8051)
