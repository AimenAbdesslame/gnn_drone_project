import plotly.graph_objects as go
import torch
import numpy as np
import dash
from typing import Optional, List, Union

def visualize_episode_timelapse(
    dataset, 
    episode_id: int, 
    title: str = "Drone Swarm Timelapse", 
    view_2d: bool = True
):
    """
    Creates an interactive scrollable timelapse for a specific episode in a PyG dataset.
    This expects the graph objects in the dataset to have a `pos` attribute containing 
    the absolute (x, y, z) global coordinates for each node.
    
    Args:
        dataset: A PyTorch Geometric dataset or list of Data/HeteroData objects.
        episode_id: The ID of the episode to visualize.
        title: The title of the plot.
        view_2d: If True, plots Y vs X (top-down) view. If False, plots full 3D.
        
    Returns:
        go.Figure: A Plotly interactive figure with a sequence slider.
    """
    
    # Filter dataset for the specific episode
    episode_graphs = []
    
    for graph in dataset:
        if hasattr(graph, 'episode_id'):
            # handle both single int/tensor episode formats
            graph_ep = graph.episode_id.item() if isinstance(graph.episode_id, torch.Tensor) else graph.episode_id
            if graph_ep == episode_id:
                episode_graphs.append(graph)
                
    if not episode_graphs:
        raise ValueError(f"No graphs found for episode_id {episode_id}")

    # Ensure graphs are sorted by time step
    if hasattr(episode_graphs[0], 'step_idx'):
        episode_graphs = sorted(episode_graphs, key=lambda g: g.step_idx.item() if isinstance(g.step_idx, torch.Tensor) else g.step_idx)
        
    # Check for position attribute
    def get_positions(graph):
        if hasattr(graph, "pos") and graph.pos is not None:
            return graph.pos.cpu().numpy()
        elif hasattr(graph, "drone") and hasattr(graph["drone"], "pos"):
            return graph["drone"].pos.cpu().numpy()
        elif hasattr(graph, "global_pos") and graph.global_pos is not None:
             return graph.global_pos.cpu().numpy()
        else:
            raise AttributeError("Graph missing 'pos' attribute. Please update data collection to save global positions.")
            
    fig = go.Figure()
    
    # Determine the global bounding box so the axes stay fixed during animation
    all_pos = []
    for g in episode_graphs:
        all_pos.append(get_positions(g))
    all_pos = np.vstack(all_pos)
    
    min_x, max_x = all_pos[:, 0].min() - 2, all_pos[:, 0].max() + 2
    min_y, max_y = all_pos[:, 1].min() - 2, all_pos[:, 1].max() + 2
    min_z, max_z = all_pos[:, 2].min() - 1, all_pos[:, 2].max() + 3
    
    # Prepare frames for each step
    frames = []
    
    for i, graph in enumerate(episode_graphs):
        pos = get_positions(graph)
        step_val = graph.step_idx.item() if hasattr(graph, 'step_idx') else i
        
        # Get edges
        edges_x, edges_y, edges_z = [], [], []
        
        if hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.numel() > 0:
            edge_idx = graph.edge_index.cpu().numpy()
        elif hasattr(graph, 'drone') and 'communicates' in graph.edge_types:
            edge_idx = graph["drone", "communicates", "drone"].edge_index.cpu().numpy()
        else:
            edge_idx = np.empty((2, 0))
            
        for e in range(edge_idx.shape[1]):
            src, dst = edge_idx[0, e], edge_idx[1, e]
            edges_x.extend([pos[src, 0], pos[dst, 0], None])
            edges_y.extend([pos[src, 1], pos[dst, 1], None])
            edges_z.extend([pos[src, 2], pos[dst, 2], None])
            
        # Handle obstacles
        obs_x, obs_y = [], []
        if hasattr(graph, 'obstacles') and graph.obstacles is not None and graph.obstacles.numel() > 0:
            obs = graph.obstacles.cpu().numpy()
            obs_x, obs_y = obs[:, 0], obs[:, 1]
            
        # Handle target slots (Heterogeneous)
        slot_x, slot_y, slot_z = [], [], []
        if hasattr(graph, 'slot') and hasattr(graph['slot'], 'x'):
            # The 'x' in slot denotes naive unshifted positions relative to a center, 
            # ideally the full absolute position is preserved, but we'll try plotting it 
            slots = graph['slot'].x.cpu().numpy()
            # If they are just relative, plotting them directly might be off by the center
            # which wasn't saved. We'll plot them if available
            slot_x, slot_y, slot_z = slots[:, 0], slots[:, 1], slots[:, 2] if slots.shape[1] > 2 else np.zeros_like(slots[:, 0])

        if view_2d:
             # X-Y View
             frame_data = [
                # Edges
                go.Scatter(x=edges_x, y=edges_y, mode='lines', line=dict(color='rgba(150,150,150,0.5)', width=1), hoverinfo='none', name="Comm Link"),
                # Obstacles
                go.Scatter(x=obs_x, y=obs_y, mode='markers', marker=dict(symbol='square', size=20, color='red', opacity=0.3), name="Obstacles", hoverinfo='none'),
                # Target Slots
                go.Scatter(x=slot_x, y=slot_y, mode='markers', marker=dict(symbol='cross', size=8, color='green', opacity=0.6), name="Target Slots", hoverinfo='none'),
                # Nodes
                go.Scatter(
                    x=pos[:, 0], y=pos[:, 1], 
                    mode='markers+text', 
                    marker=dict(size=12, color='rgba(0,114,239,0.8)', line=dict(width=2, color='DarkSlateGrey')),
                    text=[f"{n}" for n in range(len(pos))],
                    textposition="top center",
                    name="Drones",
                    hoverinfo='text',
                    hovertext=[f"Drone: {n}<br>X: {p[0]:.2f}<br>Y: {p[1]:.2f}" for n, p in enumerate(pos)]
                )
             ]
        else:
             # 3D View
             frame_data = [
                 go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode='lines', line=dict(color='rgba(150,150,150,0.5)', width=2), hoverinfo='none', name="Comm Link"),
                 go.Scatter3d(x=obs_x, y=obs_y, z=np.zeros_like(obs_x), mode='markers', marker=dict(size=10, color='red', symbol='square', opacity=0.5), name="Obstacle Base"),
                 go.Scatter3d(x=slot_x, y=slot_y, z=slot_z, mode='markers', marker=dict(size=6, color='green', symbol='cross'), name="Target Slots"),
                 go.Scatter3d(
                     x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], 
                     mode='markers+text',
                     marker=dict(size=6, color='rgba(0,114,239,0.8)'),
                     text=[f"{n}" for n in range(len(pos))],
                     name="Drones",
                     hovertext=[f"Drone: {n}<br>X: {p[0]:.2f}<br>Y: {p[1]:.2f}<br>Z: {p[2]:.2f}" for n, p in enumerate(pos)]
                 )
             ]
             
        # Just create the main trace layout frame
        frames.append(go.Frame(data=frame_data, name=f"step_{step_val}"))

    # Add the first frame's traces to the figure
    for trace in frames[0].data:
        fig.add_trace(trace)
        
    fig.frames = frames

    # Build slider
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[f"step_{g.step_idx.item() if hasattr(g, 'step_idx') else i}"], 
                  dict(mode='immediate', frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
            label=str(g.step_idx.item() if hasattr(g, 'step_idx') else i)
        ) for i, g in enumerate(episode_graphs)],
        active=0,
        transition=dict(duration=0),
        x=0, y=0,
        currentvalue=dict(font=dict(size=12), prefix='Step: ', visible=True, xanchor='center'),
        len=1.0
    )]

    # Layout configuration
    if view_2d:
        fig.update_layout(
            title=title,
            xaxis=dict(range=[min_x, max_x], title="X Position"),
            yaxis=dict(range=[min_y, max_y], title="Y Position", scaleanchor="x", scaleratio=1),
            sliders=sliders,
            updatemenus=[dict(type='buttons', showactive=False, y=0, x=-0.05,
                             buttons=[dict(label='Play',
                                           method='animate',
                                           args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])],
            showlegend=False,
            plot_bgcolor='white'
        )
    else:
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[min_x, max_x], title="X"),
                yaxis=dict(range=[min_y, max_y], title="Y"),
                zaxis=dict(range=[min_z, max_z], title="Z / Altitude"),
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=1, z=0),
                    eye=dict(x=0, y=0.01, z=2.0) # looking straight down
                )
            ),
            sliders=sliders,
            updatemenus=[dict(type='buttons', showactive=False, y=0, x=-0.05,
                             buttons=[dict(label='Play',
                                           method='animate',
                                           args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])],
            showlegend=False
        )

    return fig

