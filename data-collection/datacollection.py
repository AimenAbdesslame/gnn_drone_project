#!/usr/bin/env python
# coding: utf-8

# # Data Collection Notebook
# 
# This notebook version of `data_collection.py` keeps the same dataset-generation logic while splitting it into smaller sections that are easier to inspect, modify, and run incrementally.
# 
# It includes:
# 
# - PyTorch Geometric `.pt` export
# - episode-level train/val/test splits
# - tapered temporal sampling
# - rollout metadata per graph
# - a JSON sidecar describing the dataset recipe
# 

# In[37]:


import json
import os
from typing import Literal

import numpy as np
import pybullet as p
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data, HeteroData, InMemoryDataset

from PyFlyt.core import Aviary


# ## Global Constants
# 
# This cell defines formation labels and split-specific seed offsets. The split offsets make validation and test episodes use different random streams from training even when you start from one base seed.
# 

# In[38]:


FORMATION_NAMES = ("a", "rectangle", "triangle")
FORMATION_TO_ID = {name: idx for idx, name in enumerate(FORMATION_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}
TASK_TYPES = ("setpoint_prediction", "residual_correction", "formation_assignment_homo", "formation_assignment_hetero")


# ## Wind and Episode Initialization
# 
# These helpers generate a simple wind field and sample randomized initial conditions for each episode. The initialization function accepts a NumPy random generator so each episode can be reproduced from a seed.
# 

# In[40]:


def wind_generator(time: float, position: np.ndarray):
    """Generates an upward draft with random turbulence."""
    wind = np.zeros_like(position)
    wind[:, 2] = np.sin(time) * 0.5 + np.random.normal(0, 0.2, size=(len(position),))
    wind[:, 0] = np.cos(time / 2.0) * 0.3
    wind[:, 1] = np.sin(time / 2.0) * 0.3
    return wind


def sample_episode_initial_conditions(
    num_drones: int,
    rng: np.random.Generator,
    xy_limit: float = 10.0,
    altitude_range: tuple[float, float] = (0.5, 5.0),
):
    start_pos = rng.uniform(-xy_limit, xy_limit, size=(num_drones, 3))
    start_pos[:, 2] = rng.uniform(altitude_range[0], altitude_range[1], size=(num_drones,))

    start_orn = np.zeros((num_drones, 3))
    start_orn[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
    return start_pos, start_orn

def sample_obstacles(rng: np.random.Generator, num_obstacles: int, xy_limit: float):
    """Generate random non-overlapping vertical obstacles (cylinders)."""
    if num_obstacles == 0:
        return np.zeros((0, 2))
    return rng.uniform(-xy_limit, xy_limit, size=(num_obstacles, 2))

def resolve_formation_name(dataset_type: str):
    if dataset_type in {"a", "formation_a"}:
        return "a"
    if dataset_type in {"rectangle", "rectangular", "formation_rectangle"}:
        return "rectangle"
    if dataset_type in {"triangle", "formation_triangle"}:
        return "triangle"
    return None


# ## Formation Geometry
# 
# These helpers define how each formation is laid out in the XY plane. The setpoints are centered around the episode's mean initial position so the swarm moves into formation without teleporting the target far away from the sampled scenario.
# 

# In[41]:


def _formation_a_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 1:
        return offsets
    offsets[0] = np.array([0.0, 0.0, 0.0])
    for idx in range(1, num_drones):
        level = (idx + 1) // 2
        side = -1.0 if idx % 2 == 1 else 1.0 
        offsets[idx, 0] = side * level * spacing 
        offsets[idx, 1] = level * spacing

    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0) 
    return offsets


def _formation_rectangle_offsets(num_drones: int, spacing: float = 2.0):
    cols = int(np.ceil(np.sqrt(num_drones)))
    rows = int(np.ceil(num_drones / cols))

    offsets = np.zeros((num_drones, 3))
    for idx in range(num_drones):
        row = idx // cols
        col = idx % cols
        offsets[idx, 0] = (col - (cols - 1) / 2.0) * spacing
        offsets[idx, 1] = (row - (rows - 1) / 2.0) * spacing

    return offsets


def _formation_triangle_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))

    count = 0
    row = 0
    while count < num_drones:
        points_in_row = row + 1
        row_y = row * spacing
        row_start_x = -0.5 * row * spacing

        for col in range(points_in_row):
            if count >= num_drones:
                break
            offsets[count, 0] = row_start_x + col * spacing
            offsets[count, 1] = row_y
            count += 1

        row += 1

    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets


def apply_obstacle_avoidance(slots, obstacles, obstacle_radius, padding=1.0):
    """Pushes format slots out of obstacles if they overlap."""
    if len(obstacles) == 0:
        return slots

    safe_slots = np.copy(slots)
    min_dist = obstacle_radius + padding

    for i in range(len(safe_slots)):
        for obs in obstacles:
            diff = safe_slots[i, :2] - obs
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                direction = diff / (dist + 1e-6)
                safe_slots[i, :2] = obs + direction * min_dist

    return safe_slots


def _build_formation_setpoints(formation_name: str, start_pos: np.ndarray, obstacles=np.empty((0, 2)), obstacle_radius=0.0):
    num_drones = len(start_pos)
    formation_center = np.mean(start_pos[:, :2], axis=0)
    target_altitude = np.mean(start_pos[:, 2])

    if formation_name == "a":
        offsets = _formation_a_offsets(num_drones)
    elif formation_name == "rectangle":
        offsets = _formation_rectangle_offsets(num_drones)
    elif formation_name == "triangle":
        offsets = _formation_triangle_offsets(num_drones)
    else:
        return None, None, None

    # Step 1: Base slots (naive)
    global_slots = np.zeros((num_drones, 3))
    global_slots[:, :2] = formation_center + offsets[:, :2]

    # Step 2: Apply obstacle avoidance to shift slots dynamically
    safe_global_slots = apply_obstacle_avoidance(global_slots, obstacles, obstacle_radius, padding=1.0)

    # Step 3: Optimal Assignment
    dist_matrix = np.linalg.norm(start_pos[:, None, :2] - safe_global_slots[None, :, :2], axis=2)
    _, assigned_indices = linear_sum_assignment(dist_matrix)

    # Step 4: Build final setpoints
    setpoints = np.zeros((num_drones, 4))
    for i in range(num_drones):
        slot_idx = assigned_indices[i]
        setpoints[i, :2] = safe_global_slots[slot_idx, :2]
        setpoints[i, 2] = 0.0 # Yaw
        setpoints[i, 3] = target_altitude

    return setpoints, assigned_indices, offsets


# ## Environment Creation and Setpoint Generation
# 
# This part creates the PyFlyt `Aviary`, enables optional wind, and generates setpoints for either a formation task or a generic random-hovering style task.
# 

# In[42]:


def create_aviary(
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    environmental_wind: bool,
    obstacles: np.ndarray = np.empty((0,2)),
    obstacle_radius: float = 1.0,
    graphical: bool = False,
):
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        drone_type="quadx",
        render=graphical,
    )

    if environmental_wind:
        env.register_wind_field_function(wind_generator)

    env.set_mode(7)

    # Spawn simple cylinder obstacles if present
    if len(obstacles) > 0:
        physics_client = env._client
        for obs in obstacles:
            # Add simple stationary cylinders in pybullet
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=obstacle_radius, height=10.0, physicsClientId=physics_client)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=obstacle_radius, length=10.0, rgbaColor=[1,0,0,0.5], physicsClientId=physics_client)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[obs[0], obs[1], 5.0], physicsClientId=physics_client)
        env.register_all_new_bodies()

    return env


def build_setpoints(
    dataset_type: str,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    rng: np.random.Generator,
    obstacles: np.ndarray = np.empty((0,2)),
    obstacle_radius: float = 1.0,
):
    num_drones = len(start_pos)
    formation_name = resolve_formation_name(dataset_type)

    if formation_name is not None:
        setpoints, col_ind, offsets = _build_formation_setpoints(formation_name, start_pos, obstacles, obstacle_radius)
        if setpoints is not None:
            return setpoints, col_ind, offsets

    # Default fallback (hovering or aggressive target changing)
    setpoints = np.zeros((num_drones, 4))
    if dataset_type == "hovering":
        setpoints[:, :2] = start_pos[:, :2]
        setpoints[:, 2] = start_orn[:, 2]
        setpoints[:, 3] = start_pos[:, 2]
    else:
        radius = 10.0 if dataset_type == "aggressive" else 5.0
        setpoints[:, :2] = start_pos[:, :2] + rng.uniform(-radius, radius, size=(num_drones, 2))
        setpoints[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
        setpoints[:, 3] = rng.uniform(1.0, radius, size=(num_drones,))

    col_ind = np.arange(num_drones)
    return setpoints, col_ind, np.zeros((num_drones, 3))


# ## State Processing and Graph Construction
# 
# These helpers transform raw drone state into graph node features, target vectors, and communication edges. This is the part that turns the simulator rollout into supervised learning samples for the GNN.
# 

# In[43]:


def maybe_add_sensor_noise(
    global_pos: np.ndarray,
    global_euler: np.ndarray,
    local_lin_vel: np.ndarray,
    local_ang_vel: np.ndarray,
    noisy_sensors: bool,
    noise_variance: float,
):
    if not noisy_sensors:
        return global_pos, global_euler, local_lin_vel, local_ang_vel

    global_pos = global_pos + np.random.normal(0, noise_variance, size=3)
    global_euler = global_euler + np.random.normal(0, noise_variance, size=3)
    local_lin_vel = local_lin_vel + np.random.normal(0, noise_variance, size=3)
    local_ang_vel = local_ang_vel + np.random.normal(0, noise_variance, size=3)
    return global_pos, global_euler, local_lin_vel, local_ang_vel


def build_drone_features(
    drone,
    drone_idx: int,
    setpoint: np.ndarray,
    assigned_slot_idx: int,
    naive_offset: np.ndarray,
    task_type: str,
    noisy_sensors: bool,
    noise_variance: float,
    formation_one_hot: np.ndarray | None,
    obstacles: np.ndarray,
    include_formation_in_state: bool,
    start_pos_center: np.ndarray,
):
    state = drone.state
    global_pos = np.array(state[3], copy=True)
    global_euler = np.array(state[1], copy=True)
    local_lin_vel = np.array(state[2], copy=True)
    local_ang_vel = np.array(state[0], copy=True)

    global_pos, global_euler, local_lin_vel, local_ang_vel = maybe_add_sensor_noise(
        global_pos,
        global_euler,
        local_lin_vel,
        local_ang_vel,
        noisy_sensors,
        noise_variance,
    )

    # Inject visible obstacles into state (relative x, y of nearest 2 obstacles)
    obs_features = np.zeros(4)
    if len(obstacles) > 0:
        rel_obs = obstacles - global_pos[:2]
        dists = np.linalg.norm(rel_obs, axis=1)
        closest_idxs = np.argsort(dists)[:2]
        for i, idx in enumerate(closest_idxs):
            obs_features[i*2:(i+1)*2] = rel_obs[idx]

    gnn_input_state = np.concatenate([local_lin_vel, local_ang_vel, obs_features])
    if include_formation_in_state and formation_one_hot is not None:
        gnn_input_state = np.concatenate([gnn_input_state, formation_one_hot])

    target_global_pos = np.array([setpoint[0], setpoint[1], setpoint[3]])
    target_global_yaw = setpoint[2]

    # Target switch logic
    if task_type == "setpoint_prediction":
        global_pos_error = target_global_pos - global_pos
        yaw_error = target_global_yaw - global_euler[2]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        rotation_quaternion = p.getQuaternionFromEuler(global_euler)
        rot_matrix = np.array(p.getMatrixFromQuaternion(rotation_quaternion)).reshape(3, 3)
        local_pos_error = rot_matrix.T @ global_pos_error
        y_label = np.concatenate([local_pos_error, [yaw_error]])

    elif task_type == "residual_correction":
        naive_global_pos = np.array([
            start_pos_center[0] + naive_offset[0],
            start_pos_center[1] + naive_offset[1],
            target_global_pos[2]
        ])
        y_label = target_global_pos - naive_global_pos

    elif task_type in ("formation_assignment_homo", "formation_assignment_hetero"):
        y_label = np.array([assigned_slot_idx], dtype=np.float32)

    gnn_input_target = y_label
    motor_pwm_labels = drone.pwm

    return gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos


def build_edges(global_positions: np.ndarray, communication_radius: float):
    edges = []
    edge_attrs = []
    num_drones = len(global_positions)

    for i in range(num_drones):
        for j in range(num_drones):
            if i == j:
                continue

            rel_pos = global_positions[j] - global_positions[i]
            dist = np.linalg.norm(rel_pos)
            if dist <= communication_radius:
                edges.append([i, j])
                edge_attrs.append(rel_pos)

    return edges, edge_attrs


def collect_step_data(
    env,
    active_drones: list[int],
    setpoints,
    slot_assignments: list[int],
    naive_offsets: np.ndarray,
    task_type: str,
    noisy_sensors: bool,
    noise_variance: float,
    communication_radius: float,
    formation_one_hot: np.ndarray | None,
    obstacles: np.ndarray,
    include_formation_in_state: bool,
    start_pos_center: np.ndarray,
):
    episode_states = []
    episode_targets = []
    episode_labels = []
    global_positions = []

    for i, drone_idx in enumerate(active_drones):
        drone = env.drones[drone_idx]
        slot_idx = slot_assignments[i]
        gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos = build_drone_features(
            drone,
            drone_idx,
            setpoints[i],
            slot_idx,
            naive_offsets[slot_idx],
            task_type,
            noisy_sensors,
            noise_variance,
            formation_one_hot,
            obstacles,
            include_formation_in_state,
            start_pos_center,
        )
        episode_states.append(gnn_input_state)
        # Note: in residual/assignement targets, episode targets is the y_label.
        episode_targets.append(gnn_input_target)
        episode_labels.append(motor_pwm_labels)
        global_positions.append(global_pos)

    edges, edge_attrs = build_edges(np.array(global_positions), communication_radius)
    return episode_states, episode_targets, episode_labels, edges, edge_attrs, global_positions


# ## Split Logic, Tapered Sampling, and Metadata
# 
# This section implements the practical dataset design choices discussed earlier:
# 
# - split by episode rather than by graph
# - use separate seed streams for train, val, and test
# - optionally save more steps early and fewer later
# - write a metadata sidecar so the dataset recipe is reproducible
# 

# In[44]:


def compute_split_episode_counts(
    num_episodes: int,
    split_ratios: tuple[float, float, float],
):
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain exactly three values for train, val, test.")
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("split_ratios must sum to 1.0.")

    counts = [int(num_episodes * ratio) for ratio in split_ratios]
    remainder = num_episodes - sum(counts)
    for idx in range(remainder):
        counts[idx] += 1

    return {split_name: count for split_name, count in zip(SPLIT_NAMES, counts)}


def resolve_split_spread_scale(
    split_name: str,
    validation_spread_scale: float,
    test_spread_scale: float,
):
    if split_name == "val":
        return validation_spread_scale
    if split_name == "test":
        return test_spread_scale
    return 1.0


def should_sample_step(
    step_idx: int,
    max_steps: int,
    tapered_sampling: bool,
    dense_sampling_steps: int,
    mid_sampling_steps: int,
    mid_step_stride: int,
    late_step_stride: int,
):
    if not tapered_sampling or step_idx == max_steps - 1:
        return True

    mid_sampling_steps = max(mid_sampling_steps, dense_sampling_steps)

    if step_idx < dense_sampling_steps:
        return True
    if step_idx < mid_sampling_steps:
        return (step_idx - dense_sampling_steps) % mid_step_stride == 0
    return (step_idx - mid_sampling_steps) % late_step_stride == 0


def build_episode_seed(seed: int | None, split_name: str, split_episode_idx: int):
    if seed is None:
        return None
    return int(seed + SPLIT_SEED_OFFSETS[split_name] + split_episode_idx)


def write_dataset_metadata(metadata_path: str, metadata: dict):
    with open(metadata_path, "w", encoding="ascii") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def save_dataset(
    dataset_path: str,
    all_graphs,
    formation_names,
    split_name: str,
):
    data, slices = InMemoryDataset.collate(all_graphs)
    torch.save(
        {
            "data": data,
            "slices": slices,
            "formation_names": formation_names,
            "split_name": split_name,
        },
        dataset_path,
    )


# ## Main Dataset Generator
# 
# This is the full generation loop. It creates split-specific episodes, applies the tapered sampling policy, stores one PyG graph per retained step, and writes both split files and a metadata JSON file.
# 

# In[ ]:


def generate_dataset(
    num_episodes: int = 200,
    max_steps: int = 300,
    dataset_name: str = "formation_dataset",
    dataset_type: str = "mixed_formations",
    task_type: Literal[
        "setpoint_prediction", 
        "residual_correction", 
        "formation_assignment_homo", 
        "formation_assignment_hetero"
    ] = "setpoint_prediction",
    num_obstacles: int = 0,
    obstacle_radius: float = 1.0,
    inject_failures: bool = False,
    dynamic_formation: bool = False,
    noisy_sensors: bool = False,
    noise_variance: float = 0.01,
    environmental_wind: bool = False,
    graphical: bool = False,
    communication_radius: float = np.inf,
    include_formation_in_state: bool = True,
    mixed_formation_types: tuple = FORMATION_NAMES,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 12345,
    base_xy_limit: float = 10.0,
    altitude_range: tuple[float, float] = (0.5, 5.0),
    validation_spread_scale: float = 1.25,
    test_spread_scale: float = 1.5,
    tapered_sampling: bool = True,
    dense_sampling_steps: int = 120,
    mid_sampling_steps: int = 240,
    mid_step_stride: int = 2,
    late_step_stride: int = 5,
    conv_stopping: bool = True,
    conv_threshold: float = 0.2,
) -> tuple[dict[str, str], str]:
    if '__file__' in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.path.abspath(os.getcwd())
    repo_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(repo_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")

    dataset_prefix = os.path.join(datasets_dir, f"{dataset_name}_{dataset_type}")
    split_episode_counts = compute_split_episode_counts(num_episodes, split_ratios)
    split_graphs = {split_name: [] for split_name in SPLIT_NAMES}
    split_summaries = {}
    episode_records = []

    global_episode_id = 0
    for split_name in SPLIT_NAMES:
        split_episode_count = split_episode_counts[split_name]
        split_spread_scale = resolve_split_spread_scale(
            split_name,
            validation_spread_scale,
            test_spread_scale,
        )
        split_summaries[split_name] = {
            "num_episodes": split_episode_count,
            "num_graphs": 0,
            "spread_scale": split_spread_scale,
        }

        for split_episode_idx in range(split_episode_count):
            episode_seed = build_episode_seed(seed, split_name, split_episode_idx)
            rng = np.random.default_rng(episode_seed)

            print(
                f"[{dataset_name}] Starting {split_name} episode {split_episode_idx + 1}/{split_episode_count} ..."
            )

            num_drones = int(rng.integers(10, 21))
            episode_dataset_type = dataset_type
            if dataset_type in {"mixed_formations", "mixed", "formations"}:
                episode_dataset_type = str(rng.choice(mixed_formation_types))

            xy_limit = base_xy_limit * split_spread_scale
            start_pos, start_orn = sample_episode_initial_conditions(
                num_drones, rng, xy_limit=xy_limit, altitude_range=altitude_range
            )
            obstacles = sample_obstacles(rng, num_obstacles, xy_limit=5.0)

            env = create_aviary(start_pos, start_orn, environmental_wind, obstacles, obstacle_radius, graphical)
            setpoints, col_ind, naive_offsets = build_setpoints(episode_dataset_type, start_pos, start_orn, rng, obstacles, obstacle_radius)
            active_drones = list(range(num_drones))

            formation_name = resolve_formation_name(episode_dataset_type)
            formation_id = -1
            if formation_name is not None:
                formation_id = FORMATION_TO_ID[formation_name]

            formation_one_hot = None
            if formation_id >= 0:
                formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
                formation_one_hot[formation_id] = 1.0

            # Only setting it for active drones
            for i, drone_idx in enumerate(active_drones):
                env.set_setpoint(drone_idx, setpoints[i])

            saved_steps = 0
            steps_since_last_event = 0
            failure_injected_this_episode = False
            already_converged_for_segment = False

            for step_idx in range(max_steps):
                steps_since_last_event += 1

                # Optionally inject mid-flight failure
                should_inject_failure = inject_failures and not failure_injected_this_episode and step_idx >= max_steps // 2
                if should_inject_failure and len(active_drones) > 2:
                    failed_drone = active_drones.pop() # logically drop a drone
                    env.drones[failed_drone].set_mode(0) # kill motors physically
                    # Recalculate setpoints with N-1 drones
                    active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
                    setpoints, col_ind, naive_offsets = build_setpoints(episode_dataset_type, active_start_pos, start_orn[:len(active_drones)], rng, obstacles, obstacle_radius)
                    for i, drone_idx in enumerate(active_drones):
                        env.drones[drone_idx].set_mode(7)
                        env.set_setpoint(drone_idx, setpoints[i])
                    failure_injected_this_episode = True
                    steps_since_last_event = 0
                    already_converged_for_segment = False

                # Optionally rotate formation dynamically
                if dynamic_formation and step_idx > 0 and step_idx % 100 == 0:
                    next_shape = str(rng.choice(tuple(set(FORMATION_NAMES) - {formation_name})))
                    formation_name = next_shape
                    formation_id = FORMATION_TO_ID[formation_name]
                    formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
                    formation_one_hot[formation_id] = 1.0
                    active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
                    setpoints, col_ind, naive_offsets = build_setpoints(formation_name, active_start_pos, start_orn[:len(active_drones)], rng, obstacles, obstacle_radius)
                    for i, drone_idx in enumerate(active_drones):
                        env.set_setpoint(drone_idx, setpoints[i])
                    steps_since_last_event = 0
                    already_converged_for_segment = False

                is_converged = False
                # wait at least 50 steps after any event before evaluating convergence
                if conv_stopping and steps_since_last_event > 50: 
                    max_pos_error = 0.0
                    for i, drone_idx in enumerate(active_drones):
                        drone_pos = env.drones[drone_idx].state[3]
                        target_pos = np.array([setpoints[i][0], setpoints[i][1], setpoints[i][3]])
                        error = np.linalg.norm(drone_pos - target_pos)
                        if error > max_pos_error:
                            max_pos_error = error
                    is_converged = (max_pos_error < conv_threshold)

                if is_converged and not already_converged_for_segment:
                    print(f"      Swarm successfully converged to formation at step {step_idx}!")
                    already_converged_for_segment = True

                # Check if it's safe to stop the whole episode early
                can_stop_episode = is_converged
                if can_stop_episode:
                    if inject_failures and not failure_injected_this_episode:
                        can_stop_episode = False
                    if dynamic_formation:
                        # Dynamic formation episodes are expected to run until max_steps to gather all transitions
                        can_stop_episode = False

                # We still want to save the final frame of ANY convergence, even if we don't stop the episode
                if is_converged or should_sample_step(
                    step_idx, max_steps, tapered_sampling, dense_sampling_steps,
                    mid_sampling_steps, mid_step_stride, late_step_stride
                ):
                    # To prevent taking too many samples while "converged" but waiting for an event (e.g. failure or dynamic event),
                    # we only force sample if it's the exact moment we stop, or we follow normal tapered sampling otherwise
                    if can_stop_episode or should_sample_step(
                        step_idx, max_steps, tapered_sampling, dense_sampling_steps,
                        mid_sampling_steps, mid_step_stride, late_step_stride
                    ):
                        episode_states, episode_targets, episode_labels, edges, edge_attrs, global_positions = collect_step_data(
                            env, active_drones, setpoints, col_ind, naive_offsets, task_type,
                            noisy_sensors, noise_variance, communication_radius, formation_one_hot,
                            obstacles, include_formation_in_state, np.mean(start_pos[:, :2], axis=0)
                        )

                        x = torch.as_tensor(np.asarray(episode_states), dtype=torch.float32)
                        target = torch.as_tensor(np.asarray(episode_targets), dtype=torch.float32)
                        y = torch.as_tensor(np.asarray(episode_labels), dtype=torch.float32)

                        if edges:
                            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                            edge_attr = torch.as_tensor(np.asarray(edge_attrs), dtype=torch.float32)
                        else:
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                            edge_attr = torch.empty((0, 3), dtype=torch.float32)

                        if task_type == "formation_assignment_hetero":
                            graph = HeteroData()
                            graph["drone"].x = x
                            graph["drone"].y = y
                            graph["drone"].pos = torch.as_tensor(np.asarray(global_positions), dtype=torch.float32)
                            graph["slot"].x = torch.as_tensor(naive_offsets, dtype=torch.float32)
                            graph["drone", "communicates", "drone"].edge_index = edge_index
                            graph["drone", "communicates", "drone"].edge_attr = edge_attr

                            drones = torch.arange(len(active_drones), dtype=torch.long)
                            slots = target.view(-1).long()
                            graph["drone", "assigned_to", "slot"].edge_label_index = torch.stack([drones, slots], dim=0)

                            graph.formation_id = torch.tensor([formation_id], dtype=torch.long)
                            graph.episode_id = torch.tensor([global_episode_id], dtype=torch.long)
                            graph.step_idx = torch.tensor([step_idx], dtype=torch.long)
                            graph.num_drones = torch.tensor([len(active_drones)], dtype=torch.long)
                            graph.obstacles = torch.as_tensor(obstacles, dtype=torch.float32)
                        else:
                            graph = Data(
                                x=x, target=target, y=y, edge_index=edge_index, edge_attr=edge_attr,
                                pos=torch.as_tensor(np.asarray(global_positions), dtype=torch.float32),
                                obstacles=torch.as_tensor(obstacles, dtype=torch.float32),
                                formation_id=torch.tensor([formation_id], dtype=torch.long),
                                episode_id=torch.tensor([global_episode_id], dtype=torch.long),
                                step_idx=torch.tensor([step_idx], dtype=torch.long),
                                num_drones=torch.tensor([len(active_drones)], dtype=torch.long),
                            )
                        split_graphs[split_name].append(graph)
                        saved_steps += 1

                env.step()
                if can_stop_episode:
                    print(f"      Swarm converged perfectly at step {step_idx}! Stopping early.")
                    break;

            env.disconnect()

            split_summaries[split_name]["num_graphs"] += saved_steps
            episode_center = np.mean(start_pos[:, :2], axis=0)
            initial_xy_radius = float(np.max(np.linalg.norm(start_pos[:, :2] - episode_center, axis=1)))
            episode_records.append({
                "episode_id": global_episode_id, "split": split_name,
                "split_episode_idx": split_episode_idx, "episode_seed": episode_seed,
                "num_drones": num_drones, "episode_dataset_type": episode_dataset_type,
                "formation_name": formation_name, "initial_xy_limit": xy_limit,
                "initial_xy_radius": initial_xy_radius, "saved_steps": saved_steps,
            })
            global_episode_id += 1

    generated_files = {}
    for split_name, graphs in split_graphs.items():
        if not graphs: continue
        split_dataset_path = f"{dataset_prefix}_{split_name}.pt"
        save_dataset(split_dataset_path, graphs, FORMATION_NAMES, split_name)
        generated_files[split_name] = os.path.basename(split_dataset_path)
        print(f"Generated {split_name} dataset -> {split_dataset_path}")

    metadata_path = f"{dataset_prefix}_metadata.json"
    metadata = {
        "dataset_name": dataset_name, "dataset_type": dataset_type, "task_type": task_type,
        "config": {
            "num_episodes": num_episodes, "max_steps": max_steps, "num_obstacles": num_obstacles,
            "obstacle_radius": obstacle_radius, "inject_failures": inject_failures,
            "dynamic_formation": dynamic_formation, "noisy_sensors": noisy_sensors,"noise_variance": noise_variance,
            "environmental_wind": environmental_wind,"graphical": graphical,"communication_radius": communication_radius,
            "include_formation_in_state": include_formation_in_state,"mixed_formation_types": list(mixed_formation_types),
            "split_ratios": list(split_ratios),"seed": seed,"base_xy_limit": base_xy_limit,
            "altitude_range": list(altitude_range),"validation_spread_scale": validation_spread_scale,
            "test_spread_scale": test_spread_scale,"tapered_sampling": tapered_sampling,
            "dense_sampling_steps": dense_sampling_steps,"mid_sampling_steps": mid_sampling_steps,
            "mid_step_stride": mid_step_stride,"late_step_stride": late_step_stride,
            "conv_stopping": conv_stopping,"conv_threshold": conv_threshold,
        },
        "split_summary": split_summaries, "episodes": episode_records,
    }
    write_dataset_metadata(metadata_path, metadata)
    print(f"Generated dataset metadata -> {metadata_path}")

    return generated_files, metadata_path


# ## Example Usage
# 
# The final cell shows a practical default configuration. Leave it commented if you only want the notebook as documentation, or run it to generate split `.pt` files and the metadata JSON.
# 

# In[49]:


# Example generation call
generated_files, metadata_path =  generate_dataset(
     dataset_name="noiseless_baseline_10",
     task_type="formation_assignment_homo",
     noisy_sensors=False,
     environmental_wind=False,
     dynamic_formation=False,
     inject_failures=True,
     communication_radius=2.0, 
     num_episodes=10,   
     max_steps=1200,    # High timeout
     tapered_sampling=False, # Uniform sampling
     conv_stopping=True,     # Stop early when drones reach formation
     conv_threshold=0.2,
     obstacle_radius=0.5,
     num_obstacles=10,
)
print(generated_files)
print(metadata_path)


# ## Inspect a Generated Split
# 
# This section uses a small PyTorch Geometric `InMemoryDataset` wrapper to load one generated split exactly the way a PyG training pipeline would. It reconstructs the dataset, inspects one sample graph, and then shows how PyG batches graphs with a `DataLoader`.
# 

# In[ ]:


from pathlib import Path



from torch_geometric.loader import DataLoader





class GeneratedSplitDataset(InMemoryDataset):

    def __init__(self, split_path: str | Path):

        self.split_path = Path(split_path)

        payload = torch.load(self.split_path, weights_only=False)

        self.formation_names = payload.get("formation_names", [])

        self.split_name = payload.get("split_name", "unknown")

        super().__init__(root="")

        self.data = payload["data"]

        self.slices = payload["slices"]





inspect_dataset_name = "noiseless_baseline_mixed_formations"

inspect_split_name = "train"



notebook_dir = Path.cwd()

datasets_dir = notebook_dir.parent / "datasets"



split_path = datasets_dir / f"{inspect_dataset_name}_{inspect_split_name}.pt"

metadata_path = datasets_dir / f"{inspect_dataset_name}_metadata.json"



if not split_path.exists():

    raise FileNotFoundError(

        f"Could not find split dataset at {split_path}. Generate the dataset first or update inspect_dataset_name."

    )



dataset = GeneratedSplitDataset(split_path)



print(f"Loaded split file: {split_path.name}")

print(f"Stored split name: {dataset.split_name}")

print(f"Available formation names: {dataset.formation_names}")

print(f"Number of graphs: {len(dataset)}")



sample_graph = dataset[0]

print("\nSample graph summary:")

print(sample_graph)

print(f"x shape: {tuple(sample_graph.x.shape)}")

print(f"target shape: {tuple(sample_graph.target.shape)}")

print(f"y shape: {tuple(sample_graph.y.shape)}")
print(f"edge_index shape: {tuple(sample_graph.edge_index.shape)}")
if hasattr(sample_graph, 'edge_attr') and sample_graph.edge_attr is not None:
    print(f"edge_attr shape: {tuple(sample_graph.edge_attr.shape)}")
print(f"episode_id: {int(sample_graph.episode_id.item())}")

print(f"step_idx: {int(sample_graph.step_idx.item())}")

print(f"num_drones: {int(sample_graph.num_drones.item())}")

print(f"formation_id: {int(sample_graph.formation_id.item())}")



loader = DataLoader(dataset, batch_size=4, shuffle=False)

batch = next(iter(loader))

print("\nPyG batch summary:")

print(batch)

print(f"batched x shape: {tuple(batch.x.shape)}")

print(f"batched edge_index shape: {tuple(batch.edge_index.shape)}")

print(f"batch vector shape: {tuple(batch.batch.shape)}")

print(f"graphs in batch: {batch.num_graphs}")



if metadata_path.exists():

    with open(metadata_path, "r", encoding="ascii") as metadata_file:

        metadata = json.load(metadata_file)



    print("\nMetadata summary:")

    print(f"Metadata file: {metadata_path.name}")

    print(f"Configured dataset type: {metadata['dataset_type']}")

    print(f"Configured split files: {metadata['generated_files']}")

    print(f"Split summary: {metadata['split_summary'][inspect_split_name]}")

    print(f"First episode record: {metadata['episodes'][0]}")

else:

    print(f"\nNo metadata file found at {metadata_path}")


# ## Upload a Dataset to Google Drive
# 
# This section is intended for Google Colab. It mounts Google Drive and copies the generated split files and metadata JSON into `MyDrive/dataset/<dataset-name>/`. Update the dataset name if you want to upload a different generated dataset bundle.
# 

# In[ ]:


import shutil

from pathlib import Path



upload_dataset_name = "formation_mixed_comm_10m_mixed_formations"

google_drive_root = Path("/content/drive/MyDrive")

google_drive_dataset_dir = google_drive_root / "dataset" / upload_dataset_name



try:

    from google.colab import drive

except ImportError as exc:

    raise RuntimeError(

        "This upload cell is intended to run in Google Colab, where google.colab is available."

    ) from exc



drive.mount("/content/drive", force_remount=False)



local_datasets_dir = Path.cwd().parent / "datasets"

google_drive_dataset_dir.mkdir(parents=True, exist_ok=True)



artifacts_to_copy = sorted(local_datasets_dir.glob(f"{upload_dataset_name}*.pt"))

metadata_file = local_datasets_dir / f"{upload_dataset_name}_metadata.json"

if metadata_file.exists():

    artifacts_to_copy.append(metadata_file)



if not artifacts_to_copy:

    raise FileNotFoundError(

        f"No dataset artifacts found for {upload_dataset_name} in {local_datasets_dir}."

    )



copied_files = []

for artifact_path in artifacts_to_copy:

    destination_path = google_drive_dataset_dir / artifact_path.name

    shutil.copy2(artifact_path, destination_path)

    copied_files.append(destination_path)



print(f"Uploaded {len(copied_files)} files to {google_drive_dataset_dir}")

for copied_file in copied_files:

    print(f" - {copied_file.name}")

