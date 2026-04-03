# Auto-generated from data_collection_notebook.ipynb
# This script preserves notebook cell order and content in a Python-friendly format.

# %% [markdown]
# # Data Collection Notebook
#
# This notebook version of `data_collection.py` keeps the same dataset-generation logic while splitting it into smaller sections that are easier to inspect, modify, and run incrementally.
#
# It includes:
# - PyTorch Geometric `.pt` export
# - episode-level train/val/test splits
# - tapered temporal sampling
# - rollout metadata per graph
# - a JSON sidecar describing the dataset recipe

# %%
import json
import os

import numpy as np
import pybullet as p
import torch
from PyFlyt.core import Aviary
from torch_geometric.data import Data, InMemoryDataset

# %% [markdown]
# ## Global Constants
#
# This cell defines formation labels and split-specific seed offsets. The split offsets make validation and test episodes use different random streams from training even when you start from one base seed.

# %%
FORMATION_NAMES = ("a", "rectangle", "triangle")
FORMATION_TO_ID = {name: idx for idx, name in enumerate(FORMATION_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}

# %% [markdown]
# ## Wind and Episode Initialization
#
# These helpers generate a simple wind field and sample randomized initial conditions for each episode. The initialization function accepts a NumPy random generator so each episode can be reproduced from a seed.


# %%
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
    start_pos[:, 2] = rng.uniform(
        altitude_range[0], altitude_range[1], size=(num_drones,)
    )

    start_orn = np.zeros((num_drones, 3))
    start_orn[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
    return start_pos, start_orn


def resolve_formation_name(dataset_type: str):
    if dataset_type in {"a", "formation_a"}:
        return "a"
    if dataset_type in {"rectangle", "rectangular", "formation_rectangle"}:
        return "rectangle"
    if dataset_type in {"triangle", "formation_triangle"}:
        return "triangle"
    return None


# %% [markdown]
# ## Formation Geometry
#
# These helpers define how each formation is laid out in the XY plane. The setpoints are centered around the episode's mean initial position so the swarm moves into formation without teleporting the target far away from the sampled scenario.


# %%
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


def _build_formation_setpoints(formation_name: str, start_pos: np.ndarray):
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
        return None

    setpoints = np.zeros((num_drones, 4))
    setpoints[:, :2] = formation_center + offsets[:, :2]
    setpoints[:, 2] = 0.0
    setpoints[:, 3] = target_altitude
    return setpoints


# %% [markdown]
# ## Environment Creation and Setpoint Generation
#
# This part creates the PyFlyt `Aviary`, enables optional wind, and generates setpoints for either a formation task or a generic random-hovering style task.


# %%
def create_aviary(
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    environmental_wind: bool,
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
    return env


def build_setpoints(
    dataset_type: str,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    rng: np.random.Generator,
):
    num_drones = len(start_pos)

    formation_name = resolve_formation_name(dataset_type)
    formation_setpoints = None
    if formation_name is not None:
        formation_setpoints = _build_formation_setpoints(formation_name, start_pos)
    if formation_setpoints is not None:
        return formation_setpoints

    if dataset_type == "hovering":
        setpoints = np.zeros((num_drones, 4))
        setpoints[:, :2] = start_pos[:, :2]
        setpoints[:, 2] = start_orn[:, 2]
        setpoints[:, 3] = start_pos[:, 2]
        return setpoints

    radius = 10.0 if dataset_type == "aggressive" else 5.0
    setpoints = np.zeros((num_drones, 4))
    setpoints[:, :2] = start_pos[:, :2] + rng.uniform(
        -radius, radius, size=(num_drones, 2)
    )
    setpoints[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
    setpoints[:, 3] = rng.uniform(1.0, radius, size=(num_drones,))
    return setpoints


# %% [markdown]
# ## State Processing and Graph Construction
#
# These helpers transform raw drone state into graph node features, target vectors, and communication edges. This is the part that turns the simulator rollout into supervised learning samples for the GNN.


# %%
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
    setpoint: np.ndarray,
    noisy_sensors: bool,
    noise_variance: float,
    formation_one_hot: np.ndarray = None,
    include_formation_in_state: bool = True,
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

    target_global_pos = np.array([setpoint[0], setpoint[1], setpoint[3]])
    target_global_yaw = setpoint[2]

    global_pos_error = target_global_pos - global_pos
    yaw_error = target_global_yaw - global_euler[2]
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

    rotation_quaternion = p.getQuaternionFromEuler(global_euler)
    rot_matrix = np.array(p.getMatrixFromQuaternion(rotation_quaternion)).reshape(3, 3)
    local_pos_error = rot_matrix.T @ global_pos_error

    gnn_input_state = np.concatenate([local_lin_vel, local_ang_vel])
    if include_formation_in_state and formation_one_hot is not None:
        gnn_input_state = np.concatenate([gnn_input_state, formation_one_hot])

    gnn_input_target = np.concatenate([local_pos_error, np.array([yaw_error])])
    motor_pwm_labels = drone.pwm

    return gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos


def build_edges(global_positions: np.ndarray, communication_radius: float):
    edges = []
    num_drones = len(global_positions)

    for i in range(num_drones):
        for j in range(num_drones):
            if i == j:
                continue

            dist = np.linalg.norm(global_positions[i] - global_positions[j])
            if dist <= communication_radius:
                edges.append([i, j])

    return edges


def collect_step_data(
    env,
    setpoints,
    noisy_sensors: bool,
    noise_variance: float,
    communication_radius: float,
    formation_one_hot: np.ndarray = None,
    include_formation_in_state: bool = True,
):
    episode_states = []
    episode_targets = []
    episode_labels = []
    global_positions = []

    for i, drone in enumerate(env.drones):
        gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos = (
            build_drone_features(
                drone,
                setpoints[i],
                noisy_sensors,
                noise_variance,
                formation_one_hot,
                include_formation_in_state,
            )
        )
        episode_states.append(gnn_input_state)
        episode_targets.append(gnn_input_target)
        episode_labels.append(motor_pwm_labels)
        global_positions.append(global_pos)

    edges = build_edges(np.array(global_positions), communication_radius)
    return episode_states, episode_targets, episode_labels, edges


# %% [markdown]
# ## Split Logic, Tapered Sampling, and Metadata
#
# This section implements the practical dataset design choices discussed earlier:
# - split by episode rather than by graph
# - use separate seed streams for train, val, and test
# - optionally save more steps early and fewer later
# - write a metadata sidecar so the dataset recipe is reproducible


# %%
def compute_split_episode_counts(
    num_episodes: int,
    split_ratios: tuple[float, float, float],
):
    if len(split_ratios) != 3:
        raise ValueError(
            "split_ratios must contain exactly three values for train, val, test."
        )
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


# %% [markdown]
# ## Main Dataset Generator
#
# This is the full generation loop. It creates split-specific episodes, applies the tapered sampling policy, stores one PyG graph per retained step, and writes both split files and a metadata JSON file.


# %%
def generate_dataset(
    num_episodes=200,
    max_steps=300,
    dataset_name="formation_dataset",
    dataset_type="mixed_formations",
    noisy_sensors=False,
    noise_variance=0.01,
    environmental_wind=False,
    graphical=False,
    communication_radius=np.inf,
    include_formation_in_state=True,
    mixed_formation_types=FORMATION_NAMES,
    split_ratios=(0.8, 0.1, 0.1),
    seed=12345,
    base_xy_limit=10.0,
    altitude_range=(0.5, 5.0),
    validation_spread_scale=1.25,
    test_spread_scale=1.5,
    tapered_sampling=True,
    dense_sampling_steps=120,
    mid_sampling_steps=240,
    mid_step_stride=2,
    late_step_stride=5,
):
    script_dir = os.path.dirname(
        os.path.abspath(os.getcwd() if "__file__" not in globals() else __file__)
    )
    repo_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(repo_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

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
                num_drones,
                rng,
                xy_limit=xy_limit,
                altitude_range=altitude_range,
            )
            env = create_aviary(start_pos, start_orn, environmental_wind, graphical)
            setpoints = build_setpoints(episode_dataset_type, start_pos, start_orn, rng)

            formation_name = resolve_formation_name(episode_dataset_type)
            formation_id = -1
            if formation_name is not None:
                formation_id = FORMATION_TO_ID[formation_name]

            formation_one_hot = None
            if formation_id >= 0:
                formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
                formation_one_hot[formation_id] = 1.0

            env.set_all_setpoints(setpoints)

            saved_steps = 0
            for step_idx in range(max_steps):
                if should_sample_step(
                    step_idx,
                    max_steps,
                    tapered_sampling,
                    dense_sampling_steps,
                    mid_sampling_steps,
                    mid_step_stride,
                    late_step_stride,
                ):
                    episode_states, episode_targets, episode_labels, edges = (
                        collect_step_data(
                            env,
                            setpoints,
                            noisy_sensors,
                            noise_variance,
                            communication_radius,
                            formation_one_hot,
                            include_formation_in_state,
                        )
                    )

                    x = torch.as_tensor(np.asarray(episode_states), dtype=torch.float32)
                    target = torch.as_tensor(
                        np.asarray(episode_targets), dtype=torch.float32
                    )
                    y = torch.as_tensor(np.asarray(episode_labels), dtype=torch.float32)

                    if edges:
                        edge_index = (
                            torch.tensor(edges, dtype=torch.long).t().contiguous()
                        )
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)

                    graph = Data(
                        x=x,
                        target=target,
                        y=y,
                        edge_index=edge_index,
                        formation_id=torch.tensor([formation_id], dtype=torch.long),
                        episode_id=torch.tensor([global_episode_id], dtype=torch.long),
                        step_idx=torch.tensor([step_idx], dtype=torch.long),
                        num_drones=torch.tensor([num_drones], dtype=torch.long),
                    )
                    split_graphs[split_name].append(graph)
                    saved_steps += 1

                env.step()

            env.disconnect()

            split_summaries[split_name]["num_graphs"] += saved_steps
            episode_center = np.mean(start_pos[:, :2], axis=0)
            initial_xy_radius = float(
                np.max(np.linalg.norm(start_pos[:, :2] - episode_center, axis=1))
            )
            episode_records.append(
                {
                    "episode_id": global_episode_id,
                    "split": split_name,
                    "split_episode_idx": split_episode_idx,
                    "episode_seed": episode_seed,
                    "num_drones": num_drones,
                    "episode_dataset_type": episode_dataset_type,
                    "formation_name": formation_name,
                    "initial_xy_limit": xy_limit,
                    "initial_xy_radius": initial_xy_radius,
                    "saved_steps": saved_steps,
                }
            )
            global_episode_id += 1

    generated_files = {}
    for split_name, graphs in split_graphs.items():
        if not graphs:
            continue

        split_dataset_path = f"{dataset_prefix}_{split_name}.pt"
        save_dataset(split_dataset_path, graphs, FORMATION_NAMES, split_name)
        generated_files[split_name] = os.path.basename(split_dataset_path)
        print(f"Generated {split_name} dataset -> {split_dataset_path}")

    metadata_path = f"{dataset_prefix}_metadata.json"
    metadata = {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "generated_files": generated_files,
        "formation_names": list(FORMATION_NAMES),
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "noisy_sensors": noisy_sensors,
            "noise_variance": noise_variance,
            "environmental_wind": environmental_wind,
            "graphical": graphical,
            "communication_radius": communication_radius,
            "include_formation_in_state": include_formation_in_state,
            "mixed_formation_types": list(mixed_formation_types),
            "split_ratios": list(split_ratios),
            "seed": seed,
            "base_xy_limit": base_xy_limit,
            "altitude_range": list(altitude_range),
            "validation_spread_scale": validation_spread_scale,
            "test_spread_scale": test_spread_scale,
            "tapered_sampling": tapered_sampling,
            "dense_sampling_steps": dense_sampling_steps,
            "mid_sampling_steps": mid_sampling_steps,
            "mid_step_stride": mid_step_stride,
            "late_step_stride": late_step_stride,
        },
        "split_summary": split_summaries,
        "episodes": episode_records,
    }
    write_dataset_metadata(metadata_path, metadata)
    print(f"Generated dataset metadata -> {metadata_path}")

    return generated_files, metadata_path


# %% [markdown]
# ## Example Usage
#
# The final cell shows a practical default configuration. Leave it commented if you only want the notebook as documentation, or run it to generate split `.pt` files and the metadata JSON.

# %%
# Example generation call
generated_files, metadata_path = generate_dataset(
    num_episodes=240,
    max_steps=300,
    dataset_name="formation_mixed_comm_10m",
    dataset_type="mixed_formations",
    communication_radius=10.0,
    include_formation_in_state=True,
    tapered_sampling=True,
)
print(generated_files)
print(metadata_path)

# %% [markdown]
# ## Inspect a Generated Split
#
#
#
# This section uses a small PyTorch Geometric `InMemoryDataset` wrapper to load one generated split exactly the way a PyG training pipeline would. It reconstructs the dataset, inspects one sample graph, and then shows how PyG batches graphs with a `DataLoader`.

# %%
from pathlib import Path

from torch_geometric.loader import DataLoader


class GeneratedSplitDataset(InMemoryDataset):
    def __init__(self, split_path: str | Path):

        self.split_path = Path(split_path)

        payload = torch.load(self.split_path)

        self.formation_names = payload.get("formation_names", [])

        self.split_name = payload.get("split_name", "unknown")

        super().__init__(root="")

        self.data = payload["data"]

        self.slices = payload["slices"]


inspect_dataset_name = "formation_mixed_comm_10m_mixed_formations"

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

# %% [markdown]
# ## Upload a Dataset to Google Drive
#
#
#
# This section is intended for Google Colab. It mounts Google Drive and copies the generated split files and metadata JSON into `MyDrive/dataset/<dataset-name>/`. Update the dataset name if you want to upload a different generated dataset bundle.

# %%
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
