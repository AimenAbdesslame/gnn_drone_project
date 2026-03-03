import os
import numpy as np
import pybullet as p
from PyFlyt.core import Aviary


FORMATION_NAMES = ("a", "rectangle", "triangle")
FORMATION_TO_ID = {name: idx for idx, name in enumerate(FORMATION_NAMES)}


# Define a simple wind function matching what you see in example 09
def wind_generator(time: float, position: np.ndarray):
    """Generates an upward draft with random turbulence."""
    wind = np.zeros_like(position)
    
    # Give a base updraft force plus randomness based on position
    wind[:, 2] = np.sin(time) * 0.5 + np.random.normal(0, 0.2, size=(len(position),))
    
    # Push slightly horizontally 
    wind[:, 0] = np.cos(time / 2.0) * 0.3
    wind[:, 1] = np.sin(time / 2.0) * 0.3
    
    return wind


def sample_episode_initial_conditions(num_drones: int):
    start_pos = np.random.uniform(-10.0, 10.0, size=(num_drones, 3))
    start_pos[:, 2] = np.random.uniform(0.5, 5.0, size=(num_drones,))

    start_orn = np.zeros((num_drones, 3))
    start_orn[:, 2] = np.random.uniform(-np.pi, np.pi, size=(num_drones,))

    return start_pos, start_orn


def resolve_formation_name(dataset_type: str):
    if dataset_type in {"a", "formation_a"}:
        return "a"
    if dataset_type in {"rectangle", "rectangular", "formation_rectangle"}:
        return "rectangle"
    if dataset_type in {"triangle", "formation_triangle"}:
        return "triangle"
    return None


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


def build_setpoints(dataset_type: str, start_pos: np.ndarray, start_orn: np.ndarray):
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
    setpoints[:, :2] = start_pos[:, :2] + np.random.uniform(-radius, radius, size=(num_drones, 2))
    setpoints[:, 2] = np.random.uniform(-np.pi, np.pi, size=(num_drones,))
    setpoints[:, 3] = np.random.uniform(1.0, radius, size=(num_drones,))
    return setpoints


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
        gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos = build_drone_features(
            drone,
            setpoints[i],
            noisy_sensors,
            noise_variance,
            formation_one_hot,
            include_formation_in_state,
        )
        episode_states.append(gnn_input_state)
        episode_targets.append(gnn_input_target)
        episode_labels.append(motor_pwm_labels)
        global_positions.append(global_pos)

    edges = build_edges(np.array(global_positions), communication_radius)
    return episode_states, episode_targets, episode_labels, edges


def save_dataset(
    dataset_path: str,
    all_states,
    all_targets,
    all_labels,
    all_edges,
    all_formation_ids,
    all_formation_names,
):
    np.savez(
        dataset_path,
        states=np.array(all_states, dtype=object),
        targets=np.array(all_targets, dtype=object),
        labels=np.array(all_labels, dtype=object),
        edges=np.array(all_edges, dtype=object),
        formation_ids=np.array(all_formation_ids, dtype=np.int32),
        formation_names=np.array(all_formation_names, dtype=object),
    )


def generate_dataset(
    num_episodes=50,
    max_steps=500,
    dataset_name="formation_dataset",
    dataset_type="random",  # 'random', 'hovering', 'aggressive', 'a', 'rectangle', 'triangle', 'mixed_formations'
    noisy_sensors=False,
    noise_variance=0.01,
    environmental_wind=False,
    graphical=False,
    communication_radius=np.inf, # For edge formation
    include_formation_in_state=True,
    mixed_formation_types=FORMATION_NAMES,
):
    """
    Generate a multi-drone imitation-learning dataset from the PyFlyt PID controller.

    The simulator runs at the controller loop rate (120 Hz), and each collected sample
    contains node features, local target errors, PWM labels, and graph edges.
    Data is represented in local body-frame conventions (ENU-aligned state usage).

    Parameters
    ----------
    num_episodes : int, default=50
        Number of episodes to simulate.
    max_steps : int, default=500
        Number of simulation steps collected per episode.
    dataset_name : str, default="formation_dataset"
        Base name used in the output file path:
        ``datasets/{dataset_name}_{dataset_type}.npz``.
    dataset_type : str, default="random"
        Task/setpoint mode for each episode.
        Supported values include:
        - ``random``: random nearby targets.
        - ``hovering``: hold current position/yaw.
        - ``aggressive``: wider random target range.
        - ``a``, ``rectangle``, ``triangle``: fixed geometric formations.
        - ``mixed_formations`` (also ``mixed``, ``formations``): choose one
          formation type randomly per episode from ``mixed_formation_types``.
    noisy_sensors : bool, default=False
        If True, adds Gaussian noise to position, Euler angle, linear velocity,
        and angular velocity before building model inputs.
    noise_variance : float, default=0.01
        Standard deviation used for Gaussian sensor noise when
        ``noisy_sensors=True``.
    environmental_wind : bool, default=False
        If True, registers ``wind_generator`` as an external wind field in Aviary.
    graphical : bool, default=False
        If True, runs episodes with the PyBullet GUI enabled (rendering visible).
        If False, runs headless for faster data collection.
    communication_radius : float, default=np.inf
        Maximum inter-drone distance for adding directed communication edges.
        Use ``np.inf`` for fully connected (excluding self-loops).
    include_formation_in_state : bool, default=True
        If True, concatenates a formation one-hot vector to each node state when
        the active episode uses a recognized formation.
    mixed_formation_types : tuple[str, ...], default=FORMATION_NAMES
        Candidate formation names sampled in mixed mode. Typical values:
        ``("a", "rectangle", "triangle")``.

    Returns
    -------
    None
        Writes a compressed ``.npz`` dataset to disk containing states, targets,
        labels, edges, and formation metadata.
    """
    os.makedirs("datasets", exist_ok=True)
    dataset_path = f"../datasets/{dataset_name}_{dataset_type}.npz"

    all_states = []
    all_targets = []
    all_labels = []
    all_edges = []
    all_formation_ids = []
    all_formation_names = []

    for ep in range(num_episodes):
        print(f"[{dataset_name}] Starting Episode {ep+1}/{num_episodes} ...")

        num_drones = np.random.randint(10, 21)

        episode_dataset_type = dataset_type
        if dataset_type in {"mixed_formations", "mixed", "formations"}:
            episode_dataset_type = np.random.choice(mixed_formation_types)

        start_pos, start_orn = sample_episode_initial_conditions(num_drones)
        env = create_aviary(start_pos, start_orn, environmental_wind, graphical)
        setpoints = build_setpoints(episode_dataset_type, start_pos, start_orn)

        formation_name = resolve_formation_name(episode_dataset_type)
        formation_id = -1
        if formation_name is not None:
            formation_id = FORMATION_TO_ID[formation_name]
        formation_one_hot = None
        if formation_id >= 0:
            formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
            formation_one_hot[formation_id] = 1.0

        print(
            f"[{dataset_name}] Episode {ep+1} formation = {formation_name if formation_name is not None else episode_dataset_type}"
        )

        env.set_all_setpoints(setpoints)

        for _ in range(max_steps):
            episode_states, episode_targets, episode_labels, edges = collect_step_data(
                env,
                setpoints,
                noisy_sensors,
                noise_variance,
                communication_radius,
                formation_one_hot,
                include_formation_in_state,
            )

            all_edges.append(edges)
            all_states.append(episode_states)
            all_targets.append(episode_targets)
            all_labels.append(episode_labels)
            all_formation_ids.append(formation_id)
            all_formation_names.append(formation_name if formation_name is not None else episode_dataset_type)

            env.step()

        env.disconnect()

    save_dataset(
        dataset_path,
        all_states,
        all_targets,
        all_labels,
        all_edges,
        all_formation_ids,
        all_formation_names,
    )
    print(f"✅ Generated Dataset -> {dataset_path}")


if __name__ == "__main__":
    # Generate A-formation dataset
    generate_dataset(
        num_episodes=20,
        max_steps=400,
        dataset_name="formation_a_comm_10m",
        dataset_type="a",
        communication_radius=10.0,
    )

    # Generate rectangular formation dataset
    generate_dataset(
        num_episodes=20,
        max_steps=400,
        dataset_name="formation_rectangle_comm_10m",
        dataset_type="rectangle",
        communication_radius=10.0,
    )

    # Generate triangular formation dataset
    generate_dataset(
        num_episodes=20,
        max_steps=400,
        dataset_name="formation_triangle_comm_10m",
        dataset_type="triangle",
        communication_radius=10.0,
    )

    # Generate mixed-formation dataset with per-step formation labels
    generate_dataset(
        num_episodes=30,
        max_steps=400,
        dataset_name="formation_mixed_comm_10m",
        dataset_type="mixed_formations",
        communication_radius=10.0,
        include_formation_in_state=True,
    )
