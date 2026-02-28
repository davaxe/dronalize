import numpy as np
import polars as pl
import pytest

from preprocessing.common.trajectory.convert import convert_to_agent_data_dict
from preprocessing.core.datatypes.categories import AgentCategory


def _make_scene(
    n_agents: int = 3,
    input_len: int = 4,
    output_len: int = 4,
) -> pl.DataFrame:
    """Create a minimal scene DataFrame with n_agents each having full coverage."""
    total = input_len + output_len
    rows = []
    for agent_id in range(1, n_agents + 1):
        rows.extend(
            {
                "frame": frame,
                "id": agent_id,
                "x": float(agent_id * 10 + frame),
                "y": float(agent_id * 100 + frame),
                "vx": float(agent_id),
                "vy": float(agent_id * 0.1),
                "ax": 0.0,
                "ay": 0.0,
                "yaw": 0.0,
                "agent_category": AgentCategory.CAR,
            }
            for frame in range(total)
        )
    return pl.DataFrame(rows)


def test_output_shape_positions() -> None:
    """inp_pos and trg_pos have the correct (num_agents, time, 2) shape."""
    input_len, output_len = 4, 4
    df = _make_scene(n_agents=3, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    assert result["inp_pos"].shape == (3, input_len, 2)
    assert result["trg_pos"].shape == (3, output_len, 2)


def test_output_shape_velocities() -> None:
    """inp_vel and trg_vel have the correct shape."""
    input_len, output_len = 3, 5
    df = _make_scene(n_agents=2, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    assert result["inp_vel"].shape == (2, input_len, 2)
    assert result["trg_vel"].shape == (2, output_len, 2)


def test_output_shape_accelerations() -> None:
    """inp_acc and trg_acc have the correct shape."""
    input_len, output_len = 4, 4
    df = _make_scene(n_agents=2, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    assert result["inp_acc"].shape == (2, input_len, 2)
    assert result["trg_acc"].shape == (2, output_len, 2)


def test_output_shape_yaw() -> None:
    """inp_yaw and trg_yaw have shape (num_agents, time)."""
    input_len, output_len = 4, 4
    df = _make_scene(n_agents=3, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    assert result["inp_yaw"].shape == (3, input_len)
    assert result["trg_yaw"].shape == (3, output_len)


def test_num_nodes() -> None:
    """num_nodes matches the number of unique agents."""
    df = _make_scene(n_agents=5, input_len=3, output_len=3)
    result = convert_to_agent_data_dict(df, input_len=3, output_len=3)
    assert result["num_nodes"] == 5


def test_target_agent_is_index_zero() -> None:
    """ta_index is always 0, meaning the target agent occupies row 0."""
    df = _make_scene(n_agents=3, input_len=4, output_len=4)
    result = convert_to_agent_data_dict(df, input_len=4, output_len=4)
    assert result["ta_index"] == 0


def test_explicit_target_agent() -> None:
    """Passing a specific target_agent places that agent at index 0."""
    input_len, output_len = 4, 4
    df = _make_scene(n_agents=3, input_len=input_len, output_len=output_len)
    # Agent 3 has x = 30 + frame
    result = convert_to_agent_data_dict(
        df, input_len=input_len, output_len=output_len, target_agent=3
    )
    assert result["ta_index"] == 0
    # Verify that row 0 corresponds to agent 3's positions
    expected_x = [30.0 + f for f in range(input_len)]
    actual_x = result["inp_pos"][0, :, 0].tolist()
    for a, e in zip(actual_x, expected_x):
        assert abs(a - e) < 1e-5


def test_explicit_target_agent_insufficient_frames() -> None:
    """Specifying a target_agent without enough frames raises ValueError."""
    input_len, output_len = 4, 4
    total = input_len + output_len
    # Agent 1 has all frames, agent 2 has only 3 frames (not enough)
    rows = []
    for frame in range(total):
        rows.append({
            "frame": frame,
            "id": 1,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "yaw": 0.0,
            "agent_category": 1,
        })
    for frame in range(3):
        rows.append({
            "frame": frame,
            "id": 2,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "yaw": 0.0,
            "agent_category": 1,
        })
    df = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="does not have enough valid frames"):
        convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len, target_agent=2)


def test_no_valid_target_agent() -> None:
    """When no agent has sufficient frames, ValueError is raised."""
    # Only 2 frames per agent, but we need input_len + output_len = 8
    rows = []
    for agent_id in [1, 2]:
        for frame in [0, 1]:
            rows.append({
                "frame": frame,
                "id": agent_id,
                "x": 0.0,
                "y": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "ax": 0.0,
                "ay": 0.0,
                "yaw": 0.0,
                "agent_category": 1,
            })
    df = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="No valid target agent"):
        convert_to_agent_data_dict(df, input_len=4, output_len=4)


def test_mask_full_coverage() -> None:
    """Agents with data at every frame have all-True masks."""
    input_len, output_len = 4, 4
    df = _make_scene(n_agents=2, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    assert result["input_mask"].shape == (2, input_len)
    assert result["valid_mask"].shape == (2, output_len)
    assert result["input_mask"].all()
    assert result["valid_mask"].all()


def test_mask_sparse_agent() -> None:
    """An agent missing some frames has False in the mask at those positions."""
    input_len, output_len = 4, 4
    total = input_len + output_len
    rows = []
    # Agent 1: full coverage
    for frame in range(total):
        rows.append({
            "frame": frame,
            "id": 1,
            "x": float(frame),
            "y": 0.0,
            "vx": 1.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "yaw": 0.0,
            "agent_category": 1,
        })
    # Agent 2: present only at frames 0, 1 (missing frames 2..7)
    for frame in [0, 1]:
        rows.append({
            "frame": frame,
            "id": 2,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "yaw": 0.0,
            "agent_category": 1,
        })
    df = pl.DataFrame(rows)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    # Agent 2 is not the target (index 0), find it
    # Target is agent 1 (has full coverage), agent 2 is at index 1
    agent2_input_mask = result["input_mask"][1]
    agent2_valid_mask = result["valid_mask"][1]
    # Frames 0, 1 present; frames 2, 3 missing in input
    assert agent2_input_mask[0] is True or agent2_input_mask[0] == True  # noqa: E712
    assert agent2_input_mask[1] is True or agent2_input_mask[1] == True  # noqa: E712
    assert agent2_input_mask[2] == False  # noqa: E712
    assert agent2_input_mask[3] == False  # noqa: E712
    # All output frames missing
    assert not agent2_valid_mask.any()


def test_position_values_correct() -> None:
    """Position arrays contain the actual coordinate values from the DataFrame."""
    input_len, output_len = 3, 2
    df = _make_scene(n_agents=1, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    # Agent 1: x = 10 + frame, y = 100 + frame
    expected_inp_x = [10.0, 11.0, 12.0]
    expected_trg_x = [13.0, 14.0]
    np.testing.assert_allclose(result["inp_pos"][0, :, 0], expected_inp_x, atol=1e-5)
    np.testing.assert_allclose(result["trg_pos"][0, :, 0], expected_trg_x, atol=1e-5)

    expected_inp_y = [100.0, 101.0, 102.0]
    expected_trg_y = [103.0, 104.0]
    np.testing.assert_allclose(result["inp_pos"][0, :, 1], expected_inp_y, atol=1e-5)
    np.testing.assert_allclose(result["trg_pos"][0, :, 1], expected_trg_y, atol=1e-5)


def test_type_array_default() -> None:
    """Without category_mapping, the type array contains raw AgentCategory int values."""
    input_len, output_len = 3, 3
    df = _make_scene(n_agents=2, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    assert result["type"].shape == (2,)
    assert result["type"].dtype == np.int32
    for val in result["type"]:
        assert val == int(AgentCategory.CAR)


def test_type_array_custom_mapping() -> None:
    """Custom category_mapping remaps AgentCategory values in the type array."""
    input_len, output_len = 3, 3
    total = input_len + output_len
    rows = []
    for frame in range(total):
        rows.append({
            "frame": frame,
            "id": 1,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "yaw": 0.0,
            "agent_category": AgentCategory.CAR,
        })
    for frame in range(total):
        rows.append({
            "frame": frame,
            "id": 2,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "yaw": 0.0,
            "agent_category": AgentCategory.PEDESTRIAN,
        })
    df = pl.DataFrame(rows)
    mapping = {AgentCategory.CAR: 0, AgentCategory.PEDESTRIAN: 1}
    result = convert_to_agent_data_dict(
        df, input_len=input_len, output_len=output_len, category_mapping=mapping
    )

    types = result["type"].tolist()
    assert 0 in types
    assert 1 in types


def test_type_array_unmapped_category_returns_minus_one() -> None:
    """Category not in mapping gets -1."""
    input_len, output_len = 3, 3
    total = input_len + output_len
    rows = []
    for frame in range(total):
        rows.append({
            "frame": frame,
            "id": 1,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "yaw": 0.0,
            "agent_category": AgentCategory.BICYCLE,
        })
    df = pl.DataFrame(rows)
    mapping = {AgentCategory.CAR: 0}  # BICYCLE not in mapping
    result = convert_to_agent_data_dict(
        df, input_len=input_len, output_len=output_len, category_mapping=mapping
    )
    assert result["type"][0] == -1


def test_non_target_agents_sorted_by_id() -> None:
    """Non-target agents are sorted by ID after the target agent."""
    input_len, output_len = 3, 3
    total = input_len + output_len
    rows = []
    for agent_id in [5, 2, 9, 1]:
        for frame in range(total):
            rows.append({
                "frame": frame,
                "id": agent_id,
                "x": float(agent_id),
                "y": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "ax": 0.0,
                "ay": 0.0,
                "yaw": 0.0,
                "agent_category": 1,
            })
    df = pl.DataFrame(rows)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    # Target agent is at index 0; identify it by its x position
    target_x = result["inp_pos"][0, 0, 0]
    remaining_x = [result["inp_pos"][i, 0, 0] for i in range(1, 4)]
    # Remaining agents should be sorted by ID (= x value here), excluding the target
    all_ids = [5, 2, 9, 1]
    target_id = int(target_x)
    non_target_ids = sorted([i for i in all_ids if i != target_id])
    assert remaining_x == [float(i) for i in non_target_ids]


def test_single_agent_scene() -> None:
    """A scene with a single agent produces correct shapes."""
    input_len, output_len = 3, 3
    df = _make_scene(n_agents=1, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    assert result["num_nodes"] == 1
    assert result["ta_index"] == 0
    assert result["inp_pos"].shape == (1, input_len, 2)
    assert result["trg_pos"].shape == (1, output_len, 2)
    assert result["input_mask"].all()
    assert result["valid_mask"].all()


def test_velocity_values_correct() -> None:
    """Velocity arrays contain the values from the DataFrame."""
    input_len, output_len = 2, 2
    df = _make_scene(n_agents=1, input_len=input_len, output_len=output_len)
    result = convert_to_agent_data_dict(df, input_len=input_len, output_len=output_len)

    # Agent 1: vx = 1.0, vy = 0.1
    np.testing.assert_allclose(result["inp_vel"][0, :, 0], [1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(result["inp_vel"][0, :, 1], [0.1, 0.1], atol=1e-5)
