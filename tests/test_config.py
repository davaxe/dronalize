from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.datatypes.map_config import MapConfig

# Adjust the import path based on the actual name of your module containing the config code
from dronalize.pipeline.ops.resample import Resampling, ResamplingMethod
from dronalize.processing.config import (
    Config,
    ConfigDict,
    ExecutionConfig,
    ExecutionConfigDict,
    LoaderConfigDict,
    MapConfigDict,
    _deep_merge,  # noqa: PLC2701
    resolve_config,
    resolve_execution_config,
    resolve_loader_config,
    resolve_map_config,
)


def test_deep_merge_flat() -> None:
    """Merge flat dictionaries and prioritize override values."""
    base = {"a": 1, "b": 2}
    overrides = {"b": 3, "c": 4}

    result = _deep_merge(base, overrides)

    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested() -> None:
    """Merge nested dictionaries recursively without losing base keys."""
    base = {"nested": {"a": 1, "b": 2}, "other": 5}
    overrides = {"nested": {"b": 99, "c": 3}}

    result = _deep_merge(base, overrides)

    assert result == {"nested": {"a": 1, "b": 99, "c": 3}, "other": 5}


def test_resolve_execution_config_pydantic_base_dict_override() -> None:
    """Resolve execution configuration using a Pydantic base and TypedDict override."""
    default = ExecutionConfig(parallel=False, workers=2, chunksize=10)
    overrides: ExecutionConfigDict = {"parallel": True, "workers": 8}

    result = resolve_execution_config(default, overrides)

    assert isinstance(result, ExecutionConfig)
    assert result.parallel is True
    assert result.workers == 8
    assert result.chunksize == 10  # Preserved from default


def test_resolve_execution_config_dict_base_pydantic_override() -> None:
    """Resolve execution configuration using a TypedDict base and Pydantic override."""
    default: ExecutionConfigDict = {"parallel": True, "workers": 4, "chunksize": 50}
    overrides = ExecutionConfig(parallel=False, workers=1, chunksize=None)

    result = resolve_execution_config(default, overrides)

    assert isinstance(result, ExecutionConfig)
    assert result.parallel is False
    assert result.workers == 1
    assert result.chunksize is None


def test_resolve_map_config_partial_override() -> None:
    """Resolve map configuration with partial TypedDict overrides."""
    # Assuming MapConfig takes these arguments based on the TypedDict definition
    default = MapConfig(min_distance=1.0, interp_distance=2.0, include_map=True)
    overrides: MapConfigDict = {"include_map": False}

    result = resolve_map_config(default, overrides)

    assert isinstance(result, MapConfig)
    assert result.include_map is False
    assert abs(result.min_distance - 1.0) < 1e-6


def test_resolve_loader_config_nested_dict_override() -> None:
    """Resolve loader configuration prioritizing deeply nested dictionary overrides."""
    default = LoaderConfig(
        input_len=10,
        output_len=20,
        sample_time=0.1,
        resampling=Resampling(up=1, down=1, method=ResamplingMethod.FAST),
    )
    overrides: LoaderConfigDict = {"output_len": 50, "resampling": {"method": "spline"}}

    result = resolve_loader_config(default, overrides)
    assert result.resampling is not None

    assert isinstance(result, LoaderConfig)
    assert result.output_len == 50
    assert result.input_len == 10
    # The nested dict should deep-merge, keeping 'up' and 'down' but changing 'method'
    assert result.resampling.model_dump() == {"up": 1, "down": 1, "method": "spline"}


def test_resolve_config_full_model() -> None:
    """Resolve the root Config object combining multiple nested sub-configurations."""
    default = Config(
        loader=LoaderConfig(input_len=5, output_len=5, sample_time=0.5),
        map=MapConfig(min_distance=5.0, interp_distance=5.0, include_map=True),
        execution=ExecutionConfig(parallel=False, workers=1),
    )

    overrides: ConfigDict = {"loader": {"input_len": 100}, "execution": {"parallel": True}}

    result = resolve_config(default, overrides)

    assert isinstance(result, Config)
    assert result.loader.input_len == 100
    assert result.loader.output_len == 5  # Preserved
    assert result.execution.parallel is True
    assert result.map.include_map is True  # Untouched sub-model
