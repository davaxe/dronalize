from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

ResampleMethod: TypeAlias = Literal["linear", "cubic", "pchip"]


class ConfigBase(BaseModel):
    """Shared base class for all config models."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    @override
    def __repr_args__(self) -> Iterator[tuple[str | None, object]]:
        for name, field in super().__repr_args__():
            if field is not None:
                yield name, field


class ResolvedConfig(ConfigBase):
    """Full configuration with all fields required and defaults applied."""


ResolvedConfigT = TypeVar("ResolvedConfigT", bound=ConfigBase)


class ConfigPatch(BaseModel, Generic[ResolvedConfigT]):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    full_config_type: type[ResolvedConfigT]

    def merge_into(
        self, target: ResolvedConfigT | None, *, exclude_none: bool = True
    ) -> ResolvedConfigT:
        """Apply partial config to a full config.

        By passing `None` as the target, the partial config can be applied to a
        default full config with all non-required fields set to their default
        values. This allows converting a valid partial config into a full
        config, but will raise errors if the partial config is missing any
        required fields.

        Parameters
        ----------
        target : ResolvedConfigT or None
            Full config to apply the partial config to. If `None`, the partial
            config is applied to a default full config with all non-required
            fields set to their default values.
        exclude_none : bool, optional
            Whether to exclude fields with `None` values from the patch. This is
            passed to `BaseModel.model_dump()` when creating the patch from the
            partial config,

        Returns
        -------
        ResolvedConfigT
            Result of applying the partial config to the target full config.

        """
        if target is None:
            base = {
                name: field.get_default()
                for name, field in self.full_config_type.model_fields.items()
                if not field.is_required()
            }
        else:
            base = target.model_dump()
        patch = self.model_dump(exclude_unset=True, exclude_none=exclude_none)
        merged = deep_merge(base, patch)
        return self.full_config_type.model_validate(merged)

    @classmethod
    def from_full_config(cls, full_config: ResolvedConfigT) -> ConfigPatch[ResolvedConfigT]:
        """Create a ConfigPatch instance from a ResolvedConfig instance."""
        data = full_config.model_dump()
        return cls.model_validate(data)


def deep_merge(
    base: MutableMapping[str, object], patch: Mapping[str, object]
) -> MutableMapping[str, object]:
    """Recursively merge two mappings."""
    for key, patch_value in patch.items():
        base_value = base.get(key)

        if isinstance(base_value, MutableMapping) and isinstance(patch_value, Mapping):
            _ = deep_merge(
                cast("MutableMapping[str, object]", base_value),
                cast("Mapping[str, object]", patch_value),
            )
        else:
            base[key] = copy.deepcopy(patch_value)

    return base


TargetT = TypeVar("TargetT", bound=ConfigBase)


def apply_optional(
    patch: ConfigPatch[TargetT] | Literal[False] | None, target: TargetT | None
) -> TargetT | None:
    """Apply a patch to an optional config block."""
    if patch is None:
        return target
    if patch is False:
        return None
    return patch.merge_into(target)
