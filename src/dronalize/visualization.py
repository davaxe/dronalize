"""Visualization utilities.

This module re-exports selected plotting and visualization symbols from the
``dronalize-viz`` package for convenient access through the main package API.

This module requires the visualization extra to be installed, for example:

    pip install dronalize[viz]

!!! tip "Comming soon"
    This module is currently a placeholder and will be populated with visualization
    utilities in a future release.

!!! warning "Not yet implemented"
    This module might be renamed or restructured in the future as the
    visualization utilities are developed.
"""

from dronalize.core.optional import raise_missing_optional_dependency

try:
    ...
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="Visualization utilities", extra="viz")
