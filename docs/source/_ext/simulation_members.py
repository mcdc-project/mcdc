"""Sphinx directive for documenting Simulation-owned configuration interfaces."""

from __future__ import annotations

import inspect
from dataclasses import fields

from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.ext.napoleon.docstring import NumpyDocstring
from sphinx.pycode import ModuleAnalyzer
from sphinx.util.docutils import SphinxDirective

from mcdc.object_.settings import Settings
from mcdc.object_.technique import (
    GlobalWeightRoulette,
    ImplicitCapture,
    PopulationControl,
    WeightedEmission,
    WeightWindows,
)

DIRECT_SETTINGS = (
    "N_particle",
    "N_batch",
    "rng_seed",
    "time_boundary",
    "output_name",
    "use_progress_bar",
    "active_bank_buffer",
    "census_bank_buffer_ratio",
    "source_bank_buffer_ratio",
    "future_bank_buffer_ratio",
)

SETTINGS_METHODS = (
    "set_time_census",
    "set_eigenmode",
    # TODO: Restore file-backed source setup after its compilation path is enabled.
    # "set_source_file",
)

TECHNIQUES = (
    ("implicit_capture", ImplicitCapture),
    ("weighted_emission", WeightedEmission),
    ("global_weight_roulette", GlobalWeightRoulette),
    ("weight_windows", WeightWindows),
    ("population_control", PopulationControl),
)


def setup(app):
    app.add_directive("simulation-members", SimulationMembersDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


class SimulationMembersDirective(SphinxDirective):
    """Render internal configuration docstrings as public Simulation members."""

    has_content = False

    def run(self):
        lines = ViewList()
        settings_fields = {field.name: field for field in fields(Settings)}
        attribute_docs = ModuleAnalyzer.for_module(Settings.__module__).find_attr_docs()

        _append_line(lines, ".. rubric:: Settings")
        _append_line(lines)

        for name in DIRECT_SETTINGS:
            field = settings_fields[name]
            _append_line(lines, f".. py:attribute:: settings.{name}")
            _append_line(lines, "   :module:")
            _append_line(
                lines,
                f"   :canonical: mcdc.Simulation.settings.{name}",
            )
            _append_line(lines, f"   :type: {_public_type(field.type)}")
            _append_line(lines, f"   :value: {field.default!r}")
            _append_line(lines)
            _append_content(
                lines,
                attribute_docs[("Settings", name)],
                indent="   ",
            )
            _append_line(lines)

        for name in SETTINGS_METHODS:
            method = getattr(Settings, name)
            public_name = f"Simulation.settings.{name}"
            _append_line(
                lines,
                f".. py:method:: settings.{name}{_public_signature(method)}",
            )
            _append_line(lines, "   :module:")
            _append_line(
                lines,
                f"   :canonical: mcdc.Simulation.settings.{name}",
            )
            _append_line(lines)
            _append_content(
                lines,
                _converted_docstring(self, method, public_name),
                indent="   ",
            )
            _append_line(lines)

        _append_line(lines, ".. rubric:: Particle transport settings")
        _append_line(lines)
        _append_line(
            lines,
            "All species start inactive. Finalization activates the particle types present in the sources and preserves explicitly enabled types. Lower-energy-first transport is currently supported only for electrons, where it is enabled by default.",
        )
        _append_line(lines)

        for species in ("neutron", "electron", "proton", "photon"):
            for option, default, description in (
                ("active", False, "Enable transport for this species."),
                (
                    "prioritize_low_energy",
                    species == "electron",
                    "Request lower-energy outgoing particles to be transported first.",
                ),
            ):
                name = f"settings.{species}_transport.{option}"
                _append_line(lines, f".. py:attribute:: {name}")
                _append_line(lines, "   :module:")
                _append_line(lines, f"   :canonical: mcdc.Simulation.{name}")
                _append_line(lines, "   :type: bool")
                _append_line(lines, f"   :value: {default!r}")
                _append_line(lines)
                _append_line(lines, f"   {description}")
                _append_line(lines)

        _append_line(lines, ".. rubric:: Transport techniques")
        _append_line(lines)

        for name, technique in TECHNIQUES:
            method = technique.__call__
            public_name = f"Simulation.technique.{name}"
            _append_line(
                lines,
                f".. py:method:: technique.{name}{_public_signature(method)}",
            )
            _append_line(lines, "   :module:")
            _append_line(
                lines,
                f"   :canonical: mcdc.Simulation.technique.{name}",
            )
            _append_line(lines)
            _append_content(
                lines,
                _converted_docstring(self, method, public_name),
                indent="   ",
            )
            _append_line(lines)

        container = nodes.container(classes=["simulation-members"])
        self.state.nested_parse(lines, self.content_offset, container)
        return [container]


def _append_line(lines: ViewList, line: str = "") -> None:
    lines.append(line, "<simulation-members>")


def _append_content(lines: ViewList, content: list[str], indent: str = "") -> None:
    for line in content:
        _append_line(lines, f"{indent}{line}" if line else "")


def _converted_docstring(
    directive: SphinxDirective, obj, public_name: str
) -> list[str]:
    docstring = inspect.getdoc(obj) or ""
    return str(
        NumpyDocstring(
            docstring,
            config=directive.env.config,
            app=directive.env.app,
            what="method",
            name=public_name,
            obj=obj,
        )
    ).splitlines()


def _public_signature(function) -> str:
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())[1:]
    return str(
        signature.replace(
            parameters=parameters,
            return_annotation=inspect.Signature.empty,
        )
    )


def _public_type(annotation) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).removeprefix("typing.")
