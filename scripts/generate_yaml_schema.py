#!/usr/bin/env python3
"""Generate JSON schema for waves YAML files from node model typings."""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Some node imports indirectly load optional runtime audio deps (e.g. PortAudio).
# Stub them so schema generation remains a pure static/dev task.
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

from nodes.node_utils.node_registry import NODE_REGISTRY
from nodes.node_utils.base_node import BaseNodeModel


COMMENT_SPLIT = "#"
SOUNDS_DIR = PROJECT_ROOT / "sounds"

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover - fallback when LibYAML unavailable
    YAML_LOADER = yaml.SafeLoader


def _extract_model_field_comments(model_cls: type) -> dict[str, str]:
    """Extract inline/preceding comments for model fields from source code."""
    source_file = inspect.getsourcefile(model_cls)
    if not source_file:
        return {}

    path = Path(source_file)
    lines = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines), filename=str(path))

    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == model_cls.__name__
        ),
        None,
    )
    if class_node is None:
        return {}

    comments: dict[str, str] = {}

    for node in class_node.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue

        field_name = node.target.id
        lineno = node.lineno

        inline_comment = ""
        line_text = lines[lineno - 1]
        if COMMENT_SPLIT in line_text:
            inline_comment = line_text.split(COMMENT_SPLIT, 1)[1].strip()

        preceding_comment = ""
        cursor = lineno - 2
        chunks: list[str] = []
        while cursor >= 0:
            candidate = lines[cursor].strip()
            if candidate.startswith("#"):
                chunks.append(candidate.lstrip("#").strip())
                cursor -= 1
                continue
            if candidate == "":
                cursor -= 1
                continue
            break

        if chunks:
            preceding_comment = " ".join(reversed([chunk for chunk in chunks if chunk]))

        description = inline_comment or preceding_comment
        if description:
            comments[field_name] = description

    return comments


def _clean_model_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a schema fragment suitable as a reusable model definition."""
    out = dict(schema)
    out.pop("$defs", None)
    out.pop("title", None)
    return out


def _public_field_name(name: str) -> str:
    if name.endswith("_") and name[:-1]:
        return name[:-1]
    return name


def _model_allows_extra(model_cls: type) -> bool:
    return model_cls.model_config.get("extra") == "allow"


def _model_public_field_names(model_cls: type) -> set[str]:
    return {_public_field_name(name) for name in model_cls.model_fields}


def _load_wavable_value_schema(defs: dict[str, Any]) -> dict[str, Any]:
    from pydantic import TypeAdapter
    from nodes.wavable_value import WavableValue

    wavable_schema = TypeAdapter(WavableValue).json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
    )
    for def_name, def_schema in wavable_schema.get("$defs", {}).items():
        sanitized_def = _sanitize_for_json_schema(def_schema)
        if def_name in defs and defs[def_name] != sanitized_def:
            continue
        defs[def_name] = sanitized_def

    sanitized_schema = _sanitize_for_json_schema(wavable_schema)
    sanitized_schema.pop("$defs", None)
    sanitized_schema.pop("title", None)
    return sanitized_schema


def _iter_sound_yaml_paths() -> list[Path]:
    return sorted(SOUNDS_DIR.rglob("*.yaml"))


def _load_sound_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw_data = yaml.load(handle, Loader=YAML_LOADER)

    if raw_data is None:
        return {}
    if not isinstance(raw_data, dict):
        return {}

    if path.name == "waves.yaml":
        raw_data = dict(raw_data)
        raw_data.pop("vars", None)

    return raw_data


def _extract_reusable_sound_schemas(
    defs: dict[str, Any],
    base_properties: dict[str, Any],
    wavable_value_schema: dict[str, Any],
    node_properties: dict[str, Any],
    node_snippets: list[dict[str, Any]],
) -> None:
    registry_by_name = {definition.name: definition for definition in NODE_REGISTRY}
    base_field_names = _model_public_field_names(BaseNodeModel)

    for path in _iter_sound_yaml_paths():
        raw_data = _load_sound_yaml(path)
        for sound_name, node_spec in raw_data.items():
            if not isinstance(node_spec, dict) or len(node_spec) != 1:
                continue

            root_node_name, root_params = next(iter(node_spec.items()))
            if not isinstance(root_params, dict) or not root_params.get("is_reusable"):
                continue
            if sound_name in registry_by_name or sound_name in node_properties:
                continue

            definition = registry_by_name.get(root_node_name)
            if definition is None:
                continue

            params_def_name = f"NodeParams_reusable_{sound_name}"
            base_params_def_name = f"NodeParams_{root_node_name}"
            base_params_schema = defs.get(base_params_def_name)
            if not isinstance(base_params_schema, dict):
                continue

            reusable_schema = copy.deepcopy(base_params_schema)
            reusable_schema["description"] = f"Reusable YAML node from {path.name}."
            reusable_schema.pop("required", None)

            reusable_properties = reusable_schema.setdefault("properties", {})
            reusable_properties.pop("is_reusable", None)

            if _model_allows_extra(definition.model):
                explicit_fields = _model_public_field_names(definition.model)
                inferred_param_names = [
                    key
                    for key in root_params
                    if key not in explicit_fields and key not in base_field_names
                ]

                for key in inferred_param_names:
                    reusable_properties[key] = copy.deepcopy(wavable_value_schema)

                if "input_signal" in reusable_properties and "signal" not in reusable_properties:
                    reusable_properties["signal"] = copy.deepcopy(wavable_value_schema)
                    reusable_properties["signal"]["description"] = (
                        "Alias for input_signal when instantiating the reusable node."
                    )

            defs[params_def_name] = reusable_schema

            node_properties[sound_name] = {"$ref": f"#/$defs/{params_def_name}"}
            node_snippets.append(
                {
                    "label": sound_name,
                    "description": f"Insert reusable node from {path.name}",
                    "body": {
                        sound_name: {},
                    },
                }
            )


def _sanitize_for_json_schema(value: Any) -> Any:
    """Strip non-JSON values that Pydantic may emit in schemas."""
    if isinstance(value, dict):
        sanitized = {key: _sanitize_for_json_schema(item) for key, item in value.items()}
        if sanitized.get("$ref") == "#/$defs/BaseNodeModel":
            sanitized["$ref"] = "#/$defs/Node"
        any_of = sanitized.get("anyOf")
        if isinstance(any_of, list):
            node_options = [
                item for item in any_of
                if isinstance(item, dict) and item.get("$ref") == "#/$defs/Node"
            ]
            other_options = [
                item for item in any_of
                if not (isinstance(item, dict) and item.get("$ref") == "#/$defs/Node")
            ]
            if node_options:
                sanitized["anyOf"] = node_options + other_options
        properties = sanitized.get("properties")
        if isinstance(properties, dict):
            renamed_properties: dict[str, Any] = {}
            rename_map: dict[str, str] = {}

            for key, item in properties.items():
                if key.endswith("_") and key[:-1] and key[:-1] not in properties:
                    public_name = key[:-1]
                    rename_map[key] = public_name
                    renamed_properties[public_name] = item
                else:
                    renamed_properties[key] = item

            sanitized["properties"] = renamed_properties

            required = sanitized.get("required")
            if isinstance(required, list):
                sanitized["required"] = [rename_map.get(name, name) for name in required]

        default = sanitized.get("default")
        if isinstance(default, float) and not math.isfinite(default):
            sanitized.pop("default")
            note = f"Default: {default!r}."
            description = sanitized.get("description")
            if description:
                if note not in description:
                    sanitized["description"] = f"{description} {note}"
            else:
                sanitized["description"] = note
        elif "default" in sanitized:
            sanitized.pop("default")
        return sanitized

    if isinstance(value, list):
        return [_sanitize_for_json_schema(item) for item in value]

    return value


def generate_schema() -> dict[str, Any]:
    defs: dict[str, Any] = {}
    node_properties: dict[str, Any] = {}
    node_snippets: list[dict[str, Any]] = []
    wavable_value_schema = _load_wavable_value_schema(defs)

    for definition in NODE_REGISTRY:
        model_cls = definition.model
        model_schema = model_cls.model_json_schema(mode="validation", ref_template="#/$defs/{model}")

        for def_name, def_schema in model_schema.get("$defs", {}).items():
            def_schema = _sanitize_for_json_schema(def_schema)
            if def_name in defs and defs[def_name] != def_schema:
                # Keep the first definition to avoid ref churn across models.
                continue
            defs[def_name] = def_schema

        params_schema = _sanitize_for_json_schema(_clean_model_schema(model_schema))

        field_comments = _extract_model_field_comments(model_cls)
        properties = params_schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if "description" not in field_schema and field_name in field_comments:
                field_schema["description"] = field_comments[field_name]

        if _model_allows_extra(model_cls):
            params_schema["additionalProperties"] = copy.deepcopy(wavable_value_schema)

        params_def_name = f"NodeParams_{definition.name}"
        defs[params_def_name] = params_schema

        node_variant_name = f"NodeVariant_{definition.name}"
        defs[node_variant_name] = {
            "type": "object",
            "properties": {
                definition.name: {"$ref": f"#/$defs/{params_def_name}"},
            },
            "required": [definition.name],
            "additionalProperties": False,
            "description": f"{definition.name} node",
        }
        node_properties[definition.name] = {"$ref": f"#/$defs/{params_def_name}"}
        node_snippets.append(
            {
                "label": definition.name,
                "description": f"Insert a {definition.name} node",
                "body": {
                    definition.name: {},
                },
            }
        )

    base_properties = defs.get("BaseNodeModel", {}).get("properties", {})
    _extract_reusable_sound_schemas(
        defs=defs,
        base_properties=base_properties,
        wavable_value_schema=wavable_value_schema,
        node_properties=node_properties,
        node_snippets=node_snippets,
    )

    defs["Node"] = {
        "type": "object",
        "properties": node_properties,
        "additionalProperties": False,
        "minProperties": 1,
        "maxProperties": 1,
        "description": "A single node definition keyed by node name.",
        "defaultSnippets": node_snippets,
    }

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/waves.schema.json",
        "title": "Waves sound library schema",
        "type": "object",
        "properties": {
            "vars": {
                "type": "object",
                "description": "Optional global user variables (used in waves.yaml).",
                "additionalProperties": True,
            },
        },
        "patternProperties": {
            "^(?!vars$)[A-Za-z_][A-Za-z0-9_-]*$": {"$ref": "#/$defs/Node"},
        },
        "additionalProperties": False,
        "$defs": defs,
    }

    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSON schema for waves YAML files.")
    parser.add_argument(
        "--output",
        default="schemas/waves.schema.json",
        help="Output path for generated JSON schema.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = generate_schema()
    output_path.write_text(
        json.dumps(schema, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated schema at {output_path}")


if __name__ == "__main__":
    main()
