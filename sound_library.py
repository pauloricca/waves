from __future__ import annotations
from typing import Dict
from pydantic import RootModel, model_validator, ValidationError
import yaml
import glob
import os
from dataclasses import dataclass


try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover - fallback when LibYAML unavailable
    YAML_LOADER = yaml.SafeLoader

from nodes.node_utils.base_node import BaseNodeModel


# Dict mapping filename -> SoundLibraryModel
sound_libraries: Dict[str, SoundLibraryModel] = {}
# Flat dict of all sound names -> (filename, model) for quick lookup
sound_index: Dict[str, tuple[str, BaseNodeModel]] = {}
# Temporary storage of raw YAML data during multi-file loading (for cross-file reference resolution)
_raw_data_cache: Dict[str, dict] = {}


@dataclass
class _LibraryFileState:
    mtime: float
    raw_data: dict
    user_vars: dict | None = None


_library_state: Dict[str, _LibraryFileState] = {}


def format_validation_error(error: ValidationError, filename: str, raw_data: dict = None) -> str:
    """
    Format a Pydantic ValidationError into a clean, readable error message.
    
    Args:
        error: The ValidationError from Pydantic
        filename: The YAML file where the error occurred
        raw_data: The raw YAML data (to extract sound and node names)
    
    Returns:
        A formatted error string
    """
    lines = [f"\n{'='*60}"]
    lines.append(f"Validation Error in {filename}")
    lines.append('='*60)
    
    # Group errors by location path to avoid showing duplicate errors for Union types
    # For Union types, Pydantic creates one error per union member with type names in the path
    # We need to filter out type names like 'str', 'int', 'float', etc.
    errors_by_location = {}
    union_type_names = {'str', 'int', 'float', 'list', 'dict', 'bool', 'none', 'NoneType'}
    
    for err in error.errors():
        # Create a location key excluding union type names
        loc_key = tuple(loc for loc in err['loc'] if str(loc) not in union_type_names)
        
        # Only keep the first error for each unique location
        if loc_key not in errors_by_location:
            errors_by_location[loc_key] = err
    
    for i, (loc_key, err) in enumerate(errors_by_location.items()):
        if i > 0:
            lines.append('-'*60)
            
        location = err['loc']
        msg = err['msg']
        input_value = err.get('input', '')
        
        # Try to find which sound this error belongs to
        sound_name = None
        if raw_data and location:
            # Check if first location element is a sound name
            first = str(location[0])
            if first in raw_data:
                sound_name = first
            else:
                # Try to find by traversing the input value in raw_data
                for snd_name, snd_data in raw_data.items():
                    if _value_exists_in_dict(snd_data, input_value):
                        sound_name = snd_name
                        break
        
        # Extract path (excluding sound name, special keys, and union type names)
        path_parts = []
        for loc in location:
            loc_str = str(loc)
            if loc_str not in ['__root__', 'root'] and loc_str != sound_name and loc_str not in union_type_names:
                path_parts.append(loc_str)
        
        # Display the error
        if sound_name:
            lines.append(f"Sound: '{sound_name}'")
        
        if path_parts:
            lines.append(f"Location: {' → '.join(path_parts)}")
        
        # Simplify and improve error messages
        if 'Input should be a valid dictionary or instance of BaseNodeModel' in msg:
            clean_msg = "Expected a node definition (e.g., {'node_type': {...}})"
            if isinstance(input_value, str) and input_value.startswith('$'):
                clean_msg += f"\nNote: ${input_value[1:]} syntax only works in expressions, not as direct node parameters"
                clean_msg += "\nTo reference a node output, use: reference: {ref: " + input_value[1:] + "}"
        else:
            clean_msg = msg
        
        lines.append(f"Error: {clean_msg}")
        
        # Show the problematic value
        if input_value is not None and len(str(input_value)) < 200:
            lines.append(f"Got: {repr(input_value)}")
    
    lines.append('='*60)
    return '\n'.join(lines)


def _value_exists_in_dict(d, value, max_depth=10):
    """Helper to check if a value exists somewhere in a nested dict"""
    if max_depth <= 0:
        return False
    if not isinstance(d, dict):
        return d == value
    for v in d.values():
        if v == value:
            return True
        if isinstance(v, dict) and _value_exists_in_dict(v, value, max_depth - 1):
            return True
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and _value_exists_in_dict(item, value, max_depth - 1):
                    return True
    return False


def parse_node(data, available_sound_names=None, raw_sound_data=None, all_sound_names=None) -> BaseNodeModel:
    from nodes.node_utils.node_registry import NODE_REGISTRY
    from nodes.node_utils.node_string_parser import apply_params_to_model
    
    # Helper to check if a dict looks like a standard node definition
    def looks_like_standard_node(v):
        return isinstance(v, dict) and len(v) == 1 and isinstance(next(iter(v.values())), dict)
    
    # Helper to restructure dict for nodes with extra='allow' (select, context, expression, etc.)
    def restructure_if_needed(v):
        """
        If dict has multiple keys but one matches a node type, restructure it.
        Input:  {'select': {'test': 'note'}, '55': {...}, '57': {...}}
        Output: {'select': {'test': 'note', '55': {...}, '57': {...}}}
        """
        if not isinstance(v, dict) or len(v) <= 1:
            return v
        
        # Find node type keys
        node_type_keys = [k for k in v.keys() if any(n.name == k for n in NODE_REGISTRY)]
        sound_type_keys = [k for k in v.keys() if (available_sound_names and k in available_sound_names) or (all_sound_names and k in all_sound_names)]
        
        # If exactly one key is a node type, restructure
        if len(node_type_keys) == 1:
            node_type_key = node_type_keys[0]
            node_params = v[node_type_key].copy() if isinstance(v[node_type_key], dict) else {}
            # Add all other keys as extra params
            for k in v.keys():
                if k != node_type_key:
                    node_params[k] = v[k]
            return {node_type_key: node_params}
        elif len(sound_type_keys) == 1:
            sound_type_key = sound_type_keys[0]
            sound_params = v[sound_type_key].copy() if isinstance(v[sound_type_key], dict) else {}
            # Add all other keys as extra params
            for k in v.keys():
                if k != sound_type_key:
                    sound_params[k] = v[k]
            return {sound_type_key: sound_params}
        
        return v
    
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Each node must have exactly one node_type key: {data}")
    
    node_type, params = next(iter(data.items()))
    
    # Recursively parse params that look like nodes
    for k, v in params.items():
        # First restructure if needed
        v_restructured = restructure_if_needed(v)
        
        if looks_like_standard_node(v_restructured):
            params[k] = parse_node(v_restructured, available_sound_names, raw_sound_data, all_sound_names)
        elif isinstance(v, list):
            params[k] = [
                parse_node(restructure_if_needed(i), available_sound_names, raw_sound_data, all_sound_names) 
                if looks_like_standard_node(restructure_if_needed(i)) else i
                for i in v
            ]
    
    node_definition = next((n for n in NODE_REGISTRY if n.name == node_type), None)

    if not node_definition:
        # Check if this is a sub-patch reference - first check local file, then all files
        if available_sound_names and node_type in available_sound_names and raw_sound_data:
            # Get the raw sound data and parse it
            sound_data = raw_sound_data[node_type]
            # Parse the sound to get its base model
            base_model = parse_node(sound_data, available_sound_names, raw_sound_data, all_sound_names)
            # Apply parameters to the model
            return apply_params_to_model(base_model, params)
        elif all_sound_names and node_type in all_sound_names:
            # Reference to sound in another file - get it from the sound index
            _, sound_model = sound_index[node_type]
            
            # If the model is still None (not yet parsed), we need to parse it from raw data
            if sound_model is None:
                # Find the raw data for this sound in the cache
                for filename, raw_data in _raw_data_cache.items():
                    if node_type in raw_data:
                        # Parse it now
                        try:
                            sound_data = {node_type: raw_data[node_type]}
                            temp_library = SoundLibraryModel.model_validate(sound_data)
                            sound_model = temp_library.root[node_type]
                            # Update the index with the parsed model
                            sound_index[node_type] = (filename, sound_model)
                            break
                        except ValidationError as e:
                            print(format_validation_error(e, filename, sound_data))
                            raise
                
                if sound_model is None:
                    raise ValueError(f"Could not find or parse sound '{node_type}' from cross-file reference")
            
            # Apply parameters to the model
            return apply_params_to_model(sound_model, params)
        else:
            raise ValueError(f"Node type '{node_type}' not recognised and not found in sound library")

    model_cls = node_definition.model

    return model_cls(**params)


class SoundLibraryModel(RootModel[Dict[str, BaseNodeModel]]):
    @model_validator(mode="before")
    @classmethod
    def parse_graph(cls, data):
        # Get list of all root-level sound names for sub-patch recognition
        available_sound_names = set(data.keys())
        # Get all sound names across all files for cross-file references
        all_sound_names = set(sound_index.keys())

        # Parse all sounds with knowledge of available sound names and raw data
        return {k: parse_node(v, available_sound_names, data, all_sound_names) for k, v in data.items()}

    def __getitem__(self, key):
        return self.root[key]

    def keys(self):
        return self.root.keys()


def _apply_user_variables(user_vars: dict | None) -> None:
    if user_vars:
        from expression_globals import set_user_variables
        set_user_variables(user_vars)


def _load_raw_data_for_file(file_path: str) -> tuple[dict, dict | None]:
    with open(file_path) as file:
        raw_data = yaml.load(file, Loader=YAML_LOADER)

    if raw_data is None:
        raw_data = {}

    user_vars = None
    if os.path.basename(file_path) == 'waves.yaml':
        user_vars = raw_data.pop('vars', None)
        _apply_user_variables(user_vars)

    return raw_data, user_vars


def load_yaml_file(file_path: str) -> SoundLibraryModel:
    """Load and parse a single YAML file."""
    raw_data, user_vars = _load_raw_data_for_file(file_path)

    try:
        library = SoundLibraryModel.model_validate(raw_data)

        _library_state[os.path.basename(file_path)] = _LibraryFileState(
            mtime=os.path.getmtime(file_path),
            raw_data=raw_data,
            user_vars=user_vars,
        )

        return library

    except ValidationError as e:
        print(format_validation_error(e, os.path.basename(file_path), raw_data))
        raise
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def load_all_sound_libraries(directory: str = ".") -> Dict[str, SoundLibraryModel]:
    """Load all .yaml files in the directory into the sound library."""
    global sound_libraries, sound_index, _raw_data_cache, _library_state

    yaml_paths = glob.glob(os.path.join(directory, "*.yaml"))
    if not yaml_paths:
        raise ValueError(f"No .yaml files found in {directory}")

    files_by_name = {os.path.basename(path): path for path in yaml_paths}
    load_order = sorted(files_by_name.keys())

    # Remove any state for files that disappeared
    removed_files = set(sound_libraries.keys()) - set(files_by_name.keys())
    for filename in removed_files:
        sound_libraries.pop(filename, None)
        _library_state.pop(filename, None)
    for sound_name in list(sound_index.keys()):
        if sound_index[sound_name][0] in removed_files:
            del sound_index[sound_name]

    # First pass: gather raw data for all files so we know every sound name up-front.
    _raw_data_cache.clear()
    raw_data_by_file: Dict[str, dict] = {}
    user_vars_by_file: Dict[str, dict | None] = {}
    mtimes: Dict[str, float] = {}
    needs_parse: Dict[str, bool] = {}

    for filename in load_order:
        file_path = files_by_name[filename]
        mtime = os.path.getmtime(file_path)
        state = _library_state.get(filename)

        if state and state.mtime == mtime and filename in sound_libraries:
            raw_data = state.raw_data
            user_vars = state.user_vars
            # Ensure user variables stay applied between loads
            _apply_user_variables(user_vars)
            needs_parse[filename] = False
        else:
            raw_data, user_vars = _load_raw_data_for_file(file_path)
            needs_parse[filename] = True

        raw_data = raw_data or {}
        raw_data_by_file[filename] = raw_data
        user_vars_by_file[filename] = user_vars
        mtimes[filename] = mtime
        _raw_data_cache[filename] = raw_data

    # Remove any stale index entries for files that will be reparsed
    previous_entries: Dict[str, Dict[str, tuple[str, BaseNodeModel]]] = {}
    for sound_name in list(sound_index.keys()):
        filename, model = sound_index[sound_name]
        if needs_parse.get(filename):
            previous_entries.setdefault(filename, {})[sound_name] = (filename, model)
            del sound_index[sound_name]

    # Pre-populate the index with placeholders so cross-file references resolve
    for filename in load_order:
        raw_data = raw_data_by_file[filename]
        for sound_name in raw_data.keys():
            sound_index.setdefault(sound_name, (filename, None))

    # Second pass: validate and build models now that all names are known
    for filename in load_order:
        raw_data = raw_data_by_file[filename]
        try:
            if not needs_parse.get(filename, True):
                library = sound_libraries.get(filename)
                if library is None:
                    library = SoundLibraryModel.model_validate(raw_data)
                    sound_libraries[filename] = library
                    _library_state[filename] = _LibraryFileState(
                        mtime=mtimes[filename],
                        raw_data=raw_data,
                        user_vars=user_vars_by_file[filename],
                    )
            else:
                library = SoundLibraryModel.model_validate(raw_data)
                sound_libraries[filename] = library
                _library_state[filename] = _LibraryFileState(
                    mtime=mtimes[filename],
                    raw_data=raw_data,
                    user_vars=user_vars_by_file[filename],
                )

            # Remove sounds that no longer exist in the file
            for sound_name in list(sound_index.keys()):
                if sound_index[sound_name][0] == filename and sound_name not in raw_data:
                    del sound_index[sound_name]

            for sound_name, sound_model in library.root.items():
                existing = sound_index.get(sound_name)
                if existing and existing[0] != filename:
                    print(
                        f"Warning: Sound '{sound_name}' defined in multiple files. Using definition from {filename}"
                    )
                sound_index[sound_name] = (filename, sound_model)

            print(f"Loaded {len(library.root)} sound(s) from {filename}")
        except ValidationError as e:
            print(format_validation_error(e, filename, raw_data))
            # Restore previous index entries if validation fails
            for sound_name, entry in previous_entries.get(filename, {}).items():
                sound_index[sound_name] = entry
            for sound_name in list(sound_index.keys()):
                if sound_index[sound_name][0] == filename and sound_name not in previous_entries.get(filename, {}):
                    del sound_index[sound_name]
            return False
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            for sound_name, entry in previous_entries.get(filename, {}).items():
                sound_index[sound_name] = entry
            for sound_name in list(sound_index.keys()):
                if sound_index[sound_name][0] == filename and sound_name not in previous_entries.get(filename, {}):
                    del sound_index[sound_name]
            return False

    _raw_data_cache.clear()

    return sound_libraries


def reload_sound_library(filename: str, directory: str = ".") -> bool:
    """Reload a single YAML file and update the sound library."""
    global sound_libraries, sound_index, _raw_data_cache, _library_state

    file_path = os.path.join(directory, filename)

    if not os.path.exists(file_path):
        print(f"File {filename} not found")
        return False
    
    # Preserve old state in case reload fails
    old_library = sound_libraries.get(filename)
    old_sound_index_entries = {}
    if old_library:
        for sound_name in old_library.root.keys():
            if sound_name in sound_index and sound_index[sound_name][0] == filename:
                old_sound_index_entries[sound_name] = sound_index[sound_name]
    
    try:
        _raw_data_cache.clear()
        for existing_filename, state in _library_state.items():
            if existing_filename != filename:
                _raw_data_cache[existing_filename] = state.raw_data

        raw_data, user_vars = _load_raw_data_for_file(file_path)
        _raw_data_cache[filename] = raw_data

        try:
            library = SoundLibraryModel.model_validate(raw_data)
        except ValidationError as e:
            print(format_validation_error(e, filename, raw_data))
            raise

        for sound_name in old_sound_index_entries.keys():
            del sound_index[sound_name]

        sound_libraries[filename] = library
        for sound_name, sound_model in library.root.items():
            sound_index[sound_name] = (filename, sound_model)

        _library_state[filename] = _LibraryFileState(mtime=os.path.getmtime(file_path), raw_data=raw_data, user_vars=user_vars)

        _raw_data_cache.clear()
        return True

    except Exception as e:
        print(f"Error reloading {filename}: {e}")
        _raw_data_cache.clear()
        return False


def get_sound_model(sound_name: str) -> BaseNodeModel:
    """Get a sound model by name from any loaded library."""
    if not sound_index:
        raise ValueError("Sound library not loaded. Please load the sound library first.")
    
    if sound_name not in sound_index:
        raise ValueError(f"Sound '{sound_name}' not found in any loaded YAML file.")
    
    _, sound_model = sound_index[sound_name]
    return sound_model


def get_sound_filename(sound_name: str) -> str:
    """Get the filename that contains a sound definition."""
    if sound_name not in sound_index:
        raise ValueError(f"Sound '{sound_name}' not found in any loaded YAML file.")
    
    filename, _ = sound_index[sound_name]
    return filename