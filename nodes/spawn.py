from __future__ import annotations
import math
import numpy as np
from pydantic import ConfigDict

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import detect_triggers


class SpawnModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    trigger: WavableValue  # Signal to watch for 0→1 crossings (e.g., beat_tick from tempo node)
    signal: WavableValue  # The sound/signal to spawn on each trigger
    voices: int = 32  # Maximum number of simultaneous spawned instances
    threshold: float = 0.5  # Threshold for detecting 0→1 crossings
    duration: float = math.inf


class SpawnNode(BaseNode):
    def __init__(self, model: SpawnModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.trigger_model = model.trigger
        self.signal_model = model.signal
        self.max_voices = model.voices
        self.threshold = model.threshold
        
        # Persistent state for active spawned instances (survives hot reload)
        # We store only the data, not the node instances - nodes are ephemeral
        if do_initialise_state:
            self.state.active_instances = {}  # {instance_id: {render_args, samples_rendered}}
            self.state.instance_id_counter = 0
            self.state.last_trigger_value = 0.0  # Track previous trigger value for edge detection
        
        # Ephemeral: Node instances for active spawned sounds - recreated from state on hot reload
        # Maps instance_id -> sound_node instance
        self.active_instance_nodes = {}
        
        # Recreate node instances from state (for hot reload)
        for instance_id in self.state.active_instances.keys():
            sound_node = self.instantiate_child_node(self.signal_model, "instances", instance_id)
            self.active_instance_nodes[instance_id] = sound_node
        
        # Instantiate trigger node (not spawned, always rendered once per chunk)
        self.trigger_node = self.instantiate_child_node(self.trigger_model, "trigger")
    
    def _spawn_instance(self, trigger_sample_offset: int, **params):
        """
        Spawn a new instance of the signal.
        
        Args:
            trigger_sample_offset: Sample offset within current chunk where trigger occurred
            **params: Parameters to pass to the spawned signal
        """
        # Voice stealing: if we've reached max voices, remove the oldest instance
        if len(self.state.active_instances) >= self.max_voices:
            # Find the oldest instance (lowest instance_id)
            oldest_id = min(self.state.active_instances.keys())
            
            # Remove it from state
            del self.state.active_instances[oldest_id]
            
            # Remove from ephemeral nodes
            if oldest_id in self.active_instance_nodes:
                del self.active_instance_nodes[oldest_id]
        
        # Generate unique ID for this instance
        instance_id = self.state.instance_id_counter
        self.state.instance_id_counter += 1
        
        # Create a new instance of the signal node (ephemeral)
        sound_node = self.instantiate_child_node(self.signal_model, "instances", instance_id)
        self.active_instance_nodes[instance_id] = sound_node
        
        # Setup render args - pass gate=1 initially (sustaining)
        render_args = {
            'gate': 1.0,
            'trigger_offset': trigger_sample_offset,  # Sample offset where trigger occurred
        }
        
        # Store the active instance data (persistent state - no node objects)
        self.state.active_instances[instance_id] = {
            'render_args': render_args,
            'samples_rendered': 0,
            'trigger_offset': trigger_sample_offset,  # Track where in the chunk this started
        }
    
    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Spawn node continues indefinitely if num_samples is None
        if num_samples is None:
            from config import BUFFER_SIZE
            num_samples = BUFFER_SIZE
            self._last_chunk_samples = num_samples
        
        # Render the trigger signal
        trigger_wave = self.trigger_node.render(num_samples, context, **self.get_params_for_children(params))
        
        # Ensure trigger_wave is the right length
        if len(trigger_wave) < num_samples:
            trigger_wave = np.pad(trigger_wave, (0, num_samples - len(trigger_wave)))
        elif len(trigger_wave) > num_samples:
            trigger_wave = trigger_wave[:num_samples]
        
        # Detect trigger events (0→1 crossings)
        trigger_indices, self.state.last_trigger_value = detect_triggers(
            trigger_wave, 
            self.state.last_trigger_value, 
            self.threshold
        )
        
        # Spawn new instances for each trigger
        for trigger_idx in trigger_indices:
            self._spawn_instance(trigger_idx, **params)
        
        # Create output buffer
        output_wave = np.zeros(num_samples, dtype=np.float32)
        
        # Render all active instances and mix them together
        instances_to_remove = []
        
        for instance_id, instance_data in self.state.active_instances.items():
            render_args = instance_data['render_args']
            samples_rendered = instance_data['samples_rendered']
            trigger_offset = instance_data['trigger_offset']
            
            # Get the node instance from the ephemeral dictionary
            sound_node = self.active_instance_nodes.get(instance_id)
            if sound_node is None:
                # Node instance doesn't exist (shouldn't happen, but handle gracefully)
                instances_to_remove.append(instance_id)
                continue
            
            # Merge render_args with params
            merged_params = self.get_params_for_children(params)
            merged_params.update(render_args)
            
            # For instances spawned in this chunk, only render from their trigger offset onwards
            if samples_rendered == 0 and trigger_offset > 0:
                # This instance was just spawned mid-chunk
                samples_to_render = num_samples - trigger_offset
                instance_chunk = sound_node.render(samples_to_render, context, **merged_params)
                
                # If the instance returns an empty array, it has finished
                if len(instance_chunk) == 0:
                    instances_to_remove.append(instance_id)
                    continue
                
                # Update samples rendered counter
                instance_data['samples_rendered'] += len(instance_chunk)
                
                # Mix into output at the correct offset
                output_wave[trigger_offset:trigger_offset + len(instance_chunk)] += instance_chunk
                
                # Reset trigger offset so next chunk renders from the start
                instance_data['trigger_offset'] = 0
            else:
                # Normal rendering for instances that started in previous chunks
                instance_chunk = sound_node.render(num_samples, context, **merged_params)
                
                # If the instance returns an empty array, it has finished
                if len(instance_chunk) == 0:
                    instances_to_remove.append(instance_id)
                    continue
                
                # Update samples rendered counter
                instance_data['samples_rendered'] += len(instance_chunk)
                
                # Mix into output (pad if needed)
                if len(instance_chunk) < len(output_wave):
                    instance_chunk = np.pad(instance_chunk, (0, len(output_wave) - len(instance_chunk)))
                
                output_wave[:len(instance_chunk)] += instance_chunk
        
        # Remove finished instances
        for instance_id in instances_to_remove:
            if instance_id in self.state.active_instances:
                del self.state.active_instances[instance_id]
            
            # Also remove from ephemeral node instances
            if instance_id in self.active_instance_nodes:
                del self.active_instance_nodes[instance_id]
        
        return output_wave


SPAWN_DEFINITION = NodeDefinition("spawn", SpawnNode, SpawnModel)
