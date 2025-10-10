# GitHub Copilot Instructions

## Project Overview

This project is a python and numpy test bed for experimental sound design, to help me create sounds, sequences or compositions from scratch to be used in music production in flexible, low-level hands-on ways. The way sounds are created is through a kind of modular synthesis, where different modules (called nodes here) can be connected together procedurally to create complex sounds. The modules and connections are described in YAML format in the waves.yaml file.

There are different types of nodes, such as oscillators, filters, and effects, each with their own parameters and often parameters can be either set to scalar values or connected to the output of other nodes.

## Key Concepts

### Nodes

Nodes are objects that inherit from BaseNode (nodes/node_utils/base_node.py) and they can be instantiated by calling the instantiate_node function (nodes/node_utils/instantiate_node.py) using a definition from the yaml file, which gets translated into an object that inherits from BaseNodeModel (nodes/node_utils/base_node.py).

In short, a Node instance is defined in yaml, matching the signature of a NodeModel and is instantiated as a Node object at runtime.

A Node has a render method which takes in a number of samples to render and a set of parameters, and returns a numpy array of samples. The render method can use the parameters and the internal state of the node to generate what we call a wave. 

#### The render method

The render method is called multiple times (potentially in real time) to generate the final output wave. Each time it is called, it generates a chunk of samples (the size of which is defined by the num_samples parameter). We should take this into consideration when writing its implementation, we should take measures to ensure that this method is efficient and does not introduce artifacts in the output wave, for example as a result of value jumps between one chunk and the next. We might need to keep some internal state (or sometimes a buffer) in the node to ensure continuity between chunks.

The number of samples is optional. If not provided, the node should render the whole wave at once. This is useful for non-realtime rendering, where we want to generate the whole sound in one go. When the render function returns an empty array, it signals to the parent node that it has finished rendering and there is no more sound to be generated. This is useful for nodes that have a defined duration, such as an envelope or a sample player.

The render method received a params dictionary, which contains the parameters that are passed on by parent nodes and can be used by the node to generate the wave. In order to keep these params organised, we use the RenderArgs enum (constants.py) to define the keys that can be used in the params dictionary. This way, we can avoid hardcoding strings in the code and make it easier to understand what each parameter is for.

### Waves

Waves can be used as audio or as input for parameters of other nodes. In this sense, there is no destinction between audio and control (like in modular synths), they are all just waves. The objective is to create a unified representation of all types of signals and make the system very flexible.

### WavableValues

Some parameters of nodes can be fixed scalar values or waves – dynamic values that change over time. We call this a WavableValue (nodes/wavable_value.py). Waves are typically the output of a Node's render function but they can also be interpolated values, and in this case the parameter is a list of values to be used for interpolation. Each value can be a scalar or a list or two values, the first one is the value and the second one is the time it takes to get to that value, proportionally to the total time of the envelope. For example:
freq: [10000, [500, 0.01], [60, 0.02]] means that the frequency starts at 10000 Hz, then goes to 500 Hz in 1% of the total time of the envelope, then goes to 60 Hz in 2% of the total time of the envelope, and then stays at 60 Hz for the rest of the time. The first and last values can have their positions ommited (assumed to be  0 and 1 respectively).

## The YAML file

The yaml file (waves.yaml) is the main definitions file for the project. Each root level key identifies a different sound or sequence to be generated and the subsequent child keys define nodes and their parameters in a tree structure:

```yaml
sound_identifier:
  node_name_1:
    parameter_1: value_1
    parameter_2:
        node_name_2:
          parameter_a: value_a
          parameter_b: value_b
  node_name_3:
    (...)
```

## Running the code

To run the code we use the main waves.py file, followed by one parameter, which should match one of the root level keys in the yaml file. For example:

```bash
python waves.py sound_identifier
```

The program will then play the sound and stay idle until it detects a change in the yaml file, at which point it will re-render the sound and play it again, so that we can make changes and hear the results in real time.

There is a command-line based audio visualizer that shows the waveform of the sound being played (both in realtime and non-realtime modes).

## Creating a new type of Node

To create a new type of node, we create a new file with the node name in the nodes/ directory. The file should contain:
- a class that inherits from BaseNodeModel
- a class that inherits from BaseNode
- constant of type NodeDefinition called {NODE_NAME}_DEFINITION that attaches a node name, the model and the node class.

Then we need to add the new node to the NODE_REGISTRY in nodes/node_utils/node_registry.py.

One thing we need to be careful about is to ensure that the render method of the node works correctly in both realtime and non-realtime modes. This means that we need the node to work as expected when rendering the whole sound at once (non-realtime) and also when rendering in chunks (realtime).

## Configuration

In config.py we can set some global parameters for the project, such as the sample rate, the buffer size, etc. but also some options on the mode of operating like whether we play the sound in realtime (render in chunks) or pre-render the whole sound and play it back (non-realtime).

## Past and Future Work

In the earlier stages of the project, we only supported non-realtime playback and then slowly we started converting nodes to support realtime rendering. The goal is to eventually support all nodes in realtime mode, which will allow for more interactive sound design. The nodes that haven't been converted yet are in the nodes/non_real_time directory.

One challenge at the moment is that some nodes generate artifacts or problems when rendered in chunks, for example the normalise node, when a source min isn't supplied in the model, calculates the peak of the input wave and then scales the wave to fit within a given range. When rendering in chunks, the first chunk is used to calculate the peak, which can lead to clipping or distortion in subsequent chunks if they have higher peaks.

The objective of having realtime playback is to be able to connect the system to a MIDI controller or other input device and play the sounds live, as well as to potentially be able to tweak parameters in real time and hear the results immediately, but neither of these features are implemented yet.

### Directives for AI

- Avoid saying things like "this likely happens because ...", do further checks to confirm your hypothesis.
