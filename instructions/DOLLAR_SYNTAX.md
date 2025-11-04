# $ Syntax for Node References in Expressions

## Overview

The `$` syntax provides a concise way to reference node outputs in expressions, making it easier to use one node's output to modulate parameters of other nodes.

## Basic Usage

Instead of using the verbose `reference` node:

```yaml
# Old way (verbose)
my_sound:
  osc:
    freq:
      reference:
        ref: my_lfo
```

You can now use the `$` syntax directly in expressions:

```yaml
# New way (concise)
my_sound:
  osc:
    freq: $my_lfo
```

## Examples

### Simple Reference

```yaml
my_sound:
  mix:
    signals:
      # Define an LFO with an id
      - osc:
          id: my_lfo
          type: sin
          freq: 2
          range: [200, 800]
      
      # Reference it with $ syntax
      - osc:
          type: sin
          freq: $my_lfo
```

### Math Operations

Combine references with mathematical operations:

```yaml
my_sound:
  mix:
    signals:
      - osc:
          id: lfo
          type: sin
          freq: 1
          range: [0, 1]
      
      - osc:
          type: saw
          freq: $lfo * 440  # Modulate frequency
      
      - osc:
          type: tri
          freq: $lfo * 880 + 100  # More complex expression
```

### Multiple References

Use multiple node references in one expression:

```yaml
my_sound:
  expression:
    duration: 2
    exp: $lfo1 + $lfo2 * 0.5  # Combine two LFOs
    lfo1:
      osc:
        id: lfo1
        type: sin
        freq: 3
    lfo2:
      osc:
        id: lfo2
        type: tri
        freq: 5
```

### In Expression Nodes

Works seamlessly in expression nodes:

```yaml
fm_synth:
  expression:
    duration: 1
    exp: carrier * (1 + $modulator * 0.3)  # FM synthesis
    carrier:
      osc:
        type: sin
        freq: 440
    modulator:
      osc:
        id: modulator
        type: sin
        freq: 220
```

### As WavableValue

Use directly as parameter values (no quotes needed):

```yaml
modulated_osc:
  mix:
    signals:
      - osc:
          id: pitch_lfo
          type: sin
          freq: 4
          range: [200, 800]
      
      - osc:
          type: sin
          freq: $pitch_lfo * 2  # Direct usage, no quotes!
```

## Technical Details

### How It Works

1. The `$` prefix is preprocessed at expression compilation time
2. `$node_id` is transformed to `node_id` (strips the `$`)
3. The node output is injected into the expression context during rendering
4. The referenced node's output must already be in the RenderContext

### Requirements

- The referenced node must have an `id` field
- The referenced node must be rendered before the referencing expression
- Works in any expression context (expression nodes, WavableValues, etc.)

### Equivalent Forms

These are all equivalent:

```yaml
# Verbose reference node
freq:
  reference:
    ref: my_lfo

# $ syntax in expression node
freq:
  expression:
    exp: $my_lfo

# $ syntax as WavableValue (string expression)
freq: $my_lfo
```

## Advantages

1. **Concise**: Shorter and clearer than `reference` nodes
2. **Composable**: Easy to combine with math operations
3. **No quotes needed**: YAML handles `$` without requiring quotes
4. **Visual clarity**: The `$` makes it obvious you're referencing a node
5. **Multiple references**: Natural syntax for combining multiple nodes

## Compatibility

- Works everywhere string expressions are supported
- No breaking changes - existing `reference` nodes still work
- Can mix and match `$` syntax with traditional references
