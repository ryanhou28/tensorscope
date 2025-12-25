# Creating Custom Scenarios

Scenarios are Python modules that define a linear algebra pipeline with parameters and probed tensors. They are automatically discovered by the server when placed in the `tensorscope/scenarios/` directory.

## Basic Structure

A scenario module must define a `create_scenario()` function that returns a `Scenario` object:

```python
# tensorscope/scenarios/my_scenario.py

import numpy as np
from tensorscope.core import Scenario, Parameter
from tensorscope.operators import MatMul, SVD

def create_scenario() -> Scenario:
    scenario = Scenario(
        id="my_scenario",
        name="My Custom Scenario",
        description="A custom linear algebra demonstration"
    )

    # Define parameters (see below)
    # Build the operator graph
    # Add probes for tensors to inspect

    return scenario
```

## Parameters

Parameters allow users to interactively modify the scenario via sliders or dropdowns in the UI.

### Continuous Parameters

For numeric values with a range:

```python
scenario.add_parameter(Parameter(
    name="noise_level",
    display_name="Noise Level",
    type="continuous",
    default=0.1,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Amount of noise to add to the input"
))
```

### Discrete Parameters

For values with specific options:

```python
scenario.add_parameter(Parameter(
    name="matrix_size",
    display_name="Matrix Size",
    type="discrete",
    default=3,
    options=[2, 3, 4, 5],
    description="Dimensions of the input matrix"
))
```

## Building the Operator Graph

Use the available operators to construct your computation:

```python
from tensorscope.operators import MatMul, Transpose, SVD, LeastSquares

# Create operators
transpose = Transpose()
matmul = MatMul()
svd = SVD()

# Add to scenario graph
scenario.add_operator("transpose_A", transpose)
scenario.add_operator("AtA", matmul)
scenario.add_operator("svd", svd)

# Connect operators
scenario.connect("input_A", "transpose_A", "A")
scenario.connect("transpose_A", "AtA", "A")
scenario.connect("input_A", "AtA", "B")
scenario.connect("AtA", "svd", "A")
```

## Adding Probes

Probes mark which tensors should be available for inspection in the UI:

```python
scenario.add_probe("A", display_name="Input Matrix A", description="The input matrix")
scenario.add_probe("AtA", display_name="A^T A", description="The Gram matrix")
scenario.add_probe("x", display_name="Solution", description="The least squares solution")
```

## Available Operators

### Basic Operations
- `MatMul` - Matrix multiplication
- `Transpose` - Matrix transpose
- `Add` - Element-wise addition
- `Subtract` - Element-wise subtraction
- `Scale` - Scalar multiplication
- `Norm` - Frobenius or L2 norm

### Decompositions
- `SVD` - Singular value decomposition (returns U, S, Vt)
- `Eigendecomposition` - Eigenvalue decomposition for symmetric matrices
- `QR` - QR decomposition
- `Cholesky` - Cholesky decomposition for positive definite matrices

### Solvers
- `LeastSquares` - Least squares solution for overdetermined systems
- `LinearSolve` - Direct solve for square systems
- `NormalEquations` - Solve via normal equations
- `Inverse` - Matrix inverse
- `PseudoInverse` - Moore-Penrose pseudoinverse

## Example: Complete Scenario

See `least_squares_2d.py` in this directory for a complete working example that demonstrates:

- Parameter definition (noise level, condition number, random seed)
- Building a multi-step operator graph
- Probing intermediate tensors
- A data generator function for creating inputs

## Testing Your Scenario

After creating your scenario, you can test it directly in Python:

```python
from tensorscope.scenarios.my_scenario import create_scenario

scenario = create_scenario()

# Run with default parameters
results = scenario.run({})

# Run with custom parameters
results = scenario.run({
    "noise_level": 0.5,
    "matrix_size": 4
})

# Access tensor summaries
for name, tensor in results.items():
    print(f"{name}: shape={tensor.shape}, norm={tensor.stats.get('norm')}")
```

Then start the server with `./dev.sh` and your scenario will appear in the sidebar.
