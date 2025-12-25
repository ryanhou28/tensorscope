# Tensorscope

An interactive tool for building linear algebra intuition through probe-anywhere operator graphs with rich visualizations.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ (for the web frontend)
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tensorscope.git
cd tensorscope

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the Python package in development mode
pip install -e ".[dev]"

# Install frontend dependencies
cd web
npm install
cd ..
```

### Running the Application

Start both the API server and web frontend:

```bash
./dev.sh
```

Or run them individually:

```bash
# API server only (http://localhost:8000)
./dev.sh --api

# Web frontend only (http://localhost:5173)
./dev.sh --web
```

Then open http://localhost:5173 in your browser.

### Verify Installation

```bash
# Test that the Python package is installed correctly
python -c "import tensorscope; print('Tensorscope installed successfully!')"

# Run the test suite
pytest
```

## Usage

1. **Select a Scenario**: Choose a scenario from the sidebar (e.g., "Least Squares 2D")
2. **View the Graph**: The operator graph shows the flow of tensors through operations
3. **Inspect Tensors**: Click on tensor edges in the graph or items in the tensor list
4. **Adjust Parameters**: Use the sliders to modify scenario parameters
5. **Explore Visualizations**: Switch between Data, Heatmap, Stem, and other visualization tabs

## Creating New Scenarios

Scenarios are defined in Python and automatically discovered by the server. See the [Creating Scenarios Guide](tensorscope/scenarios/CREATING_SCENARIOS.md) for detailed instructions, examples, and available operators.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tensorscope

# Run specific test file
pytest tests/test_operators/test_basic.py
```

### Code Formatting

```bash
# Format Python code
black tensorscope tests

# Lint Python code
ruff check tensorscope tests

# Type check
mypy tensorscope
```

### Frontend Development

```bash
cd web

# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Type check
npm run tsc
```