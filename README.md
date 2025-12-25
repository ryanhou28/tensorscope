# Tensorscope

A "linear algebra microscope" — an interactive tool for building LA intuition through probe-anywhere operator graphs with rich visualizations.

## Overview

Tensorscope lets you define linear algebra operator graphs in Python and visualize them through an interactive web UI. You can:

- Build operator graphs (matrix multiplication, SVD, least squares, etc.)
- Probe any intermediate tensor to see its values and statistics
- Visualize tensors as heatmaps, singular value plots, covariance ellipses, and more
- Adjust parameters with sliders and see results update live

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for web frontend, optional for API-only usage)

### Installation

```bash
# Clone the repository
git clone https://github.com/ryanhou/tensorscope.git
cd tensorscope

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python package in development mode
pip install -e ".[dev]"

# Install frontend dependencies
cd web
npm install
cd ..
```

### Running

```bash
# Start both API and web servers
./dev.sh

# Or start servers individually
./dev.sh --api   # API server only (http://localhost:8000)
./dev.sh --web   # Web server only (http://localhost:5173)
```

Then open http://localhost:5173 in your browser. Select a scenario from the sidebar, and click on tensors to inspect their values.

### Verify Installation

```bash
# Test that the package is installed correctly
python -c "import tensorscope; print(tensorscope.__version__)"
```

## Project Structure

```
tensorscope/
├── tensorscope/           # Python package
│   ├── core/              # Core abstractions (TrackedTensor, Operator, Graph)
│   ├── operators/         # LA operators (MatMul, SVD, LeastSquares, etc.)
│   ├── scenarios/         # Pre-built learning scenarios
│   └── server/            # FastAPI backend
├── web/                   # React frontend (Vite + TypeScript + D3)
├── tests/                 # Test suite
├── pyproject.toml         # Python package configuration
└── dev.sh                 # Development server script
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black tensorscope tests
ruff check tensorscope tests
```