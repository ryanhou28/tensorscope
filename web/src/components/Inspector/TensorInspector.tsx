import { useState, useEffect, useMemo } from 'react';
import type { TensorSummary, TensorData } from '../../types';
import { fetchTensorData } from '../../api/client';
import { getVisualizersForTensor } from '../../lib/visualizerRegistry';

interface TensorInspectorProps {
  tensor: TensorSummary;
}

// Format a number for display
function formatNumber(n: number): string {
  if (Number.isInteger(n)) return n.toString();
  if (Math.abs(n) < 0.001 || Math.abs(n) >= 10000) {
    return n.toExponential(2);
  }
  return n.toFixed(4);
}

// Format stat value for display
function formatStatValue(value: unknown): string {
  if (typeof value === 'number') {
    return formatNumber(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((v) => (typeof v === 'number' ? formatNumber(v) : String(v))).join(', ')}]`;
  }
  return String(value);
}

// Component to display tensor data as a matrix/vector
function TensorDataView({ data, shape }: { data: unknown[]; shape: number[] }) {
  if (shape.length === 1) {
    // Vector: display as a column
    const values = data as number[];
    return (
      <div className="tensor-data tensor-vector">
        <table>
          <tbody>
            {values.map((val, i) => (
              <tr key={i}>
                <td className="index">{i}</td>
                <td className="value">{formatNumber(val)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (shape.length === 2) {
    // Matrix: display as a grid
    const rows = data as number[][];
    return (
      <div className="tensor-data tensor-matrix">
        <table>
          <thead>
            <tr>
              <th></th>
              {Array.from({ length: shape[1] }, (_, j) => (
                <th key={j}>{j}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                <td className="index">{i}</td>
                {row.map((val, j) => (
                  <td key={j} className="value">
                    {formatNumber(val)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // Higher dimensions: just show as JSON
  return (
    <div className="tensor-data">
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export function TensorInspector({ tensor }: TensorInspectorProps) {
  const [tensorData, setTensorData] = useState<TensorData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>('data');

  // Get available visualizers for this tensor
  const visualizers = useMemo(() => getVisualizersForTensor(tensor), [tensor]);

  // Fetch tensor data when tensor changes
  useEffect(() => {
    setIsLoading(true);
    setError(null);
    setTensorData(null);

    fetchTensorData(tensor.id)
      .then((data) => {
        setTensorData(data);
      })
      .catch((err) => {
        console.error('Failed to fetch tensor data:', err);
        setError(err.message || 'Failed to load tensor data');
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [tensor.id]);

  // Reset active tab when tensor changes if current tab is not available
  useEffect(() => {
    const availableTabs = ['data', ...visualizers.map((v) => v.id)];
    if (!availableTabs.includes(activeTab)) {
      setActiveTab('data');
    }
  }, [tensor.id, visualizers, activeTab]);

  // Render the active visualizer
  const renderVisualization = () => {
    if (!tensorData) {
      return null;
    }

    if (activeTab === 'data') {
      return <TensorDataView data={tensorData.data} shape={tensorData.shape} />;
    }

    const activeVisualizer = visualizers.find((v) => v.id === activeTab);
    if (!activeVisualizer) {
      return <div className="visualizer-empty">Visualizer not available</div>;
    }

    const VisualizerComponent = activeVisualizer.component;

    // Prepare data for visualizers
    // Most visualizers expect a 'data' prop with the tensor data
    const visualizerProps: Record<string, unknown> = {
      width: 400,
      height: 300,
    };

    // Handle different visualizer data requirements
    if (activeVisualizer.id === 'heatmap') {
      // Heatmap expects 2D array
      visualizerProps.data = tensorData.data;
    } else if (activeVisualizer.id === 'stem' || activeVisualizer.id === 'singular_values') {
      // Stem and singular values expect 1D array
      if (Array.isArray(tensorData.data) && tensorData.shape.length === 1) {
        visualizerProps.data = tensorData.data;
      } else if (Array.isArray(tensorData.data) && tensorData.shape.length === 2) {
        // Flatten if needed
        visualizerProps.data = (tensorData.data as number[][]).flat();
      }
    } else if (activeVisualizer.id === 'ellipse') {
      // Ellipse expects 2x2 matrix
      visualizerProps.matrix = tensorData.data;
    }

    return (
      <div className="visualizer-container">
        <VisualizerComponent {...visualizerProps} />
      </div>
    );
  };

  return (
    <div className="tensor-inspector">
      {/* Header */}
      <div className="inspector-header">
        <h3>{tensor.name}</h3>
        <div className="tensor-meta-inline">
          <span className="tensor-shape">[{tensor.shape.join(' x ')}]</span>
          <span className="tensor-kind">{tensor.kind}</span>
          <span className="tensor-dtype">{tensor.dtype}</span>
        </div>
      </div>

      {/* Tags */}
      {tensor.tags.length > 0 && (
        <div className="tensor-tags">
          {tensor.tags.map((tag) => (
            <span key={tag} className="tensor-tag">
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="visualizer-tabs">
        <button
          className={`viz-tab ${activeTab === 'data' ? 'active' : ''}`}
          onClick={() => setActiveTab('data')}
        >
          Data
        </button>
        {visualizers.map((viz) => (
          <button
            key={viz.id}
            className={`viz-tab ${activeTab === viz.id ? 'active' : ''}`}
            onClick={() => setActiveTab(viz.id)}
            title={viz.description}
          >
            {viz.name}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="inspector-content-area">
        {isLoading ? (
          <div className="loading-state">Loading tensor data...</div>
        ) : error ? (
          <div className="error-state">{error}</div>
        ) : (
          <div className="inspector-visualization">{renderVisualization()}</div>
        )}
      </div>

      {/* Stats Panel */}
      <div className="inspector-stats">
        <h4>Statistics</h4>
        <div className="stats-grid">
          {Object.entries(tensor.stats).map(([key, value]) => (
            <div key={key} className="stat-item">
              <span className="stat-label">{key.replace(/_/g, ' ')}</span>
              <span className="stat-value">{formatStatValue(value)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default TensorInspector;
