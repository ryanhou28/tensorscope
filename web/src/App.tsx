import { useEffect, useState } from 'react';
import { useStore } from './store';
import { useWebSocket } from './api/websocket';
import { fetchTensorData } from './api/client';
import type { WSServerMessage, TensorData } from './types';
import './App.css';

// Format a number for display
function formatNumber(n: number): string {
  if (Number.isInteger(n)) return n.toString();
  if (Math.abs(n) < 0.001 || Math.abs(n) >= 10000) {
    return n.toExponential(2);
  }
  return n.toFixed(3);
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
                  <td key={j} className="value">{formatNumber(val)}</td>
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

function App() {
  const {
    scenarios,
    currentScenario,
    isLoadingScenarios,
    scenarioError,
    tensors,
    selectedTensorId,
    loadScenarios,
    selectScenario,
    selectTensor,
    updateTensor,
  } = useStore();

  const [tensorData, setTensorData] = useState<TensorData | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);

  // Load scenarios on mount
  useEffect(() => {
    loadScenarios();
  }, [loadScenarios]);

  // Fetch tensor data when selection changes
  useEffect(() => {
    if (!selectedTensorId) {
      setTensorData(null);
      return;
    }

    setIsLoadingData(true);
    fetchTensorData(selectedTensorId)
      .then(setTensorData)
      .catch((err) => {
        console.error('Failed to fetch tensor data:', err);
        setTensorData(null);
      })
      .finally(() => setIsLoadingData(false));
  }, [selectedTensorId]);

  // WebSocket connection for real-time updates
  const { isConnected } = useWebSocket(
    `ws://${window.location.host}/ws`,
    {
      onMessage: (message: WSServerMessage) => {
        if (message.type === 'tensor_update') {
          updateTensor(message.tensor_id, message.summary);
        }
      },
      onConnect: () => console.log('WebSocket connected'),
      onDisconnect: () => console.log('WebSocket disconnected'),
    }
  );

  const selectedTensor = selectedTensorId ? tensors[selectedTensorId] : null;

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>Tensorscope</h1>
        <span className={`ws-status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </header>

      <div className="main-container">
        {/* Sidebar - Scenario List */}
        <aside className="sidebar">
          <h2>Scenarios</h2>
          {isLoadingScenarios && <p className="loading">Loading...</p>}
          {scenarioError && <p className="error">{scenarioError}</p>}
          <ul className="scenario-list">
            {scenarios.map((scenario) => (
              <li
                key={scenario.id}
                className={currentScenario?.id === scenario.id ? 'selected' : ''}
                onClick={() => selectScenario(scenario.id)}
              >
                <strong>{scenario.name}</strong>
                <p>{scenario.description}</p>
              </li>
            ))}
          </ul>
        </aside>

        {/* Main Panel */}
        <main className="main-panel">
          {!currentScenario ? (
            <div className="placeholder">
              <p>Select a scenario from the sidebar to begin</p>
            </div>
          ) : (
            <>
              <div className="scenario-header">
                <h2>{currentScenario.name}</h2>
                <p>{currentScenario.description}</p>
              </div>

              {/* Tensor List */}
              <div className="tensor-section">
                <h3>Tensors</h3>
                <div className="tensor-grid">
                  {Object.entries(tensors).map(([id, tensor]) => (
                    <div
                      key={id}
                      className={`tensor-card ${selectedTensorId === id ? 'selected' : ''}`}
                      onClick={() => selectTensor(id)}
                    >
                      <strong>{tensor.name}</strong>
                      <span className="tensor-shape">
                        [{tensor.shape.join(' x ')}]
                      </span>
                      <span className="tensor-kind">{tensor.kind}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Tensor Inspector */}
              {selectedTensor && (
                <div className="inspector">
                  <h3>Inspector: {selectedTensor.name}</h3>
                  <div className="inspector-content">
                    {/* Tensor Data Display */}
                    <div className="inspector-data">
                      <h4>Data</h4>
                      {isLoadingData ? (
                        <p className="loading">Loading data...</p>
                      ) : tensorData ? (
                        <TensorDataView data={tensorData.data} shape={tensorData.shape} />
                      ) : (
                        <p className="error">Failed to load data</p>
                      )}
                    </div>

                    {/* Tensor Metadata */}
                    <div className="inspector-meta">
                      <h4>Properties</h4>
                      <div className="inspector-details">
                        <p><strong>Kind:</strong> {selectedTensor.kind}</p>
                        <p><strong>Shape:</strong> [{selectedTensor.shape.join(', ')}]</p>
                        <p><strong>Dtype:</strong> {selectedTensor.dtype}</p>
                        <p><strong>Tags:</strong> {selectedTensor.tags.join(', ') || 'none'}</p>
                      </div>
                      <h4>Stats</h4>
                      <div className="stats">
                        <pre>{JSON.stringify(selectedTensor.stats, null, 2)}</pre>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
