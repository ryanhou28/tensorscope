import { useEffect, useCallback, useRef } from 'react';
import { useStore } from './store';
import { useWebSocket } from './api/websocket';
import { OperatorGraph } from './components/Graph';
import { TensorInspector } from './components/Inspector';
import { ParameterPanel } from './components/Controls';
import type { WSServerMessage } from './types';
import './App.css';

function App() {
  const {
    scenarios,
    currentScenario,
    isLoadingScenarios,
    scenarioError,
    tensors,
    selectedTensorId,
    parameters,
    loadScenarios,
    selectScenario,
    selectTensor,
    updateTensor,
    updateParameter,
    setTensors,
  } = useStore();

  // Track if we're waiting for a parameter update response
  const isUpdatingRef = useRef(false);

  // Load scenarios on mount
  useEffect(() => {
    loadScenarios();
  }, [loadScenarios]);

  // WebSocket connection for real-time updates
  const { isConnected, updateParam } = useWebSocket(`ws://${window.location.host}/ws`, {
    onMessage: (message: WSServerMessage) => {
      if (message.type === 'tensor_update') {
        updateTensor(message.tensor_id, message.summary);
        isUpdatingRef.current = false;
      } else if (message.type === 'tensors_update') {
        // Bulk update all tensors (e.g., after parameter change)
        setTensors(message.tensors);
        isUpdatingRef.current = false;
      } else if (message.type === 'error') {
        console.error('WebSocket error:', message.message);
        isUpdatingRef.current = false;
      }
    },
    onConnect: () => console.log('WebSocket connected'),
    onDisconnect: () => console.log('WebSocket disconnected'),
  });

  // Handle parameter changes - update store and send via WebSocket
  const handleParameterChange = useCallback(
    (name: string, value: number | string) => {
      updateParameter(name, value);

      // Send update via WebSocket if connected and scenario is loaded
      if (isConnected && currentScenario) {
        isUpdatingRef.current = true;
        updateParam(currentScenario.id, name, value);
      }
    },
    [isConnected, currentScenario, updateParameter, updateParam]
  );

  const selectedTensor = selectedTensorId ? tensors[selectedTensorId] : null;

  // Handle tensor selection from graph edges
  const handleEdgeSelect = useCallback(
    (tensorId: string) => {
      // Find the tensor by name from the tensors map
      const matchingTensor = Object.entries(tensors).find(
        ([, tensor]) => tensor.name === tensorId || tensor.id === tensorId
      );
      if (matchingTensor) {
        selectTensor(matchingTensor[0]);
      }
    },
    [tensors, selectTensor]
  );

  // Handle node selection - could select first output tensor
  const handleNodeSelect = useCallback(
    (nodeId: string) => {
      // Find tensors that were output by this node
      // For now, we can select based on naming convention
      const matchingTensor = Object.entries(tensors).find(([, tensor]) =>
        tensor.name.toLowerCase().includes(nodeId.toLowerCase())
      );
      if (matchingTensor) {
        selectTensor(matchingTensor[0]);
      }
    },
    [tensors, selectTensor]
  );

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

          {/* Tensor List (below scenarios when a scenario is selected) */}
          {currentScenario && Object.keys(tensors).length > 0 && (
            <div className="tensor-list-section">
              <h2>Tensors</h2>
              <ul className="tensor-list">
                {Object.entries(tensors).map(([id, tensor]) => (
                  <li
                    key={id}
                    className={selectedTensorId === id ? 'selected' : ''}
                    onClick={() => selectTensor(id)}
                  >
                    <strong>{tensor.name}</strong>
                    <span className="tensor-shape">[{tensor.shape.join('x')}]</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>

        {/* Main Content Area */}
        <div className="content-area">
          {!currentScenario ? (
            <div className="placeholder">
              <p>Select a scenario from the sidebar to begin</p>
            </div>
          ) : (
            <>
              {/* Scenario Header */}
              <div className="scenario-header">
                <h2>{currentScenario.name}</h2>
                <p>{currentScenario.description}</p>
              </div>

              {/* Parameter Controls */}
              {currentScenario.parameters && currentScenario.parameters.length > 0 && (
                <ParameterPanel
                  parameters={currentScenario.parameters}
                  values={parameters}
                  onChange={handleParameterChange}
                  disabled={!isConnected}
                />
              )}

              {/* Graph and Inspector Layout */}
              <div className="graph-inspector-layout">
                {/* Operator Graph Panel */}
                <div className="graph-panel">
                  <div className="panel-header">
                    <h3>Operator Graph</h3>
                  </div>
                  <div className="graph-container">
                    <OperatorGraph
                      graph={currentScenario.graph}
                      selectedTensorId={selectedTensorId}
                      onNodeSelect={handleNodeSelect}
                      onEdgeSelect={handleEdgeSelect}
                    />
                  </div>
                </div>

                {/* Inspector Panel */}
                <div className="inspector-panel">
                  <div className="panel-header">
                    <h3>Inspector</h3>
                  </div>
                  <div className="inspector-container">
                    {selectedTensor ? (
                      <TensorInspector tensor={selectedTensor} />
                    ) : (
                      <div className="inspector-placeholder">
                        <p>Click on a tensor in the graph or list to inspect it</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
