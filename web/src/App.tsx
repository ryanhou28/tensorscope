import { useEffect, useCallback, useRef, useState } from 'react';
import { useStore } from './store';
import { useWebSocket } from './api/websocket';
import { OperatorGraph } from './components/Graph';
import { TensorInspector } from './components/Inspector';
import { ParameterPanel } from './components/Controls';
import { ToastContainer } from './components/Toast';
import { GraphSkeleton, Spinner } from './components/Loading';
import type { WSServerMessage } from './types';
import './App.css';

function App() {
  const {
    scenarios,
    currentScenario,
    isLoadingScenarios,
    isLoadingScenario,
    isRunningScenario,
    scenarioError,
    tensors,
    selectedTensorId,
    parameters,
    isUpdatingParams,
    toasts,
    sidebarCollapsed,
    loadScenarios,
    selectScenario,
    selectTensor,
    updateTensor,
    updateParameter,
    setTensors,
    setUpdatingParams,
    addToast,
    removeToast,
    toggleSidebar,
  } = useStore();

  // Track WebSocket reconnection attempts
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 10;

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
        setUpdatingParams(false);
      } else if (message.type === 'tensors_update') {
        // Bulk update all tensors (e.g., after parameter change)
        setTensors(message.tensors);
        isUpdatingRef.current = false;
        setUpdatingParams(false);
      } else if (message.type === 'error') {
        console.error('WebSocket error:', message.message);
        addToast({ type: 'error', message: message.message });
        isUpdatingRef.current = false;
        setUpdatingParams(false);
      }
    },
    onConnect: () => {
      console.log('WebSocket connected');
      setReconnectAttempts(0);
      if (reconnectAttempts > 0) {
        addToast({ type: 'success', message: 'Reconnected to server' });
      }
    },
    onDisconnect: () => {
      console.log('WebSocket disconnected');
      setReconnectAttempts((prev) => prev + 1);
    },
    onError: () => {
      if (reconnectAttempts === 0) {
        addToast({
          type: 'warning',
          message: 'Connection lost. Attempting to reconnect...',
          duration: 3000,
        });
      }
    },
    maxReconnectAttempts,
  });

  // Handle parameter changes - update store and send via WebSocket
  const handleParameterChange = useCallback(
    (name: string, value: number | string) => {
      updateParameter(name, value);

      // Send update via WebSocket if connected and scenario is loaded
      if (isConnected && currentScenario) {
        isUpdatingRef.current = true;
        setUpdatingParams(true);
        updateParam(currentScenario.id, name, value);
      }
    },
    [isConnected, currentScenario, updateParameter, updateParam, setUpdatingParams]
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

  // Determine connection status display
  const getConnectionStatus = () => {
    if (isConnected) {
      return { text: 'Connected', className: 'connected' };
    }
    if (reconnectAttempts > 0 && reconnectAttempts < maxReconnectAttempts) {
      return { text: `Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`, className: 'reconnecting' };
    }
    if (reconnectAttempts >= maxReconnectAttempts) {
      return { text: 'Connection Failed', className: 'failed' };
    }
    return { text: 'Disconnected', className: 'disconnected' };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <button
            className="sidebar-toggle"
            onClick={toggleSidebar}
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="20" height="20">
              {sidebarCollapsed ? (
                <path d="M4 6h16M4 12h16M4 18h16" />
              ) : (
                <path d="M4 6h16M4 12h10M4 18h16" />
              )}
            </svg>
          </button>
          <h1>Tensorscope</h1>
        </div>
        <div className="header-right">
          {(isRunningScenario || isUpdatingParams) && (
            <div className="updating-indicator">
              <Spinner size="small" />
              <span>Updating...</span>
            </div>
          )}
          <span className={`ws-status ${connectionStatus.className}`}>
            <span className="ws-status-dot" />
            {connectionStatus.text}
          </span>
        </div>
      </header>

      <div className="main-container">
        {/* Sidebar - Scenario List */}
        <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
          <div className="sidebar-content">
            <div className="sidebar-section">
              <h2>Scenarios</h2>
              {isLoadingScenarios && (
                <div className="loading-indicator">
                  <Spinner size="small" />
                  <span>Loading...</span>
                </div>
              )}
              {scenarioError && (
                <div className="error-message">
                  <span className="error-icon">!</span>
                  <span>{scenarioError}</span>
                </div>
              )}
              <ul className="scenario-list">
                {scenarios.map((scenario) => (
                  <li
                    key={scenario.id}
                    className={`${currentScenario?.id === scenario.id ? 'selected' : ''} ${
                      isLoadingScenario && currentScenario?.id !== scenario.id ? 'disabled' : ''
                    }`}
                    onClick={() => !isLoadingScenario && selectScenario(scenario.id)}
                  >
                    <strong>{scenario.name}</strong>
                    <p>{scenario.description}</p>
                  </li>
                ))}
              </ul>
            </div>

            {/* Tensor List (below scenarios when a scenario is selected) */}
            {currentScenario && Object.keys(tensors).length > 0 && (
              <div className="sidebar-section tensor-list-section">
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
          </div>
        </aside>

        {/* Main Content Area */}
        <div className="content-area">
          {isLoadingScenario ? (
            <div className="loading-overlay">
              <Spinner size="large" />
              <p>Loading scenario...</p>
            </div>
          ) : !currentScenario ? (
            <div className="placeholder">
              <div className="placeholder-content">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="48" height="48">
                  <path d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25" />
                </svg>
                <p>Select a scenario from the sidebar to begin</p>
                <p className="placeholder-hint">Explore linear algebra operations with interactive visualizations</p>
              </div>
            </div>
          ) : (
            <>
              {/* Scenario Header */}
              <div className="scenario-header">
                <div className="scenario-title">
                  <h2>{currentScenario.name}</h2>
                  {isRunningScenario && <Spinner size="small" />}
                </div>
                <p>{currentScenario.description}</p>
              </div>

              {/* Parameter Controls */}
              {currentScenario.parameters && currentScenario.parameters.length > 0 && (
                <ParameterPanel
                  parameters={currentScenario.parameters}
                  values={parameters}
                  onChange={handleParameterChange}
                  disabled={!isConnected || isUpdatingParams}
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
                    {isRunningScenario && Object.keys(tensors).length === 0 ? (
                      <GraphSkeleton />
                    ) : (
                      <OperatorGraph
                        graph={currentScenario.graph}
                        selectedTensorId={selectedTensorId}
                        onNodeSelect={handleNodeSelect}
                        onEdgeSelect={handleEdgeSelect}
                      />
                    )}
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
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="32" height="32">
                          <path d="M15.75 15.75l-2.489-2.489m0 0a3.375 3.375 0 10-4.773-4.773 3.375 3.375 0 004.774 4.774zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
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

      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} onDismiss={removeToast} />
    </div>
  );
}

export default App;
