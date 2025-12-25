import { useMemo, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import OperatorNode, { type OperatorNodeData } from './OperatorNode';
import type { Graph, GraphNode, GraphEdge } from '../../types';

// Register custom node types
const nodeTypes = {
  operator: OperatorNode,
};

interface OperatorGraphProps {
  graph: Graph | undefined;
  selectedNodeId?: string | null;
  selectedTensorId?: string | null;
  onNodeSelect?: (nodeId: string) => void;
  onEdgeSelect?: (tensorId: string) => void;
}

// Layout constants
const HORIZONTAL_SPACING = 280;
const VERTICAL_SPACING = 100;

/**
 * Calculate node positions using a layered layout with barycenter ordering.
 * This reduces edge crossings by positioning nodes near their connected neighbors.
 */
function calculateLayout(nodes: GraphNode[], edges: GraphEdge[]): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();

  if (nodes.length === 0) return positions;

  // Build adjacency lists
  const inDegree = new Map<string, number>();
  const outEdges = new Map<string, string[]>();
  const inEdges = new Map<string, string[]>();

  nodes.forEach((node) => {
    inDegree.set(node.id, 0);
    outEdges.set(node.id, []);
    inEdges.set(node.id, []);
  });

  edges.forEach((edge) => {
    inDegree.set(edge.to_node, (inDegree.get(edge.to_node) || 0) + 1);
    outEdges.get(edge.from_node)?.push(edge.to_node);
    inEdges.get(edge.to_node)?.push(edge.from_node);
  });

  // Assign layers using topological sort
  const layers: string[][] = [];
  const nodeLayer = new Map<string, number>();
  const visited = new Set<string>();

  // Start with nodes that have no incoming edges
  let currentLayer = nodes.filter((n) => inDegree.get(n.id) === 0).map((n) => n.id);

  while (currentLayer.length > 0) {
    layers.push(currentLayer);
    currentLayer.forEach((nodeId) => {
      nodeLayer.set(nodeId, layers.length - 1);
      visited.add(nodeId);
    });

    const nextLayer: string[] = [];
    currentLayer.forEach((nodeId) => {
      outEdges.get(nodeId)?.forEach((targetId) => {
        if (!visited.has(targetId)) {
          const remaining = inDegree.get(targetId)! - 1;
          inDegree.set(targetId, remaining);
          if (remaining === 0) {
            nextLayer.push(targetId);
          }
        }
      });
    });
    currentLayer = nextLayer;
  }

  // Handle any remaining nodes (cycles or disconnected)
  nodes.forEach((node) => {
    if (!visited.has(node.id)) {
      const lastLayer = layers.length > 0 ? layers.length : 0;
      if (!layers[lastLayer]) {
        layers.push([]);
      }
      layers[lastLayer].push(node.id);
      nodeLayer.set(node.id, lastLayer);
    }
  });

  // Barycenter ordering: reorder nodes within each layer to minimize edge crossings
  // by positioning each node at the average position of its neighbors
  const nodeYPosition = new Map<string, number>();

  // Initialize first layer positions
  layers[0]?.forEach((nodeId, index) => {
    nodeYPosition.set(nodeId, index);
  });

  // Forward pass: order based on incoming edges
  for (let i = 1; i < layers.length; i++) {
    const layer = layers[i];
    const barycenters: { id: string; bc: number }[] = [];

    for (const nodeId of layer) {
      const parents = inEdges.get(nodeId) || [];
      if (parents.length > 0) {
        const avg = parents.reduce((sum, p) => sum + (nodeYPosition.get(p) || 0), 0) / parents.length;
        barycenters.push({ id: nodeId, bc: avg });
      } else {
        barycenters.push({ id: nodeId, bc: 0 });
      }
    }

    // Sort by barycenter
    barycenters.sort((a, b) => a.bc - b.bc);

    // Update layer order and positions
    layers[i] = barycenters.map((b) => b.id);
    layers[i].forEach((nodeId, index) => {
      nodeYPosition.set(nodeId, index);
    });
  }

  // Backward pass: refine based on outgoing edges
  for (let i = layers.length - 2; i >= 0; i--) {
    const layer = layers[i];
    const barycenters: { id: string; bc: number }[] = [];

    for (const nodeId of layer) {
      const children = outEdges.get(nodeId) || [];
      if (children.length > 0) {
        const avg = children.reduce((sum, c) => sum + (nodeYPosition.get(c) || 0), 0) / children.length;
        barycenters.push({ id: nodeId, bc: avg });
      } else {
        barycenters.push({ id: nodeId, bc: nodeYPosition.get(nodeId) || 0 });
      }
    }

    // Sort by barycenter
    barycenters.sort((a, b) => a.bc - b.bc);

    // Update layer order and positions
    layers[i] = barycenters.map((b) => b.id);
    layers[i].forEach((nodeId, index) => {
      nodeYPosition.set(nodeId, index);
    });
  }

  // Calculate final positions
  const maxLayerSize = Math.max(...layers.map((l) => l.length));

  layers.forEach((layer, layerIndex) => {
    // Center the layer vertically
    const layerHeight = layer.length * VERTICAL_SPACING;
    const maxHeight = maxLayerSize * VERTICAL_SPACING;
    const offsetY = (maxHeight - layerHeight) / 2;

    layer.forEach((nodeId, nodeIndex) => {
      positions.set(nodeId, {
        x: layerIndex * HORIZONTAL_SPACING,
        y: offsetY + nodeIndex * VERTICAL_SPACING,
      });
    });
  });

  return positions;
}

/**
 * Convert backend graph format to React Flow format
 */
function convertToReactFlow(
  graph: Graph,
  selectedNodeId?: string | null,
  selectedTensorId?: string | null
): { nodes: Node[]; edges: Edge[] } {
  const positions = calculateLayout(graph.nodes, graph.edges);

  const nodes: Node[] = graph.nodes.map((node) => {
    const pos = positions.get(node.id) || { x: 0, y: 0 };
    const nodeData: OperatorNodeData = {
      label: node.name,
      operatorType: node.id.split('_')[0] || node.id, // Extract type from id like "matmul_1"
      inputs: node.inputs,
      outputs: node.outputs,
      tags: node.tags,
      isSelected: node.id === selectedNodeId,
    };
    return {
      id: node.id,
      type: 'operator',
      position: pos,
      data: nodeData,
    };
  });

  const edges: Edge[] = graph.edges.map((edge, index) => {
    // Create a tensor ID from the edge
    const tensorId = `${edge.from_node}:${edge.from_output}`;
    const isSelected = tensorId === selectedTensorId ||
                       edge.from_output === selectedTensorId ||
                       edge.to_input === selectedTensorId;

    return {
      id: `edge-${index}`,
      source: edge.from_node,
      sourceHandle: edge.from_output,
      target: edge.to_node,
      targetHandle: edge.to_input,
      animated: isSelected,
      style: {
        stroke: isSelected ? '#4a90d9' : '#666',
        strokeWidth: isSelected ? 2 : 1,
      },
      label: edge.from_output,
      labelStyle: {
        fill: '#888',
        fontSize: 10,
      },
      labelBgStyle: {
        fill: '#1a1a2e',
        fillOpacity: 0.8,
      },
      data: {
        tensorId: edge.from_output,
      },
    };
  });

  return { nodes, edges };
}

export function OperatorGraph({
  graph,
  selectedNodeId,
  selectedTensorId,
  onNodeSelect,
  onEdgeSelect,
}: OperatorGraphProps) {
  // Convert graph to React Flow format
  const { initialNodes, initialEdges } = useMemo(() => {
    if (!graph) {
      return { initialNodes: [] as Node[], initialEdges: [] as Edge[] };
    }
    const { nodes, edges } = convertToReactFlow(graph, selectedNodeId, selectedTensorId);
    return { initialNodes: nodes, initialEdges: edges };
  }, [graph, selectedNodeId, selectedTensorId]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  // Handle node click
  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      onNodeSelect?.(node.id);
    },
    [onNodeSelect]
  );

  // Handle edge click - select the tensor flowing through the edge
  const handleEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      // Use the edge label as the tensor ID (which is the output name)
      if (edge.label && typeof edge.label === 'string') {
        onEdgeSelect?.(edge.label);
      }
    },
    [onEdgeSelect]
  );

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="operator-graph-empty">
        <p>No operator graph available</p>
        <p className="hint">Select a scenario to view its operator graph</p>
      </div>
    );
  }

  return (
    <div className="operator-graph">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        defaultEdgeOptions={{
          type: 'smoothstep',
        }}
      >
        <Background color="#333" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
}

export default OperatorGraph;
