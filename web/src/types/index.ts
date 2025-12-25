// TypeScript interfaces matching backend schemas

export type TensorKind = 'vector' | 'matrix' | 'image' | 'sparse_matrix' | 'pointcloud';

export interface Parameter {
  name: string;
  display_name: string;
  type: 'continuous' | 'discrete';
  default: number | string;
  description: string;
  min?: number;
  max?: number;
  step?: number;
  options?: (number | string)[];
}

export interface Probe {
  key: string;
  display_name: string;
  description: string;
}

export interface GraphNode {
  id: string;
  name: string;
  inputs: string[];
  outputs: string[];
  tags: string[];
}

export interface GraphEdge {
  from_node: string;
  from_output: string;
  to_node: string;
  to_input: string;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ScenarioInfo {
  id: string;
  name: string;
  description: string;
}

export interface ScenarioDetail {
  id: string;
  name: string;
  description: string;
  parameters: Parameter[];
  probes: Probe[];
  graph?: Graph;
}

export interface TensorSummary {
  id: string;
  name: string;
  kind: TensorKind;
  tags: string[];
  shape: number[];
  dtype: string;
  stats: Record<string, unknown>;
  recommended_views: string[];
}

export interface TensorData {
  id: string;
  name: string;
  shape: number[];
  dtype: string;
  data: unknown[];
}

export interface TensorSlice {
  id: string;
  name: string;
  full_shape: number[];
  slice_shape: number[];
  row_range: [number, number];
  col_range: [number, number];
  data: unknown[];
}

export interface RunScenarioRequest {
  parameters: Record<string, number | string>;
}

export interface RunScenarioResponse {
  scenario_id: string;
  parameters: Record<string, unknown>;
  tensors: Record<string, TensorSummary>;
}

// WebSocket message types

export type WSClientMessage =
  | { type: 'subscribe'; tensor_id: string; view?: string }
  | { type: 'unsubscribe'; tensor_id: string }
  | { type: 'update_param'; scenario_id: string; param: string; value: number | string };

export type WSServerMessage =
  | { type: 'tensor_update'; tensor_id: string; summary: TensorSummary }
  | { type: 'graph_update'; nodes: GraphNode[]; edges: GraphEdge[] }
  | { type: 'error'; message: string };
