// REST API client for Tensorscope backend

import type {
  ScenarioInfo,
  ScenarioDetail,
  TensorSummary,
  TensorData,
  TensorSlice,
  RunScenarioRequest,
  RunScenarioResponse,
} from '../types';

const API_BASE = '/api';

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API error ${response.status}: ${error}`);
  }

  return response.json();
}

export async function fetchScenarios(): Promise<ScenarioInfo[]> {
  return fetchJSON<ScenarioInfo[]>(`${API_BASE}/scenarios`);
}

export async function fetchScenario(id: string): Promise<ScenarioDetail> {
  return fetchJSON<ScenarioDetail>(`${API_BASE}/scenarios/${id}`);
}

export async function runScenario(
  id: string,
  params: RunScenarioRequest = { parameters: {} }
): Promise<RunScenarioResponse> {
  return fetchJSON<RunScenarioResponse>(`${API_BASE}/scenarios/${id}/run`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function fetchTensorSummary(id: string): Promise<TensorSummary> {
  return fetchJSON<TensorSummary>(`${API_BASE}/tensors/${id}/summary`);
}

export async function fetchTensorData(id: string): Promise<TensorData> {
  return fetchJSON<TensorData>(`${API_BASE}/tensors/${id}/data`);
}

export async function fetchTensorSlice(
  id: string,
  rowStart = 0,
  rowEnd?: number,
  colStart = 0,
  colEnd?: number
): Promise<TensorSlice> {
  const params = new URLSearchParams();
  params.set('row_start', String(rowStart));
  if (rowEnd !== undefined) params.set('row_end', String(rowEnd));
  params.set('col_start', String(colStart));
  if (colEnd !== undefined) params.set('col_end', String(colEnd));

  return fetchJSON<TensorSlice>(`${API_BASE}/tensors/${id}/slice?${params}`);
}
