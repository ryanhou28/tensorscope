// Zustand store for Tensorscope state management

import { create } from 'zustand';
import type {
  ScenarioInfo,
  ScenarioDetail,
  TensorSummary,
  Parameter,
} from '../types';
import { fetchScenarios, fetchScenario, runScenario } from '../api/client';

interface TensorscapeState {
  // Scenarios
  scenarios: ScenarioInfo[];
  currentScenario: ScenarioDetail | null;
  isLoadingScenarios: boolean;
  scenarioError: string | null;

  // Tensors
  tensors: Record<string, TensorSummary>;
  selectedTensorId: string | null;

  // Parameters (current values for the active scenario)
  parameters: Record<string, number | string>;

  // Actions
  loadScenarios: () => Promise<void>;
  selectScenario: (id: string) => Promise<void>;
  clearScenario: () => void;
  updateParameter: (name: string, value: number | string) => void;
  runCurrentScenario: () => Promise<void>;
  selectTensor: (id: string | null) => void;
  updateTensor: (id: string, summary: TensorSummary) => void;
  setTensors: (tensors: Record<string, TensorSummary>) => void;
}

export const useStore = create<TensorscapeState>((set, get) => ({
  // Initial state
  scenarios: [],
  currentScenario: null,
  isLoadingScenarios: false,
  scenarioError: null,
  tensors: {},
  selectedTensorId: null,
  parameters: {},

  // Actions
  loadScenarios: async () => {
    set({ isLoadingScenarios: true, scenarioError: null });
    try {
      const scenarios = await fetchScenarios();
      set({ scenarios, isLoadingScenarios: false });
    } catch (error) {
      set({
        scenarioError: error instanceof Error ? error.message : 'Failed to load scenarios',
        isLoadingScenarios: false,
      });
    }
  },

  selectScenario: async (id: string) => {
    set({ isLoadingScenarios: true, scenarioError: null });
    try {
      const scenario = await fetchScenario(id);

      // Initialize parameters with defaults
      const defaultParams: Record<string, number | string> = {};
      scenario.parameters.forEach((param: Parameter) => {
        defaultParams[param.name] = param.default;
      });

      set({
        currentScenario: scenario,
        parameters: defaultParams,
        isLoadingScenarios: false,
        tensors: {},
        selectedTensorId: null,
      });

      // Run the scenario to get initial tensors
      await get().runCurrentScenario();
    } catch (error) {
      set({
        scenarioError: error instanceof Error ? error.message : 'Failed to load scenario',
        isLoadingScenarios: false,
      });
    }
  },

  clearScenario: () => {
    set({
      currentScenario: null,
      parameters: {},
      tensors: {},
      selectedTensorId: null,
    });
  },

  updateParameter: (name: string, value: number | string) => {
    set((state) => ({
      parameters: { ...state.parameters, [name]: value },
    }));
  },

  runCurrentScenario: async () => {
    const { currentScenario, parameters } = get();
    if (!currentScenario) return;

    try {
      const response = await runScenario(currentScenario.id, { parameters });
      set({ tensors: response.tensors });
    } catch (error) {
      console.error('Failed to run scenario:', error);
    }
  },

  selectTensor: (id: string | null) => {
    set({ selectedTensorId: id });
  },

  updateTensor: (id: string, summary: TensorSummary) => {
    set((state) => ({
      tensors: { ...state.tensors, [id]: summary },
    }));
  },

  setTensors: (tensors: Record<string, TensorSummary>) => {
    set({ tensors });
  },
}));
