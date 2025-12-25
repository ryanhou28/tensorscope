// Zustand store for Tensorscope state management

import { create } from 'zustand';
import type {
  ScenarioInfo,
  ScenarioDetail,
  TensorSummary,
  Parameter,
} from '../types';
import { fetchScenarios, fetchScenario, runScenario } from '../api/client';

// Toast notification types
export interface Toast {
  id: string;
  type: 'error' | 'warning' | 'success' | 'info';
  message: string;
  duration?: number;
}

interface TensorscapeState {
  // Scenarios
  scenarios: ScenarioInfo[];
  currentScenario: ScenarioDetail | null;
  isLoadingScenarios: boolean;
  isLoadingScenario: boolean;
  isRunningScenario: boolean;
  scenarioError: string | null;

  // Tensors
  tensors: Record<string, TensorSummary>;
  selectedTensorId: string | null;

  // Parameters (current values for the active scenario)
  parameters: Record<string, number | string>;
  isUpdatingParams: boolean;

  // Toast notifications
  toasts: Toast[];

  // UI State
  sidebarCollapsed: boolean;

  // Actions
  loadScenarios: () => Promise<void>;
  selectScenario: (id: string) => Promise<void>;
  clearScenario: () => void;
  updateParameter: (name: string, value: number | string) => void;
  runCurrentScenario: () => Promise<void>;
  selectTensor: (id: string | null) => void;
  updateTensor: (id: string, summary: TensorSummary) => void;
  setTensors: (tensors: Record<string, TensorSummary>) => void;
  setUpdatingParams: (updating: boolean) => void;

  // Toast actions
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;

  // UI actions
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

export const useStore = create<TensorscapeState>((set, get) => ({
  // Initial state
  scenarios: [],
  currentScenario: null,
  isLoadingScenarios: false,
  isLoadingScenario: false,
  isRunningScenario: false,
  scenarioError: null,
  tensors: {},
  selectedTensorId: null,
  parameters: {},
  isUpdatingParams: false,
  toasts: [],
  sidebarCollapsed: false,

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
    set({ isLoadingScenario: true, scenarioError: null });
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
        isLoadingScenario: false,
        tensors: {},
        selectedTensorId: null,
      });

      // Run the scenario to get initial tensors
      await get().runCurrentScenario();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load scenario';
      set({
        scenarioError: message,
        isLoadingScenario: false,
      });
      get().addToast({ type: 'error', message });
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

    set({ isRunningScenario: true });
    try {
      const response = await runScenario(currentScenario.id, { parameters });
      set({ tensors: response.tensors, isRunningScenario: false });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to run scenario';
      console.error('Failed to run scenario:', error);
      set({ isRunningScenario: false });
      get().addToast({ type: 'error', message });
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

  setUpdatingParams: (updating: boolean) => {
    set({ isUpdatingParams: updating });
  },

  // Toast actions
  addToast: (toast: Omit<Toast, 'id'>) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    const newToast: Toast = { ...toast, id };
    set((state) => ({ toasts: [...state.toasts, newToast] }));

    // Auto-remove after duration (default 5 seconds)
    const duration = toast.duration ?? 5000;
    if (duration > 0) {
      setTimeout(() => {
        get().removeToast(id);
      }, duration);
    }
  },

  removeToast: (id: string) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },

  // UI actions
  toggleSidebar: () => {
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }));
  },

  setSidebarCollapsed: (collapsed: boolean) => {
    set({ sidebarCollapsed: collapsed });
  },
}));
