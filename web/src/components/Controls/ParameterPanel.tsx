// ParameterPanel.tsx - Renders all parameters for the current scenario

import type { Parameter } from '../../types';
import { ParameterSlider } from './ParameterSlider';
import { ParameterSelect } from './ParameterSelect';

interface ParameterPanelProps {
  parameters: Parameter[];
  values: Record<string, number | string>;
  onChange: (name: string, value: number | string) => void;
  disabled?: boolean;
}

export function ParameterPanel({
  parameters,
  values,
  onChange,
  disabled = false,
}: ParameterPanelProps) {
  if (parameters.length === 0) {
    return (
      <div className="parameter-panel parameter-panel-empty">
        <p>No adjustable parameters</p>
      </div>
    );
  }

  return (
    <div className={`parameter-panel ${disabled ? 'disabled' : ''}`}>
      <div className="parameter-panel-header">
        <h3>Parameters</h3>
      </div>
      <div className="parameter-list">
        {parameters.map((param) => {
          const value = values[param.name] ?? param.default;

          // Use select for discrete options, slider for continuous
          if (param.type === 'discrete' && param.options && param.options.length > 0) {
            return (
              <ParameterSelect
                key={param.name}
                parameter={param}
                value={value}
                onChange={(newValue) => onChange(param.name, newValue)}
              />
            );
          }

          // Default to slider for continuous parameters
          return (
            <ParameterSlider
              key={param.name}
              parameter={param}
              value={typeof value === 'number' ? value : parseFloat(String(value)) || 0}
              onChange={(newValue) => onChange(param.name, newValue)}
            />
          );
        })}
      </div>
    </div>
  );
}
