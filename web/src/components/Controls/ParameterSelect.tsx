// ParameterSelect.tsx - Dropdown for parameters with discrete options

import type { Parameter } from '../../types';

interface ParameterSelectProps {
  parameter: Parameter;
  value: number | string;
  onChange: (value: number | string) => void;
}

export function ParameterSelect({
  parameter,
  value,
  onChange,
}: ParameterSelectProps) {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const rawValue = e.target.value;
    // Try to parse as number if the original value was numeric
    const numValue = parseFloat(rawValue);
    const newValue = !isNaN(numValue) ? numValue : rawValue;
    onChange(newValue);
  };

  const options = parameter.options ?? [];

  return (
    <div className="parameter-select">
      <div className="parameter-header">
        <label className="parameter-label" htmlFor={`param-${parameter.name}`}>
          {parameter.display_name}
        </label>
        <span className="parameter-value">{String(value)}</span>
      </div>
      <select
        id={`param-${parameter.name}`}
        value={String(value)}
        onChange={handleChange}
        className="select-input"
      >
        {options.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
      </select>
      {parameter.description && (
        <p className="parameter-description">{parameter.description}</p>
      )}
    </div>
  );
}
