// ParameterSlider.tsx - Slider component for numeric parameters

import { useState, useCallback, useRef, useEffect } from 'react';
import type { Parameter } from '../../types';

interface ParameterSliderProps {
  parameter: Parameter;
  value: number;
  onChange: (value: number) => void;
  debounceMs?: number;
}

export function ParameterSlider({
  parameter,
  value,
  onChange,
  debounceMs = 150,
}: ParameterSliderProps) {
  const [localValue, setLocalValue] = useState(value);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync local value with prop when prop changes externally
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = parseFloat(e.target.value);
      setLocalValue(newValue);

      // Debounce the onChange callback
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        onChange(newValue);
      }, debounceMs);
    },
    [onChange, debounceMs]
  );

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const min = parameter.min ?? 0;
  const max = parameter.max ?? 1;
  const step = parameter.step ?? (max - min) / 100;

  // Calculate percentage for gradient
  const percentage = ((localValue - min) / (max - min)) * 100;

  return (
    <div className="parameter-slider">
      <div className="parameter-header">
        <label className="parameter-label" htmlFor={`param-${parameter.name}`}>
          {parameter.display_name}
        </label>
        <span className="parameter-value">{localValue.toFixed(2)}</span>
      </div>
      <input
        id={`param-${parameter.name}`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={localValue}
        onChange={handleChange}
        className="slider-input"
        style={{
          background: `linear-gradient(to right, #4a90d9 0%, #4a90d9 ${percentage}%, #333 ${percentage}%, #333 100%)`,
        }}
      />
      <div className="parameter-range">
        <span>{min}</span>
        <span>{max}</span>
      </div>
      {parameter.description && (
        <p className="parameter-description">{parameter.description}</p>
      )}
    </div>
  );
}
