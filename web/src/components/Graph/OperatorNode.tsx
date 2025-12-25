import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

export interface OperatorNodeData {
  label: string;
  operatorType: string;
  inputs: string[];
  outputs: string[];
  tags: string[];
  isSelected?: boolean;
  [key: string]: unknown;
}

interface OperatorNodeProps {
  data: OperatorNodeData;
  selected?: boolean;
}

function OperatorNode({ data, selected }: OperatorNodeProps) {
  const { label, operatorType, inputs, outputs, tags, isSelected } = data;

  return (
    <div
      className={`operator-node ${selected || isSelected ? 'selected' : ''}`}
    >
      {/* Input handles */}
      {inputs.map((input: string, index: number) => (
        <Handle
          key={`input-${input}`}
          type="target"
          position={Position.Left}
          id={input}
          style={{
            top: `${((index + 1) / (inputs.length + 1)) * 100}%`,
          }}
          className="node-handle input-handle"
        />
      ))}

      {/* Node content */}
      <div className="node-content">
        <div className="node-type">{operatorType}</div>
        <div className="node-label">{label}</div>
        {tags.length > 0 && (
          <div className="node-tags">
            {tags.slice(0, 2).map((tag: string) => (
              <span key={tag} className="node-tag">
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Output handles */}
      {outputs.map((output: string, index: number) => (
        <Handle
          key={`output-${output}`}
          type="source"
          position={Position.Right}
          id={output}
          style={{
            top: `${((index + 1) / (outputs.length + 1)) * 100}%`,
          }}
          className="node-handle output-handle"
        />
      ))}
    </div>
  );
}

export default memo(OperatorNode);
