import { useMemo } from 'react';

export interface Ellipse2DProps {
  /** 2x2 covariance or AᵀA matrix */
  matrix: number[][];
  width?: number;
  height?: number;
  showAxes?: boolean;
  showEigenvalues?: boolean;
}

/**
 * Visualizes a 2x2 positive semi-definite matrix as an ellipse.
 * The ellipse represents the error surface or covariance structure.
 * Principal axes are shown with their corresponding eigenvalues.
 */
export function Ellipse2D({
  matrix,
  width = 300,
  height = 300,
  showAxes = true,
  showEigenvalues = true,
}: Ellipse2DProps) {
  // Compute eigendecomposition of 2x2 symmetric matrix
  const { eigenvalues, eigenvectors, angle, semiMajor, semiMinor } = useMemo(() => {
    if (!matrix || matrix.length !== 2 || matrix[0]?.length !== 2) {
      return {
        eigenvalues: [0, 0],
        eigenvectors: [[1, 0], [0, 1]],
        angle: 0,
        semiMajor: 0,
        semiMinor: 0,
      };
    }

    const a = matrix[0][0];
    const b = matrix[0][1];
    const c = matrix[1][0];
    const d = matrix[1][1];

    // For symmetric matrix, b should equal c
    const bAvg = (b + c) / 2;

    // Eigenvalues of 2x2 symmetric matrix: [[a, b], [b, d]]
    // λ = (a + d)/2 ± sqrt(((a - d)/2)² + b²)
    const trace = a + d;
    const det = a * d - bAvg * bAvg;
    const discriminant = Math.sqrt(Math.max(0, trace * trace / 4 - det));

    const lambda1 = trace / 2 + discriminant;
    const lambda2 = trace / 2 - discriminant;

    // Eigenvector for lambda1
    let v1x: number, v1y: number;
    if (Math.abs(bAvg) > 1e-10) {
      v1x = lambda1 - d;
      v1y = bAvg;
    } else if (Math.abs(a - d) > 1e-10) {
      v1x = 1;
      v1y = 0;
    } else {
      v1x = 1;
      v1y = 0;
    }

    // Normalize
    const norm1 = Math.sqrt(v1x * v1x + v1y * v1y);
    if (norm1 > 1e-10) {
      v1x /= norm1;
      v1y /= norm1;
    }

    // Second eigenvector is perpendicular (for symmetric matrix)
    const v2x = -v1y;
    const v2y = v1x;

    // Angle of first principal axis
    const theta = Math.atan2(v1y, v1x);

    // Semi-axes lengths (sqrt of eigenvalues for ellipse from covariance)
    const major = Math.sqrt(Math.max(0, lambda1));
    const minor = Math.sqrt(Math.max(0, lambda2));

    return {
      eigenvalues: [lambda1, lambda2],
      eigenvectors: [[v1x, v1y], [v2x, v2y]],
      angle: theta * (180 / Math.PI),
      semiMajor: major,
      semiMinor: minor,
    };
  }, [matrix]);

  // Calculate scaling to fit in viewport
  const scale = useMemo(() => {
    const maxRadius = Math.max(semiMajor, semiMinor);
    if (maxRadius < 1e-10) return 1;
    const padding = 40;
    const availableSize = Math.min(width, height) - 2 * padding;
    return availableSize / (2 * maxRadius);
  }, [semiMajor, semiMinor, width, height]);

  const cx = width / 2;
  const cy = height / 2;

  if (!matrix || matrix.length !== 2 || matrix[0]?.length !== 2) {
    return <div className="visualizer-empty">Requires a 2x2 matrix</div>;
  }

  const scaledMajor = semiMajor * scale;
  const scaledMinor = semiMinor * scale;

  // Principal axis endpoints
  const axis1End = {
    x: cx + eigenvectors[0][0] * scaledMajor,
    y: cy - eigenvectors[0][1] * scaledMajor, // SVG y is inverted
  };
  const axis1Start = {
    x: cx - eigenvectors[0][0] * scaledMajor,
    y: cy + eigenvectors[0][1] * scaledMajor,
  };
  const axis2End = {
    x: cx + eigenvectors[1][0] * scaledMinor,
    y: cy - eigenvectors[1][1] * scaledMinor,
  };
  const axis2Start = {
    x: cx - eigenvectors[1][0] * scaledMinor,
    y: cy + eigenvectors[1][1] * scaledMinor,
  };

  return (
    <div className="visualizer ellipse-visualizer">
      <svg width={width} height={height}>
        {/* Grid lines */}
        <g className="grid" opacity={0.2}>
          <line x1={0} y1={cy} x2={width} y2={cy} stroke="#666" strokeDasharray="4,4" />
          <line x1={cx} y1={0} x2={cx} y2={height} stroke="#666" strokeDasharray="4,4" />
        </g>

        {/* Ellipse */}
        <ellipse
          cx={cx}
          cy={cy}
          rx={scaledMajor}
          ry={scaledMinor}
          transform={`rotate(${-angle}, ${cx}, ${cy})`}
          fill="rgba(74, 144, 217, 0.3)"
          stroke="#4a90d9"
          strokeWidth={2}
        />

        {/* Principal axes */}
        {showAxes && (
          <g className="principal-axes">
            {/* First principal axis (major) */}
            <line
              x1={axis1Start.x}
              y1={axis1Start.y}
              x2={axis1End.x}
              y2={axis1End.y}
              stroke="#e74c3c"
              strokeWidth={2}
              markerEnd="url(#arrowhead-red)"
            />
            {/* Second principal axis (minor) */}
            <line
              x1={axis2Start.x}
              y1={axis2Start.y}
              x2={axis2End.x}
              y2={axis2End.y}
              stroke="#27ae60"
              strokeWidth={2}
              markerEnd="url(#arrowhead-green)"
            />
          </g>
        )}

        {/* Arrowhead markers */}
        <defs>
          <marker
            id="arrowhead-red"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#e74c3c" />
          </marker>
          <marker
            id="arrowhead-green"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#27ae60" />
          </marker>
        </defs>

        {/* Center point */}
        <circle cx={cx} cy={cy} r={4} fill="#333" />
      </svg>

      {/* Eigenvalue labels */}
      {showEigenvalues && (
        <div className="eigenvalue-labels" style={{ marginTop: 10, fontSize: 12 }}>
          <div style={{ display: 'flex', gap: 20, justifyContent: 'center' }}>
            <span>
              <span style={{ color: '#e74c3c', fontWeight: 'bold' }}>λ₁</span> ={' '}
              {eigenvalues[0].toFixed(4)}
            </span>
            <span>
              <span style={{ color: '#27ae60', fontWeight: 'bold' }}>λ₂</span> ={' '}
              {eigenvalues[1].toFixed(4)}
            </span>
          </div>
          <div style={{ textAlign: 'center', color: '#666', marginTop: 4 }}>
            Condition: {eigenvalues[1] > 1e-10 ? (eigenvalues[0] / eigenvalues[1]).toFixed(2) : '∞'}
          </div>
        </div>
      )}
    </div>
  );
}

export default Ellipse2D;
