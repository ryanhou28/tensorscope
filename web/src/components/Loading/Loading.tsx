interface SpinnerProps {
  size?: 'small' | 'medium' | 'large';
  className?: string;
}

export function Spinner({ size = 'medium', className = '' }: SpinnerProps) {
  return (
    <div className={`spinner spinner-${size} ${className}`} role="status">
      <span className="sr-only">Loading...</span>
    </div>
  );
}

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  variant?: 'text' | 'rect' | 'circle';
  className?: string;
}

export function Skeleton({
  width,
  height,
  variant = 'rect',
  className = '',
}: SkeletonProps) {
  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  return (
    <div
      className={`skeleton skeleton-${variant} ${className}`}
      style={style}
      aria-hidden="true"
    />
  );
}

interface GraphSkeletonProps {
  className?: string;
}

export function GraphSkeleton({ className = '' }: GraphSkeletonProps) {
  return (
    <div className={`graph-skeleton ${className}`}>
      <div className="skeleton-nodes">
        <Skeleton width={120} height={60} className="skeleton-node" />
        <Skeleton width={120} height={60} className="skeleton-node" />
        <Skeleton width={120} height={60} className="skeleton-node" />
      </div>
      <div className="skeleton-edges">
        <Skeleton width="100%" height={2} className="skeleton-edge" />
        <Skeleton width="100%" height={2} className="skeleton-edge" />
      </div>
      <div className="skeleton-loading-text">
        <Spinner size="small" />
        <span>Loading graph...</span>
      </div>
    </div>
  );
}

interface InspectorSkeletonProps {
  className?: string;
}

export function InspectorSkeleton({ className = '' }: InspectorSkeletonProps) {
  return (
    <div className={`inspector-skeleton ${className}`}>
      <div className="skeleton-header">
        <Skeleton width={150} height={24} variant="text" />
        <Skeleton width={80} height={16} variant="text" />
      </div>
      <div className="skeleton-tabs">
        <Skeleton width={60} height={32} />
        <Skeleton width={80} height={32} />
        <Skeleton width={70} height={32} />
      </div>
      <div className="skeleton-content">
        <Skeleton width="100%" height={200} />
      </div>
      <div className="skeleton-stats">
        <Skeleton width="45%" height={40} />
        <Skeleton width="45%" height={40} />
        <Skeleton width="45%" height={40} />
        <Skeleton width="45%" height={40} />
      </div>
    </div>
  );
}

export default Spinner;
