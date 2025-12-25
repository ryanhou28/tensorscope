import { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';

export interface VectorStemProps {
  values: number[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  showZeroLine?: boolean;
  color?: string;
}

/**
 * Stem plot visualization for vectors.
 * Shows index vs value with vertical stems from zero line.
 */
export function VectorStem({
  values,
  width = 400,
  height = 200,
  margin = { top: 20, right: 20, bottom: 40, left: 60 },
  showZeroLine = true,
  color = '#4a90d9',
}: VectorStemProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Calculate dimensions
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Data bounds
  const { minVal, maxVal } = useMemo(() => {
    if (!values || values.length === 0) {
      return { minVal: -1, maxVal: 1 };
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    // Ensure zero is included in the range
    return {
      minVal: Math.min(min, 0),
      maxVal: Math.max(max, 0),
    };
  }, [values]);

  useEffect(() => {
    if (!svgRef.current || !values || values.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // X scale
    const xScale = d3
      .scaleLinear()
      .domain([0, values.length - 1])
      .range([0, innerWidth]);

    // Y scale
    const yScale = d3
      .scaleLinear()
      .domain([minVal, maxVal])
      .range([innerHeight, 0])
      .nice();

    // Zero line position
    const zeroY = yScale(0);

    // Draw zero line
    if (showZeroLine) {
      g.append('line')
        .attr('class', 'zero-line')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', zeroY)
        .attr('y2', zeroY)
        .attr('stroke', '#999')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4');
    }

    // Draw stems
    g.selectAll('.stem')
      .data(values)
      .join('line')
      .attr('class', 'stem')
      .attr('x1', (_, i) => xScale(i))
      .attr('x2', (_, i) => xScale(i))
      .attr('y1', zeroY)
      .attr('y2', (d) => yScale(d))
      .attr('stroke', color)
      .attr('stroke-width', 2);

    // Draw dots at the top of stems
    g.selectAll('.stem-dot')
      .data(values)
      .join('circle')
      .attr('class', 'stem-dot')
      .attr('cx', (_, i) => xScale(i))
      .attr('cy', (d) => yScale(d))
      .attr('r', 4)
      .attr('fill', color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .on('mouseover', function (event, d) {
        // Get the index from the DOM node position
        const nodes = g.selectAll('.stem-dot').nodes();
        const idx = nodes.indexOf(this);
        d3.select(this).attr('r', 6);
        if (tooltipRef.current) {
          tooltipRef.current.style.display = 'block';
          tooltipRef.current.style.left = `${event.pageX + 10}px`;
          tooltipRef.current.style.top = `${event.pageY - 10}px`;
          tooltipRef.current.textContent = `[${idx}]: ${d.toFixed(4)}`;
        }
      })
      .on('mouseout', function () {
        d3.select(this).attr('r', 4);
        if (tooltipRef.current) {
          tooltipRef.current.style.display = 'none';
        }
      });

    // X axis
    const xAxis = d3.axisBottom(xScale).ticks(Math.min(values.length, 10)).tickFormat(d3.format('d'));

    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 35)
      .attr('fill', '#666')
      .attr('text-anchor', 'middle')
      .text('Index');

    // Y axis
    const yAxis = d3.axisLeft(yScale).ticks(5);

    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -45)
      .attr('fill', '#666')
      .attr('text-anchor', 'middle')
      .text('Value');

  }, [values, innerWidth, innerHeight, margin, minVal, maxVal, showZeroLine, color]);

  if (!values || values.length === 0) {
    return <div className="visualizer-empty">No vector data to display</div>;
  }

  return (
    <div className="visualizer vector-stem-visualizer" style={{ position: 'relative' }}>
      <svg ref={svgRef} width={width} height={height} />
      <div
        ref={tooltipRef}
        style={{
          display: 'none',
          position: 'fixed',
          background: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '12px',
          pointerEvents: 'none',
          zIndex: 1000,
        }}
      />
    </div>
  );
}

export default VectorStem;
