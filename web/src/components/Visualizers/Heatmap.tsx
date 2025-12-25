import { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';

export interface HeatmapProps {
  data: number[][];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  colorScheme?: 'diverging' | 'sequential';
  showLabels?: boolean;
}

export function Heatmap({
  data,
  width = 400,
  height = 300,
  margin = { top: 30, right: 60, bottom: 40, left: 50 },
  colorScheme = 'diverging',
  showLabels = true,
}: HeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Calculate dimensions
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Get data bounds
  const { minVal, maxVal, rows, cols } = useMemo(() => {
    if (!data || data.length === 0) {
      return { minVal: 0, maxVal: 1, rows: 0, cols: 0 };
    }
    const flat = data.flat();
    return {
      minVal: Math.min(...flat),
      maxVal: Math.max(...flat),
      rows: data.length,
      cols: data[0]?.length || 0,
    };
  }, [data]);

  // Create color scale
  const colorScale = useMemo(() => {
    if (colorScheme === 'diverging') {
      // Diverging scale centered at 0
      const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal));
      return d3.scaleSequential(d3.interpolateRdBu).domain([absMax, -absMax]);
    } else {
      // Sequential scale from min to max
      return d3.scaleSequential(d3.interpolateViridis).domain([minVal, maxVal]);
    }
  }, [minVal, maxVal, colorScheme]);

  useEffect(() => {
    if (!svgRef.current || !data || data.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Cell dimensions
    const cellWidth = innerWidth / cols;
    const cellHeight = innerHeight / rows;

    // Draw cells
    const cellsGroup = g.append('g').attr('class', 'cells');

    data.forEach((row, i) => {
      row.forEach((value, j) => {
        cellsGroup
          .append('rect')
          .attr('x', j * cellWidth)
          .attr('y', i * cellHeight)
          .attr('width', cellWidth)
          .attr('height', cellHeight)
          .attr('fill', colorScale(value))
          .attr('stroke', '#fff')
          .attr('stroke-width', 0.5)
          .on('mouseover', function (event) {
            d3.select(this).attr('stroke', '#000').attr('stroke-width', 2);
            if (tooltipRef.current) {
              tooltipRef.current.style.display = 'block';
              tooltipRef.current.style.left = `${event.pageX + 10}px`;
              tooltipRef.current.style.top = `${event.pageY - 10}px`;
              tooltipRef.current.textContent = `[${i}, ${j}]: ${value.toFixed(4)}`;
            }
          })
          .on('mouseout', function () {
            d3.select(this).attr('stroke', '#fff').attr('stroke-width', 0.5);
            if (tooltipRef.current) {
              tooltipRef.current.style.display = 'none';
            }
          });
      });
    });

    // Add row labels (y-axis)
    if (showLabels) {
      const yAxis = g.append('g').attr('class', 'y-axis');

      for (let i = 0; i < rows; i++) {
        yAxis
          .append('text')
          .attr('x', -5)
          .attr('y', i * cellHeight + cellHeight / 2)
          .attr('text-anchor', 'end')
          .attr('dominant-baseline', 'middle')
          .attr('font-size', Math.min(12, cellHeight * 0.8))
          .attr('fill', '#666')
          .text(i);
      }

      // Add column labels (x-axis)
      const xAxis = g.append('g').attr('class', 'x-axis');

      for (let j = 0; j < cols; j++) {
        xAxis
          .append('text')
          .attr('x', j * cellWidth + cellWidth / 2)
          .attr('y', -5)
          .attr('text-anchor', 'middle')
          .attr('font-size', Math.min(12, cellWidth * 0.8))
          .attr('fill', '#666')
          .text(j);
      }
    }

    // Add color legend
    const legendWidth = 20;
    const legendHeight = innerHeight;
    const legendGroup = svg
      .append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - margin.right + 10},${margin.top})`);

    // Create gradient for legend
    const defs = svg.append('defs');
    const gradient = defs
      .append('linearGradient')
      .attr('id', 'heatmap-gradient')
      .attr('x1', '0%')
      .attr('y1', '100%')
      .attr('x2', '0%')
      .attr('y2', '0%');

    // Add gradient stops
    const nStops = 10;
    for (let i = 0; i <= nStops; i++) {
      const t = i / nStops;
      const value = minVal + t * (maxVal - minVal);
      gradient
        .append('stop')
        .attr('offset', `${t * 100}%`)
        .attr('stop-color', colorScale(value));
    }

    // Draw legend rect
    legendGroup
      .append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'url(#heatmap-gradient)')
      .attr('stroke', '#ccc');

    // Add legend labels
    legendGroup
      .append('text')
      .attr('x', legendWidth + 5)
      .attr('y', 10)
      .attr('font-size', 10)
      .attr('fill', '#666')
      .text(maxVal.toFixed(2));

    legendGroup
      .append('text')
      .attr('x', legendWidth + 5)
      .attr('y', legendHeight)
      .attr('font-size', 10)
      .attr('fill', '#666')
      .text(minVal.toFixed(2));

  }, [data, innerWidth, innerHeight, rows, cols, colorScale, margin, showLabels, minVal, maxVal, width]);

  if (!data || data.length === 0) {
    return <div className="visualizer-empty">No data to display</div>;
  }

  return (
    <div className="visualizer heatmap-visualizer" style={{ position: 'relative' }}>
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

export default Heatmap;
