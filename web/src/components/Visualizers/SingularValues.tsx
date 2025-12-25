import { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';

export interface SingularValuesProps {
  values: number[];
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
  logScale?: boolean;
  showCumulativeEnergy?: boolean;
}

export function SingularValues({
  values,
  width = 400,
  height = 250,
  margin = { top: 20, right: 50, bottom: 40, left: 60 },
  logScale = false,
  showCumulativeEnergy = true,
}: SingularValuesProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  // Calculate dimensions
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Calculate cumulative energy
  const { cumulativeEnergy } = useMemo(() => {
    if (!values || values.length === 0) {
      return { totalEnergy: 0, cumulativeEnergy: [] };
    }
    const squaredValues = values.map((v) => v * v);
    const total = squaredValues.reduce((a, b) => a + b, 0);
    let cumSum = 0;
    const cumulative = squaredValues.map((v) => {
      cumSum += v;
      return total > 0 ? cumSum / total : 0;
    });
    return { totalEnergy: total, cumulativeEnergy: cumulative };
  }, [values]);

  useEffect(() => {
    if (!svgRef.current || !values || values.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // X scale (bar positions)
    const xScale = d3
      .scaleBand()
      .domain(values.map((_, i) => i.toString()))
      .range([0, innerWidth])
      .padding(0.2);

    // Y scale for bars (singular values)
    const maxVal = Math.max(...values);
    const minVal = logScale ? Math.min(...values.filter((v) => v > 0)) : 0;

    const yScale = logScale
      ? d3.scaleLog().domain([minVal, maxVal]).range([innerHeight, 0]).nice()
      : d3.scaleLinear().domain([0, maxVal]).range([innerHeight, 0]).nice();

    // Y scale for cumulative energy (right axis)
    const yScaleEnergy = d3.scaleLinear().domain([0, 1]).range([innerHeight, 0]);

    // Draw bars
    g.selectAll('.bar')
      .data(values)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (_, i) => xScale(i.toString()) || 0)
      .attr('y', (d) => yScale(Math.max(d, logScale ? minVal : 0)))
      .attr('width', xScale.bandwidth())
      .attr('height', (d) => {
        const yVal = yScale(Math.max(d, logScale ? minVal : 0));
        return innerHeight - yVal;
      })
      .attr('fill', '#4a90d9')
      .attr('opacity', 0.8);

    // Draw cumulative energy line
    if (showCumulativeEnergy && cumulativeEnergy.length > 0) {
      const line = d3
        .line<number>()
        .x((_, i) => (xScale(i.toString()) || 0) + xScale.bandwidth() / 2)
        .y((d) => yScaleEnergy(d));

      g.append('path')
        .datum(cumulativeEnergy)
        .attr('class', 'cumulative-line')
        .attr('fill', 'none')
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', 2)
        .attr('d', line);

      // Add dots on the line
      g.selectAll('.cumulative-dot')
        .data(cumulativeEnergy)
        .join('circle')
        .attr('class', 'cumulative-dot')
        .attr('cx', (_, i) => (xScale(i.toString()) || 0) + xScale.bandwidth() / 2)
        .attr('cy', (d) => yScaleEnergy(d))
        .attr('r', 3)
        .attr('fill', '#e74c3c');
    }

    // X axis
    const xAxis = d3.axisBottom(xScale).tickValues(
      values.length <= 10
        ? values.map((_, i) => i.toString())
        : values
            .map((_, i) => i)
            .filter((i) => i % Math.ceil(values.length / 10) === 0)
            .map((i) => i.toString())
    );

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

    // Y axis (left - singular values)
    const yAxis = logScale
      ? d3.axisLeft(yScale).ticks(5, '.0e')
      : d3.axisLeft(yScale).ticks(5);

    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -45)
      .attr('fill', '#666')
      .attr('text-anchor', 'middle')
      .text('Singular Value');

    // Y axis (right - cumulative energy)
    if (showCumulativeEnergy) {
      const yAxisRight = d3
        .axisRight(yScaleEnergy)
        .ticks(5)
        .tickFormat((d) => `${(d as number * 100).toFixed(0)}%`);

      g.append('g')
        .attr('class', 'y-axis-right')
        .attr('transform', `translate(${innerWidth},0)`)
        .call(yAxisRight)
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', 40)
        .attr('fill', '#e74c3c')
        .attr('text-anchor', 'middle')
        .text('Cumulative Energy');
    }

    // Legend
    const legend = svg
      .append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${margin.left + 10},${margin.top + 5})`);

    legend
      .append('rect')
      .attr('width', 12)
      .attr('height', 12)
      .attr('fill', '#4a90d9')
      .attr('opacity', 0.8);

    legend
      .append('text')
      .attr('x', 16)
      .attr('y', 10)
      .attr('font-size', 10)
      .attr('fill', '#666')
      .text('σᵢ');

    if (showCumulativeEnergy) {
      legend
        .append('line')
        .attr('x1', 0)
        .attr('y1', 24)
        .attr('x2', 12)
        .attr('y2', 24)
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', 2);

      legend
        .append('text')
        .attr('x', 16)
        .attr('y', 28)
        .attr('font-size', 10)
        .attr('fill', '#666')
        .text('Σσᵢ²/Σσ²');
    }
  }, [values, innerWidth, innerHeight, margin, logScale, showCumulativeEnergy, cumulativeEnergy]);

  if (!values || values.length === 0) {
    return <div className="visualizer-empty">No singular values to display</div>;
  }

  return (
    <div className="visualizer singular-values-visualizer">
      <svg ref={svgRef} width={width} height={height} />
    </div>
  );
}

export default SingularValues;
