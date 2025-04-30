import {
    useRef, useEffect, useMemo,
  } from 'react';
  import * as d3 from 'd3';
  
  interface PieData {
    label: string;
    value: number;
  }
  
  interface PieChartProps {
    data: PieData[];
    width: number;
    height: number;
  }
  
  export default function PieChart({
    data, width, height,
  }: PieChartProps) {
    const svgRef = useRef<SVGSVGElement | null>(null);
  
    const total = useMemo(() => data.reduce((sum, d) => sum + d.value, 0), [data]);
  
    useEffect(() => {
      if (!data || !svgRef.current) return;
  
      const radius = Math.min(width, height) / 2;
      const labelOffset = 1.15; // Adjust this factor to control the label distance from the center
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove(); // Clear previous SVG content
  
      svg.attr('width', width).attr('height', height);
  
      const chartGroup = svg
        .append('g')
        .attr('transform', `translate(${width / 2}, ${height / 2})`);
  
      const color = d3.scaleOrdinal(d3.schemeTableau10);
  
      const pieGenerator = d3.pie<PieData>()
        .value((d) => d.value);
  
      const arcGenerator = d3.arc<d3.PieArcDatum<PieData>>()
        .innerRadius(0)
        .outerRadius(radius);
  
      chartGroup
        .selectAll('path')
        .data(pieGenerator(data))
        .enter()
        .append('path')
        .attr('d', arcGenerator)
        .attr('fill', (d) => color(d.data.label) as string);
  
      // Add labels to each slice
      chartGroup
        .selectAll('text')
        .data(pieGenerator(data))
        .enter()
        .append('text')
        .attr('transform', (d) => {
          const [x, y] = arcGenerator.centroid(d);
          return `translate(${x * labelOffset}, ${y * labelOffset})`;
        })
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', 'white')
        .each(function(d) {
          const text = d3.select(this);
          const percentage = ((d.data.value / total) * 100).toFixed(1);
  
          // Add the top line with the label
          text.append('tspan')
            .attr('x', 0)
            .attr('dy', '-0.4em')
            .text(d.data.label);
  
          // Add the bottom line with percentage and count
          text.append('tspan')
            .attr('x', 0)
            .attr('dy', '1.2em')
            .text(`${percentage}% (${d.data.value})`);
        });
  
    }, [data, total, width, height]);
  
    return <svg ref={svgRef} />;
  }
  