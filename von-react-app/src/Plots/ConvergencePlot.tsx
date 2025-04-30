import * as d3 from 'd3';
import {
  useMemo, useRef, useEffect,
} from 'react';
import { MinType, convergenceColor } from '../constants.ts';
import Calc, { MinimizationPoint } from '../EcatTypes/Calc.ts';
import { Cmap } from './SiteLabel.ts';

function statusColor(status: string): string {
  if (status.includes('Converged')) return 'limegreen';
  if (status.includes('Fail') || status.includes('Error')) return 'red';
  return 'orange';
}

function pickPowerForFit(
  t0_s: number,
  t1_s: number,
  E0_eV: number,
  E1_eV: number,
  points: MinimizationPoint[],
): number {
  const fractions = [...Array(24).keys()].map((power) => {
    const zero = E1_eV * (1 + 2 ** -power) - (2 ** -power) * E0_eV;
    const range = Math.log(E1_eV - zero) - Math.log(E0_eV - zero);
    const refSlope = range / (t1_s - t0_s);
    const res = points.map((point) => {
      const linePoint = refSlope * (point.time_s - t0_s) + Math.log(E0_eV - zero);
      return ((Math.log(point.Etot_eV - zero) - linePoint) / range) ** 2;
    });
    return Math.abs(res.reduce((a, b) => a + b, 0));
  });
  return fractions.indexOf(Math.min(...fractions));
}

type ConvergencePlotProps = {
  dimensions: {height: number, width: number};
  calc: Calc | null;
  convLegendToggle: {[key: string]: boolean};
  setConvLegendToggle: React.Dispatch<React.SetStateAction<{[key: string]: boolean}>>;
};

export default function ConvergencePlot({
  dimensions, calc, convLegendToggle, setConvLegendToggle,
}: ConvergencePlotProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const lationLabel = calc?.geo === 'bulk' ? 'Lattice' : 'Ionic';
  const cmaps: Cmap[] = [
    { name: 'Elec', fill: 'pink' },
    { name: lationLabel, fill: 'gradient' },
  ];

  // Fetch the electronic minimization data
  const elecData = useMemo(() => {
    if (!calc?.minData) return [];
    return calc.minData.filter((d) => [MinType.ElecConverged, MinType.ElecNotConverged].includes(d.type)).slice(1);
  }, [calc]);

  // Fetch the latice or ionic minimization data
  const lationData = useMemo(() => {
    if (!calc?.minData) return [];
    return calc.minData.filter((d) => [MinType.Ionic, MinType.Lattice].includes(d.type));
  }, [calc]);

  // Determine the bounds of the plot in the x and y directions
  const allData = useMemo(() => lationData.concat(elecData), [elecData, lationData]);
  const energyMin_eV = useMemo(() => Math.min(...allData.map((d) => d.Etot_eV)), [allData]);
  const energyMax_eV = useMemo(() => Math.max(...allData.map((d) => d.Etot_eV)), [allData]);
  const tMin_s = useMemo(() => Math.min(...allData.map((d) => d.time_s)), [allData]);
  const tMax_s = useMemo(() => Math.max(...allData.map((d) => d.time_s)), [allData]);
  const useHours = tMax_s < 3600 * 48;
  const tFactor = useHours ? 1 / 3600 : 1 / (3600 * 24);

  const yOffset_eV = useMemo(() => {
    const power = pickPowerForFit(tMin_s, tMax_s, energyMax_eV, energyMin_eV, lationData);
    return -energyMin_eV * (1 + 2 ** -power) + (2 ** -power) * energyMax_eV;
  }, [calc, convLegendToggle]);

  const f = (value_eV: number): number => value_eV + yOffset_eV;
  // Choose sensible bounds for the y axis to give a little space around the data
  const y_min = f(energyMin_eV) * 0.8;
  const y_max = f(energyMax_eV) * 1.3;

  function fillColor(cmap: Cmap) {
    if (cmap.fill !== 'gradient') return cmap.fill;
    if (lationData.some((d) => d.forces_max_L2_eVpA)) return 'url(#lationGradient)';
    return 'cyan';
  }

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous SVG content
    svg.attr('width', dimensions.width)
      .attr('height', dimensions.height);
    // Gradient for lattice points
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'lationGradient')
      .attr('x1', '0%')
      .attr('x2', '100%')
      .attr('y1', '0%')
      .attr('y2', '0%');
    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', 'red');
    gradient.append('stop')
      .attr('offset', '33%')
      .attr('stop-color', 'orange');
    gradient.append('stop')
      .attr('offset', '67%')
      .attr('stop-color', 'yellow');
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', 'limegreen');

    const margin = {
      top: 50, right: 100, bottom: 80, left: 80,
    };
    const width = dimensions.width - margin.left - margin.right;
    const height = dimensions.height - margin.top - margin.bottom;

    const xScale = d3.scaleLinear()
      .range([0, width])
      .domain([0, tMax_s * tFactor * 1.03]);
    const yScale = d3.scaleLog()
      .range([height, 0])
      .domain([y_min, y_max]);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    /// ///////////////////////////////////////////////////////////////////////
    // Axes
    // X Axis
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale)
        .tickPadding(5)
        .tickSize(10)
        .tickSizeOuter(0)
        .ticks(7))
      .style('font-size', '1em')
      .attr('fill', 'white');
    // Axis Label
    g.append('text')
      .attr('transform', `translate(0,${height})`)
      .attr('fill', 'white')
      .attr('y', 40)
      .attr('x', width / 2)
      .attr('dy', '0.71em')
      .attr('text-anchor', 'middle')
      .style('font-size', '1.2em')
      .text(`Time (${useHours ? 'h' : 'd'})`);

    // Y Axis
    g.append('g')
      .call(d3.axisLeft(yScale)
        .tickPadding(5)
        .tickSize(10)
        .tickSizeOuter(0))
      .selectAll('text')
      .style('font-size', '1.5em')
      .attr('fill', 'white');
    // Axis Label
    g.append('text')
      .attr('fill', 'white')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -80)
      .attr('dy', '0.71em')
      .attr('text-anchor', 'middle')
      .style('font-size', '1.2em') // You can also adjust the font size of the axis label here
      .text('Energy (eV)');

    /// ///////////////////////////////////////////////////////////////////////
    // Data points
    const elecMin = g.selectAll('.elec-min')
      .data(elecData)
      .enter()
      .append('g')
      .attr('class', 'elec-min-group');

    elecMin.append('circle')
      .attr('cx', (d) => xScale(d.time_s * tFactor))
      .attr('cy', (d) => yScale(f(d.Etot_eV)))
      .attr('r', 2)
      .attr('fill', cmaps.find((cmap) => cmap.name === 'Elec')!.fill);

    const lationMin = g.selectAll('.lation-min')
      .data(lationData)
      .enter()
      .append('g')
      .attr('class', 'lation-min-group');

    function circleColor(d: MinimizationPoint) {
      if (d.forces_max_L2_eVpA) return convergenceColor(d.forces_max_L2_eVpA * 1000);
      return 'cyan';
    }

    lationMin.append('circle')
      .attr('cx', (d) => xScale(d.time_s * tFactor))
      .attr('cy', (d) => yScale(f(d.Etot_eV)))
      .attr('r', 6)
      .attr('fill', (d) => circleColor(d))
      .attr('fill-opacity', 0.6)
      .attr('stroke', (d) => circleColor(d))
      .on('mouseover', (event, d) => {
        if (d.forces_max_L2_eVpA != null && convLegendToggle[lationLabel]) {
          const tooltip = document.getElementById('tooltip');
          if (tooltip && svgRef.current) {
            const svgPosition = svgRef.current.getBoundingClientRect();
            tooltip.style.opacity = '1';
            tooltip.innerHTML = `${(d.forces_max_L2_eVpA * 1000).toFixed(0)}`;
            tooltip.style.color = convergenceColor(d.forces_max_L2_eVpA * 1000);
            tooltip.style.left = `${event.pageX - svgPosition.left}px`;
            tooltip.style.top = `${event.pageY - svgPosition.top}px`;
            tooltip.style.transform = 'translateX(20%) translateY(-120%)';
          }
        }
      })
      .on('mouseout', () => {
        const tooltip = document.getElementById('tooltip');
        if (tooltip) {
          tooltip.style.opacity = '0';
        }
      });

    /// ///////////////////////////////////////////////////////////////////////
    // Status text
    if (calc?.lastModified) {
      svg.append('text')
        .attr('transform', 'translate(0, 25)')
        .attr('dominant-baseline', 'hanging')
        .style('font-size', '1em')
        .attr('fill', 'DarkTurquoise')
        .text(`${calc.formattedlastModifiedAgo} ago`);

      svg.append('text')
        .attr('transform', 'translate(0, 5)')
        .attr('dominant-baseline', 'hanging')
        .style('font-size', '1em')
        .attr('fill', `${statusColor(calc.status!)}`)
        .text(`${calc.statusIcon} ${calc.status!.replaceAll('_', ' ')}`);
    }

    /// ///////////////////////////////////////////////////////////////////////
    // Legend Title
    // Create a text element to append tspsans to
    const legendOffset = width + margin.left + margin.right - 10;

    if (calc) {
      const textElement = svg.append('text')
        .attr('transform', `translate(${legendOffset}, 23)`)
        .attr('dominant-baseline', 'hanging')
        .attr('text-anchor', 'end')
        .style('font-size', '1.5em')
        .attr('fill', 'white');

      let dyValue = 0; // Initial dy value
      textElement.append('tspan')
        .attr('dy', `${dyValue}em`)
        .style('font-size', '1em')
        .text(calc.geo === 'bulk' ? 'bulk ' : `${calc.adsorbate} `);
      // We alternate subscript and normal text by adjusting the dy and font-size
      // Regular expression to match groups of letters or numbers
      const regex = /(\d+|[a-zA-Z]+)/g;
      let match = regex.exec(calc.material);
      while (match !== null) {
        // Check if the segment is numeric
        if (Number.isNaN(parseInt(match[0], 10))) {
          // Append letter group normally
          textElement.append('tspan')
            .attr('dy', `${dyValue}em`)
            .style('font-size', '1em')
            .attr('fill', calc.geo === 'bulk' ? 'gold' : 'DarkTurquoise')
            .text(match[0]);
        } else {
          // Append number group as subscript
          textElement.append('tspan')
            .attr('dy', '0.5em')
            .style('font-size', '0.8em')
            .attr('fill', calc.geo === 'bulk' ? 'gold' : 'DarkTurquoise')
            .text(match[0]);
          dyValue = -0.4;
        }
        match = regex.exec(calc.material);
      }
    }

    /// ///////////////////////////////////////////////////////////////////////
    // Legend
    const labelFontSize_em = 1;
    const rectSize = 20;

    // Hint Text
    svg.append('text')
      .attr('transform', `translate(${legendOffset}, ${cmaps.length * (rectSize + 5) + margin.top + 10})`)
      .attr('dominant-baseline', 'hanging')
      .attr('text-anchor', 'end')
      .style('font-size', `${labelFontSize_em * 0.8}em`)
      .style('font-style', 'italic')
      .attr('fill', 'silver')
      .text('Click to toggle');

    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${legendOffset - rectSize}, 20)`)
      .selectAll('g')
      .data(cmaps)
      .enter()
      .append('g')
      .attr('transform', (_, i) => `translate(0, ${i * (rectSize + 5) + margin.top - 15})`)
      .style('cursor', 'pointer')
      .on('click', (_, cmap) => {
        setConvLegendToggle((currentToggle) => ({ ...currentToggle, [cmap.name]: !currentToggle[cmap.name] }));
      });

    legend.append('rect')
      .attr('class', 'legend-rect')
      .style('pointer-events', 'all')
      .attr('width', rectSize)
      .attr('height', rectSize)
      .attr('fill', (cmap) => (convLegendToggle[cmap.name.replace(' ', '')] ? fillColor(cmap) : 'none'))
      .attr('stroke', (cmap) => fillColor(cmap))
      .attr('stroke-width', 2);

    // The text to the left of the rectangles
    legend.append('text')
      .attr('x', -10)
      .attr('y', 10)
      .attr('dy', '0.35em')
      .style('font-size', `${labelFontSize_em}em`)
      .attr('text-anchor', 'end')
      .attr('fill', 'white')
      .text((cmap) => cmap.name.replace(/(?<=.)[A-Z]/g, ' $&'));
  }, [dimensions, calc]);

  // Update the opacity of levels based on the convLegendToggle state
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const g = svg.select('g');
    g.selectAll('.elec-min-group').attr('opacity', convLegendToggle.Elec ? 1 : 0);
    g.selectAll('.lation-min-group').attr('opacity', convLegendToggle[lationLabel] ? 1 : 0);

    // When a legend is toggled off, make the legend check box appear hollow
    svg.selectAll('.legend-rect')
      .attr('fill', (d) => {
        const cmap = d as Cmap;
        return convLegendToggle[cmap.name] ? fillColor(cmap) : 'none';
      });
  }, [convLegendToggle, calc]);

  return (
    <div style={{ position: 'relative' }}>
      <div
        id="tooltip"
        style={{
          position: 'absolute',
          opacity: 0,
          pointerEvents: 'none',
          background: 'rgba(0, 0, 0, 0.75)',
          padding: '5px',
          borderRadius: '5px',
          fontSize: '1em',
          zIndex: 10,
        }}
      >
        Tooltip
      </div>
      <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />
    </div>
  );
}
