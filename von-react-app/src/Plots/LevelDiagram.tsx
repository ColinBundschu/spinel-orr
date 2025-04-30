import {
  useRef, useEffect, useMemo,
} from 'react';
import * as d3 from 'd3';
import Study from '../EcatTypes/Study.ts';
import { CONVERGED_COLOR, convergenceColor } from '../constants.ts';
import Step from '../EcatTypes/Step.ts';
import { Cmap } from './SiteLabel.ts';

interface LevelDiagramProps {
  study: Study;
  strain_Pct: number;
  potential_V: number;
  dimensions: {height: number, width: number};
  mepSteps: (Step | null)[];
  steps: Step[];
  cmaps: Cmap[];
  levelsLegendToggle: {[key: string]: boolean};
  setLevelsLegendToggle: React.Dispatch<React.SetStateAction<{[key: string]: boolean}>>;
  hideConverged: boolean;
}

interface Barrier {
  index: number;
  Estart_eV: number;
  Eend_eV: number;
  population: string | null;
}

interface CmapGroup {
  name: string;
  base: Cmap | null;
  H: Cmap | null;
}

function renderYAxis(
  yScale: d3.ScaleLinear<number, number>, 
  svg: d3.Selection<SVGSVGElement | null, unknown, null, undefined>, 
  margin: { left: number; top: number }, 
  height: number
): void {
  // Remove any existing Y-axis elements
  svg.selectAll('.custom-y-axis').remove();

  // Create a group for the Y-axis
  const yAxisGroup = svg.append('g')
    .attr('class', 'custom-y-axis');

  // Add the vertical white line along the axis
  yAxisGroup.append('line')
    .attr('class', 'y-axis-line')
    .attr('x1', margin.left) // Vertical line at the left margin
    .attr('x2', margin.left)
    .attr('y1', 0) // Start from the top of the chart area
    .attr('y2', height + margin.top) // Extend to the bottom of the chart area
    .attr('stroke', 'white') // White line
    .attr('stroke-width', 2); // Adjust line thickness

  // Define tick values
  const ticks = yScale.ticks(10);
  const tickFormat = yScale.tickFormat(10);

  // Add tick lines
  yAxisGroup.selectAll('.y-grid-line')
    .data(ticks)
    .enter()
    .append('line')
    .attr('class', 'y-grid-line')
    .attr('x1', margin.left - 8)
    .attr('x2', margin.left)
    .attr('y1', (d) => yScale(d))
    .attr('y2', (d) => yScale(d))
    .attr('stroke', '#ddd')
    .attr('stroke-width', 1)

  // Add tick labels
  yAxisGroup.selectAll('.y-tick-label')
    .data(ticks)
    .enter()
    .append('text')
    .attr('class', 'y-tick-label')
    .attr('x', margin.left - 12) // Position to the left of the grid
    .attr('y', (d) => yScale(d))
    .attr('dy', '0.32em')
    .attr('text-anchor', 'end')
    .attr('fill', 'white')
    .style('font-size', '1em')
    .text((d) => tickFormat(d));

  // Add the Y-axis label
  yAxisGroup.append('text')
  .attr('class', 'y-axis-label')
  .attr('x', -height / 2) // Center the label along the axis
  .attr('y', 0) // Offset the label to the left of the axis
  .attr('transform', 'rotate(-90)') // Rotate the label for vertical orientation
  .attr('text-anchor', 'middle') // Align the start (left) of the text
  .attr('dominant-baseline', 'hanging') // Vertically center the text
  .attr('fill', 'white') // White text
  .style('font-size', '1.2em') // Adjust font size
  .text('Energy (eV)');
}

export default function LevelDiagram({
  study, strain_Pct, potential_V, dimensions, mepSteps, steps, cmaps,
  levelsLegendToggle, setLevelsLegendToggle, hideConverged,
}: LevelDiagramProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  const groupedCmaps = useMemo(() => {
    // Use reduce to group cmaps by group name
    const groupAccumulator = cmaps.reduce((acc: { [key: string]: CmapGroup }, cmap) => {
      const groupName = cmap.name.replace('*H ', '') || 'clean';
      if (!acc[groupName]) {
        acc[groupName] = { name: groupName, base: null, H: null };
      }
      if (cmap.name === groupName) {
        acc[groupName].base = cmap;
      } else {
        acc[groupName].H = cmap;
      }
      return acc;
    }, {});
    // Convert the accumulator object into an array of its values (CmapGroup objects)
    return Object.values(groupAccumulator);
  }, [cmaps]);

  // Disable page scrolling when hovering over the SVG
  useEffect(() => {
    const disableScroll = (event: WheelEvent) => {
      event.preventDefault();
    };

    const svgElement = svgRef.current;
    const handleMouseEnter = () => {
      window.addEventListener('wheel', disableScroll, { passive: false });
    };

    const handleMouseLeave = () => {
      window.removeEventListener('wheel', disableScroll);
    };

    if (svgElement) {
      svgElement.addEventListener('mouseenter', handleMouseEnter);
      svgElement.addEventListener('mouseleave', handleMouseLeave);
    }

    return () => {
      if (svgElement) {
        svgElement.removeEventListener('mouseenter', handleMouseEnter);
        svgElement.removeEventListener('mouseleave', handleMouseLeave);
      }
      window.removeEventListener('wheel', disableScroll);
    };
  }, []);

  // Calculate the energies for the steps at the current strain and potential
  const minEnergies = useMemo(() => mepSteps.map(
    (step) => step?.Elevel_eV(strain_Pct, potential_V) ?? Infinity,
  ), [mepSteps]);

  const markovPops = useMemo(() => {
    if (minEnergies.some((x) => x === null || minEnergies.length < 3 || !Number.isFinite(x))) return null;
    return study.calculateSteadyStatePopulation(minEnergies);
  }, [minEnergies]);

  // Calculate the heights of the energy barriers along the minimum energy path
  const mepBarriers = useMemo(() => minEnergies.reduce((acc: Barrier[], currentValue, i, arr) => {
    // Skip the last element as we are comparing each element with its next one.
    if (i === arr.length - 1) return acc;
    const nextValue = arr[i + 1];
    if (currentValue !== null && nextValue !== null && Number.isFinite(currentValue) && Number.isFinite(nextValue)) {
      acc.push({
        index: i,
        Estart_eV: currentValue,
        Eend_eV: nextValue,
        population: markovPops ? `${Math.round(100 * markovPops[i])}%` : null,
      });
    }
    return acc;
  }, []), [minEnergies]);

  // Main D3 rendering function
  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous SVG content
    svg.attr('width', dimensions.width)
      .attr('height', dimensions.height);

    const margin = {
      top: 20, right: 105, bottom: 60, left: 70,
    };
    const width = dimensions.width - margin.left - margin.right;
    const height = dimensions.height - margin.top - margin.bottom;

    const min_E = Math.min(...minEnergies);
    const max_E = Math.max(
      ...steps
        .filter((step) => levelsLegendToggle[step.calc.siteLabel.colorMap.name])
        .map((step) => step.Elevel_eV(strain_Pct, potential_V)!),
    );

    const xScale = d3.scaleLinear()
      .range([0, width])
      .domain([0.5, study.maxStepIndex + 1.5]);
    const yMin = min_E * 1.1 - max_E * 0.1;
    const yMax = max_E * 1.1 - min_E * 0.1;
    const yScale = d3.scaleLinear()
      .range([height, 0])
      .domain([yMin, yMax]);

    // Add a clipPath to restrict drawing outside the axes
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('x', 0)
      .attr('y', 0);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .attr('clip-path', 'url(#clip)'); // Apply the clipping path here

    const labelOffsetAbove = 3;
    const rectHeight = 3;
    // Define a zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 10]) // Limit zoom levels
      .translateExtent([[0, 0], [width, height]]) // Limit translation
      .on('zoom', (event) => {
        const transform = event.transform;

        // Get mouse position in SVG coordinates
        const [_, mouseY] = d3.pointer(event, svg.node());
        const cursorY = mouseY - margin.top; // Adjust for margins

        // Get the y value of the cursor in the plot
        const cursorDomainValue = yScale.invert(cursorY);

        // Position of the mouse as a fraction of the SVG draw space
        const fracFromTop = cursorY / height;

        // Calculate the new domain range scaled by zoom factor
        const scaledYRange = (yMax - yMin) / transform.k;

        // New domain boundaries
        const rawAdjustedDomainBottom = Math.max(cursorDomainValue - scaledYRange * (1 - fracFromTop), yMin);
        const adjustedDomainTop = Math.min(rawAdjustedDomainBottom + scaledYRange, yMax);
        const adjustedDomainBottom = adjustedDomainTop - scaledYRange;

        // Update the scale's domain
        yScale.domain([adjustedDomainBottom, adjustedDomainTop]);

        // Update axes and chart elements
        renderYAxis(yScale, svg, margin, height);

        // Update the position of all elements based on the new yScale
        g.selectAll<SVGRectElement, Step>('.level-group rect')
          .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) - rectHeight / 2);
        g.selectAll<SVGTextElement, Step>('.label-above')
          .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) - rectHeight / 2 - labelOffsetAbove);
        g.selectAll<SVGTextElement, Step>('.label-below-energy')
          .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) + labelOffsetBelow);
        g.selectAll<SVGTextElement, Step>('.label-below-force')
          .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) + labelOffsetBelow);
        g.selectAll<SVGRectElement, Barrier>('.barrier-rect-right')
          .attr('y', (barrier) => yScale(barrier.Eend_eV) - rectHeight / 2);
        g.selectAll<SVGRectElement, Barrier>('.barrier-rect-left')
          .attr('y', (barrier) => yScale(barrier.Estart_eV) - rectHeight / 2);
        g.selectAll<SVGLineElement, Barrier>('.barrier-group line')
          .attr('y1', (barrier) => yScale(barrier.Estart_eV))
          .attr('y2', (barrier) => yScale(barrier.Eend_eV));
        g.selectAll<SVGTextElement, Barrier>('.barrier-group text')
          .attr('y', (barrier) => yScale(barrier.Estart_eV) + labelOffsetBelow);
    });

    if (svgRef.current) {
      const svg = d3.select<SVGSVGElement, unknown>(svgRef.current); // Explicitly type svg selection
      svg.call(zoom); // Apply zoom behavior
    }

    // Axes
    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${margin.left},${margin.top + height})`)
      .call(d3.axisBottom(xScale)
        .tickValues(d3.range(Math.ceil(xScale.domain()[0]), Math.floor(xScale.domain()[1]) + 1, 1))
        .tickFormat(d3.format('d'))
        .tickPadding(5)
        .tickSize(10)
        .tickSizeOuter(0))
      .style('font-size', '1.1em')
      .append('text')
      .attr('fill', 'white')
      .attr('y', 40)
      .attr('x', width / 2)
      .attr('dy', '0.71em')
      .attr('text-anchor', 'middle')
      .text('Reaction Coordinate');

    renderYAxis(yScale, svg, margin, height);

    /// ///////////////////////////////////////////////////////////////////////
    // MEP barriers
    const rectWidth = xScale(0.7 + xScale.domain()[0]);
    const labelFontSize_em = 1;
    const labelOffsetBelow = 4;

    const barriers = g.selectAll('.barriers')
      .data(mepBarriers)
      .enter()
      .append('g')
      .attr('class', 'barrier-group');

    // The rectangles for the energy level itself - hidden by the actual levels when drawn
    barriers.append('rect')
      .attr('class', 'barrier-rect-right') // Add a class for the "above" label
      .attr('x', (barrier) => xScale(barrier.index + 2) - rectWidth / 2)
      .attr('y', (barrier) => yScale(barrier.Eend_eV) - rectHeight / 2)
      .attr('width', rectWidth)
      .attr('height', rectHeight)
      .attr('fill', 'peru');

    barriers.append('rect')
      .attr('class', 'barrier-rect-left') // Add a class for the "above" label
      .attr('x', (barrier) => xScale(barrier.index + 1) - rectWidth / 2)
      .attr('y', (barrier) => yScale(barrier.Estart_eV) - rectHeight / 2)
      .attr('width', rectWidth)
      .attr('height', rectHeight)
      .attr('fill', 'peru');

    // Markov population labels
    barriers.append('text')
      .text((barrier) => barrier.population ?? '')
      .attr('x', (barrier) => xScale(barrier.index + 1) - rectWidth / 2)
      .attr('y', (barrier) => yScale(barrier.Estart_eV) + labelOffsetBelow)
      .attr('dominant-baseline', 'hanging')
      .attr('fill', 'peru')
      .style('font-size', `${labelFontSize_em}em`);

    barriers.append('line')
      .attr('x1', (barrier) => xScale(barrier.index + 1) + rectWidth / 2)
      .attr('y1', (barrier) => yScale(barrier.Estart_eV))
      .attr('x2', (barrier) => xScale(barrier.index + 2) - rectWidth / 2)
      .attr('y2', (barrier) => yScale(barrier.Eend_eV))
      .attr('stroke', 'peru')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5'); // Creates the dotted effect, adjust numbers to change the pattern

    /// ///////////////////////////////////////////////////////////////////////
    // Levels

    // Bind the energy levels and use a group to bundle each rectangle with its labels
    const levels = g.selectAll('.level')
      .data(steps.filter((step) => !hideConverged || !step.calc.isConverged))
      .enter()
      .append('g')
      .attr('opacity', (step) => (levelsLegendToggle[step.calc.siteLabel.colorMap.name] ? 1 : 0))
      .attr('class', 'level-group');

    // The rectangles for the energy level itself
    levels.append('rect')
      .attr('x', (step) => xScale(step.index + 1) - rectWidth / 2)
      .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) - rectHeight / 2)
      .attr('width', rectWidth)
      .attr('height', rectHeight)
      .attr('fill', (step) => step.calc.siteLabel.colorMap.fill);

    // The label above each rectangle
    levels.append('text')
      .attr('class', 'label-above') // Add a class for the "above" label
      .text((step) => `${step.studyIndex}${step.calc.adLabel}`)
      .attr('x', (step) => xScale(step.index + 1))
      .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) - rectHeight / 2 - labelOffsetAbove)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .style('font-size', `${labelFontSize_em}em`);

    // The label below each rectangle for Elevel_eV
    levels.append('text')
      .attr('class', 'label-below-energy') // Add a class for the "below energy" label
      .text((step) => step.Elevel_eV(strain_Pct, potential_V)!.toFixed(2))
      .attr('dominant-baseline', 'hanging')
      .attr('x', (step) => xScale(step.index + 1))
      .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) + labelOffsetBelow)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .style('font-size', `${labelFontSize_em}em`);

    // The secondary label below for L2MaxForce_meVpA
    levels.append('text')
      .attr('class', 'label-below-force') // Add a class for the "below force" label
      .text((step) => `${step.calc.statusIcon}${step.calc.roundedMaxForce_meVpA}`)
      .attr('dominant-baseline', 'hanging')
      .attr('x', (step) => xScale(step.index + 1) + 15)
      .attr('y', (step) => yScale(step.Elevel_eV(strain_Pct, potential_V)!) + labelOffsetBelow)
      .attr('text-anchor', 'start')
      .attr('fill', (step) => step.calc.isConverged ? CONVERGED_COLOR : convergenceColor(step.calc.roundedMaxForce_meVpA))
      .style('font-size', `${labelFontSize_em * 0.7}em`); // Smaller font size for superscript appearance

    /// ///////////////////////////////////////////////////////////////////////
    // Legend Title
    // Create a text element to append tspsans to
    const legendOffset = width + margin.left + margin.right - 10;
    const textElement = svg.append('text')
      .attr('transform', `translate(${legendOffset}, 20)`)
      .attr('dominant-baseline', 'hanging')
      .attr('text-anchor', 'end')
      .style('font-size', '1.5em')
      .attr('fill', 'white');

    // We alternate subscript and normal text by adjusting the dy and font-size
    // Regular expression to match groups of letters or numbers
    const regex = /(\d+|[a-zA-Z]+)/g;
    let dyValue = 0; // Initial dy value
    let match = regex.exec(study.material);
    while (match !== null) {
      // Check if the segment is numeric
      if (Number.isNaN(parseInt(match[0], 10))) {
        // Append letter group normally
        textElement.append('tspan')
          .attr('dy', `${dyValue}em`)
          .style('font-size', '1em')
          .text(match[0]);
      } else {
        // Append number group as subscript
        textElement.append('tspan')
          .attr('dy', '0.5em')
          .style('font-size', '0.8em')
          .text(match[0]);
        dyValue = -0.4;
      }
      match = regex.exec(study.material);
    }

    /// ///////////////////////////////////////////////////////////////////////
    // Legend
    const rectSize = 20;

    // Hint Text
    svg.append('text')
      .attr('transform', `translate(${legendOffset}, ${groupedCmaps.length * (rectSize + 5) + 84})`)
      .attr('dominant-baseline', 'hanging')
      .attr('text-anchor', 'end')
      .style('font-size', `${labelFontSize_em * 0.8}em`)
      .style('font-style', 'italic')
      .attr('fill', 'silver')
      .text('Click to toggle');

    // *H Label text
    svg.append('text')
      .attr('transform', `translate(${legendOffset - 3}, 67)`)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .attr('fill', 'white')
      .text('*H');

    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${legendOffset - 2 * rectSize - 5}, 80)`)
      .selectAll('g')
      .data(groupedCmaps)
      .enter()
      .append('g')
      .attr('transform', (_, i) => `translate(0, ${i * (rectSize + 5)})`);

    legend.each(function addRectangles(d) {
      // For each group, add rectangles for base and H cmaps
      const group = d3.select(this);

      // The text to the left of the rectangles
      group.append('text')
        .attr('x', -10)
        .attr('y', 10)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('fill', 'white')
        .text(d.name);

      if (d.base) { // Check if the base cmap exists in the group
        group.append('rect')
          .attr('class', 'legend-rect-base')
          .style('pointer-events', 'all')
          .attr('x', 0)
          .attr('width', rectSize)
          .attr('height', rectSize)
          .attr('fill', (levelsLegendToggle[d.base.name] ? d.base.fill : 'none'))
          .attr('stroke', d.base.fill)
          .attr('stroke-width', 2)
          .style('cursor', 'pointer')
          .on('click', () => {
            setLevelsLegendToggle((current) => ({ ...current, [d.base!.name]: !current[d.base!.name] }));
          });
      }

      if (d.H) { // Check if the *H cmap exists in the group
        group.append('rect')
          .attr('class', 'legend-rect-H')
          .style('pointer-events', 'all')
          .attr('x', 25) // Offset the second rectangle to place it next to the first
          .attr('width', rectSize)
          .attr('height', rectSize)
          .attr('fill', (levelsLegendToggle[d.H.name] ? d.H.fill : 'none'))
          .attr('stroke', d.H.fill)
          .attr('stroke-width', 2)
          .style('cursor', 'pointer')
          .on('click', () => {
            setLevelsLegendToggle((currentToggle) => ({ ...currentToggle, [d.H!.name]: !currentToggle[d.H!.name] }));
          });
      }
    });
  }, [study, steps, minEnergies, mepBarriers, dimensions, potential_V,
    strain_Pct, cmaps, levelsLegendToggle]);

  return <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />;
}
