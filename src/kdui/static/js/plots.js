// Charts drawn in the browser (live progress, previews, comparison summaries),
// on the vendored ECharts. Every function takes scientific data (arrays,
// FieldSlice), never a chart configuration, and returns a handle
// { chart, update(data), dispose() }. Colours follow src/kd/viz/style.py.
// Figures of a finished run are KD's own report figures, not redrawn here.

import * as echarts from "../vendor/echarts/echarts.esm.min.js";
import { fmtNumber } from "./ui.js";

/** style.py SERIES, assigned in order, never by rank. */
export const SERIES = ["#139d96", "#c9583f", "#6a45c0", "#b58a00"];
/** style.py CMAP_SEQ (viridis) as stops. */
export const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
/** style.py CMAP_DIV (kd_div), for signed differences on a symmetric scale. */
export const KD_DIV = ["#0b4f4b", "#139d96", "#8fd0cb", "#f0efec", "#e8a08e", "#c9583f", "#6e2415"];

const INK = "#1a1a18";
const INK_2 = "#52514e";
const GRID = "#e1e0d9";
const AXIS = "#c3c2b7";
const FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif';

const TOOLTIP = {
  backgroundColor: "#ffffff",
  borderColor: "#d6d5cd",
  borderWidth: 1,
  textStyle: { color: INK, fontFamily: FONT, fontSize: 12 },
  extraCssText: "box-shadow:0 2px 8px #1a1a1814;border-radius:4px;",
};

/** Log-axis tick: "1e-7" for powers of ten, else two significant digits. */
const logLabel = (v) => (Number.isInteger(Math.log10(v)) ? `1e${Math.log10(v)}` : fmtNumber(v, 2));

function axis(name, { type = "value", log = false, data = null } = {}) {
  return {
    type: log ? "log" : type,
    data,
    name,
    nameLocation: "middle",
    nameGap: type === "category" ? 40 : 32,
    nameTextStyle: { color: INK_2, fontFamily: FONT, fontSize: 12 },
    axisLine: { lineStyle: { color: AXIS } },
    axisTick: { lineStyle: { color: AXIS } },
    axisLabel: { color: INK_2, fontFamily: FONT, fontSize: 11, hideOverlap: true, formatter: type === "category" ? undefined : log ? logLabel : (v) => fmtNumber(v, 3) },
    splitLine: { lineStyle: { color: GRID } },
    scale: !log,
  };
}

const base = (grid = {}) => ({
  animation: false,
  textStyle: { fontFamily: FONT, color: INK },
  grid: { left: 64, right: 24, top: 18, bottom: 46, ...grid },
});

/**
 * Dispose a chart that may still be painting. zrender paints a large scene
 * (a 256 by 201 heatmap) over several animation frames and continues while its
 * redraw id is unchanged; dispose() alone leaves that frame callback reading a
 * removed layer ("Cannot read properties of null (reading 'layerStack')") when
 * the user leaves the page mid-paint. Painting the cleared chart at once gives
 * a new redraw id, so the pending frame stops.
 */
function disposeChart(chart) {
  chart.clear();
  chart.getZr().refreshImmediately();
  chart.dispose();
}

function mount(el, makeOption, data, initOpts = {}) {
  const chart = echarts.init(el, null, { renderer: "canvas", ...initOpts });
  chart.setOption(makeOption(data));
  const observer = new ResizeObserver(() => {
    if (chart.isDisposed()) observer.disconnect();
    else chart.resize();
  });
  observer.observe(el);
  return {
    chart,
    update(next) {
      chart.setOption(makeOption(next), { notMerge: true });
    },
    dispose() {
      observer.disconnect();
      disposeChart(chart);
    },
  };
}

const pairs = (x, y) => x.map((v, i) => [v, y[i]]);

/**
 * Line chart. data: {
 *   series: [{ name, x: number[], y: number[], step: false }],
 *   xLabel, yLabel, logY: false,
 *   marks: [{ x, y, label }]      points drawn on top (e.g. where the best equation changed),
 *   remainingFrom: x             shade from x to xMax as "remaining budget",
 *   xMax                          end of the x axis (default: data end)
 * }
 * One y axis only. Several series get a legend and a label at the line end.
 */
export function linePlot(el, data) {
  return mount(el, lineOption, data);
}

function lineOption(d) {
  const many = d.series.length > 1;
  const series = d.series.map((s, i) => ({
    type: "line",
    name: s.name,
    data: pairs(s.x, s.y),
    step: s.step ? "end" : false,
    showSymbol: false,
    lineStyle: { width: 2, color: SERIES[i % SERIES.length] },
    itemStyle: { color: SERIES[i % SERIES.length] },
    endLabel: many ? { show: true, formatter: s.name, color: INK, fontFamily: FONT } : undefined,
    // Lines that end at the same value (a periodic field at its boundary) would
    // print their names on top of each other.
    labelLayout: many ? { moveOverlap: "shiftY" } : undefined,
    markArea:
      i === 0 && d.remainingFrom !== undefined && d.remainingFrom !== null
        ? { silent: true, itemStyle: { color: "#f5f4f0" }, label: { color: INK_2, fontFamily: FONT }, data: [[{ name: "remaining budget", xAxis: d.remainingFrom }, { xAxis: d.xMax ?? "max" }]] }
        : undefined,
  }));
  if (d.marks && d.marks.length) {
    series.push({
      type: "scatter",
      name: "marks",
      data: d.marks.map((m) => ({ value: [m.x, m.y], label: m.label })),
      symbolSize: 9,
      itemStyle: { color: SERIES[1], borderColor: "#ffffff", borderWidth: 2 },
      tooltip: { show: false },
      z: 5,
    });
  }
  return {
    ...base({ right: many ? 96 : 24, top: many ? 34 : 18 }),
    legend: many ? { top: 0, textStyle: { color: INK_2, fontFamily: FONT }, data: d.series.map((s) => s.name) } : undefined,
    tooltip: {
      ...TOOLTIP,
      trigger: "axis",
      axisPointer: { type: "cross", lineStyle: { color: INK_2, type: "dashed" }, label: { formatter: (p) => fmtNumber(p.value) } },
      valueFormatter: (v) => fmtNumber(v),
    },
    xAxis: { ...axis(d.xLabel), max: d.xMax ?? undefined },
    yAxis: axis(d.yLabel, { log: d.logY }),
    series,
  };
}

/**
 * Scatter chart. data: { series: [{ name, x, y, labels? }], xLabel, yLabel,
 * logX: false, logY: false }. labels[i] is shown in the tooltip of point i.
 */
export function scatterPlot(el, data) {
  return mount(el, scatterOption, data);
}

function scatterOption(d) {
  const many = d.series.length > 1;
  return {
    ...base({ top: many ? 34 : 18 }),
    legend: many ? { top: 0, textStyle: { color: INK_2, fontFamily: FONT } } : undefined,
    tooltip: {
      ...TOOLTIP,
      trigger: "item",
      formatter: (p) => {
        const s = d.series[p.seriesIndex];
        const label = s.labels ? `${s.labels[p.dataIndex]}<br>` : "";
        return `${label}${d.xLabel} ${fmtNumber(p.value[0])}<br>${d.yLabel} ${fmtNumber(p.value[1])}`;
      },
    },
    xAxis: axis(d.xLabel, { log: d.logX }),
    yAxis: axis(d.yLabel, { log: d.logY }),
    series: d.series.map((s, i) => ({
      type: "scatter",
      name: s.name,
      data: pairs(s.x, s.y),
      symbolSize: 8,
      itemStyle: { color: SERIES[i % SERIES.length], opacity: 0.75 },
      large: s.x.length > 2000,
    })),
  };
}

/**
 * One value per labelled row on a shared axis (log by default), e.g. the
 * re-measured nmse of several runs. data: { rows: [{ label, value, note }],
 * xLabel, logX: true }. A row with value null is listed as "not measured".
 */
export function dotPlot(el, data) {
  el.style.height = `${Math.max(120, 60 + 34 * data.rows.length)}px`;
  return mount(el, dotOption, data);
}

function dotOption(d) {
  const labels = d.rows.map((r) => (r.value === null ? `${r.label} (not measured)` : r.label));
  return {
    ...base({ left: 16, right: 80, containLabel: true }),
    tooltip: {
      ...TOOLTIP,
      trigger: "item",
      formatter: (p) => {
        const r = d.rows[p.dataIndex];
        return `${r.label}<br>${d.xLabel} ${fmtNumber(r.value)}${r.note ? `<br>${r.note}` : ""}`;
      },
    },
    xAxis: axis(d.xLabel, { log: d.logX !== false }),
    yAxis: { ...axis("", { type: "category", data: labels }), inverse: true, axisLabel: { color: INK, fontFamily: FONT, fontSize: 12 }, splitLine: { show: true, lineStyle: { color: GRID } } },
    series: [
      {
        type: "scatter",
        data: d.rows.map((r, i) => (r.value === null ? null : [r.value, i])),
        symbolSize: 10,
        itemStyle: { color: INK, borderColor: "#ffffff", borderWidth: 2 },
        label: { show: true, position: "right", formatter: (p) => fmtNumber(p.value[0]), color: INK_2, fontFamily: FONT },
      },
    ],
  };
}

const coordLabel = (v) => fmtNumber(v, 3);

/**
 * Heatmap of a grid FieldSlice over two axes (rows follow axes[0], drawn
 * upward; columns follow axes[1], drawn to the right, as kd's plot_field draws
 * u(x, t)). opts: { diverging: false } uses kd_div on a scale symmetric about
 * zero (for signed differences); otherwise viridis over the shown range.
 * null values (non-finite) are left blank.
 */
export function fieldHeatmap(el, slice, opts = {}) {
  // Twice the pixel density hides the hairline seams between narrow cells.
  return mount(el, (s) => heatmapOption(s, opts), slice, { devicePixelRatio: Math.max(2, window.devicePixelRatio || 1) });
}

function valueRange(values, diverging) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === Infinity) return [0, 1];
  if (diverging) {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    return [-m, m];
  }
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

function heatmapOption(s, { diverging = false } = {}) {
  const [rowAxis, colAxis] = s.axes;
  const rows = s.coordinates[rowAxis];
  const cols = s.coordinates[colAxis];
  const cells = [];
  const flat = [];
  s.values.forEach((row, i) =>
    row.forEach((v, j) => {
      cells.push([j, i, v === null ? "-" : v]);
      flat.push(v);
    }),
  );
  const [lo, hi] = valueRange(flat, diverging);
  return {
    ...base({ right: 86, left: 70 }),
    tooltip: {
      ...TOOLTIP,
      trigger: "item",
      formatter: (p) => `${rowAxis} ${coordLabel(rows[p.value[1]])}, ${colAxis} ${coordLabel(cols[p.value[0]])}<br>${s.field} ${p.value[2] === "-" ? "not finite" : fmtNumber(p.value[2])}`,
    },
    xAxis: { ...axis(colAxis, { type: "category", data: cols.map(coordLabel) }), splitLine: { show: false }, axisTick: { show: false }, axisLabel: { color: INK_2, fontFamily: FONT, fontSize: 11, interval: Math.max(0, Math.ceil(cols.length / 8) - 1) } },
    yAxis: { ...axis(rowAxis, { type: "category", data: rows.map(coordLabel) }), splitLine: { show: false }, axisTick: { show: false }, axisLabel: { color: INK_2, fontFamily: FONT, fontSize: 11, interval: Math.max(0, Math.ceil(rows.length / 6) - 1) } },
    visualMap: {
      min: lo,
      max: hi,
      calculable: false,
      orient: "vertical",
      right: 8,
      top: "middle",
      itemHeight: 180,
      itemWidth: 12,
      text: [fmtNumber(hi, 3), fmtNumber(lo, 3)],
      textStyle: { color: INK_2, fontFamily: FONT, fontSize: 11 },
      inRange: { color: diverging ? KD_DIV : VIRIDIS },
    },
    series: [{ type: "heatmap", data: cells, progressive: 0, emphasis: { disabled: true } }],
  };
}

/**
 * Draw any FieldSlice the way its shape asks: one axis gives a line of the
 * values, a grid over two axes a heatmap, scattered points a scatter coloured
 * by value. opts are passed to fieldHeatmap.
 */
export function fieldPlot(el, slice, opts = {}) {
  if (slice.axes.length === 1) {
    const axisName = slice.axes[0];
    return linePlot(el, { series: [{ name: slice.field, x: slice.coordinates[axisName], y: slice.values }], xLabel: axisName, yLabel: slice.field });
  }
  if (slice.topology === "grid") return fieldHeatmap(el, slice, opts);
  return mount(el, scatterFieldOption, slice);
}

function scatterFieldOption(s) {
  const [a, b] = s.axes;
  const xs = s.coordinates[a];
  const ys = s.coordinates[b];
  const [lo, hi] = valueRange(s.values, false);
  return {
    ...base({ right: 86 }),
    tooltip: { ...TOOLTIP, trigger: "item", formatter: (p) => `${a} ${coordLabel(p.value[0])}, ${b} ${coordLabel(p.value[1])}<br>${s.field} ${fmtNumber(p.value[2])}` },
    xAxis: axis(a),
    yAxis: axis(b),
    visualMap: { min: lo, max: hi, dimension: 2, right: 8, top: "middle", itemHeight: 180, itemWidth: 12, text: [fmtNumber(hi, 3), fmtNumber(lo, 3)], textStyle: { color: INK_2, fontFamily: FONT, fontSize: 11 }, inRange: { color: VIRIDIS } },
    series: [{ type: "scatter", symbolSize: 6, data: s.values.map((v, i) => (v === null ? null : [xs[i], ys[i], v])).filter(Boolean), large: s.values.length > 2000 }],
  };
}

/** Dispose every chart drawn inside `root` (the shell calls this on unmount). */
export function disposeWithin(root) {
  for (const el of root.querySelectorAll("[_echarts_instance_]")) {
    const chart = echarts.getInstanceByDom(el);
    if (chart) disposeChart(chart);
  }
}
