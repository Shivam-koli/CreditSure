/* Shared UI builders: cards, badges, gauges, scales, reason lists and Chart.js helpers. Every
   colour is read live from the CSS custom properties on <html> (see css/app.css), so nothing here
   needs to change when the theme switches - Chart.js instances are simply destroyed and rebuilt
   (see rerenderCharts in app.js) so their canvas colours match the new theme too. */

const LABELS = { ELIGIBLE: "Eligible", MANUAL_REVIEW: "Manual review", NOT_ELIGIBLE: "Not eligible" };

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
/* the Settings "Reduce motion" toggle can't stop a Chart.js canvas animation via CSS alone,
   so every chart builder below checks this and passes animation:false when it's on */
function motionOK() { return document.documentElement.dataset.motion !== "reduced"; }
function statusColor(decision) {
  return { ELIGIBLE: cssVar("--good"), MANUAL_REVIEW: cssVar("--warn"), NOT_ELIGIBLE: cssVar("--bad"),
           stable: cssVar("--good"), watch: cssVar("--warn"), alert: cssVar("--bad") }[decision] || cssVar("--accent");
}

function topbar(title, subtitle, segments = [], note = "", action = null) {
  const seg = segments.map(([k, v]) => `<div class="seg"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  const noteHtml = note ? `<div class="seg"><div class="note">${note}</div></div>` : "";
  const strip = seg || noteHtml ? `<div class="strip">${seg}${noteHtml}</div>` : "";
  const actionHtml = action ? `<button class="btn-action" id="${action.id}">${action.icon ? `<span>${action.icon}</span>` : ""}${esc(action.label)}</button>` : "";
  return `<div class="topbar"><div class="title-wrap"><div class="page-title">${esc(title)}</div>
          ${subtitle ? `<div class="page-sub">${esc(subtitle)}</div>` : ""}</div>
          <div class="title-row">${strip}${actionHtml}</div></div>`;
}

function cardTitle(title, sub = "") {
  return `<div class="ctitle">${esc(title)}</div>${sub ? `<div class="csub">${sub}</div>` : ""}`;
}

function badge(decision) {
  const c = statusColor(decision);
  return `<span class="badge" style="color:${c}"><span class="dot"></span>${LABELS[decision] || decision}</span>`;
}
function statusBadge(status) {
  const c = statusColor(status);
  return `<span class="badge" style="color:${c}"><span class="dot"></span>${esc(String(status).toUpperCase())}</span>`;
}

function kpi(label, value, sub = "") {
  return `<div class="kpi"><div class="k">${esc(label)}</div><div class="v">${value}</div><div class="s">${sub}</div></div>`;
}
function kpiRow(items, cols) {
  cols = cols || Math.min(items.length, 5);
  return `<div class="grid grid-${cols}">${items.map(([k, v, s]) => kpi(k, v, s)).join("")}</div>`;
}

function row2(label, value, color) {
  return `<div class="row2"><span class="l">${esc(label)}</span><span class="r"${color ? ` style="color:${color}"` : ""}>${value}</span></div>`;
}

function scale(value, lo, hi, stops, labels, markerLabel = "") {
  const frac = value === null || value === undefined ? 0 : Math.min(Math.max((value - lo) / (hi - lo), 0), 1);
  const grad = stops.map(([p, c]) => `${c} ${Math.round(p * 100)}%`).join(", ");
  return `<div class="scale" style="background:linear-gradient(90deg,${grad})">
          <div class="mk" style="left:calc(${(frac * 100).toFixed(1)}% - 2px)"></div></div>
          <div class="scale-l"><span>${labels[0]}</span><span>${labels[1]}</span><span>${labels[2]}</span></div>
          ${markerLabel ? `<div class="small" style="margin-top:4px">${markerLabel}</div>` : ""}`;
}

function reasons(items, color, unitFn) {
  if (!items || !items.length) return `<div class="small">Nothing material.</div>`;
  const top = Math.max(...items.map((i) => Math.abs(i.impact))) || 1;
  return items.map((it) => {
    const w = 18 + 82 * Math.abs(it.impact) / top;
    return `<div class="reason"><div class="top"><span>${esc(it.text)}</span><span class="imp">${unitFn(it)}</span></div>
            <div class="bar" style="width:${w.toFixed(0)}%;background:${color};box-shadow:0 0 8px ${color}"></div></div>`;
  }).join("");
}

/* ---------- semicircle gauge (ported from the earlier Streamlit app's gauge_svg) ---------- */
function _pt(cx, cy, r, a) { return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; }
function _arc(cx, cy, r, f0, f1) {
  const a0 = Math.PI * (1 - f0), a1 = Math.PI * (1 - f1);
  const [x0, y0] = _pt(cx, cy, r, a0), [x1, y1] = _pt(cx, cy, r, a1);
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}
function gaugeSVG({ value, vmax, centerText, centerSub, color, marker = null, markerLabel = "", ticks = [] }) {
  const sage = cssVar("--accent"), moss = cssVar("--moss"), mint = cssVar("--text"), muted = cssVar("--muted");
  const pine = cssVar("--pine"), deep = cssVar("--deep");
  const cx = 200, cy = 205, r = 150;
  const f = Math.min(Math.max(value / vmax, 0), 1);
  const uid = Math.random().toString(36).slice(2, 8);
  let s = `<svg viewBox="-46 0 492 250" width="100%" style="max-width:500px;display:block;margin:0 auto">
    <defs><filter id="glow${uid}" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <radialGradient id="core${uid}" cx="50%" cy="100%" r="80%"><stop offset="0%" stop-color="${pine}"/>
    <stop offset="100%" stop-color="${deep}" stop-opacity="0"/></radialGradient></defs>`;
  for (const [rr, op, dash] of [[r + 34, .18, "2 6"], [r + 20, .28, ""], [r - 26, .22, "1 5"]]) {
    s += `<path d="${_arc(cx, cy, rr, 0, 1)}" fill="none" stroke="${sage}" stroke-opacity="${op}" stroke-width="1.2" stroke-dasharray="${dash}"/>`;
  }
  s += `<path d="M ${cx - r + 14} ${cy} A ${r - 14} ${r - 14} 0 0 1 ${cx + r - 14} ${cy} Z" fill="url(#core${uid})" opacity=".9"/>`;
  s += `<path d="${_arc(cx, cy, r, 0, 1)}" fill="none" stroke="${moss}" stroke-width="16" stroke-linecap="round" opacity=".7"/>`;
  if (f > 0.002) s += `<path d="${_arc(cx, cy, r, 0, f)}" fill="none" stroke="${color}" stroke-width="16" stroke-linecap="round" filter="url(#glow${uid})"/>`;
  for (const [tv, tl] of ticks) {
    const tf = Math.min(Math.max(tv / vmax, 0), 1);
    const [x, y] = _pt(cx, cy, r + 46, Math.PI * (1 - tf));
    s += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${muted}" font-size="11" text-anchor="middle" font-family="Inter">${tl}</text>`;
  }
  if (marker !== null) {
    const mf = Math.min(Math.max(marker / vmax, 0), 1), a = Math.PI * (1 - mf);
    const [x0, y0] = _pt(cx, cy, r - 14, a), [x1, y1] = _pt(cx, cy, r + 14, a), [lx, ly] = _pt(cx, cy, r + 27, a);
    s += `<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y1.toFixed(1)}" stroke="${mint}" stroke-width="3" stroke-linecap="round"/>`;
    const anchor = mf > 0.6 ? "end" : mf < 0.4 ? "start" : "middle";
    s += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" fill="${mint}" font-size="11.5" text-anchor="${anchor}" font-family="Inter">${markerLabel}</text>`;
  }
  s += `<text x="${cx}" y="${cy - 40}" fill="${mint}" font-size="50" font-weight="600" text-anchor="middle" font-family="Space Grotesk, Inter, sans-serif">${centerText}</text>`;
  s += `<text x="${cx}" y="${cy - 12}" fill="${muted}" font-size="13.5" text-anchor="middle" font-family="Inter">${centerSub}</text>`;
  s += `</svg>`;
  return s;
}

/* ---------- Chart.js helpers ---------- */
window._charts = window._charts || {};
function destroyChart(id) { if (window._charts[id]) { window._charts[id].destroy(); delete window._charts[id]; } }

function baseFont() { return { family: "Inter, sans-serif", size: 12, color: cssVar("--muted") }; }

function barHChart(canvasId, labels, values, highlightLabel, fmt, xrange, height) {
  destroyChart(canvasId);
  const el = document.getElementById(canvasId);
  if (height) el.parentElement.style.height = height + "px";
  const accent = cssVar("--accent"), moss = cssVar("--moss"), text = cssVar("--text");
  const colors = labels.map((l) => (l === highlightLabel ? accent : moss));
  window._charts[canvasId] = new Chart(el, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 6, maxBarThickness: 28 }] },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: motionOK() ? undefined : false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => fmt(ctx.parsed.x) } },
        datalabels: false },
      scales: {
        x: { min: xrange ? xrange[0] : undefined, max: xrange ? xrange[1] : undefined,
             grid: { color: cssVar("--card-border") }, ticks: { color: cssVar("--muted"), callback: (v) => fmt(v) } },
        y: { grid: { display: false }, ticks: { color: text, font: { family: "Inter" } } },
      },
    },
  });
}

function lineChart(canvasId, labels, datasets, opts = {}) {
  destroyChart(canvasId);
  const el = document.getElementById(canvasId);
  if (opts.height) el.parentElement.style.height = opts.height + "px";
  window._charts[canvasId] = new Chart(el, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      animation: motionOK() ? undefined : false,
      plugins: { legend: { position: "bottom", labels: { color: cssVar("--muted"), boxWidth: 10, font: { size: 11 } } } },
      scales: Object.assign({
        x: { grid: { color: cssVar("--card-border") }, ticks: { color: cssVar("--muted") } },
        y: { grid: { color: cssVar("--card-border") }, ticks: { color: cssVar("--muted") } },
      }, opts.scales || {}),
    },
  });
}

function sparkline(canvasId, values, color, height = 30) {
  destroyChart(canvasId);
  const el = document.getElementById(canvasId);
  if (!el) return;
  el.parentElement.style.height = height + "px";
  window._charts[canvasId] = new Chart(el, {
    type: "line",
    data: { labels: values.map((_, i) => i), datasets: [{ data: values, borderColor: color, backgroundColor: color + "26",
             fill: true, tension: .35, pointRadius: 0, borderWidth: 1.8 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: motionOK() ? { duration: 250 } : false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } } },
  });
}

/* a small inline bar with a marker, for a KPI card's "where does this number sit" context -
   same visual language as scale() but compact enough to live under a KPI value */
function miniScale(value, lo, hi, color) {
  const frac = Math.min(Math.max((value - lo) / (hi - lo), 0), 1);
  return `<div class="mini-scale" style="background:linear-gradient(90deg, var(--row), ${color}55)">
          <div class="mk" style="left:calc(${(frac * 100).toFixed(1)}% - 1.5px);background:${color};box-shadow:0 0 6px ${color}"></div></div>`;
}

/* initials avatar, coloured either by the default accent gradient (css class) or an explicit colour
   (used for queue rows, where the colour signals the decision rather than the brand) */
function avatarBadge(initials, color, small = true) {
  const style = color ? ` style="background:${color};box-shadow:0 0 10px ${color}55"` : "";
  return `<div class="avatar${small ? " sm" : ""}"${style}>${esc(initials)}</div>`;
}

function tileRow(items) {
  return `<div class="tile-row">${items.map(([l, v]) => `<div class="tile"><div class="tl">${esc(l)}</div><div class="tv">${v}</div></div>`).join("")}</div>`;
}

function radarChart(canvasId, profile, height = 260) {
  destroyChart(canvasId);
  const el = document.getElementById(canvasId);
  el.parentElement.style.height = height + "px";
  const accent = cssVar("--accent"), text = cssVar("--text"), muted = cssVar("--muted"), border = cssVar("--card-border");
  window._charts[canvasId] = new Chart(el, {
    type: "radar",
    data: {
      labels: Object.keys(profile),
      datasets: [{ data: Object.values(profile), backgroundColor: accent + "38", borderColor: accent, borderWidth: 2,
                   pointBackgroundColor: text, pointRadius: 3 }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: motionOK() ? undefined : false,
      plugins: { legend: { display: false } },
      scales: { r: { min: 0, max: 100, ticks: { display: false }, grid: { color: border }, angleLines: { color: border },
                     pointLabels: { color: muted, font: { size: 11 } } } },
    },
  });
}
