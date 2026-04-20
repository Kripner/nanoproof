import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

const ROW_HEIGHT = 14;
const ROW_GAP = 2;
const LABEL_WIDTH = 80;
const HEADER_HEIGHT = 40;
const POLL_INTERVAL_LIVE = 2000;
const MIN_BAR_PX = 1;

const COLORS = {
  llm: '#58a6ff',
  lean: '#f85149',
  phase_collect: '#3fb950',
  phase_eval: '#d29922',
  phase_train: '#a371f7',
  background: '#0d1117',
  rowBg: '#161b22',
  label: '#8b949e',
  grid: '#21262d',
};

type PhaseName = 'collect' | 'eval' | 'train' | string;

interface WirePhase {
  name: PhaseName;
  action: 'start' | 'end';
  t: number;
}

interface WireData {
  actors: Record<string, { llm: number[]; lean: number[] }>;
  phases: WirePhase[];
  mode?: 'live' | 'standalone';
  cursor?: number;
}

interface ActorData {
  // Flat interleaved [start, end, start, end, ...], sorted by start.
  llm: Float64Array;
  lean: Float64Array;
}

interface ProfilerData {
  actorIds: string[];
  actors: Record<string, ActorData>;
  phases: WirePhase[];
  minTime: number;
  maxTime: number;
  cursor: number;
}

interface Props {
  mode: 'live' | 'standalone';
}

function buildActorData(llm: number[], lean: number[]): ActorData {
  return { llm: sortPairs(llm), lean: sortPairs(lean) };
}

// Sort a flat interleaved [s0,e0,s1,e1,...] array by start ascending.
function sortPairs(flat: number[]): Float64Array {
  const n = flat.length >> 1;
  const idx = new Array<number>(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => flat[a * 2] - flat[b * 2]);
  const out = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    out[i * 2] = flat[idx[i] * 2];
    out[i * 2 + 1] = flat[idx[i] * 2 + 1];
  }
  return out;
}

// Merge two sorted interleaved arrays into a new sorted one.
function mergeSorted(a: Float64Array, b: Float64Array): Float64Array {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const out = new Float64Array(a.length + b.length);
  let i = 0, j = 0, k = 0;
  while (i < a.length && j < b.length) {
    if (a[i] <= b[j]) {
      out[k++] = a[i++]; out[k++] = a[i++];
    } else {
      out[k++] = b[j++]; out[k++] = b[j++];
    }
  }
  while (i < a.length) out[k++] = a[i++];
  while (j < b.length) out[k++] = b[j++];
  return out;
}

// Binary search: return the smallest index i (in pairs) such that end[i] > t,
// i.e. the first event that could be visible for viewStart=t.
function firstVisiblePair(arr: Float64Array, t: number): number {
  // Events are sorted by start. An event is visible iff start<=viewEnd && end>=viewStart.
  // We want to skip events whose end < viewStart. Since events in a row of one
  // type are non-overlapping (an actor is in exactly one state at a time),
  // end is also monotonic, so we can binary search on end.
  let lo = 0, hi = arr.length >> 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid * 2 + 1] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function computeBounds(data: ProfilerData | null): { min: number; max: number } | null {
  if (!data) return null;
  if (data.minTime === Infinity) return null;
  return { min: data.minTime, max: data.maxTime };
}

function scanBounds(actors: Record<string, ActorData>, phases: WirePhase[]) {
  let min = Infinity, max = -Infinity;
  for (const aid in actors) {
    const { llm, lean } = actors[aid];
    if (llm.length) {
      if (llm[0] < min) min = llm[0];
      if (llm[llm.length - 1] > max) max = llm[llm.length - 1];
    }
    if (lean.length) {
      if (lean[0] < min) min = lean[0];
      if (lean[lean.length - 1] > max) max = lean[lean.length - 1];
    }
  }
  for (const ph of phases) {
    if (ph.t < min) min = ph.t;
    if (ph.t > max) max = ph.t;
  }
  return { min, max };
}

export function ProfilerPanel({ mode }: Props) {
  const [data, setData] = useState<ProfilerData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const headerCanvasRef = useRef<HTMLCanvasElement>(null);
  const bodyCanvasRef = useRef<HTMLCanvasElement>(null);
  const [viewportWidth, setViewportWidth] = useState(0);

  const [viewStart, setViewStart] = useState(0);
  const [viewEnd, setViewEnd] = useState(60);
  const [autoFollow, setAutoFollow] = useState(true);

  const dragRef = useRef<{ startX: number; startY: number; viewStart: number; viewEnd: number; scrollTop: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  // Fetch data (with delta polling in live mode).
  useEffect(() => {
    let cancelled = false;
    let lastCursor = -Infinity;

    const endpoint = mode === 'standalone'
      ? '/api/instrumentation/file'
      : '/api/instrumentation';

    const fetchData = async () => {
      try {
        const url = mode === 'live' && lastCursor !== -Infinity
          ? `${endpoint}?since=${lastCursor}`
          : endpoint;
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to fetch instrumentation data');
        const wire: WireData = await res.json();
        if (cancelled) return;
        applyWire(wire, lastCursor !== -Infinity);
        if (typeof wire.cursor === 'number') lastCursor = wire.cursor;
        setError(null);
      } catch {
        if (!cancelled) setError('Cannot load instrumentation data');
      }
    };

    const applyWire = (wire: WireData, isDelta: boolean) => {
      setData(prev => {
        const baseActors = isDelta && prev ? prev.actors : {};
        const mergedActors: Record<string, ActorData> = { ...baseActors };
        for (const aid in wire.actors) {
          const w = wire.actors[aid];
          const incoming = buildActorData(w.llm, w.lean);
          const existing = mergedActors[aid];
          if (!existing) {
            mergedActors[aid] = incoming;
          } else {
            mergedActors[aid] = {
              llm: mergeSorted(existing.llm, incoming.llm),
              lean: mergeSorted(existing.lean, incoming.lean),
            };
          }
        }
        const mergedPhases = isDelta && prev ? prev.phases.concat(wire.phases) : wire.phases;
        const { min, max } = scanBounds(mergedActors, mergedPhases);
        const actorIds = Object.keys(mergedActors).sort((a, b) => Number(a) - Number(b));
        return {
          actorIds,
          actors: mergedActors,
          phases: mergedPhases,
          minTime: min,
          maxTime: max,
          cursor: wire.cursor ?? 0,
        };
      });
    };

    fetchData();
    if (mode === 'live') {
      const interval = setInterval(fetchData, POLL_INTERVAL_LIVE);
      return () => { cancelled = true; clearInterval(interval); };
    }
    return () => { cancelled = true; };
  }, [mode]);

  // Auto-follow: keep view fitted to data bounds when new data arrives.
  useEffect(() => {
    const bounds = computeBounds(data);
    if (!bounds || !autoFollow) return;
    const range = bounds.max - bounds.min;
    const padding = Math.max(range * 0.05, 5);
    setViewStart(bounds.min);
    setViewEnd(bounds.max + padding);
  }, [data, autoFollow]);

  const dataOrigin = useMemo(() => {
    const bounds = computeBounds(data);
    return bounds ? bounds.min : 0;
  }, [data]);

  // Track container width via a callback ref so we set up the observer as
  // soon as the scroll container mounts (it's rendered conditionally).
  const attachScrollRef = useCallback((el: HTMLDivElement | null) => {
    scrollRef.current = el;
    if (observerRef.current) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }
    if (!el) return;
    setViewportWidth(el.clientWidth);
    const observer = new ResizeObserver(() => setViewportWidth(el.clientWidth));
    observer.observe(el);
    observerRef.current = observer;
  }, []);

  const actorIds = data?.actorIds ?? [];
  const numRows = actorIds.length;
  const bodyHeight = Math.max(numRows * (ROW_HEIGHT + ROW_GAP) + 8, 120);

  // Render header canvas.
  useEffect(() => {
    const canvas = headerCanvasRef.current;
    if (!canvas || viewportWidth === 0 || !data) return;
    const dpr = window.devicePixelRatio || 1;
    const width = viewportWidth;
    const height = HEADER_HEIGHT;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    const duration = viewEnd - viewStart;
    if (duration <= 0) return;
    const timelineWidth = width - LABEL_WIDTH;
    const timeToX = (t: number) => LABEL_WIDTH + ((t - viewStart) / duration) * timelineWidth;

    // Time axis
    const tickInterval = computeTickInterval(duration, timelineWidth);
    const firstTick = Math.ceil((viewStart - dataOrigin) / tickInterval) * tickInterval + dataOrigin;
    ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = COLORS.label;
    for (let t = firstTick; t <= viewEnd; t += tickInterval) {
      const x = timeToX(t);
      if (x < LABEL_WIDTH - 20 || x > width + 20) continue;
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.fillStyle = COLORS.label;
      ctx.fillText(formatTime(t - dataOrigin, tickInterval), x, height - 8);
    }

    // Phase start markers on the header (thin solid lines, no labels;
    // the legend above the chart explains the color meanings). Alpha is
    // attenuated when many lines of the same color are packed into the
    // viewport, so a dense run of phase transitions doesn't paint over the
    // whole chart.
    const phaseAlpha = computePhaseAlphas(data.phases, viewStart, viewEnd, timelineWidth);
    ctx.lineWidth = 1;
    for (const ph of data.phases) {
      if (ph.action !== 'start') continue;
      const x = timeToX(ph.t);
      if (x < LABEL_WIDTH || x > width) continue;
      ctx.globalAlpha = phaseAlpha[ph.name] ?? 1;
      ctx.strokeStyle = phaseColor(ph.name);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Left gutter (separator between labels and timeline)
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, LABEL_WIDTH, height);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(LABEL_WIDTH, 0);
    ctx.lineTo(LABEL_WIDTH, height);
    ctx.stroke();
  }, [data, viewStart, viewEnd, dataOrigin, viewportWidth]);

  // Render body canvas (rows + phase bars on top).
  useEffect(() => {
    const canvas = bodyCanvasRef.current;
    if (!canvas || viewportWidth === 0 || !data) return;

    const dpr = window.devicePixelRatio || 1;
    const width = viewportWidth;
    const height = bodyHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    const duration = viewEnd - viewStart;
    if (duration <= 0) return;
    const timelineWidth = width - LABEL_WIDTH;
    const timeToX = (t: number) => LABEL_WIDTH + ((t - viewStart) / duration) * timelineWidth;

    // Light vertical grid lines to match header
    const tickInterval = computeTickInterval(duration, timelineWidth);
    const firstTick = Math.ceil((viewStart - dataOrigin) / tickInterval) * tickInterval + dataOrigin;
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let t = firstTick; t <= viewEnd; t += tickInterval) {
      const x = timeToX(t);
      if (x < LABEL_WIDTH || x > width) continue;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Rows + events. Per-pixel time-weighted buckets so that when many events
    // collapse into one pixel column, the dominant type (by total time) wins
    // the color instead of whichever type was drawn last.
    const colLlm = new Float32Array(timelineWidth);
    const colLean = new Float32Array(timelineWidth);
    ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
    for (let i = 0; i < actorIds.length; i++) {
      const aid = actorIds[i];
      const y = i * (ROW_HEIGHT + ROW_GAP);

      ctx.fillStyle = COLORS.rowBg;
      ctx.fillRect(LABEL_WIDTH, y, timelineWidth, ROW_HEIGHT);

      ctx.fillStyle = COLORS.label;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`Actor ${aid}`, LABEL_WIDTH - 8, y + ROW_HEIGHT / 2);

      colLlm.fill(0);
      colLean.fill(0);
      accumulateColumns(data.actors[aid].llm, viewStart, viewEnd, timeToX, LABEL_WIDTH, timelineWidth, colLlm);
      accumulateColumns(data.actors[aid].lean, viewStart, viewEnd, timeToX, LABEL_WIDTH, timelineWidth, colLean);
      paintDominant(ctx, colLlm, colLean, COLORS.llm, COLORS.lean, LABEL_WIDTH, y + 1, ROW_HEIGHT - 2);
    }

    // Phase vertical lines (drawn on top of rows so they're visible).
    // Both starts and ends are thin (1px); starts are solid, ends are dashed.
    // Draw ends first so the solid starts always paint on top of any
    // overlapping dashed end at the same x coordinate. Alpha is attenuated
    // when many same-color lines pack into the viewport so they don't fully
    // overpower the underlying actor bars.
    const bodyPhaseAlpha = computePhaseAlphas(data.phases, viewStart, viewEnd, timelineWidth);
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (const ph of data.phases) {
      if (ph.action !== 'end') continue;
      const x = timeToX(ph.t);
      if (x < LABEL_WIDTH || x > width) continue;
      ctx.globalAlpha = bodyPhaseAlpha[ph.name] ?? 1;
      ctx.strokeStyle = phaseColor(ph.name);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const ph of data.phases) {
      if (ph.action !== 'start') continue;
      const x = timeToX(ph.t);
      if (x < LABEL_WIDTH || x > width) continue;
      ctx.globalAlpha = bodyPhaseAlpha[ph.name] ?? 1;
      ctx.strokeStyle = phaseColor(ph.name);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }, [data, viewStart, viewEnd, dataOrigin, viewportWidth, bodyHeight, actorIds]);

  // Wheel handler: ctrl/meta + wheel = zoom; plain wheel = native vertical scroll.
  // Attached natively to get passive:false so preventDefault works for ctrl-wheel.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const canvas = bodyCanvasRef.current;
      const rect = (canvas ?? el).getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const fraction = (mouseX - LABEL_WIDTH) / Math.max(rect.width - LABEL_WIDTH, 1);
      const clamped = Math.max(0, Math.min(1, fraction));
      const duration = viewEnd - viewStart;
      const zoomFactor = e.deltaY > 0 ? 1.2 : 1 / 1.2;
      const newDuration = Math.max(duration * zoomFactor, 0.05);
      const pivot = viewStart + clamped * duration;
      setAutoFollow(false);
      setViewStart(pivot - clamped * newDuration);
      setViewEnd(pivot + (1 - clamped) * newDuration);
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, [viewStart, viewEnd]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setAutoFollow(false);
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      viewStart,
      viewEnd,
      scrollTop: scrollRef.current?.scrollTop ?? 0,
    };
    setDragging(true);
  }, [viewStart, viewEnd]);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current || !bodyCanvasRef.current) return;
      const rect = bodyCanvasRef.current.getBoundingClientRect();
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      const timelineWidth = Math.max(rect.width - LABEL_WIDTH, 1);
      const duration = dragRef.current.viewEnd - dragRef.current.viewStart;
      const timeDelta = -(dx / timelineWidth) * duration;
      setViewStart(dragRef.current.viewStart + timeDelta);
      setViewEnd(dragRef.current.viewEnd + timeDelta);
      // Vertical drag scrolls the container (inverse of dy: drag down -> scroll up).
      if (scrollRef.current) {
        scrollRef.current.scrollTop = dragRef.current.scrollTop - dy;
      }
    };
    const onUp = () => {
      dragRef.current = null;
      setDragging(false);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [dragging]);

  if (error) {
    return (
      <div className="main" style={{ padding: 16 }}>
        <div className="card" style={{ padding: 24, textAlign: 'center', color: 'var(--accent-red)' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="main" style={{ padding: 16 }}>
        <div className="card" style={{ padding: 24, textAlign: 'center' }}>
          Loading...
        </div>
      </div>
    );
  }

  const hasData = actorIds.length > 0 || data.phases.length > 0;

  return (
    <div className="main" style={{ padding: 16 }}>
      <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
          <div className="card-title" style={{ margin: 0 }}>Actor Timelines</div>
          <div style={{ display: 'flex', gap: 12, fontSize: 'var(--font-sm)', flexWrap: 'wrap' }}>
            <LegendSwatch color={COLORS.llm} label="LLM Inference" />
            <LegendSwatch color={COLORS.lean} label="Lean Verification" />
            <LegendDash color={COLORS.phase_collect} label="collect" />
            <LegendDash color={COLORS.phase_eval} label="eval" />
            <LegendDash color={COLORS.phase_train} label="train" />
          </div>
          <div style={{ marginLeft: 'auto', fontSize: 'var(--font-xs)', color: 'var(--text-secondary)' }}>
            Ctrl+wheel to zoom, drag to pan
          </div>
          <button
            className={`tab-btn ${autoFollow ? 'active' : ''}`}
            onClick={() => setAutoFollow(!autoFollow)}
            style={{ fontSize: 'var(--font-xs)' }}
          >
            Auto-follow
          </button>
        </div>

        {!hasData ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-secondary)' }}>
            No instrumentation data yet. Timeline events will appear during collection and evaluation.
          </div>
        ) : (
          <div
            ref={attachScrollRef}
            style={{ flex: 1, minHeight: 0, overflowY: 'auto', overflowX: 'hidden', position: 'relative' }}
          >
            <div style={{
              position: 'sticky',
              top: 0,
              zIndex: 2,
              background: COLORS.background,
              borderBottom: `1px solid ${COLORS.grid}`,
            }}>
              <canvas ref={headerCanvasRef} style={{ display: 'block' }} />
            </div>
            <canvas
              ref={bodyCanvasRef}
              onMouseDown={handleMouseDown}
              style={{ display: 'block', cursor: dragging ? 'grabbing' : 'grab' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// Spread each visible event's duration across the pixel columns it covers,
// adding the contribution to `col`. Each event-time value fills exactly as
// many columns as it visually covers, so column sums are directly comparable
// across event types (they're both in "seconds of wall time per column").
// Events narrower than one pixel count as a minimum MIN_BAR_PX wide, so a
// single sub-pixel event still shows up and competes with neighbors.
function accumulateColumns(
  arr: Float64Array,
  viewStart: number,
  viewEnd: number,
  timeToX: (t: number) => number,
  leftClip: number,
  timelineWidth: number,
  col: Float32Array,
) {
  const n = arr.length >> 1;
  if (n === 0) return;
  const startIdx = firstVisiblePair(arr, viewStart);
  for (let i = startIdx; i < n; i++) {
    const s = arr[i * 2];
    if (s > viewEnd) break;
    const e = arr[i * 2 + 1];
    const duration = e - s;
    let x1 = timeToX(s);
    let x2 = timeToX(e);
    if (x2 - x1 < MIN_BAR_PX) x2 = x1 + MIN_BAR_PX;
    // Clip to the visible timeline area, convert to col-relative indices.
    let p1 = Math.floor(x1 - leftClip);
    let p2 = Math.ceil(x2 - leftClip);
    if (p1 < 0) p1 = 0;
    if (p2 > timelineWidth) p2 = timelineWidth;
    if (p2 <= p1) continue;
    const perCol = duration / (p2 - p1);
    for (let p = p1; p < p2; p++) col[p] += perCol;
  }
}

// Walk the two column accumulators in lockstep; for each column pick the
// dominant color; emit contiguous runs of the same color as single fillRects.
function paintDominant(
  ctx: CanvasRenderingContext2D,
  colA: Float32Array,
  colB: Float32Array,
  colorA: string,
  colorB: string,
  leftClip: number,
  y: number,
  height: number,
) {
  const w = colA.length;
  let runStart = -1;
  let runIsA = false;
  const flush = (endPx: number) => {
    if (runStart < 0) return;
    ctx.fillStyle = runIsA ? colorA : colorB;
    ctx.fillRect(leftClip + runStart, y, endPx - runStart, height);
    runStart = -1;
  };
  for (let p = 0; p < w; p++) {
    const a = colA[p], b = colB[p];
    if (a <= 0 && b <= 0) { flush(p); continue; }
    const isA = a >= b;
    if (runStart < 0) { runStart = p; runIsA = isA; }
    else if (isA !== runIsA) { flush(p); runStart = p; runIsA = isA; }
  }
  flush(w);
}

// Pixel spacing at which a color is considered "comfortably sparse" and
// rendered at full opacity. Below this, alpha tapers down so dense bands of
// phase lines fade instead of painting a solid wall over the actor rows.
const PHASE_TARGET_SPACING_PX = 25;
const PHASE_MIN_ALPHA = 0.15;

// For each phase name, return the alpha to use for vertical markers in the
// current viewport, attenuated by how densely they're packed.
function computePhaseAlphas(
  phases: WirePhase[],
  viewStart: number,
  viewEnd: number,
  timelineWidth: number,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const ph of phases) {
    if (ph.t < viewStart || ph.t > viewEnd) continue;
    counts[ph.name] = (counts[ph.name] ?? 0) + 1;
  }
  const out: Record<string, number> = {};
  for (const name in counts) {
    const avgSpacing = timelineWidth / counts[name];
    out[name] = Math.max(PHASE_MIN_ALPHA, Math.min(1, avgSpacing / PHASE_TARGET_SPACING_PX));
  }
  return out;
}

function phaseColor(name: string): string {
  if (name === 'collect') return COLORS.phase_collect;
  if (name === 'eval') return COLORS.phase_eval;
  return COLORS.phase_train;
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ width: 12, height: 12, borderRadius: 2, background: color, display: 'inline-block' }} />
      {label}
    </span>
  );
}

function LegendDash({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ width: 12, height: 3, borderTop: `2px dashed ${color}`, display: 'inline-block' }} />
      {label}
    </span>
  );
}

function computeTickInterval(duration: number, width: number): number {
  const targetTicks = Math.max(width / 100, 3);
  const rawInterval = duration / targetTicks;
  const niceIntervals = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30,
    60, 120, 300, 600, 1800, 3600, 7200, 21600, 43200, 86400,
  ];
  for (const ni of niceIntervals) {
    if (ni >= rawInterval) return ni;
  }
  return Math.ceil(rawInterval / 86400) * 86400;
}

// Format a seconds-from-origin value for tick labels. `tickInterval` determines
// the precision: when ticks are sub-second we show 2 decimals, when they're
// multi-second we round, and for longer spans we drop irrelevant smaller units
// (e.g. on an hourly axis we don't clutter each label with stray seconds).
function formatTime(seconds: number, tickInterval: number): string {
  if (seconds < 0) seconds = 0;
  if (tickInterval < 0.1) return `${seconds.toFixed(2)}s`;
  if (tickInterval < 1) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const mins = Math.floor(seconds / 60);
  const remSecs = Math.round(seconds - mins * 60);
  if (seconds < 3600) {
    return tickInterval >= 60 ? `${mins}m` : `${mins}m${String(remSecs).padStart(2, '0')}s`;
  }
  const hrs = Math.floor(mins / 60);
  const remMins = mins - hrs * 60;
  if (tickInterval >= 3600) return `${hrs}h`;
  return `${hrs}h${String(remMins).padStart(2, '0')}m`;
}
