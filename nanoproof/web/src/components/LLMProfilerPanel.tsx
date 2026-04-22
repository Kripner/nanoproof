import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

// Row layout: top portion is the queue-depth line plot, bottom strip is the
// state bar. The line plot needs room to resolve integer queue depths, the
// bar just needs to be readable.
const ROW_HEIGHT = 46;
const STATE_BAR_HEIGHT = 10;
const ROW_GAP = 4;
const LABEL_WIDTH = 80;
const HEADER_HEIGHT = 40;
const POLL_INTERVAL_LIVE = 2000;
const MIN_BAR_PX = 1;
const MAX_VIEW_DURATION = 3600;
const NAV_STEP = 1800;

const COLORS = {
  inferencing: '#58a6ff',
  waiting: '#484f58',
  queueLine: '#f0c64a',
  phase_collect: '#3fb950',
  phase_eval: '#d29922',
  phase_train: '#a371f7',
  background: '#0d1117',
  rowBg: '#161b22',
  label: '#8b949e',
  grid: '#21262d',
  nowCursor: 'rgba(230, 237, 243, 0.55)',
};

const PHASE_OVERLAY_ALPHA = 0.07;

type PhaseName = 'collect' | 'eval' | 'train' | string;

interface WirePhase {
  name: PhaseName;
  action: 'start' | 'end';
  t: number;
}

interface WireRank {
  inferencing: number[]; // flat [s0,e0,s1,e1,...]
  sample_t: number[];
  sample_n: number[];
}

interface WireData {
  ranks: Record<string, WireRank>;
  phases: WirePhase[];
  mode?: 'live' | 'standalone';
  cursor?: number;
}

interface RankData {
  inferencing: Float64Array; // sorted by start
  sampleT: Float64Array;     // sorted ascending
  sampleN: Float64Array;     // aligned with sampleT
}

interface LLMData {
  rankIds: string[];
  ranks: Record<string, RankData>;
  phases: WirePhase[];
  minTime: number;
  maxTime: number;
  maxQueueDepth: number;
  cursor: number;
}

interface Props {
  mode: 'live' | 'standalone';
}

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

function mergeSortedPairs(a: Float64Array, b: Float64Array): Float64Array {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const out = new Float64Array(a.length + b.length);
  let i = 0, j = 0, k = 0;
  while (i < a.length && j < b.length) {
    if (a[i] <= b[j]) { out[k++] = a[i++]; out[k++] = a[i++]; }
    else              { out[k++] = b[j++]; out[k++] = b[j++]; }
  }
  while (i < a.length) out[k++] = a[i++];
  while (j < b.length) out[k++] = b[j++];
  return out;
}

// Samples arrive in time order per rank (server uses a monotonic seq on top of
// an ordered deque). The delta shipped to us is therefore already sorted and
// strictly after the previous batch — so concat is safe without a re-sort.
function appendSamples(
  oldT: Float64Array, oldN: Float64Array,
  newT: number[], newN: number[],
): { t: Float64Array; n: Float64Array } {
  if (newT.length === 0) return { t: oldT, n: oldN };
  const t = new Float64Array(oldT.length + newT.length);
  const n = new Float64Array(oldN.length + newN.length);
  t.set(oldT, 0);
  n.set(oldN, 0);
  for (let i = 0; i < newT.length; i++) {
    t[oldT.length + i] = newT[i];
    n[oldN.length + i] = newN[i];
  }
  return { t, n };
}

function buildRankData(w: WireRank): RankData {
  const t = new Float64Array(w.sample_t.length);
  const n = new Float64Array(w.sample_n.length);
  for (let i = 0; i < w.sample_t.length; i++) t[i] = w.sample_t[i];
  for (let i = 0; i < w.sample_n.length; i++) n[i] = w.sample_n[i];
  return { inferencing: sortPairs(w.inferencing), sampleT: t, sampleN: n };
}

function firstVisiblePair(arr: Float64Array, t: number): number {
  let lo = 0, hi = arr.length >> 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid * 2 + 1] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function scanBounds(ranks: Record<string, RankData>, phases: WirePhase[]) {
  let min = Infinity, max = -Infinity, maxN = 0;
  for (const rid in ranks) {
    const { inferencing, sampleT, sampleN } = ranks[rid];
    if (inferencing.length) {
      if (inferencing[0] < min) min = inferencing[0];
      if (inferencing[inferencing.length - 1] > max) max = inferencing[inferencing.length - 1];
    }
    if (sampleT.length) {
      if (sampleT[0] < min) min = sampleT[0];
      if (sampleT[sampleT.length - 1] > max) max = sampleT[sampleT.length - 1];
    }
    for (let i = 0; i < sampleN.length; i++) {
      if (sampleN[i] > maxN) maxN = sampleN[i];
    }
  }
  for (const ph of phases) {
    if (ph.t < min) min = ph.t;
    if (ph.t > max) max = ph.t;
  }
  return { min, max, maxN };
}

function computeBounds(data: LLMData | null): { min: number; max: number } | null {
  if (!data) return null;
  if (data.minTime === Infinity) return null;
  return { min: data.minTime, max: data.maxTime };
}

export function LLMProfilerPanel({ mode }: Props) {
  const [data, setData] = useState<LLMData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const chartRef = useRef<HTMLDivElement | null>(null);
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
  const [hoverX, setHoverX] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    let lastCursor = -Infinity;

    const endpoint = mode === 'standalone'
      ? '/api/llm_instrumentation/file'
      : '/api/llm_instrumentation';

    const fetchData = async () => {
      try {
        const url = mode === 'live' && lastCursor !== -Infinity
          ? `${endpoint}?since=${lastCursor}`
          : endpoint;
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to fetch LLM instrumentation');
        const wire: WireData = await res.json();
        if (cancelled) return;
        applyWire(wire, lastCursor !== -Infinity);
        if (typeof wire.cursor === 'number') lastCursor = wire.cursor;
        setError(null);
      } catch {
        if (!cancelled) setError('Cannot load LLM instrumentation');
      }
    };

    const applyWire = (wire: WireData, isDelta: boolean) => {
      setData(prev => {
        const baseRanks = isDelta && prev ? prev.ranks : {};
        const mergedRanks: Record<string, RankData> = { ...baseRanks };
        for (const rid in wire.ranks) {
          const w = wire.ranks[rid];
          const incoming = buildRankData(w);
          const existing = mergedRanks[rid];
          if (!existing) {
            mergedRanks[rid] = incoming;
          } else {
            const samples = appendSamples(existing.sampleT, existing.sampleN, w.sample_t, w.sample_n);
            mergedRanks[rid] = {
              inferencing: mergeSortedPairs(existing.inferencing, incoming.inferencing),
              sampleT: samples.t,
              sampleN: samples.n,
            };
          }
        }
        const mergedPhases = isDelta && prev ? prev.phases.concat(wire.phases) : wire.phases;
        const { min, max, maxN } = scanBounds(mergedRanks, mergedPhases);
        const rankIds = Object.keys(mergedRanks).sort((a, b) => Number(a) - Number(b));
        return {
          rankIds,
          ranks: mergedRanks,
          phases: mergedPhases,
          minTime: min,
          maxTime: max,
          maxQueueDepth: maxN,
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

  useEffect(() => {
    const bounds = computeBounds(data);
    if (!bounds || !autoFollow) return;
    const range = bounds.max - bounds.min;
    const padding = Math.min(Math.max(range * 0.05, 5), 60);
    const end = bounds.max + padding;
    const start = Math.max(bounds.min, end - MAX_VIEW_DURATION);
    setViewStart(start);
    setViewEnd(end);
  }, [data, autoFollow]);

  const dataOrigin = useMemo(() => {
    const bounds = computeBounds(data);
    return bounds ? bounds.min : 0;
  }, [data]);

  const phasePairs = useMemo(
    () => pairPhases(data?.phases ?? []),
    [data?.phases],
  );

  const [nowSec, setNowSec] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (mode !== 'live') return;
    const id = setInterval(() => setNowSec(Date.now() / 1000), 500);
    return () => clearInterval(id);
  }, [mode]);

  const attachChartRef = useCallback((el: HTMLDivElement | null) => {
    chartRef.current = el;
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

  const rankIds = data?.rankIds ?? [];
  const numRows = rankIds.length;
  const bodyHeight = Math.max(numRows * (ROW_HEIGHT + ROW_GAP) + 8, 120);
  const maxQ = Math.max(data?.maxQueueDepth ?? 0, 1);

  // Header canvas
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

    paintPhaseOverlays(ctx, phasePairs, timeToX, LABEL_WIDTH, width, 0, height);

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

    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, LABEL_WIDTH, height);
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(LABEL_WIDTH, 0);
    ctx.lineTo(LABEL_WIDTH, height);
    ctx.stroke();
  }, [data, viewStart, viewEnd, dataOrigin, viewportWidth, phasePairs]);

  // Body canvas
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

    // Grid
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

    ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
    for (let i = 0; i < rankIds.length; i++) {
      const rid = rankIds[i];
      const rank = data.ranks[rid];
      const rowTop = i * (ROW_HEIGHT + ROW_GAP);
      const stateBarY = rowTop + ROW_HEIGHT - STATE_BAR_HEIGHT;
      const plotTop = rowTop + 2;
      const plotHeight = ROW_HEIGHT - STATE_BAR_HEIGHT - 4;

      ctx.fillStyle = COLORS.rowBg;
      ctx.fillRect(LABEL_WIDTH, rowTop, timelineWidth, ROW_HEIGHT);

      ctx.fillStyle = COLORS.label;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`Rank ${rid}`, LABEL_WIDTH - 8, rowTop + ROW_HEIGHT / 2);

      // State bar: waiting baseline, then inferencing intervals painted over.
      ctx.fillStyle = COLORS.waiting;
      ctx.fillRect(LABEL_WIDTH, stateBarY, timelineWidth, STATE_BAR_HEIGHT);
      paintIntervals(ctx, rank.inferencing, viewStart, viewEnd, timeToX,
                     LABEL_WIDTH, timelineWidth, stateBarY, STATE_BAR_HEIGHT,
                     COLORS.inferencing);

      // Queue-depth line plot (step function — samples are every ~200ms and
      // depth is a discrete integer count).
      paintQueuePlot(ctx, rank.sampleT, rank.sampleN, viewStart, viewEnd,
                     timeToX, LABEL_WIDTH, plotTop, plotHeight,
                     maxQ, COLORS.queueLine);
    }

    // Phase overlays + lines (context for collect / train / eval).
    paintPhaseOverlays(ctx, phasePairs, timeToX, LABEL_WIDTH, width, 0, height);
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
  }, [data, viewStart, viewEnd, dataOrigin, viewportWidth, bodyHeight, rankIds, phasePairs, maxQ]);

  useEffect(() => {
    const el = chartRef.current;
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
      const newDuration = Math.max(Math.min(duration * zoomFactor, MAX_VIEW_DURATION), 0.05);
      const pivot = viewStart + clamped * duration;
      setAutoFollow(false);
      setViewStart(pivot - clamped * newDuration);
      setViewEnd(pivot + (1 - clamped) * newDuration);
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, [viewStart, viewEnd, data]);

  const shiftView = useCallback((delta: number) => {
    setAutoFollow(false);
    const bounds = computeBounds(data);
    const duration = viewEnd - viewStart;
    let newStart = viewStart + delta;
    let newEnd = viewEnd + delta;
    if (bounds) {
      if (newEnd < bounds.min + duration * 0.1) {
        newStart = bounds.min;
        newEnd = bounds.min + duration;
      }
      if (newStart > bounds.max - duration * 0.1) {
        newEnd = bounds.max;
        newStart = bounds.max - duration;
      }
    }
    setViewStart(newStart);
    setViewEnd(newEnd);
  }, [data, viewStart, viewEnd]);

  const goStart = useCallback(() => {
    const bounds = computeBounds(data);
    if (!bounds) return;
    setAutoFollow(false);
    const duration = Math.min(viewEnd - viewStart, MAX_VIEW_DURATION);
    setViewStart(bounds.min);
    setViewEnd(bounds.min + duration);
  }, [data, viewStart, viewEnd]);

  const goEnd = useCallback(() => setAutoFollow(true), []);

  const handleHeaderMouseMove = useCallback((e: React.MouseEvent) => {
    const chart = chartRef.current;
    if (!chart) return;
    const rect = chart.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < LABEL_WIDTH || x > rect.width) {
      setHoverX(null);
      return;
    }
    setHoverX(x);
  }, []);

  const handleHeaderMouseLeave = useCallback(() => setHoverX(null), []);

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

  const hasData = rankIds.length > 0;

  return (
    <div className="main" style={{ padding: 16 }}>
      <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
          <div className="card-title" style={{ margin: 0 }}>LLM Inference Timelines</div>
          <div style={{ display: 'flex', gap: 12, fontSize: 'var(--font-sm)', flexWrap: 'wrap' }}>
            <LegendSwatch color={COLORS.inferencing} label="Inferencing" />
            <LegendSwatch color={COLORS.waiting} label="Waiting" />
            <LegendLine color={COLORS.queueLine} label={`Queue depth (max ${data.maxQueueDepth})`} />
            <LegendDash color={COLORS.phase_collect} label="collect" />
            <LegendDash color={COLORS.phase_eval} label="eval" />
            <LegendDash color={COLORS.phase_train} label="train" />
          </div>
          <div style={{ marginLeft: 'auto', fontSize: 'var(--font-xs)', color: 'var(--text-secondary)' }}>
            Ctrl+wheel to zoom, drag to pan
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            <button className="tab-btn" onClick={goStart} style={{ fontSize: 'var(--font-xs)' }}>Start</button>
            <button className="tab-btn" onClick={() => shiftView(-NAV_STEP)} style={{ fontSize: 'var(--font-xs)' }}>-30m</button>
            <button className="tab-btn" onClick={() => shiftView(NAV_STEP)} style={{ fontSize: 'var(--font-xs)' }}>+30m</button>
            <button className="tab-btn" onClick={goEnd} style={{ fontSize: 'var(--font-xs)' }}>End</button>
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
            No LLM data yet. Per-rank inference servers need a moment to report in.
          </div>
        ) : (
          <div
            ref={attachChartRef}
            style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
          >
            <div
              style={{ flexShrink: 0, borderBottom: `1px solid ${COLORS.grid}`, background: COLORS.background }}
              onMouseMove={handleHeaderMouseMove}
              onMouseLeave={handleHeaderMouseLeave}
            >
              <canvas ref={headerCanvasRef} style={{ display: 'block' }} />
            </div>
            <div
              ref={scrollRef}
              style={{ flex: 1, minHeight: 0, overflowY: 'auto', overflowX: 'hidden' }}
            >
              <canvas
                ref={bodyCanvasRef}
                onMouseDown={handleMouseDown}
                style={{ display: 'block', cursor: dragging ? 'grabbing' : 'grab' }}
              />
            </div>
            <NowCursor
              mode={mode}
              now={nowSec}
              viewStart={viewStart}
              viewEnd={viewEnd}
              viewportWidth={viewportWidth}
            />
            <HoverCursor
              hoverX={hoverX}
              viewStart={viewStart}
              viewEnd={viewEnd}
              viewportWidth={viewportWidth}
              dataOrigin={dataOrigin}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function paintIntervals(
  ctx: CanvasRenderingContext2D,
  arr: Float64Array,
  viewStart: number,
  viewEnd: number,
  timeToX: (t: number) => number,
  leftClip: number,
  timelineWidth: number,
  y: number,
  height: number,
  color: string,
) {
  const n = arr.length >> 1;
  if (n === 0) return;
  const col = new Float32Array(timelineWidth);
  const startIdx = firstVisiblePair(arr, viewStart);
  for (let i = startIdx; i < n; i++) {
    const s = arr[i * 2];
    if (s > viewEnd) break;
    const e = arr[i * 2 + 1];
    let x1 = timeToX(s);
    let x2 = timeToX(e);
    if (x2 - x1 < MIN_BAR_PX) x2 = x1 + MIN_BAR_PX;
    let p1 = Math.floor(x1 - leftClip);
    let p2 = Math.ceil(x2 - leftClip);
    if (p1 < 0) p1 = 0;
    if (p2 > timelineWidth) p2 = timelineWidth;
    if (p2 <= p1) continue;
    for (let p = p1; p < p2; p++) col[p] = 1;
  }
  ctx.fillStyle = color;
  let runStart = -1;
  for (let p = 0; p < timelineWidth; p++) {
    if (col[p] > 0) {
      if (runStart < 0) runStart = p;
    } else if (runStart >= 0) {
      ctx.fillRect(leftClip + runStart, y, p - runStart, height);
      runStart = -1;
    }
  }
  if (runStart >= 0) ctx.fillRect(leftClip + runStart, y, timelineWidth - runStart, height);
}

// Queue depth as a step-function polyline: each sample holds until the next.
// Y axis runs from plotTop+height (depth=0) up to plotTop (depth=maxQ).
function paintQueuePlot(
  ctx: CanvasRenderingContext2D,
  sampleT: Float64Array,
  sampleN: Float64Array,
  viewStart: number,
  viewEnd: number,
  timeToX: (t: number) => number,
  leftClip: number,
  plotTop: number,
  plotHeight: number,
  maxQ: number,
  color: string,
) {
  if (sampleT.length === 0) return;
  // Find first sample whose time >= viewStart (binary search on sorted array).
  // Back up one so we have the held value at the left edge of the view.
  let lo = 0, hi = sampleT.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sampleT[mid] < viewStart) lo = mid + 1;
    else hi = mid;
  }
  const startIdx = Math.max(0, lo - 1);
  const y0 = plotTop + plotHeight;
  const depthToY = (n: number) => y0 - (n / maxQ) * plotHeight;

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  let started = false;
  let prevY = y0;
  for (let i = startIdx; i < sampleT.length; i++) {
    const t = sampleT[i];
    if (t > viewEnd) {
      // Extend the last segment to the right edge so the line doesn't stop short.
      const xEnd = timeToX(viewEnd);
      if (xEnd >= leftClip) ctx.lineTo(xEnd, prevY);
      break;
    }
    const x = timeToX(t);
    const y = depthToY(sampleN[i]);
    const xClipped = Math.max(x, leftClip);
    if (!started) {
      ctx.moveTo(xClipped, y);
      started = true;
    } else {
      // Step: horizontal to xClipped at prevY, then vertical to new y.
      ctx.lineTo(xClipped, prevY);
      ctx.lineTo(xClipped, y);
    }
    prevY = y;
    if (i === sampleT.length - 1) {
      const xEnd = timeToX(viewEnd);
      if (xEnd > xClipped) ctx.lineTo(xEnd, prevY);
    }
  }
  if (started) ctx.stroke();
}

const PHASE_TARGET_SPACING_PX = 25;
const PHASE_MIN_ALPHA = 0.15;

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

function pairPhases(phases: WirePhase[]): Array<{ name: string; start: number; end: number }> {
  const sorted = [...phases].sort((a, b) => a.t - b.t);
  const pending = new Map<string, number[]>();
  const pairs: Array<{ name: string; start: number; end: number }> = [];
  for (const ph of sorted) {
    if (ph.action === 'start') {
      const stack = pending.get(ph.name) ?? [];
      stack.push(ph.t);
      pending.set(ph.name, stack);
    } else if (ph.action === 'end') {
      const stack = pending.get(ph.name);
      if (stack && stack.length) {
        const start = stack.pop()!;
        pairs.push({ name: ph.name, start, end: ph.t });
      }
    }
  }
  return pairs;
}

function paintPhaseOverlays(
  ctx: CanvasRenderingContext2D,
  pairs: Array<{ name: string; start: number; end: number }>,
  timeToX: (t: number) => number,
  leftClip: number,
  rightClip: number,
  top: number,
  height: number,
) {
  if (pairs.length === 0) return;
  ctx.globalAlpha = PHASE_OVERLAY_ALPHA;
  for (const p of pairs) {
    let x1 = timeToX(p.start);
    let x2 = timeToX(p.end);
    if (x2 < leftClip || x1 > rightClip) continue;
    if (x1 < leftClip) x1 = leftClip;
    if (x2 > rightClip) x2 = rightClip;
    if (x2 - x1 < 1) continue;
    ctx.fillStyle = phaseColor(p.name);
    ctx.fillRect(x1, top, x2 - x1, height);
  }
  ctx.globalAlpha = 1;
}

interface NowCursorProps {
  mode: 'live' | 'standalone';
  now: number;
  viewStart: number;
  viewEnd: number;
  viewportWidth: number;
}

function NowCursor({ mode, now, viewStart, viewEnd, viewportWidth }: NowCursorProps) {
  if (mode !== 'live') return null;
  const duration = viewEnd - viewStart;
  if (duration <= 0 || viewportWidth <= 0) return null;
  if (now < viewStart || now > viewEnd) return null;
  const timelineWidth = viewportWidth - LABEL_WIDTH;
  if (timelineWidth <= 0) return null;
  const x = LABEL_WIDTH + ((now - viewStart) / duration) * timelineWidth;
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: 0,
        bottom: 0,
        width: 1,
        background: COLORS.nowCursor,
        pointerEvents: 'none',
        zIndex: 5,
      }}
    />
  );
}

interface HoverCursorProps {
  hoverX: number | null;
  viewStart: number;
  viewEnd: number;
  viewportWidth: number;
  dataOrigin: number;
}

function HoverCursor({ hoverX, viewStart, viewEnd, viewportWidth, dataOrigin }: HoverCursorProps) {
  if (hoverX === null) return null;
  const duration = viewEnd - viewStart;
  if (duration <= 0 || viewportWidth <= 0) return null;
  const timelineWidth = viewportWidth - LABEL_WIDTH;
  if (timelineWidth <= 0) return null;
  const t = viewStart + ((hoverX - LABEL_WIDTH) / timelineWidth) * duration;
  const label = formatHoverTime(t - dataOrigin);
  const APPROX_LABEL_WIDTH = 90;
  const flipLeft = hoverX + 6 + APPROX_LABEL_WIDTH > viewportWidth - 4;
  return (
    <>
      <div
        style={{
          position: 'absolute',
          left: hoverX,
          top: 0,
          bottom: 0,
          width: 1,
          background: 'rgba(230, 237, 243, 0.35)',
          pointerEvents: 'none',
          zIndex: 6,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: flipLeft ? undefined : hoverX + 6,
          right: flipLeft ? viewportWidth - hoverX + 6 : undefined,
          top: 4,
          fontSize: 10,
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          background: 'rgba(13, 17, 23, 0.9)',
          color: 'var(--text-primary, #e6edf3)',
          padding: '2px 5px',
          borderRadius: 2,
          border: `1px solid ${COLORS.grid}`,
          pointerEvents: 'none',
          whiteSpace: 'nowrap',
          zIndex: 6,
        }}
      >
        {label}
      </div>
    </>
  );
}

function formatHoverTime(seconds: number): string {
  if (seconds < 0) seconds = 0;
  if (seconds < 60) return `${seconds.toFixed(3)}s`;
  const mins = Math.floor(seconds / 60);
  const remSecs = seconds - mins * 60;
  if (seconds < 3600) return `${mins}m ${remSecs.toFixed(2)}s`;
  const hrs = Math.floor(mins / 60);
  const remMins = mins - hrs * 60;
  return `${hrs}h ${remMins}m ${remSecs.toFixed(1)}s`;
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ width: 12, height: 12, borderRadius: 2, background: color, display: 'inline-block' }} />
      {label}
    </span>
  );
}

function LegendLine({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ width: 16, height: 2, background: color, display: 'inline-block' }} />
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
