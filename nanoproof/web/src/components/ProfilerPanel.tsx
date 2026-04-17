import { useState, useEffect, useRef, useCallback } from 'react'
import { InstrumentationData } from '../types'

const ROW_HEIGHT = 28;
const ROW_GAP = 4;
const LABEL_WIDTH = 80;
const HEADER_HEIGHT = 40;
const POLL_INTERVAL_LIVE = 2000;

const COLORS = {
  llm: '#58a6ff',
  lean: '#f85149',
  phase_collect: '#3fb950',
  phase_eval: '#d29922',
  phase_train: '#58a6ff',
  background: '#0d1117',
  rowBg: '#161b22',
  label: '#8b949e',
  grid: '#21262d',
  phaseLabel: '#e6edf3',
};

interface Props {
  mode: 'live' | 'standalone';
}

export function ProfilerPanel({ mode }: Props) {
  const [data, setData] = useState<InstrumentationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // View state: time range in seconds (absolute timestamps)
  const [viewStart, setViewStart] = useState(0);
  const [viewEnd, setViewEnd] = useState(60);
  const [autoFollow, setAutoFollow] = useState(true);
  const [dataOrigin, setDataOrigin] = useState(0); // earliest timestamp in data

  // Drag state
  const dragRef = useRef<{ startX: number; viewStart: number; viewEnd: number } | null>(null);

  // Fetch data
  useEffect(() => {
    const endpoint = mode === 'standalone'
      ? '/api/instrumentation/file'
      : '/api/instrumentation';

    const fetchData = async () => {
      try {
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error('Failed to fetch instrumentation data');
        const d: InstrumentationData = await res.json();
        setData(d);
        setError(null);
      } catch (e) {
        setError('Cannot load instrumentation data');
      }
    };

    fetchData();
    if (mode === 'live') {
      const interval = setInterval(fetchData, POLL_INTERVAL_LIVE);
      return () => clearInterval(interval);
    }
  }, [mode]);

  // Compute data origin and auto-follow
  useEffect(() => {
    if (!data) return;

    let minTime = Infinity;
    let maxTime = -Infinity;

    for (const actor of Object.values(data.actors)) {
      for (const ev of actor.events) {
        if (ev.start < minTime) minTime = ev.start;
        if (ev.end > maxTime) maxTime = ev.end;
      }
    }
    for (const ph of data.phases) {
      if (ph.time < minTime) minTime = ph.time;
      if (ph.time > maxTime) maxTime = ph.time;
    }

    if (minTime === Infinity) return;
    setDataOrigin(minTime);

    if (autoFollow) {
      const range = maxTime - minTime;
      const padding = Math.max(range * 0.05, 5);
      setViewStart(minTime);
      setViewEnd(maxTime + padding);
    }
  }, [data, autoFollow]);

  // Canvas rendering
  useEffect(() => {
    if (!data || !canvasRef.current || !containerRef.current) return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    const dpr = window.devicePixelRatio || 1;

    const width = container.clientWidth;
    const actorIds = Object.keys(data.actors).sort((a, b) => Number(a) - Number(b));
    const numRows = actorIds.length;
    const height = Math.max(HEADER_HEIGHT + numRows * (ROW_HEIGHT + ROW_GAP) + 20, 200);

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    const timelineWidth = width - LABEL_WIDTH;
    const duration = viewEnd - viewStart;
    if (duration <= 0) return;

    const timeToX = (t: number) => LABEL_WIDTH + ((t - viewStart) / duration) * timelineWidth;

    // Draw time axis
    ctx.fillStyle = COLORS.label;
    ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';

    const tickInterval = computeTickInterval(duration, timelineWidth);
    const firstTick = Math.ceil((viewStart - dataOrigin) / tickInterval) * tickInterval + dataOrigin;
    for (let t = firstTick; t <= viewEnd; t += tickInterval) {
      const x = timeToX(t);
      if (x < LABEL_WIDTH || x > width) continue;

      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, HEADER_HEIGHT);
      ctx.lineTo(x, height);
      ctx.stroke();

      const label = formatTime(t - dataOrigin);
      ctx.fillStyle = COLORS.label;
      ctx.fillText(label, x, HEADER_HEIGHT - 8);
    }

    // Draw phase events as vertical lines
    for (const ph of data.phases) {
      if (ph.action !== 'start') continue;
      const x = timeToX(ph.time);
      if (x < LABEL_WIDTH || x > width) continue;

      const color = ph.name === 'collect' ? COLORS.phase_collect
        : ph.name === 'eval' ? COLORS.phase_eval
        : COLORS.phase_train;

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, HEADER_HEIGHT);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 10px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(ph.name, x + 3, HEADER_HEIGHT + 12);
      ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
    }

    // Draw phase end markers
    for (const ph of data.phases) {
      if (ph.action !== 'end') continue;
      const x = timeToX(ph.time);
      if (x < LABEL_WIDTH || x > width) continue;

      const color = ph.name === 'collect' ? COLORS.phase_collect
        : ph.name === 'eval' ? COLORS.phase_eval
        : COLORS.phase_train;

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(x, HEADER_HEIGHT);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw actor rows
    for (let i = 0; i < actorIds.length; i++) {
      const aid = actorIds[i];
      const y = HEADER_HEIGHT + i * (ROW_HEIGHT + ROW_GAP);

      // Row background
      ctx.fillStyle = COLORS.rowBg;
      ctx.fillRect(LABEL_WIDTH, y, timelineWidth, ROW_HEIGHT);

      // Label
      ctx.fillStyle = COLORS.label;
      ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`Actor ${aid}`, LABEL_WIDTH - 8, y + ROW_HEIGHT / 2 + 4);

      // Events
      const events = data.actors[aid].events;
      for (const ev of events) {
        if (ev.end < viewStart || ev.start > viewEnd) continue;

        const x1 = Math.max(timeToX(ev.start), LABEL_WIDTH);
        const x2 = Math.min(timeToX(ev.end), width);
        const barWidth = Math.max(x2 - x1, 1); // at least 1px

        ctx.fillStyle = ev.type === 'llm' ? COLORS.llm : COLORS.lean;
        ctx.fillRect(x1, y + 2, barWidth, ROW_HEIGHT - 4);
      }
    }
  }, [data, viewStart, viewEnd, dataOrigin]);

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setAutoFollow(false);

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const fraction = (mouseX - LABEL_WIDTH) / (rect.width - LABEL_WIDTH);
    const clampedFraction = Math.max(0, Math.min(1, fraction));

    const duration = viewEnd - viewStart;
    const zoomFactor = e.deltaY > 0 ? 1.2 : 1 / 1.2;
    const newDuration = Math.max(duration * zoomFactor, 0.1);

    const pivot = viewStart + clampedFraction * duration;
    const newStart = pivot - clampedFraction * newDuration;
    const newEnd = pivot + (1 - clampedFraction) * newDuration;

    setViewStart(newStart);
    setViewEnd(newEnd);
  }, [viewStart, viewEnd]);

  // Mouse drag pan
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setAutoFollow(false);
    dragRef.current = { startX: e.clientX, viewStart, viewEnd };
  }, [viewStart, viewEnd]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const dx = e.clientX - dragRef.current.startX;
    const timelineWidth = rect.width - LABEL_WIDTH;
    const duration = dragRef.current.viewEnd - dragRef.current.viewStart;
    const timeDelta = -(dx / timelineWidth) * duration;

    setViewStart(dragRef.current.viewStart + timeDelta);
    setViewEnd(dragRef.current.viewEnd + timeDelta);
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      // Trigger re-render by updating a dummy state or just rely on the data dependency
      setData(d => d ? { ...d } : d);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

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

  const hasData = Object.keys(data.actors).length > 0 || data.phases.length > 0;

  return (
    <div className="main" style={{ padding: 16 }}>
      <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12 }}>
          <div className="card-title" style={{ margin: 0 }}>Actor Timelines</div>
          <div style={{ display: 'flex', gap: 12, fontSize: 'var(--font-sm)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 12, height: 12, borderRadius: 2, background: COLORS.llm, display: 'inline-block' }} />
              LLM Inference
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 12, height: 12, borderRadius: 2, background: COLORS.lean, display: 'inline-block' }} />
              Lean Verification
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 12, height: 3, borderTop: `2px dashed ${COLORS.phase_collect}`, display: 'inline-block' }} />
              collect
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 12, height: 3, borderTop: `2px dashed ${COLORS.phase_eval}`, display: 'inline-block' }} />
              eval
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 12, height: 3, borderTop: `2px dashed ${COLORS.phase_train}`, display: 'inline-block' }} />
              train
            </span>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            <button
              className={`tab-btn ${autoFollow ? 'active' : ''}`}
              onClick={() => setAutoFollow(!autoFollow)}
              style={{ fontSize: 'var(--font-xs)' }}
            >
              Auto-follow
            </button>
          </div>
        </div>

        {!hasData ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-secondary)' }}>
            No instrumentation data yet. Timeline events will appear during collection and evaluation.
          </div>
        ) : (
          <div
            ref={containerRef}
            style={{ flex: 1, minHeight: 200, cursor: dragRef.current ? 'grabbing' : 'grab' }}
          >
            <canvas
              ref={canvasRef}
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{ display: 'block' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function computeTickInterval(duration: number, width: number): number {
  const targetTicks = Math.max(width / 100, 3);
  const rawInterval = duration / targetTicks;

  // Snap to nice intervals
  const niceIntervals = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
  for (const ni of niceIntervals) {
    if (ni >= rawInterval) return ni;
  }
  return Math.ceil(rawInterval / 3600) * 3600;
}

function formatTime(seconds: number): string {
  if (seconds < 0) seconds = 0;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins < 60) return `${mins}m${secs.toFixed(0)}s`;
  const hrs = Math.floor(mins / 60);
  const remainMins = mins % 60;
  return `${hrs}h${remainMins}m`;
}
