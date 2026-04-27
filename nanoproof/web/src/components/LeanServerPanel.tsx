import { useEffect, useRef, useState } from 'react';
import { LeanServerStatus } from '../types';

interface LeanServerPanelProps {
  server: LeanServerStatus;
  servers?: LeanServerStatus[];
}

const STOPPING_RED_MS = 10_000;

// Returns true once stopping_processes has been continuously nonzero for
// STOPPING_RED_MS. Brief teardown spikes after a recycle stay green; only
// a wedged janitor crosses the threshold.
function useStoppingWedged(stopping: number): boolean {
  const firstNonzeroRef = useRef<number | null>(null);
  const [wedged, setWedged] = useState(false);

  useEffect(() => {
    if (stopping === 0) {
      firstNonzeroRef.current = null;
      setWedged(false);
      return;
    }
    if (firstNonzeroRef.current === null) {
      firstNonzeroRef.current = Date.now();
    }
    const elapsed = Date.now() - firstNonzeroRef.current;
    if (elapsed >= STOPPING_RED_MS) {
      setWedged(true);
      return;
    }
    const handle = window.setTimeout(
      () => setWedged(true),
      STOPPING_RED_MS - elapsed,
    );
    return () => window.clearTimeout(handle);
  }, [stopping]);

  return wedged;
}

function SingleLeanServer({ server, compact = false }: { server: LeanServerStatus; compact?: boolean }) {
  const stoppingWedged = useStoppingWedged(server.stopping_processes);
  // When disconnected, show minimal info
  if (!server.connected) {
    return (
      <div className={`lean-server-item ${compact ? 'compact' : ''}`}>
        <div className="lean-server-header">
          <span className="lean-server-url">{server.address}:{server.port}</span>
          <span className="lean-status disconnected">○</span>
        </div>
        {server.error && !compact && (
          <div className="lean-error-compact">{server.error}</div>
        )}
      </div>
    );
  }

  const avgCpu = server.cpu_percent.length > 0
    ? server.cpu_percent.reduce((a, b) => a + b, 0) / server.cpu_percent.length
    : 0;

  if (compact) {
    // Compact view for multi-server display
    return (
      <div className="lean-server-item compact">
        <div className="lean-server-header">
          <span className="lean-server-url">{server.address}:{server.port}</span>
          <span className="lean-status connected">●</span>
        </div>
        <div className="lean-compact-stats">
          <div className="lean-compact-stat">
            <span className="lean-compact-label">Proc</span>
            <span className="lean-compact-value">{server.available_processes}/{server.max_processes}</span>
          </div>
          <div className="lean-compact-stat">
            <span className="lean-compact-label">Start</span>
            <span className="lean-compact-value">{server.starting_processes}</span>
          </div>
          <div className="lean-compact-stat">
            <span className="lean-compact-label">Stop</span>
            <span
              className="lean-compact-value"
              style={stoppingWedged ? { color: 'var(--danger, #e53935)' } : undefined}
              title={stoppingWedged ? 'stopping_processes nonzero >10s: janitor wedged' : undefined}
            >
              {server.stopping_processes}
            </span>
          </div>
          <div className="lean-compact-stat">
            <span className="lean-compact-label">Idle60s</span>
            <span className="lean-compact-value">{server.idle_too_long_60s}</span>
          </div>
          <div className="lean-compact-stat">
            <span className="lean-compact-label">CPU</span>
            <span className="lean-compact-value">{avgCpu.toFixed(0)}%</span>
          </div>
          <div className="lean-compact-stat">
            <span className="lean-compact-label">RAM</span>
            <span className="lean-compact-value">{server.ram_percent.toFixed(0)}%</span>
          </div>
        </div>
        <div className="lean-compact-bars">
          <div className="lean-bar-row">
            <span className="lean-bar-label">CPU</span>
            <div className="lean-bar-tiny">
              <div
                className={`lean-bar-fill ${avgCpu > 80 ? 'high' : avgCpu > 50 ? 'medium' : 'low'}`}
                style={{ width: `${avgCpu}%` }}
              />
            </div>
          </div>
          <div className="lean-bar-row">
            <span className="lean-bar-label">RAM</span>
            <div className="lean-bar-tiny">
              <div
                className={`lean-bar-fill ${server.ram_percent > 80 ? 'high' : server.ram_percent > 50 ? 'medium' : 'low'}`}
                style={{ width: `${server.ram_percent}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Full view for single server
  return (
    <div className="lean-server-item">
      <div className="lean-server-header">
        <span className="lean-server-url">{server.address}:{server.port}</span>
        <span className="lean-status connected">● Connected</span>
      </div>

      <div className="lean-stats">
        <div className="lean-stat">
          <div className="lean-stat-value">
            {server.available_processes}/{server.max_processes}
          </div>
          <div className="lean-stat-label">Available</div>
        </div>

        <div className="lean-stat">
          <div className="lean-stat-value">
            {server.starting_processes}
          </div>
          <div className="lean-stat-label">Starting</div>
        </div>

        <div className="lean-stat">
          <div
            className="lean-stat-value"
            style={stoppingWedged ? { color: 'var(--danger, #e53935)' } : undefined}
            title={stoppingWedged ? 'stopping_processes nonzero >10s: janitor wedged' : undefined}
          >
            {server.stopping_processes}
          </div>
          <div className="lean-stat-label">Stopping</div>
        </div>

        <div className="lean-stat">
          <div className="lean-stat-value">
            {server.idle_too_long_60s}
          </div>
          <div className="lean-stat-label">Idle &gt;60s</div>
        </div>

        <div className="lean-stat">
          <div className="lean-stat-value">
            {avgCpu.toFixed(0)}%
          </div>
          <div className="lean-stat-label">CPU Avg</div>
          <div className="lean-bar">
            <div 
              className={`lean-bar-fill ${avgCpu > 80 ? 'high' : avgCpu > 50 ? 'medium' : 'low'}`}
              style={{ width: `${avgCpu}%` }}
            />
          </div>
        </div>

        <div className="lean-stat">
          <div className="lean-stat-value">
            {server.ram_used_gb.toFixed(1)}/{server.ram_total_gb.toFixed(0)} GB
          </div>
          <div className="lean-stat-label">RAM ({server.ram_percent.toFixed(0)}%)</div>
          <div className="lean-bar">
            <div 
              className={`lean-bar-fill ${server.ram_percent > 80 ? 'high' : server.ram_percent > 50 ? 'medium' : 'low'}`}
              style={{ width: `${server.ram_percent}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export function LeanServerPanel({ server, servers }: LeanServerPanelProps) {
  // If we have multiple servers, show them in a grid
  if (servers && servers.length > 0) {
    return (
      <div className="card lean-server-card">
        <div className="card-title">Lean Servers</div>
        <div className="lean-servers-grid">
          {servers.map((s, i) => (
            <SingleLeanServer key={`${s.address}:${s.port}-${i}`} server={s} compact={true} />
          ))}
        </div>
      </div>
    );
  }

  // Single server (local mode) or not configured
  if (!server.address) {
    return (
      <div className="card">
        <div className="card-title">Lean Server</div>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
          Not configured
        </div>
      </div>
    );
  }

  return (
    <div className="card lean-server-card">
      <div className="card-title">Lean Server</div>
      <SingleLeanServer server={server} compact={false} />
    </div>
  );
}
