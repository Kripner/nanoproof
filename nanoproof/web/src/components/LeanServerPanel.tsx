import { LeanServerStatus } from '../types';

interface LeanServerPanelProps {
  server: LeanServerStatus;
  servers?: LeanServerStatus[];
}

function SingleLeanServer({ server, compact = false }: { server: LeanServerStatus; compact?: boolean }) {
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

  const processUtilization = server.max_processes > 0 
    ? (server.used_processes / server.max_processes) * 100 
    : 0;
  
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
            <span className="lean-compact-label">CPU</span>
            <span className="lean-compact-value">{avgCpu.toFixed(0)}%</span>
          </div>
          <div className="lean-compact-stat">
            <span className="lean-compact-label">RAM</span>
            <span className="lean-compact-value">{server.ram_percent.toFixed(0)}%</span>
          </div>
        </div>
        <div className="lean-compact-bars">
          <div className="lean-bar-tiny">
            <div 
              className={`lean-bar-fill ${processUtilization > 80 ? 'high' : processUtilization > 50 ? 'medium' : 'low'}`}
              style={{ width: `${processUtilization}%` }}
            />
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
          <div className="lean-bar">
            <div 
              className={`lean-bar-fill ${processUtilization > 80 ? 'high' : processUtilization > 50 ? 'medium' : 'low'}`}
              style={{ width: `${processUtilization}%` }}
            />
          </div>
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
