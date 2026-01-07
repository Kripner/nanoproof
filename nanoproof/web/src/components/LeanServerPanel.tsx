import { LeanServerStatus } from '../types';

interface LeanServerPanelProps {
  server: LeanServerStatus;
}

export function LeanServerPanel({ server }: LeanServerPanelProps) {
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

  // When disconnected, show minimal info
  if (!server.connected) {
    return (
      <div className="card lean-server-card">
        <div className="card-title">
          Lean Server
          <span className="lean-status disconnected">○ Disconnected</span>
        </div>
        
        <div className="lean-address">
          {server.address}:{server.port}
        </div>

        {server.error && (
          <div className="lean-error">
            {server.error}
          </div>
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

  return (
    <div className="card lean-server-card">
      <div className="card-title">
        Lean Server
        <span className="lean-status connected">● Connected</span>
      </div>

      <div className="lean-address">
        {server.address}:{server.port}
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
