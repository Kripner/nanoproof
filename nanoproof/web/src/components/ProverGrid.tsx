import { ProverServer } from '../types';

interface ProverGridProps {
  servers: Record<string, ProverServer>;
}

export function ProverGrid({ servers }: ProverGridProps) {
  const serverList = Object.values(servers);

  if (serverList.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
        No prover servers registered
      </div>
    );
  }

  return (
    <div className="prover-grid">
      {serverList.map((server) => (
        <div key={server.address} className="prover-server">
          <div className="prover-server-header">
            {server.address}
            {!server.connected && <span style={{ color: 'var(--accent-red)' }}> (disconnected)</span>}
          </div>
          <div className="prover-server-stats">
            {server.games_solved}/{server.games_played} solved | {server.transitions_collected} transitions
          </div>
          <div className="thread-grid">
            {server.threads.map((thread) => (
              <div
                key={thread.id}
                className={`thread thread-${thread.state}`}
                title={`Thread ${thread.id}: ${thread.state}`}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

