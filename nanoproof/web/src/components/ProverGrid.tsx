import { ProverServer, LocalActor } from '../types';

interface ProverGridProps {
  servers: Record<string, ProverServer>;
  localActors?: Record<string, LocalActor>;
  onActorClick?: (actorId: number) => void;
}

export function ProverGrid({ servers, localActors, onActorClick }: ProverGridProps) {
  const serverList = Object.values(servers);
  const actorList = Object.values(localActors || {});

  // If we have prover servers, show them
  if (serverList.length > 0) {
    return (
      <div className="prover-grid">
        {serverList.map((server) => {
          const running = server.threads.filter(t => t.state === 'running').length;
          const blocked = server.threads.filter(t => t.state === 'blocked').length;
          const error = server.threads.filter(t => t.state === 'error').length;
          
          return (
            <div key={server.address} className="prover-server">
              <div className="prover-server-header">
                {server.address}
                {!server.connected && <span style={{ color: 'var(--accent-red)' }}> (disconnected)</span>}
              </div>
              <div className="prover-server-stats">
                <span style={{ color: 'var(--accent-green)' }}>{running} running</span>
                {blocked > 0 && <span style={{ color: 'var(--accent-orange)' }}> · {blocked} blocked</span>}
                {error > 0 && <span style={{ color: 'var(--accent-red)' }}> · {error} error</span>}
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
          );
        })}
      </div>
    );
  }

  // If we have local actors, show them
  if (actorList.length > 0) {
    const running = actorList.filter(a => a.state === 'running').length;
    const blocked = actorList.filter(a => a.state === 'blocked').length;
    const error = actorList.filter(a => a.state === 'error').length;
    
    return (
      <div className="prover-grid">
        <div className="prover-server local-actors">
          <div className="prover-server-header">
            Local Actors ({actorList.length})
          </div>
          <div className="prover-server-stats">
            <span style={{ color: 'var(--accent-green)' }}>{running} running</span>
            {blocked > 0 && <span style={{ color: 'var(--accent-orange)' }}> · {blocked} blocked</span>}
            {error > 0 && <span style={{ color: 'var(--accent-red)' }}> · {error} error</span>}
          </div>
          <div className="thread-grid">
            {actorList.map((actor) => {
              const stateLabel = actor.state === 'blocked' 
                ? `⏳ ${actor.current_theorem || 'Reconnecting...'}`
                : actor.state === 'error'
                ? '❌ Error'
                : actor.state === 'running'
                ? '🟢 Running'
                : '⏸️ Idle';
              
              return (
                <div
                  key={actor.id}
                  className={`thread thread-${actor.state} clickable`}
                  title={`Actor ${actor.id}: ${stateLabel} (${actor.games_solved}/${actor.games_played} solved) - Click to view logs`}
                  onClick={() => onActorClick?.(actor.id)}
                />
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  // No provers and no actors
  return (
    <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
      No prover servers or local actors active
    </div>
  );
}
