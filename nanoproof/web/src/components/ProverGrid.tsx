import { LocalActor } from '../types';

interface ProverGridProps {
  localActors?: Record<string, LocalActor>;
}

export function ProverGrid({ localActors }: ProverGridProps) {
  const actorList = Object.values(localActors || {});

  if (actorList.length > 0) {
    const running = actorList.filter(a => a.state === 'running').length;
    const blocked = actorList.filter(a => a.state === 'blocked').length;
    const retry = actorList.filter(a => a.state === 'retry').length;
    const error = actorList.filter(a => a.state === 'error').length;

    return (
      <div className="prover-grid">
        <div className="prover-server local-actors">
          <div className="prover-server-header">
            Actors ({actorList.length})
          </div>
          <div className="prover-server-stats">
            <span style={{ color: 'var(--accent-green)' }}>{running} running</span>
            {blocked > 0 && <span style={{ color: 'var(--accent-orange)' }}> · {blocked} blocked</span>}
            {retry > 0 && <span style={{ color: 'var(--accent-orange)' }}> · {retry} retrying</span>}
            {error > 0 && <span style={{ color: 'var(--accent-red)' }}> · {error} error</span>}
          </div>
          <div className="thread-grid">
            {actorList.map((actor) => {
              const stateLabel = actor.state === 'blocked'
                ? `⏳ ${actor.current_theorem || 'Reconnecting...'}`
                : actor.state === 'retry'
                ? '🔄 Retrying'
                : actor.state === 'error'
                ? '❌ Error'
                : actor.state === 'running'
                ? '🟢 Running'
                : '⏸️ Idle';

              return (
                <div
                  key={actor.id}
                  className={`thread thread-${actor.state}`}
                  title={`Actor ${actor.id}: ${stateLabel} (${actor.games_solved}/${actor.games_played} solved)`}
                />
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
      No actors active
    </div>
  );
}
