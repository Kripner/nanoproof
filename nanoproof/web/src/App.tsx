import { useState, useEffect } from 'react'
import { MonitorState, LogEntry } from './types'
import { StatsPanel } from './components/StatsPanel'
import { ProverGrid } from './components/ProverGrid'
import { GPUPanel } from './components/GPUPanel'
import { EvalHistory } from './components/EvalHistory'
import { LogViewer } from './components/LogViewer'

const POLL_INTERVAL = 1000;

function App() {
  const [state, setState] = useState<MonitorState | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) throw new Error('Failed to fetch state');
        const data = await res.json();
        setState(data);
        setError(null);
      } catch (e) {
        setError('Cannot connect to server');
      }
    };

    const fetchLogs = async () => {
      try {
        const res = await fetch('/api/logs/_merged');
        if (!res.ok) return;
        const data = await res.json();
        setLogs(data.logs.slice(-100));
      } catch (e) {
        // Ignore log fetch errors
      }
    };

    fetchState();
    fetchLogs();

    const stateInterval = setInterval(fetchState, POLL_INTERVAL);
    const logsInterval = setInterval(fetchLogs, POLL_INTERVAL);

    return () => {
      clearInterval(stateInterval);
      clearInterval(logsInterval);
    };
  }, []);

  const openLogStream = (component: string) => {
    window.open(`/api/logs/${component}`, '_blank');
  };

  if (error) {
    return (
      <div className="app">
        <div className="header">
          <h1>NanoProof Monitor</h1>
        </div>
        <div style={{ padding: 40, textAlign: 'center', color: 'var(--accent-red)' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="app">
        <div className="header">
          <h1>NanoProof Monitor</h1>
        </div>
        <div style={{ padding: 40, textAlign: 'center' }}>
          Loading...
        </div>
      </div>
    );
  }

  const phaseClass = `phase-badge phase-${state.phase}`;

  return (
    <div className="app">
      <div className="header">
        <h1>NanoProof Monitor</h1>
        <span className={phaseClass}>{state.phase}</span>
        <span style={{ color: 'var(--text-secondary)' }}>Step {state.step}</span>
        <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: 12 }}>
          Buffer: {state.replay_buffer_size.toLocaleString()} transitions
        </span>
      </div>

      <div className="main">
        <StatsPanel collection={state.collection} training={state.training} phase={state.phase} />
        
        <div className="card">
          <div className="card-title">Prover Servers</div>
          <ProverGrid servers={state.prover_servers} />
        </div>

        <GPUPanel gpus={state.gpus} />
        
        <EvalHistory history={state.eval_history} />

        <LogViewer logs={logs} onOpenStream={openLogStream} />
      </div>
    </div>
  );
}

export default App

