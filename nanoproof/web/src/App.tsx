import { useState, useEffect, useCallback } from 'react'
import { MonitorState, LogEntry } from './types'
import { StatsPanel } from './components/StatsPanel'
import { ProverGrid } from './components/ProverGrid'
import { GPUPanel } from './components/GPUPanel'
import { LogViewer } from './components/LogViewer'
import { ReplayBufferPanel } from './components/ReplayBufferPanel'
import { LeanServerPanel } from './components/LeanServerPanel'

const POLL_INTERVAL = 1000;

function App() {
  const [state, setState] = useState<MonitorState | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logComponents, setLogComponents] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedActor, setSelectedActor] = useState<string | null>(null);

  const copyOutputDir = useCallback(() => {
    if (state?.output_dir) {
      navigator.clipboard.writeText(state.output_dir).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  }, [state?.output_dir]);

  const handleActorClick = useCallback((actorId: number) => {
    setSelectedActor(`actor_${actorId}`);
  }, []);

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
        setLogs(data.logs.slice(-500));
      } catch (e) {
        // Ignore log fetch errors
      }
    };

    const fetchLogComponents = async () => {
      try {
        const res = await fetch('/api/logs');
        if (!res.ok) return;
        const data = await res.json();
        setLogComponents(data.components || []);
      } catch (e) {
        // Ignore errors
      }
    };

    fetchState();
    fetchLogs();
    fetchLogComponents();

    const stateInterval = setInterval(fetchState, POLL_INTERVAL);
    const logsInterval = setInterval(fetchLogs, POLL_INTERVAL);
    const componentsInterval = setInterval(fetchLogComponents, POLL_INTERVAL * 5);

    return () => {
      clearInterval(stateInterval);
      clearInterval(logsInterval);
      clearInterval(componentsInterval);
    };
  }, []);


  if (error) {
    return (
      <div className="app">
        <div className="header">
          <h1>nanoproof</h1>
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
          <h1>nanoproof</h1>
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
        <h1>nanoproof</h1>
        <span className={phaseClass}>{state.phase}</span>
        <span style={{ color: 'var(--text-secondary)' }}>Step {state.step}</span>
        <div className="header-right">
          {state.output_dir && (
            <button 
              className={`output-dir-badge ${copied ? 'copied' : ''}`} 
              title={`Click to copy: ${state.output_dir}`}
              onClick={copyOutputDir}
            >
              {copied ? '✓ Copied!' : `📁 ${state.output_dir.split('/').slice(-2).join('/')}`}
            </button>
          )}
        </div>
      </div>

      <div className="main">
        {/* Row 1: Stats + Provers + Lean Servers */}
        <div className="row row-top">
          <StatsPanel 
            collection={state.collection} 
            training={state.training} 
            phase={state.phase}
            replayBufferSize={state.replay_buffer_size}
            evalProgress={state.eval_progress}
            evalHistory={state.eval_history}
          />
          
          <div className="card">
            <div className="card-title">Provers</div>
            <ProverGrid 
              servers={state.prover_servers} 
              localActors={state.local_actors}
              onActorClick={handleActorClick}
            />
          </div>

          <LeanServerPanel server={state.lean_server} servers={state.lean_servers} />
        </div>

        {/* Row 2: GPUs */}
        <div className="row">
          <GPUPanel gpus={state.gpus} />
        </div>

        {/* Row 3: Data */}
        <div className="row">
          <ReplayBufferPanel />
        </div>

        {/* Row 4: Logs */}
        <div className="row row-logs">
          <LogViewer 
            logs={logs} 
            components={logComponents}
            selectedActor={selectedActor}
            onActorSelect={setSelectedActor}
          />
        </div>
      </div>
    </div>
  );
}

export default App
