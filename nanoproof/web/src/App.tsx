import { useState, useEffect, useCallback } from 'react'
import { MonitorState } from './types'
import { StatsPanel } from './components/StatsPanel'
import { ProverGrid } from './components/ProverGrid'
import { GPUPanel } from './components/GPUPanel'
import { LogViewer } from './components/LogViewer'
import { DataPanel } from './components/DataPanel'
import { LeanServerPanel } from './components/LeanServerPanel'
import { ProfilerPanel } from './components/ProfilerPanel'
import { LLMProfilerPanel } from './components/LLMProfilerPanel'
import { TheoremsPanel } from './components/TheoremsPanel'

const POLL_INTERVAL = 1000;

function App() {
  const [state, setState] = useState<MonitorState | null>(null);
  const [stdoutLines, setStdoutLines] = useState<string[]>([]);
  const [stderrLines, setStderrLines] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<'monitor' | 'profiler' | 'llm' | 'data' | 'theorems'>('monitor');

  // Default to profiler tab in standalone mode
  useEffect(() => {
    if (state?.mode === 'standalone') {
      setTab('profiler');
    }
  }, [state?.mode]);

  const copyOutputDir = useCallback(() => {
    if (state?.output_dir) {
      navigator.clipboard.writeText(state.output_dir).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  }, [state?.output_dir]);

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
        const [stdoutRes, stderrRes] = await Promise.all([
          fetch('/api/stdout'),
          fetch('/api/stderr'),
        ]);
        if (stdoutRes.ok) {
          const data = await stdoutRes.json();
          setStdoutLines(data.lines || []);
        }
        if (stderrRes.ok) {
          const data = await stderrRes.json();
          setStderrLines(data.lines || []);
        }
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
        <div className="tab-bar">
          {state.mode !== 'standalone' && (
            <button className={`tab-btn ${tab === 'monitor' ? 'active' : ''}`}
                    onClick={() => setTab('monitor')}>Monitor</button>
          )}
          <button className={`tab-btn ${tab === 'profiler' ? 'active' : ''}`}
                  onClick={() => setTab('profiler')}>Profiler</button>
          <button className={`tab-btn ${tab === 'llm' ? 'active' : ''}`}
                  onClick={() => setTab('llm')}>LLM Profiler</button>
          <button className={`tab-btn ${tab === 'data' ? 'active' : ''}`}
                  onClick={() => setTab('data')}>Data</button>
          <button className={`tab-btn ${tab === 'theorems' ? 'active' : ''}`}
                  onClick={() => setTab('theorems')}>Theorems</button>
        </div>
        {tab === 'monitor' && (
          <>
            <span className={phaseClass}>{state.phase}</span>
            <span style={{ color: 'var(--text-secondary)' }}>Step {state.step}</span>
          </>
        )}
        <div className="header-right">
          {state.output_dir && (
            <button
              className={`output-dir-badge ${copied ? 'copied' : ''}`}
              title={`Click to copy: ${state.output_dir}`}
              onClick={copyOutputDir}
            >
              {copied ? '✓ Copied!' : state.output_dir.split('/').slice(-2).join('/')}
            </button>
          )}
        </div>
      </div>

      {tab === 'monitor' && (
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

            {Object.keys(state.local_actors).length > 0 && (
              <div className="card">
                <div className="card-title">Provers</div>
                <ProverGrid localActors={state.local_actors} />
              </div>
            )}

            <LeanServerPanel server={state.lean_server} servers={state.lean_servers} />
          </div>

          {/* Row 2: GPUs */}
          <div className="row">
            <GPUPanel gpus={state.gpus} />
          </div>

          {/* Row 3: Logs */}
          <div className="row row-logs">
            <LogViewer
              stdoutLines={stdoutLines}
              stderrLines={stderrLines}
            />
          </div>
        </div>
      )}

      {tab === 'profiler' && (
        <ProfilerPanel mode={state.mode} />
      )}

      {tab === 'llm' && (
        <LLMProfilerPanel mode={state.mode} />
      )}

      {tab === 'data' && (
        <DataPanel />
      )}

      {tab === 'theorems' && (
        <TheoremsPanel />
      )}
    </div>
  );
}

export default App
