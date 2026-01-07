import { useState, useEffect } from 'react';
import { ReplayBufferFile, Transition, TacticEntry } from '../types';
import { Modal } from './Modal';

interface ReplayBufferPanelProps {
  outputDir: string | null;
}

type Tab = 'live' | 'saved' | 'tactics';

interface ModalData {
  type: 'transition' | 'tactic';
  state: string;
  tactic: string;
  success?: boolean;
}

export function ReplayBufferPanel({ outputDir }: ReplayBufferPanelProps) {
  const [tab, setTab] = useState<Tab>('live');
  const [files, setFiles] = useState<ReplayBufferFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [savedTransitions, setSavedTransitions] = useState<Transition[]>([]);
  const [liveTransitions, setLiveTransitions] = useState<Transition[]>([]);
  const [tactics, setTactics] = useState<TacticEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalData, setModalData] = useState<ModalData | null>(null);

  // Fetch replay buffer files list
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const res = await fetch('/api/replay_buffers');
        if (res.ok) {
          const data = await res.json();
          setFiles(data.files);
          // Auto-select latest file if none selected
          if (data.files.length > 0 && !selectedFile) {
            const latest = data.files[data.files.length - 1];
            setSelectedFile(latest.name);
          }
        }
      } catch (e) {
        // Ignore
      }
    };
    fetchFiles();
    const interval = setInterval(fetchFiles, 5000);
    return () => clearInterval(interval);
  }, [selectedFile]);

  // Fetch saved transitions when file is selected
  useEffect(() => {
    if (!selectedFile) return;
    const fetchTransitions = async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/replay_buffers/${selectedFile}`);
        if (res.ok) {
          const data = await res.json();
          setSavedTransitions(data.transitions || []);
        }
      } catch (e) {
        // Ignore
      }
      setLoading(false);
    };
    fetchTransitions();
  }, [selectedFile]);

  // Fetch live transitions (during collection)
  useEffect(() => {
    const fetchLiveTransitions = async () => {
      try {
        const res = await fetch('/api/live_transitions');
        if (res.ok) {
          const data = await res.json();
          setLiveTransitions(data.transitions || []);
        }
      } catch (e) {
        // Ignore
      }
    };
    fetchLiveTransitions();
    const interval = setInterval(fetchLiveTransitions, 2000);
    return () => clearInterval(interval);
  }, []);

  // Fetch tactics
  useEffect(() => {
    const fetchTactics = async () => {
      try {
        const res = await fetch('/api/tactics');
        if (res.ok) {
          const data = await res.json();
          setTactics(data.tactics || []);
        }
      } catch (e) {
        // Ignore
      }
    };
    fetchTactics();
    const interval = setInterval(fetchTactics, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="card replay-buffer-panel">
      <div className="replay-header">
        <div className="card-title" style={{ marginBottom: 0 }}>
          Data
          {outputDir && (
            <span className="output-dir" title={outputDir}>
              {outputDir}
            </span>
          )}
        </div>
        <div className="replay-tabs">
          <button 
            className={tab === 'live' ? 'active' : ''} 
            onClick={() => setTab('live')}
          >
            Live ({liveTransitions.length})
          </button>
          <button 
            className={tab === 'saved' ? 'active' : ''} 
            onClick={() => setTab('saved')}
          >
            Saved ({savedTransitions.length})
          </button>
          <button 
            className={tab === 'tactics' ? 'active' : ''} 
            onClick={() => setTab('tactics')}
          >
            Tactics ({tactics.length})
          </button>
        </div>
      </div>

      {tab === 'live' && (
        <div className="replay-content">
          <div className="replay-list">
            {liveTransitions.length === 0 ? (
              <div className="replay-empty">No live transitions yet (proofs will appear here during collection)</div>
            ) : (
              liveTransitions.slice(-50).reverse().map((t, i) => (
                <div 
                  key={i} 
                  className="transition-item clickable"
                  onClick={() => setModalData({ type: 'transition', state: String(t[0]), tactic: String(t[1]) })}
                >
                  <div className="transition-state">{String(t[0])}</div>
                  <div className="transition-tactic">→ {t[1]}</div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {tab === 'saved' && (
        <div className="replay-content">
          {files.length > 0 && (
            <div className="replay-file-select">
              <select 
                value={selectedFile || ''} 
                onChange={(e) => setSelectedFile(e.target.value)}
              >
                {files.map((f) => (
                  <option key={f.name} value={f.name}>
                    Step {f.step} ({(f.size / 1024).toFixed(1)} KB)
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="replay-list">
            {loading ? (
              <div className="replay-loading">Loading...</div>
            ) : savedTransitions.length === 0 ? (
              <div className="replay-empty">No transitions in saved buffer</div>
            ) : (
              savedTransitions.slice(-50).map((t, i) => (
                <div 
                  key={i} 
                  className="transition-item clickable"
                  onClick={() => setModalData({ type: 'transition', state: String(t[0]), tactic: String(t[1]) })}
                >
                  <div className="transition-state">{String(t[0])}</div>
                  <div className="transition-tactic">→ {t[1]}</div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {tab === 'tactics' && (
        <div className="tactics-content">
          <div className="tactics-list">
            {tactics.length === 0 ? (
              <div className="replay-empty">No tactics logged yet</div>
            ) : (
              tactics.slice(-50).reverse().map((t, i) => (
                <div 
                  key={i} 
                  className={`tactic-item clickable ${t.success ? 'success' : 'failure'}`}
                  onClick={() => setModalData({ type: 'tactic', state: t.state, tactic: t.tactic, success: t.success })}
                >
                  <span className="tactic-status">{t.success ? '✓' : '✗'}</span>
                  <span className="tactic-text">{t.tactic}</span>
                  <span className="tactic-state" title={t.state}>{t.state}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      <Modal 
        isOpen={modalData !== null} 
        onClose={() => setModalData(null)}
        title={modalData?.type === 'tactic' 
          ? `Tactic ${modalData?.success ? '(Success)' : '(Failed)'}` 
          : 'Transition'}
      >
        {modalData && (
          <>
            <div className="modal-section">
              <div className="modal-label">State</div>
              <div className="modal-code state">{modalData.state}</div>
            </div>
            <div className="modal-section">
              <div className="modal-label">Tactic</div>
              <div className="modal-code tactic">{modalData.tactic}</div>
            </div>
          </>
        )}
      </Modal>
    </div>
  );
}
