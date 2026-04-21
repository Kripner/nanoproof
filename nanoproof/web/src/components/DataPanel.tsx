import { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { TacticEntry } from '../types';

const POLL_INTERVAL = 2000;
const PAGE_SIZE = 200;

interface TransitionRow {
  context: string;
  tactic: string;
  value_target: number;
}

interface ModalData {
  type: 'transition' | 'tactic';
  state: string;
  tactic: string;
  status?: 'success' | 'error' | 'cycle';
  value?: number;
}

export function DataPanel() {
  const [steps, setSteps] = useState<number[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [transitions, setTransitions] = useState<TransitionRow[]>([]);
  const [transitionsTotal, setTransitionsTotal] = useState(0);
  const [tactics, setTactics] = useState<TacticEntry[]>([]);
  const [tacticsTotal, setTacticsTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [modalData, setModalData] = useState<ModalData | null>(null);

  useEffect(() => {
    let alive = true;
    const fetchSteps = async () => {
      try {
        const res = await fetch('/api/collections');
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        const newSteps: number[] = data.steps || [];
        setSteps(newSteps);
        setSelectedStep((prev) => {
          if (prev !== null && newSteps.includes(prev)) return prev;
          return newSteps.length > 0 ? newSteps[newSteps.length - 1] : null;
        });
      } catch {
        // ignore
      }
    };
    fetchSteps();
    const t = setInterval(fetchSteps, POLL_INTERVAL);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    if (selectedStep === null) {
      setTransitions([]);
      setTransitionsTotal(0);
      setTactics([]);
      setTacticsTotal(0);
      return;
    }
    let alive = true;
    setLoading(true);
    const load = async () => {
      try {
        const [rbRes, tacticsRes] = await Promise.all([
          fetch(`/api/collections/${selectedStep}/replay_buffer?limit=${PAGE_SIZE}`),
          fetch(`/api/collections/${selectedStep}/generated_tactics?limit=${PAGE_SIZE}`),
        ]);
        if (!alive) return;
        if (rbRes.ok) {
          const d = await rbRes.json();
          setTransitions(d.transitions || []);
          setTransitionsTotal(d.total ?? (d.transitions?.length || 0));
        } else {
          setTransitions([]);
          setTransitionsTotal(0);
        }
        if (tacticsRes.ok) {
          const d = await tacticsRes.json();
          setTactics(d.tactics || []);
          setTacticsTotal(d.total ?? (d.tactics?.length || 0));
        } else {
          setTactics([]);
          setTacticsTotal(0);
        }
      } catch {
        if (!alive) return;
        setTransitions([]);
        setTransitionsTotal(0);
        setTactics([]);
        setTacticsTotal(0);
      } finally {
        if (alive) setLoading(false);
      }
    };
    load();
    return () => {
      alive = false;
    };
  }, [selectedStep]);

  const sortedSteps = [...steps].sort((a, b) => b - a);

  return (
    <div className="data-panel">
      <div className="data-sidebar">
        <div className="data-sidebar-title">Collections</div>
        {sortedSteps.length === 0 ? (
          <div className="data-empty">No collection steps yet</div>
        ) : (
          <div className="data-step-list">
            {sortedSteps.map((s) => (
              <button
                key={s}
                className={`data-step-btn ${s === selectedStep ? 'active' : ''}`}
                onClick={() => setSelectedStep(s)}
              >
                step {s.toString().padStart(5, '0')}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="data-main">
        {selectedStep === null ? (
          <div className="data-empty">Select a collection step</div>
        ) : (
          <>
            <div className="data-section">
              <div className="data-section-title">
                Replay buffer additions
                <span className="data-section-count">
                  {loading ? '…' : `${transitions.length} / ${transitionsTotal}`}
                </span>
              </div>
              <div className="replay-list">
                {loading ? (
                  <div className="replay-loading">Loading...</div>
                ) : transitions.length === 0 ? (
                  <div className="replay-empty">No transitions at this step</div>
                ) : (
                  transitions.map((t, i) => (
                    <div
                      key={i}
                      className="transition-item clickable"
                      onClick={() =>
                        setModalData({
                          type: 'transition',
                          state: t.context,
                          tactic: t.tactic,
                          value: t.value_target,
                        })
                      }
                    >
                      <div className="transition-state">{t.context}</div>
                      <div className="transition-tactic">
                        <span className="transition-value">
                          {typeof t.value_target === 'number' ? t.value_target.toFixed(2) : '?'}
                        </span>
                        → {t.tactic}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="data-section">
              <div className="data-section-title">
                Generated tactics
                <span className="data-section-count">
                  {loading ? '…' : `${tactics.length} / ${tacticsTotal}`}
                </span>
              </div>
              <div className="tactics-list">
                {loading ? (
                  <div className="replay-loading">Loading...</div>
                ) : tactics.length === 0 ? (
                  <div className="replay-empty">No tactics at this step</div>
                ) : (
                  tactics.map((t, i) => {
                    const statusClass =
                      t.status === 'success' ? 'success' : t.status === 'cycle' ? 'cycle' : 'failure';
                    const statusIcon =
                      t.status === 'success' ? '✓' : t.status === 'cycle' ? '↻' : '✗';
                    return (
                      <div
                        key={i}
                        className={`tactic-item clickable ${statusClass}`}
                        onClick={() =>
                          setModalData({
                            type: 'tactic',
                            state: t.state,
                            tactic: t.tactic,
                            status: t.status,
                          })
                        }
                      >
                        <span className="tactic-status">{statusIcon}</span>
                        <span className="tactic-text">{t.tactic}</span>
                        <span className="tactic-state" title={t.state}>
                          {t.state}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <Modal
        isOpen={modalData !== null}
        onClose={() => setModalData(null)}
        title={
          modalData?.type === 'tactic'
            ? `Tactic (${
                modalData?.status === 'success'
                  ? 'Success'
                  : modalData?.status === 'cycle'
                  ? 'Cycle'
                  : 'Error'
              })`
            : 'Transition'
        }
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
            {modalData.type === 'transition' && modalData.value !== undefined && (
              <div className="modal-section">
                <div className="modal-label">Value Target</div>
                <div className="modal-code value">{modalData.value.toFixed(4)}</div>
              </div>
            )}
          </>
        )}
      </Modal>
    </div>
  );
}
