import { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { TacticEntry } from '../types';

const POLL_INTERVAL = 2000;
const PAGE_SIZE = 200;

interface ProofSummary {
  name: string | null;
  theorem: string;
  num_iterations: number;
  num_transitions: number;
  full_tree_depth: number;
  full_tree_size: number;
  simplified_tree_depth: number;
  simplified_tree_size: number;
}

interface StepEntry {
  step: number;
  num_proofs: number;
  num_transitions: number;
}

interface NodeDict {
  id: string;
  action: string | number | null;
  prior: number | null;
  state: string[];
  reward: number | null;
  to_play: number; // 1 = OR, 2 = AND
  is_solved: boolean;
  visit_count: number;
  evaluations: number;
  value_sum: number;
  value_target: number | null;
  children: Record<string, NodeDict> | null;
}

interface ProofDetail {
  theorem: string;
  name: string | null;
  num_iterations: number;
  full_tree: NodeDict | null;
  simplified_tree: NodeDict | null;
  transitions: [string, string, number][];
}

export function DataPanel() {
  const [stepEntries, setStepEntries] = useState<StepEntry[]>([]);
  const [totalProofs, setTotalProofs] = useState(0);
  const [totalTransitions, setTotalTransitions] = useState(0);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [proofs, setProofs] = useState<ProofSummary[]>([]);
  const [tactics, setTactics] = useState<TacticEntry[]>([]);
  const [tacticsTotal, setTacticsTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedProofIndex, setSelectedProofIndex] = useState<number | null>(null);
  const [proofDetail, setProofDetail] = useState<ProofDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    const fetchSteps = async () => {
      try {
        const res = await fetch('/api/collections');
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        const entries: StepEntry[] = data.entries || [];
        setStepEntries(entries);
        setTotalProofs(data.total_proofs ?? 0);
        setTotalTransitions(data.total_transitions ?? 0);
        setSelectedStep((prev) => {
          const stepNums = entries.map((e) => e.step);
          if (prev !== null && stepNums.includes(prev)) return prev;
          return stepNums.length > 0 ? stepNums[stepNums.length - 1] : null;
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
      setProofs([]);
      setTactics([]);
      setTacticsTotal(0);
      return;
    }
    let alive = true;
    setLoading(true);
    const load = async () => {
      try {
        const [proofsRes, tacticsRes] = await Promise.all([
          fetch(`/api/collections/${selectedStep}/collected`),
          fetch(`/api/collections/${selectedStep}/generated_tactics?limit=${PAGE_SIZE}`),
        ]);
        if (!alive) return;
        if (proofsRes.ok) {
          const d = await proofsRes.json();
          setProofs(d.proofs || []);
        } else {
          setProofs([]);
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
        setProofs([]);
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

  useEffect(() => {
    if (selectedStep === null || selectedProofIndex === null) {
      setProofDetail(null);
      return;
    }
    let alive = true;
    setDetailLoading(true);
    const load = async () => {
      try {
        const res = await fetch(
          `/api/collections/${selectedStep}/collected/${selectedProofIndex}`,
        );
        if (!alive) return;
        if (res.ok) {
          const d = await res.json();
          setProofDetail(d);
        } else {
          setProofDetail(null);
        }
      } catch {
        if (alive) setProofDetail(null);
      } finally {
        if (alive) setDetailLoading(false);
      }
    };
    load();
    return () => {
      alive = false;
    };
  }, [selectedStep, selectedProofIndex]);

  const sortedEntries = [...stepEntries].sort((a, b) => b.step - a.step);
  const selectedProof = selectedProofIndex !== null ? proofs[selectedProofIndex] : null;

  return (
    <div className="data-panel">
      <div className="data-sidebar">
        <div className="data-sidebar-title">
          <span>Collections</span>
          <span className="data-sidebar-totals">
            {totalProofs} proofs · {totalTransitions} trans.
          </span>
        </div>
        {sortedEntries.length === 0 ? (
          <div className="data-empty">No collection steps yet</div>
        ) : (
          <div className="data-step-list">
            {sortedEntries.map((e) => (
              <button
                key={e.step}
                className={`data-step-btn ${e.step === selectedStep ? 'active' : ''}`}
                onClick={() => setSelectedStep(e.step)}
              >
                <span className="data-step-name">
                  step {e.step.toString().padStart(5, '0')}
                </span>
                <span className="data-step-counts">
                  {e.num_proofs}p · {e.num_transitions}t
                </span>
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
                Collected proofs
                <span className="data-section-count">
                  {loading ? '…' : `${proofs.length}`}
                </span>
              </div>
              <div className="replay-list">
                {loading ? (
                  <div className="replay-loading">Loading...</div>
                ) : proofs.length === 0 ? (
                  <div className="replay-empty">No proofs at this step</div>
                ) : (
                  proofs.map((p, i) => (
                    <div
                      key={i}
                      className="proof-item clickable"
                      onClick={() => setSelectedProofIndex(i)}
                    >
                      <span className="proof-name">{p.name ?? `proof #${i}`}</span>
                      <span className="proof-meta">
                        {p.num_transitions} trans · {p.num_iterations} iters ·
                        {' '}full d{p.full_tree_depth}/s{p.full_tree_size} ·
                        {' '}simp d{p.simplified_tree_depth}/s{p.simplified_tree_size}
                      </span>
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
                      <div key={i} className={`tactic-item ${statusClass}`}>
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
        isOpen={selectedProofIndex !== null}
        onClose={() => setSelectedProofIndex(null)}
        title={selectedProof?.name ?? `Proof #${selectedProofIndex ?? ''}`}
      >
        {detailLoading || !proofDetail ? (
          <div className="modal-section">Loading…</div>
        ) : (
          <>
            <div className="modal-section">
              <div className="modal-label">Theorem</div>
              <div className="modal-code state">{proofDetail.theorem}</div>
            </div>

            <div className="modal-section">
              <div className="modal-label">
                Simplified tree
                {selectedProof && (
                  <span className="modal-sublabel">
                    {' '}(depth {selectedProof.simplified_tree_depth},
                    size {selectedProof.simplified_tree_size})
                  </span>
                )}
              </div>
              <TreeView node={proofDetail.simplified_tree} />
            </div>

            <div className="modal-section">
              <div className="modal-label">
                Full tree
                {selectedProof && (
                  <span className="modal-sublabel">
                    {' '}(depth {selectedProof.full_tree_depth},
                    size {selectedProof.full_tree_size})
                  </span>
                )}
              </div>
              <TreeView node={proofDetail.full_tree} />
            </div>

            <div className="modal-section">
              <div className="modal-label">
                Transitions ({proofDetail.transitions.length})
              </div>
              <div className="transitions-list">
                {proofDetail.transitions.length === 0 ? (
                  <div className="replay-empty">No transitions</div>
                ) : (
                  proofDetail.transitions.map(([ctx, tactic, value], i) => (
                    <div key={i} className="transition-item">
                      <div className="transition-state">{ctx}</div>
                      <div className="transition-tactic">
                        <span className="transition-value">{value.toFixed(2)}</span>
                        → {tactic}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </>
        )}
      </Modal>
    </div>
  );
}

function TreeView({ node }: { node: NodeDict | null }) {
  if (!node) {
    return <div className="replay-empty">Tree not available</div>;
  }
  return (
    <div className="tree-view">
      <TreeNode node={node} depth={0} />
    </div>
  );
}

function TreeNode({ node, depth }: { node: NodeDict; depth: number }) {
  const children = node.children ? Object.entries(node.children) : [];
  const hasChildren = children.length > 0;
  const [expanded, setExpanded] = useState(depth < 6);
  const [wrapped, setWrapped] = useState(false);

  const kind = node.to_play === 1 ? 'OR' : 'AND';
  const stateStr = node.state.length > 0 ? node.state.join(' │ ') : '∅';
  const solvedChildren = children.filter(([, c]) => c.is_solved);
  const solvedActionSet = new Set(solvedChildren.map(([a]) => a));

  const toggleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (hasChildren) setExpanded(!expanded);
  };

  return (
    <div className="tree-node">
      <div
        className={`tree-row ${wrapped ? 'wrapped' : ''}`}
        onClick={() => setWrapped(!wrapped)}
      >
        <span
          className="tree-toggle"
          onClick={toggleExpand}
          role="button"
        >
          {hasChildren ? (expanded ? '▾' : '▸') : ' '}
        </span>
        <span className={`tree-kind tree-kind-${kind.toLowerCase()}`}>{kind}</span>
        <span className="tree-depth">d{depth}</span>
        {node.is_solved && <span className="tree-solved">✓</span>}
        {node.value_target !== null && (
          <span className="tree-value">v={node.value_target.toFixed(2)}</span>
        )}
        {node.action !== null && (
          <span className="tree-action">{String(node.action)}</span>
        )}
        <span className="tree-state" title={stateStr}>
          {stateStr}
        </span>
      </div>
      {expanded && hasChildren && (
        <div className="tree-children">
          {children.map(([action, child]) => (
            <div
              key={child.id}
              className={`tree-branch ${solvedActionSet.has(action) ? 'solved' : ''}`}
            >
              <TreeNode node={child} depth={depth + 1} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
