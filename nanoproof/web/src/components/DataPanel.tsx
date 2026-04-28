import { useState, useEffect } from 'react';
import { Modal } from './Modal';
import { TacticEntry } from '../types';

const POLL_INTERVAL = 2000;
const PAGE_SIZE = 200;

type Outcome = 'proven' | 'unproven' | 'error';

interface AttemptSummary {
  dataset: string;
  id: string;
  theorem: string;
  outcome: Outcome;
  error: string | null;
  num_simulations: number;
  num_iterations: number;
  num_transitions: number;
  full_tree_depth: number;
  full_tree_size: number;
  simplified_tree_depth: number;
  simplified_tree_size: number;
}

interface StepEntry {
  step: number;
  num_attempts: number;
  num_proven: number;
  num_unproven: number;
  num_errors: number;
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

interface AttemptDetail {
  theorem: string;
  dataset: string;
  id: string;
  outcome: Outcome;
  error: string | null;
  num_simulations: number;
  num_iterations: number;
  full_tree: NodeDict | null;
  simplified_tree: NodeDict | null;
  transitions: [string, string, number][];
}

interface TrainSample {
  source: 'rl' | 'sft' | null;
  is_value: boolean;
  tokens: string[];
  losses: (number | null)[];
}

interface CollectedTransition {
  id: string | null;
  state: string;
  tactic: string;
  value: number;
}

const TRAIN_LOSS_CLAMP = 5.0;

export function DataPanel() {
  const [stepEntries, setStepEntries] = useState<StepEntry[]>([]);
  const [totalAttempts, setTotalAttempts] = useState(0);
  const [totalProven, setTotalProven] = useState(0);
  const [totalErrors, setTotalErrors] = useState(0);
  const [totalTransitions, setTotalTransitions] = useState(0);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [attempts, setAttempts] = useState<AttemptSummary[]>([]);
  const [tactics, setTactics] = useState<TacticEntry[]>([]);
  const [tacticsTotal, setTacticsTotal] = useState(0);
  const [trainSamples, setTrainSamples] = useState<TrainSample[]>([]);
  const [trainSamplesTotal, setTrainSamplesTotal] = useState(0);
  const [collectedTransitions, setCollectedTransitions] = useState<CollectedTransition[]>([]);
  const [collectedTransitionsTotal, setCollectedTransitionsTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedAttemptIndex, setSelectedAttemptIndex] = useState<number | null>(null);
  const [attemptDetail, setAttemptDetail] = useState<AttemptDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    const fetchSteps = async () => {
      try {
        const res = await fetch('/api/steps');
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        const entries: StepEntry[] = data.entries || [];
        setStepEntries(entries);
        setTotalAttempts(data.total_attempts ?? 0);
        setTotalProven(data.total_proven ?? 0);
        setTotalErrors(data.total_errors ?? 0);
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
      setAttempts([]);
      setTactics([]);
      setTacticsTotal(0);
      setTrainSamples([]);
      setTrainSamplesTotal(0);
      setCollectedTransitions([]);
      setCollectedTransitionsTotal(0);
      return;
    }
    let alive = true;
    setLoading(true);
    const load = async () => {
      try {
        const [attemptsRes, transitionsRes, tacticsRes, trainRes] = await Promise.all([
          fetch(`/api/steps/${selectedStep}/theorems`),
          fetch(`/api/steps/${selectedStep}/transitions?limit=${PAGE_SIZE}`),
          fetch(`/api/steps/${selectedStep}/generated_tactics?limit=${PAGE_SIZE}`),
          fetch(`/api/steps/${selectedStep}/train_data?limit=${PAGE_SIZE}`),
        ]);
        if (!alive) return;
        if (attemptsRes.ok) {
          const d = await attemptsRes.json();
          setAttempts(d.attempts || []);
        } else {
          setAttempts([]);
        }
        if (transitionsRes.ok) {
          const d = await transitionsRes.json();
          setCollectedTransitions(d.transitions || []);
          setCollectedTransitionsTotal(d.total ?? (d.transitions?.length || 0));
        } else {
          setCollectedTransitions([]);
          setCollectedTransitionsTotal(0);
        }
        if (tacticsRes.ok) {
          const d = await tacticsRes.json();
          setTactics(d.tactics || []);
          setTacticsTotal(d.total ?? (d.tactics?.length || 0));
        } else {
          setTactics([]);
          setTacticsTotal(0);
        }
        if (trainRes.ok) {
          const d = await trainRes.json();
          setTrainSamples(d.samples || []);
          setTrainSamplesTotal(d.total ?? (d.samples?.length || 0));
        } else {
          setTrainSamples([]);
          setTrainSamplesTotal(0);
        }
      } catch {
        if (!alive) return;
        setAttempts([]);
        setTactics([]);
        setTacticsTotal(0);
        setTrainSamples([]);
        setTrainSamplesTotal(0);
        setCollectedTransitions([]);
        setCollectedTransitionsTotal(0);
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
    if (selectedStep === null || selectedAttemptIndex === null) {
      setAttemptDetail(null);
      return;
    }
    let alive = true;
    setDetailLoading(true);
    const load = async () => {
      try {
        const res = await fetch(
          `/api/steps/${selectedStep}/theorems/${selectedAttemptIndex}`,
        );
        if (!alive) return;
        if (res.ok) {
          const d = await res.json();
          setAttemptDetail(d);
        } else {
          setAttemptDetail(null);
        }
      } catch {
        if (alive) setAttemptDetail(null);
      } finally {
        if (alive) setDetailLoading(false);
      }
    };
    load();
    return () => {
      alive = false;
    };
  }, [selectedStep, selectedAttemptIndex]);

  const sortedEntries = [...stepEntries].sort((a, b) => b.step - a.step);
  const selectedAttempt = selectedAttemptIndex !== null ? attempts[selectedAttemptIndex] : null;

  return (
    <div className="data-panel">
      <div className="data-sidebar">
        <div className="data-sidebar-title">
          <span>Steps</span>
          <span className="data-sidebar-totals">
            {totalProven}/{totalAttempts} proven · {totalErrors} err · {totalTransitions} trans.
          </span>
        </div>
        {sortedEntries.length === 0 ? (
          <div className="data-empty">No steps yet</div>
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
                  {e.num_proven}/{e.num_attempts} · {e.num_errors}e · {e.num_transitions}t
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="data-main">
        {selectedStep === null ? (
          <div className="data-empty">Select a step</div>
        ) : (
          <>
            <div className="data-section">
              <div className="data-section-title">
                Collected transitions
                <span className="data-section-count">
                  {loading ? '…' : `${collectedTransitions.length} / ${collectedTransitionsTotal}`}
                </span>
              </div>
              <div className="collected-transitions-list">
                {loading ? (
                  <div className="replay-loading">Loading...</div>
                ) : collectedTransitions.length === 0 ? (
                  <div className="replay-empty">No transitions at this step</div>
                ) : (
                  collectedTransitions.map((t, i) => (
                    <div key={i} className="transition-item">
                      <div className="transition-state">{t.state}</div>
                      <div className="transition-tactic">
                        <span className="transition-value">{t.value.toFixed(2)}</span>
                        → {t.tactic}
                        {t.id && (
                          <span className="transition-proof"> ({t.id})</span>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="data-section">
              <div className="data-section-title">
                Attempts
                <span className="data-section-count">
                  {loading ? '…' : `${attempts.length}`}
                </span>
              </div>
              <div className="replay-list">
                {loading ? (
                  <div className="replay-loading">Loading...</div>
                ) : attempts.length === 0 ? (
                  <div className="replay-empty">No attempts at this step</div>
                ) : (
                  attempts.map((a, i) => (
                    <div
                      key={i}
                      className="proof-item clickable"
                      onClick={() => setSelectedAttemptIndex(i)}
                    >
                      <span className="proof-name">
                        <span className={`outcome-badge outcome-${a.outcome}`}>{a.outcome}</span>
                        {' '}{a.dataset}/{a.id}
                      </span>
                      <span className="proof-meta">
                        {a.num_simulations} max sims
                        {a.outcome === 'proven' && (
                          <>
                            {' · '}{a.num_iterations} sims
                            {' · '}full d{a.full_tree_depth}/s{a.full_tree_size}
                            {' · '}simp d{a.simplified_tree_depth}/s{a.simplified_tree_size}
                          </>
                        )}
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
                  tactics.map((entry, i) => {
                    const totalSamples = entry.tactics.reduce(
                      (a, t) => a + t.count,
                      0,
                    );
                    const successSamples = entry.tactics
                      .filter((t) => t.status === 'success')
                      .reduce((a, t) => a + t.count, 0);
                    return (
                      <div key={i} className="tactic-group">
                        <div className="tactic-group-state" title={entry.state}>
                          <span className="tactic-group-count">
                            {successSamples}/{totalSamples}
                          </span>
                          {entry.state}
                        </div>
                        <div className="tactic-group-tactics">
                          {entry.tactics.map((t, j) => {
                            const statusClass =
                              t.status === 'success'
                                ? 'success'
                                : t.status === 'cycle'
                                ? 'cycle'
                                : 'failure';
                            const statusIcon =
                              t.status === 'success'
                                ? '✓'
                                : t.status === 'cycle'
                                ? '↻'
                                : '✗';
                            return (
                              <div key={j} className={`tactic-item ${statusClass}`}>
                                <span className="tactic-status">{statusIcon}</span>
                                {t.count > 1 && (
                                  <span className="tactic-dup-count">
                                    {t.count}x
                                  </span>
                                )}
                                <span className="tactic-text">{t.tactic}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            <div className="data-section">
              <div className="data-section-title">
                Training samples
                <span className="data-section-count">
                  {loading ? '…' : `${trainSamples.length} / ${trainSamplesTotal}`}
                </span>
              </div>
              <div className="train-samples-list">
                {loading ? (
                  <div className="replay-loading">Loading...</div>
                ) : trainSamples.length === 0 ? (
                  <div className="replay-empty">No train samples at this step</div>
                ) : (
                  trainSamples.map((s, i) => <TrainSampleRow key={i} sample={s} />)
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <Modal
        isOpen={selectedAttemptIndex !== null}
        onClose={() => setSelectedAttemptIndex(null)}
        title={
          selectedAttempt
            ? `${selectedAttempt.dataset}/${selectedAttempt.id}`
            : `Attempt #${selectedAttemptIndex ?? ''}`
        }
      >
        {detailLoading || !attemptDetail ? (
          <div className="modal-section">Loading…</div>
        ) : (
          <>
            <div className="modal-section">
              <div className="modal-label">
                Outcome
                <span className="modal-sublabel">
                  {' '}({attemptDetail.num_simulations} sims, {attemptDetail.num_iterations} iters)
                </span>
              </div>
              <div className="modal-code state">
                <span className={`outcome-badge outcome-${attemptDetail.outcome}`}>
                  {attemptDetail.outcome}
                </span>
                {attemptDetail.error && (
                  <>
                    {'\n'}
                    {attemptDetail.error}
                  </>
                )}
              </div>
            </div>

            <div className="modal-section">
              <div className="modal-label">Theorem</div>
              <div className="modal-code state">{attemptDetail.theorem}</div>
            </div>

            {attemptDetail.outcome === 'proven' && (
              <>
                <div className="modal-section">
                  <div className="modal-label">
                    Simplified tree
                    {selectedAttempt && (
                      <span className="modal-sublabel">
                        {' '}(depth {selectedAttempt.simplified_tree_depth},
                        size {selectedAttempt.simplified_tree_size})
                      </span>
                    )}
                  </div>
                  <TreeView node={attemptDetail.simplified_tree} />
                </div>

                <div className="modal-section">
                  <div className="modal-label">
                    Full tree
                    {selectedAttempt && (
                      <span className="modal-sublabel">
                        {' '}(depth {selectedAttempt.full_tree_depth},
                        size {selectedAttempt.full_tree_size})
                      </span>
                    )}
                  </div>
                  <TreeView node={attemptDetail.full_tree} />
                </div>

                <div className="modal-section">
                  <div className="modal-label">
                    Transitions ({attemptDetail.transitions.length})
                  </div>
                  <div className="transitions-list">
                    {attemptDetail.transitions.length === 0 ? (
                      <div className="replay-empty">No transitions</div>
                    ) : (
                      attemptDetail.transitions.map(([ctx, tactic, value], i) => (
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
          </>
        )}
      </Modal>
    </div>
  );
}

function TrainSampleRow({ sample }: { sample: TrainSample }) {
  const sourceClass =
    sample.source === 'sft' ? 'train-sample-kind-sft' : sample.source === 'rl' ? 'train-sample-kind-rl' : '';
  const sourceLabel = sample.source ?? '?';
  const renderedTokens = decodeBpeTokens(sample.tokens);
  return (
    <div className="train-sample">
      <span className={`train-sample-kind ${sourceClass}`}>
        {sourceLabel}
        {sample.is_value && ' · value'}
      </span>
      <div className="train-sample-tokens">
        {renderedTokens.map((text, i) => {
          const loss = sample.losses[i];
          const alpha = loss == null ? 0 : Math.min(loss / TRAIN_LOSS_CLAMP, 1);
          const style =
            alpha > 0 ? { background: `rgba(248, 81, 73, ${alpha.toFixed(3)})` } : undefined;
          const title = loss == null ? undefined : `loss=${loss.toFixed(3)}`;
          return (
            <span key={i} className="train-token" style={style} title={title}>
              {text}
            </span>
          );
        })}
      </div>
    </div>
  );
}

const BYTE_DECODER: Map<string, number> = (() => {
  // Inverse of HuggingFace tokenizers' byte_to_unicode: each printable char
  // in a BPE token maps back to the single source byte it stood for.
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const inBs = new Set(bs);
  const cs: number[] = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!inBs.has(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const m = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) m.set(String.fromCodePoint(cs[i]), bs[i]);
  return m;
})();

function decodeBpeTokens(tokens: string[]): string[] {
  // Decode HF byte-level BPE tokens back to UTF-8 strings. Multi-byte chars
  // sometimes straddle token boundaries (e.g. ↔ split into "âĨ" + "Ķ"), so we
  // run a single TextDecoder in stream mode across all tokens; the char ends
  // up attributed to the token whose final byte completes the sequence.
  const decoder = new TextDecoder('utf-8');
  const utf8 = new TextEncoder();
  const out: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const bytes: number[] = [];
    for (const ch of tok) {
      const b = BYTE_DECODER.get(ch);
      if (b !== undefined) {
        bytes.push(b);
      } else {
        // Special tokens (e.g. <|tactic|>, ↔ when added as a single special
        // token) bypass byte-level encoding; pass them through as raw UTF-8.
        for (const eb of utf8.encode(ch)) bytes.push(eb);
      }
    }
    const isLast = i === tokens.length - 1;
    out.push(decoder.decode(new Uint8Array(bytes), { stream: !isLast }));
  }
  return out;
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
