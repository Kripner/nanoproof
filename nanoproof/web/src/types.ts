export interface GPU {
  id: number;
  name: string;
  utilization: number;
  memory_used: number;
  memory_total: number;
  inference_queue_size: number;
  avg_wait_time_ms: number;
}

export interface CollectionStats {
  num_actors: number;
  samples_collected: number;
  target_samples: number;
  proofs_attempted: number;
  proofs_successful: number;
  success_rate: number;
  expansions: number;
  elapsed: number;
  total_elapsed: number;
  wait_time_min: number;
  wait_time_max: number;
  wait_time_median: number;
}

export interface TrainingStats {
  step: number;
  loss: number;
  num_tokens: number;
  learning_rate: number;
}

export interface EvalResult {
  step: number;
  dataset: string;
  success_rate: number;
  solved: number;
  total: number;
  errors: number;
  timestamp: number;
}

export interface EvalProgress {
  dataset: string;
  current: number;
  total: number;
  solved: number;
  errors: number;
  active: boolean;
  progress_percent: number;
}

export interface MonitorState {
  mode: 'live' | 'standalone';
  phase: 'idle' | 'collecting' | 'evaluating' | 'training';
  step: number;
  replay_buffer_size: number;
  output_dir: string | null;
  collection: CollectionStats;
  training: TrainingStats;
  eval_history: EvalResult[];
  eval_progress: EvalProgress;
  local_actors: Record<string, LocalActor>;
  gpus: GPU[];
  lean_server: LeanServerStatus;
  lean_servers: LeanServerStatus[];
}

export interface TacticAttempt {
  tactic: string;
  status: 'success' | 'error' | 'cycle';
  count: number;
}

export interface TacticEntry {
  state: string;
  tactics: TacticAttempt[];
}

export interface LocalActor {
  id: number;
  state: 'idle' | 'running' | 'blocked' | 'retry' | 'error';
  games_played: number;
  games_solved: number;
  current_theorem: string;
}

export interface LeanServerStatus {
  address: string;
  port: number;
  connected: boolean;
  available_processes: number;
  used_processes: number;
  max_processes: number;
  starting_processes: number;
  stopping_processes: number;
  total_processes: number;
  idle_too_long_60s: number;
  cpu_percent: number[];
  ram_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  error: string;
}

// Timeline instrumentation types

export interface TimelineEvent {
  type: "llm" | "lean";
  start: number;
  end: number;
}

export interface PhaseEvent {
  type: "phase";
  name: string;
  action: "start" | "end";
  time: number;
}

export interface ActorTimeline {
  events: TimelineEvent[];
}

export interface InstrumentationData {
  actors: Record<string, ActorTimeline>;
  phases: PhaseEvent[];
  mode?: "live" | "standalone";
}

export type Outcome = "proven" | "unproven" | "error";

export interface TheoremAttempt {
  step: number;
  outcome: Outcome;
  error: string | null;
  num_simulations: number;
  num_iterations: number;
  num_transitions: number;
  weight_after: number;
}

export interface TheoremHistory {
  dataset: string;
  id: string;
  history: TheoremAttempt[];
  current_weight: number;
}

