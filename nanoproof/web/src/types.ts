export interface ThreadStatus {
  id: number;
  state: 'idle' | 'running' | 'blocked' | 'error';
  games_played: number;
  games_solved: number;
}

export interface ProverServer {
  address: string;
  num_threads: number;
  threads: ThreadStatus[];
  connected: boolean;
  games_played: number;
  games_solved: number;
  transitions_collected: number;
}

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

export interface LogEntry {
  timestamp: string;
  component: string;
  message: string;
  level: string;
}

export interface MonitorState {
  phase: 'idle' | 'collecting' | 'evaluating' | 'training';
  step: number;
  replay_buffer_size: number;
  output_dir: string | null;
  collection: CollectionStats;
  training: TrainingStats;
  eval_history: EvalResult[];
  eval_progress: EvalProgress;
  prover_servers: Record<string, ProverServer>;
  local_actors: Record<string, LocalActor>;
  gpus: GPU[];
  lean_server: LeanServerStatus;
}

export interface ReplayBufferFile {
  name: string;
  size: number;
  step: number;
}

export interface Transition {
  0: string;  // state
  1: string;  // tactic
  2: number;  // value
}

export interface TacticEntry {
  success: boolean;
  state: string;
  tactic: string;
}

export interface LocalActor {
  id: number;
  state: 'idle' | 'running' | 'blocked' | 'error';
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
  cpu_percent: number[];
  ram_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  error: string;
}

