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
  collection: CollectionStats;
  training: TrainingStats;
  eval_history: EvalResult[];
  prover_servers: Record<string, ProverServer>;
  gpus: GPU[];
}

