import { GPU } from '../types';

interface GPUPanelProps {
  gpus: GPU[];
}

export function GPUPanel({ gpus }: GPUPanelProps) {
  if (gpus.length === 0) {
    return (
      <div className="card">
        <div className="card-title">GPUs</div>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
          No GPU data available
        </div>
      </div>
    );
  }

  return (
    <div className="card gpu-card">
      <div className="card-title">GPUs</div>
      {gpus.map((gpu) => {
        const memPercent = gpu.memory_total > 0 ? (gpu.memory_used / gpu.memory_total) * 100 : 0;
        const memClass = memPercent > 90 ? 'high' : memPercent > 70 ? 'medium' : 'low';
        
        return (
          <div key={gpu.id} className="gpu-item">
            <div className="gpu-header">
              <span className="gpu-name">GPU {gpu.id}: {gpu.name}</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>
                {gpu.utilization.toFixed(0)}%
              </span>
            </div>
            <div className="gpu-stats">
              <span>Mem: {(gpu.memory_used / 1024).toFixed(1)}/{(gpu.memory_total / 1024).toFixed(1)} GB</span>
              <span>Queue: {gpu.inference_queue_size}</span>
              <span>Wait: {gpu.avg_wait_time_ms.toFixed(1)}ms</span>
            </div>
            <div className="gpu-bar">
              <div 
                className={`gpu-bar-fill ${memClass}`}
                style={{ width: `${memPercent}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

