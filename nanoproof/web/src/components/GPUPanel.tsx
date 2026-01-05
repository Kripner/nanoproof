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
        const utilClass = gpu.utilization > 90 ? 'high' : gpu.utilization > 70 ? 'medium' : 'low';
        
        return (
          <div key={gpu.id} className="gpu-item">
            <div className="gpu-header">
              <span className="gpu-name">GPU {gpu.id}: {gpu.name}</span>
            </div>
            <div className="gpu-metrics">
              <div className="gpu-metric">
                <div className="gpu-metric-header">
                  <span className="gpu-metric-label">Utilization</span>
                  <span className={`gpu-metric-value ${utilClass}`}>{gpu.utilization.toFixed(0)}%</span>
                </div>
                <div className="gpu-bar">
                  <div 
                    className={`gpu-bar-fill ${utilClass}`}
                    style={{ width: `${gpu.utilization}%` }}
                  />
                </div>
              </div>
              <div className="gpu-metric">
                <div className="gpu-metric-header">
                  <span className="gpu-metric-label">Memory</span>
                  <span className={`gpu-metric-value ${memClass}`}>
                    {(gpu.memory_used / 1024).toFixed(1)}/{(gpu.memory_total / 1024).toFixed(1)} GB
                  </span>
                </div>
                <div className="gpu-bar">
                  <div 
                    className={`gpu-bar-fill ${memClass}`}
                    style={{ width: `${memPercent}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="gpu-stats">
              <span>Queue: {gpu.inference_queue_size}</span>
              <span>Wait: {gpu.avg_wait_time_ms.toFixed(1)}ms</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
