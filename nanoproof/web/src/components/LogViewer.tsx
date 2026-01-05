import { useEffect, useRef } from 'react';
import { LogEntry } from '../types';

interface LogViewerProps {
  logs: LogEntry[];
  onOpenStream: (component: string) => void;
}

export function LogViewer({ logs, onOpenStream }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  // Get unique components for buttons
  const components = new Set<string>();
  for (const log of logs) {
    if (log.component && log.component !== '_merged') {
      components.add(log.component);
    }
  }

  return (
    <div className="card logs-panel">
      <div className="logs-header">
        <div className="card-title" style={{ marginBottom: 0 }}>Logs</div>
        <div className="logs-buttons">
          <button onClick={() => onOpenStream('_merged')}>All Logs</button>
          {Array.from(components).slice(0, 5).map(c => (
            <button key={c} onClick={() => onOpenStream(c)}>{c}</button>
          ))}
        </div>
      </div>
      <div className="logs-container" ref={containerRef}>
        {logs.map((log, i) => (
          <div key={i} className={`log-entry ${log.level === 'error' ? 'error' : ''}`}>
            <span className="log-time">{log.timestamp}</span>
            <span className="log-component">[{log.component}]</span>
            <span className="log-message">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

