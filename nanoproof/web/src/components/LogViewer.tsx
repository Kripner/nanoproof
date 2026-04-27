import { useEffect, useRef, useState, useCallback } from 'react';

interface LogViewerProps {
  stdoutLines: string[];
  stderrLines: string[];
}

type Stream = 'stderr' | 'stdout';

export function LogViewer({ stdoutLines, stderrLines }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [stream, setStream] = useState<Stream>('stderr');
  const [isAtBottom, setIsAtBottom] = useState(true);

  const handleScroll = useCallback(() => {
    if (containerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
      setIsAtBottom(scrollHeight - scrollTop - clientHeight < 50);
    }
  }, []);

  const lines = stream === 'stderr' ? stderrLines : stdoutLines;

  useEffect(() => {
    if (containerRef.current && isAtBottom) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [lines, isAtBottom]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
      setIsAtBottom(true);
    }
  }, [stream]);

  return (
    <div className="card logs-panel">
      <div className="logs-header">
        <div className="card-title" style={{ marginBottom: 0 }}>Logs</div>
        <div className="logs-buttons">
          <button
            className={stream === 'stderr' ? 'active' : ''}
            onClick={() => setStream('stderr')}
          >
            stderr
          </button>
          <button
            className={stream === 'stdout' ? 'active' : ''}
            onClick={() => setStream('stdout')}
          >
            stdout
          </button>
        </div>
      </div>
      <div className="logs-container" ref={containerRef} onScroll={handleScroll}>
        {lines.map((line, i) => (
          <div key={i} className="log-entry">
            <span className="log-message">{line}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
