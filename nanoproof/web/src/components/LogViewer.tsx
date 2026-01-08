import { useEffect, useRef, useState, useCallback } from 'react';
import { LogEntry } from '../types';

interface LogViewerProps {
  logs: LogEntry[];
  selectedActor?: string | null;
  onActorSelect?: (actor: string | null) => void;
}

export function LogViewer({ logs, selectedActor, onActorSelect }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [filter, setFilter] = useState<string | null>(null);
  const [actorFilter, setActorFilter] = useState<string | null>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  // Track whether user is scrolled to the bottom
  const handleScroll = useCallback(() => {
    if (containerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
      // Consider "at bottom" if within 50px of the bottom
      setIsAtBottom(scrollHeight - scrollTop - clientHeight < 50);
    }
  }, []);

  // Sync with external selectedActor prop
  useEffect(() => {
    if (selectedActor !== undefined) {
      if (selectedActor) {
        setActorFilter(selectedActor);
        setFilter(null);
      }
    }
  }, [selectedActor]);

  // Only auto-scroll if user is at the bottom
  useEffect(() => {
    if (containerRef.current && isAtBottom) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, isAtBottom]);

  // Scroll to bottom when filter changes
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
      setIsAtBottom(true);
    }
  }, [filter, actorFilter]);

  // Separate actors from other components
  const actors: string[] = [];
  const otherComponents: string[] = [];
  
  for (const log of logs) {
    if (log.component && log.component !== '_merged') {
      if (log.component.startsWith('actor_')) {
        if (!actors.includes(log.component)) {
          actors.push(log.component);
        }
      } else {
        if (!otherComponents.includes(log.component)) {
          otherComponents.push(log.component);
        }
      }
    }
  }
  
  // Sort actors numerically
  actors.sort((a, b) => {
    const numA = parseInt(a.replace('actor_', ''));
    const numB = parseInt(b.replace('actor_', ''));
    return numA - numB;
  });

  // Filter logs based on selected component or actor
  const filteredLogs = (() => {
    if (actorFilter) {
      return logs.filter(log => log.component === actorFilter);
    }
    if (filter === '_actors') {
      return logs.filter(log => log.component?.startsWith('actor_'));
    }
    if (filter) {
      return logs.filter(log => log.component === filter);
    }
    return logs;
  })();

  const handleActorSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (value === '') {
      setActorFilter(null);
      setFilter(null);
      onActorSelect?.(null);
    } else if (value === '_all_actors') {
      setActorFilter(null);
      setFilter('_actors');
      onActorSelect?.(null);
    } else {
      setActorFilter(value);
      setFilter(null);
      onActorSelect?.(value);
    }
  };

  const handleComponentClick = (component: string | null) => {
    setFilter(component);
    setActorFilter(null);
    onActorSelect?.(null);
  };

  const isActorSelected = actorFilter !== null || filter === '_actors';

  return (
    <div className="card logs-panel">
      <div className="logs-header">
        <div className="card-title" style={{ marginBottom: 0 }}>Logs</div>
        <div className="logs-buttons">
          <button 
            className={filter === null && actorFilter === null ? 'active' : ''} 
            onClick={() => handleComponentClick(null)}
          >
            All
          </button>
          {otherComponents.map(c => (
            <button 
              key={c} 
              className={filter === c ? 'active' : ''} 
              onClick={() => handleComponentClick(c)}
            >
              {c}
            </button>
          ))}
          {actors.length > 0 && (
            <select 
              className={`actor-select ${isActorSelected ? 'active' : ''}`}
              value={actorFilter || (filter === '_actors' ? '_all_actors' : '')}
              onChange={handleActorSelect}
            >
              <option value="">Actors ({actors.length})</option>
              <option value="_all_actors">All Actors</option>
              {actors.map(a => (
                <option key={a} value={a}>
                  {a.replace('actor_', 'Actor ')}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>
      <div className="logs-container" ref={containerRef} onScroll={handleScroll}>
        {filteredLogs.map((log, i) => (
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
