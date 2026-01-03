"""
Terminal UI for monitoring the RL training loop.

Provides a beautiful, informative view of:
- Current phase (collecting, evaluating, training)
- Replay buffer size
- Collection stats (actors, samples, proofs, wait times, throughput)
- Training stats (loss)
- Evaluation results history

Uses ANSI scroll regions to keep the monitor fixed at the top of the terminal
while logs scroll independently at the bottom.
"""

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from statistics import median
from typing import Literal, TextIO

# -----------------------------------------------------------------------------
# Terminal control with scroll regions
# -----------------------------------------------------------------------------

# ANSI escape codes
SAVE_CURSOR = "\033[s"
RESTORE_CURSOR = "\033[u"
CURSOR_HOME = "\033[H"
CLEAR_LINE = "\033[2K"
CLEAR_SCREEN = "\033[2J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def set_scroll_region(top: int, bottom: int) -> str:
    """Set the scroll region (1-indexed, inclusive)."""
    return f"\033[{top};{bottom}r"


def move_cursor(row: int, col: int = 1) -> str:
    """Move cursor to position (1-indexed)."""
    return f"\033[{row};{col}H"


def reset_scroll_region() -> str:
    """Reset scroll region to full terminal."""
    return "\033[r"


# Global state for scroll region management
_terminal_height = 24
_monitor_lines = 0  # Number of lines reserved for the monitor
_scroll_region_active = False
_terminal_lock = threading.Lock()


def _get_terminal_height() -> int:
    """Get the terminal height, with a fallback."""
    try:
        return os.get_terminal_size().lines
    except OSError:
        return 24


def _setup_scroll_region(monitor_height: int):
    """Set up a scroll region, reserving the top for the monitor."""
    global _terminal_height, _monitor_lines, _scroll_region_active
    
    _terminal_height = _get_terminal_height()
    _monitor_lines = monitor_height
    
    # Set scroll region from below monitor to bottom of terminal
    scroll_top = monitor_height + 1
    scroll_bottom = _terminal_height
    
    if scroll_top < scroll_bottom:
        # Save cursor, set scroll region, move cursor to scroll region, restore
        sys.stdout.write(
            set_scroll_region(scroll_top, scroll_bottom) +
            move_cursor(scroll_bottom)  # Position cursor at bottom of scroll region
        )
        sys.stdout.flush()
        _scroll_region_active = True


def _teardown_scroll_region():
    """Reset the scroll region to full terminal."""
    global _scroll_region_active
    if _scroll_region_active:
        sys.stdout.write(reset_scroll_region() + SHOW_CURSOR)
        sys.stdout.flush()
        _scroll_region_active = False


# -----------------------------------------------------------------------------
# Logging utilities
# -----------------------------------------------------------------------------

_log_lock = threading.Lock()
_log_file: TextIO | None = None
_tactics_file: TextIO | None = None


def configure_logging(output_dir: str | None):
    """
    Configure logging to write to files in the output directory.
    
    Args:
        output_dir: Directory where log files will be written.
                   If None, logging goes to console (for debugging).
    """
    global _log_file, _tactics_file
    
    with _log_lock:
        # Close any existing files
        if _log_file is not None:
            _log_file.close()
            _log_file = None
        if _tactics_file is not None:
            _tactics_file.close()
            _tactics_file = None
        
        if output_dir is not None:
            _log_file = open(os.path.join(output_dir, "logs.txt"), "a")
            _tactics_file = open(os.path.join(output_dir, "tactics.txt"), "a")


def log(msg: str, component: str | None = None, actor_id: int | None = None):
    """
    Thread-safe logging with optional component/actor prefix.
    Writes to log file if configured, otherwise to console (in the scroll region).
    
    Args:
        msg: The message to log
        component: Component name (e.g., "BatchedTacticModel", "Collection")
        actor_id: Actor thread ID if applicable
    """
    if actor_id is not None:
        prefix = f"[Actor {actor_id}]"
    elif component is not None:
        prefix = f"[{component}]"
    else:
        prefix = ""

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    with _log_lock:
        line = f"[{timestamp}] {prefix} {msg}" if prefix else f"[{timestamp}] {msg}"
        if _log_file is not None:
            _log_file.write(line + "\n")
            _log_file.flush()
        else:
            # When writing to console, we need to be in the scroll region
            with _terminal_lock:
                if _scroll_region_active:
                    # We're in scroll region mode - just print (cursor should be in scroll region)
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                else:
                    print(line, flush=True)


def log_error(msg: str, exception: Exception | None = None, component: str | None = None, actor_id: int | None = None):
    """
    Log an error with optional exception details.
    """
    if exception is not None:
        error_detail = f"{type(exception).__name__}: {exception}"
        full_msg = f"ERROR: {msg} - {error_detail}"
    else:
        full_msg = f"ERROR: {msg}"

    log(full_msg, component=component, actor_id=actor_id)

    if exception is not None:
        with _log_lock:
            if _log_file is not None:
                traceback.print_exc(file=_log_file)
                _log_file.flush()
            else:
                traceback.print_exc()


def log_tactic(state: str, tactic: str, success: bool):
    """
    Log a generated tactic to the tactics file.
    
    Args:
        state: The proof state that was used to generate the tactic
        tactic: The generated tactic
        success: Whether the tactic was successfully applied
    """
    with _log_lock:
        if _tactics_file is not None:
            status = "✓" if success else "✗"
            # Write state on one line (replace newlines with \\n for readability)
            state_oneline = state.replace("\n", "\\n")
            _tactics_file.write(f"{status}\t{state_oneline}\t{tactic}\n")
            _tactics_file.flush()


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def colorize(text: str, *codes: str) -> str:
    """Apply ANSI color codes to text."""
    return "".join(codes) + str(text) + Colors.RESET


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_rate(count: float, duration: float, unit: str = "") -> str:
    """Format a rate (count per second)."""
    if duration <= 0:
        return "—"
    rate = count / duration
    if rate >= 1000:
        return f"{rate / 1000:.1f}k{unit}/s"
    elif rate >= 1:
        return f"{rate:.1f}{unit}/s"
    else:
        return f"{rate:.3f}{unit}/s"


@dataclass
class CollectionStats:
    """Statistics for the current collection phase."""
    num_actors: int = 0
    samples_collected: int = 0
    target_samples: int = 0
    proofs_attempted: int = 0
    proofs_successful: int = 0
    expansions: int = 0
    start_time: float = 0.0

    # Thread wait times at BatchedTacticModel (in seconds)
    wait_times: list[float] = field(default_factory=list)
    _wait_times_lock: threading.Lock = field(default_factory=threading.Lock)

    def record_wait_time(self, wait_time: float):
        with self._wait_times_lock:
            self.wait_times.append(wait_time)

    def get_wait_time_stats(self) -> tuple[float, float, float]:
        """Returns (min, max, median) wait times, or (0, 0, 0) if no data."""
        with self._wait_times_lock:
            if not self.wait_times:
                return (0.0, 0.0, 0.0)
            return (min(self.wait_times), max(self.wait_times), median(self.wait_times))

    def reset(self):
        self.samples_collected = 0
        self.target_samples = 0
        self.proofs_attempted = 0
        self.proofs_successful = 0
        self.expansions = 0
        self.start_time = time.time()
        with self._wait_times_lock:
            self.wait_times = []

    @property
    def success_rate(self) -> float:
        if self.proofs_attempted == 0:
            return 0.0
        return self.proofs_successful / self.proofs_attempted

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0


@dataclass
class EvalResult:
    """Result from an evaluation run."""
    step: int
    dataset: str
    success_rate: float
    solved: int
    total: int
    errors: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingStats:
    """Statistics for the current training step."""
    step: int = 0
    loss: float = 0.0
    num_tokens: int = 0
    learning_rate: float = 0.0


Phase = Literal["idle", "collecting", "evaluating", "training"]


class RLMonitor:
    """
    Monitor for the RL training loop.
    
    Tracks and displays metrics for collection, training, and evaluation phases.
    Uses ANSI scroll regions to keep the monitor fixed at the top of the terminal
    while logs scroll independently at the bottom.
    Thread-safe for use with parallel actors.
    """

    def __init__(self, num_actors: int = 0, enabled: bool = True, refresh_interval: float = 0.5):
        self.enabled = enabled
        self._lock = threading.Lock()

        # Current state
        self.phase: Phase = "idle"
        self.step: int = 0
        self.replay_buffer_size: int = 0

        # Collection stats
        self.collection = CollectionStats(num_actors=num_actors)

        # Training stats
        self.training = TrainingStats()

        # Evaluation history
        self.eval_history: deque[EvalResult] = deque(maxlen=20)

        # Display settings
        self.terminal_width = 80
        self._monitor_height = 0  # Height of the monitor display
        self._scroll_region_setup = False
        try:
            self.terminal_width = os.get_terminal_size().columns
        except OSError:
            pass
        
        # Background refresh thread
        self._refresh_interval = refresh_interval
        self._display_thread: threading.Thread | None = None
        self._stop_display = threading.Event()
        self._display_lock = threading.Lock()  # Prevents concurrent display updates

    def set_phase(self, phase: Phase):
        with self._lock:
            self.phase = phase
            if phase == "collecting":
                self.collection.reset()
        
        # Stop background display when leaving collecting phase
        if phase != "collecting":
            self._stop_background_display()

    def set_step(self, step: int):
        with self._lock:
            self.step = step

    def set_replay_buffer_size(self, size: int):
        with self._lock:
            self.replay_buffer_size = size

    def _start_background_display(self):
        """Start a background thread that periodically updates the display."""
        if not self.enabled:
            return
        self._stop_display.clear()
        
        def display_loop():
            while not self._stop_display.is_set():
                self.display()
                self._stop_display.wait(self._refresh_interval)
        
        self._display_thread = threading.Thread(target=display_loop, daemon=True)
        self._display_thread.start()
    
    def _stop_background_display(self):
        """Stop the background display thread."""
        if self._display_thread is not None:
            self._stop_display.set()
            self._display_thread.join(timeout=1.0)
            self._display_thread = None

    # Collection phase methods
    def start_collection(self, target_samples: int, num_actors: int):
        with self._lock:
            self.phase = "collecting"
            self.collection.reset()
            self.collection.target_samples = target_samples
            self.collection.num_actors = num_actors
        
        # Start background display updates during collection
        self._start_background_display()

    def record_proof_attempt(self, successful: bool, transitions: int = 0):
        with self._lock:
            self.collection.proofs_attempted += 1
            if successful:
                self.collection.proofs_successful += 1
                self.collection.samples_collected += transitions

    def record_expansion(self):
        with self._lock:
            self.collection.expansions += 1

    def record_batch_wait(self, wait_time: float):
        self.collection.record_wait_time(wait_time)

    # Training phase methods
    def update_training(self, step: int, loss: float, num_tokens: int = 0, lr: float = 0.0):
        with self._lock:
            self.phase = "training"
            self.training.step = step
            self.training.loss = loss
            self.training.num_tokens = num_tokens
            self.training.learning_rate = lr

    # Evaluation phase methods
    def record_eval(self, step: int, dataset: str, success_rate: float, solved: int, total: int, errors: int):
        with self._lock:
            self.eval_history.append(EvalResult(
                step=step,
                dataset=dataset,
                success_rate=success_rate,
                solved=solved,
                total=total,
                errors=errors,
            ))

    def _setup_terminal(self, num_lines: int):
        """Set up the terminal with a scroll region, reserving top lines for monitor."""
        global _scroll_region_active
        
        with _terminal_lock:
            terminal_height = _get_terminal_height()
            self._monitor_height = num_lines
            
            # Hide cursor and clear screen
            sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN + CURSOR_HOME)
            
            # Print blank lines for the monitor region
            sys.stdout.write("\n" * num_lines)
            
            # Set scroll region from below monitor to bottom
            scroll_top = num_lines + 1
            scroll_bottom = terminal_height
            
            if scroll_top < scroll_bottom:
                sys.stdout.write(set_scroll_region(scroll_top, scroll_bottom))
                sys.stdout.write(move_cursor(scroll_bottom))  # Position in scroll region
                _scroll_region_active = True
            
            sys.stdout.flush()
            self._scroll_region_setup = True

    def display(self):
        """Update the monitor display in the fixed top region."""
        if not self.enabled:
            return

        with self._display_lock:
            with self._lock:
                lines = self._build_display()
            
            num_lines = len(lines)
            
            with _terminal_lock:
                # First time setup - establish the scroll region
                if not self._scroll_region_setup:
                    self._setup_terminal(num_lines)
                
                # Save cursor position (we're in the scroll region)
                sys.stdout.write(SAVE_CURSOR)
                
                # Move to top of screen and draw the monitor
                sys.stdout.write(CURSOR_HOME)
                for i, line in enumerate(lines):
                    # Move to line i+1 (1-indexed), clear it, write content
                    sys.stdout.write(move_cursor(i + 1) + CLEAR_LINE + line)
                
                # If monitor shrank, clear any leftover lines
                if num_lines < self._monitor_height:
                    for i in range(num_lines, self._monitor_height):
                        sys.stdout.write(move_cursor(i + 1) + CLEAR_LINE)
                
                # Restore cursor to scroll region
                sys.stdout.write(RESTORE_CURSOR)
                sys.stdout.flush()
                
                self._monitor_height = num_lines

    def cleanup(self):
        """Clean up terminal state - reset scroll region and show cursor."""
        self._stop_background_display()
        with _terminal_lock:
            if self._scroll_region_setup:
                # Move cursor below monitor before resetting
                terminal_height = _get_terminal_height()
                sys.stdout.write(
                    reset_scroll_region() +
                    move_cursor(terminal_height) +
                    SHOW_CURSOR +
                    "\n"
                )
                sys.stdout.flush()
                self._scroll_region_setup = False

    def _build_display(self) -> list[str]:
        """Build the display lines (must be called with lock held)."""
        lines = []
        width = min(self.terminal_width, 100)

        # Header
        lines.append(self._make_header(width))
        lines.append("")

        # Phase indicator
        lines.append(self._make_phase_indicator())
        lines.append("")

        # Main stats based on current phase
        if self.phase == "collecting":
            lines.extend(self._make_collection_display())
        elif self.phase == "training":
            lines.extend(self._make_training_display())
        elif self.phase == "evaluating":
            lines.append(colorize("  Evaluating model...", Colors.YELLOW))

        lines.append("")

        # Replay buffer
        lines.append(self._make_buffer_display())

        # Eval history (if any)
        if self.eval_history:
            lines.append("")
            lines.extend(self._make_eval_history())

        lines.append("")
        lines.append(colorize("─" * width, Colors.DIM))

        return lines

    def _make_header(self, width: int) -> str:
        title = " NANOPROOF RL "
        padding = (width - len(title)) // 2
        return (
                colorize("─" * padding, Colors.CYAN) +
                colorize(title, Colors.BOLD, Colors.BRIGHT_CYAN) +
                colorize("─" * (width - padding - len(title)), Colors.CYAN)
        )

    def _make_phase_indicator(self) -> str:
        phase_colors = {
            "idle": (Colors.DIM, "⏸ IDLE"),
            "collecting": (Colors.BRIGHT_GREEN, "🔄 COLLECTING"),
            "evaluating": (Colors.BRIGHT_YELLOW, "📊 EVALUATING"),
            "training": (Colors.BRIGHT_BLUE, "🧠 TRAINING"),
        }
        color, label = phase_colors.get(self.phase, (Colors.WHITE, self.phase.upper()))
        step_str = colorize(f"Step {self.step}", Colors.DIM)
        return f"  {colorize(label, Colors.BOLD, color)}  {step_str}"

    def _make_collection_display(self) -> list[str]:
        lines = []
        c = self.collection

        # Progress bar
        progress = c.samples_collected / c.target_samples if c.target_samples > 0 else 0
        progress = min(1.0, progress)
        bar_width = 40
        filled = int(bar_width * progress)
        bar = colorize("█" * filled, Colors.GREEN) + colorize("░" * (bar_width - filled), Colors.DIM)
        lines.append(f"  Progress: [{bar}] {progress * 100:.1f}%")
        lines.append(f"  Samples:  {colorize(c.samples_collected, Colors.BRIGHT_WHITE)}/{c.target_samples}")
        lines.append("")

        # Actor stats
        lines.append(colorize("  Actors & Throughput", Colors.BOLD))
        elapsed = c.elapsed
        lines.append(f"    Actors running:     {colorize(c.num_actors, Colors.BRIGHT_CYAN)}")
        lines.append(f"    Proofs attempted:   {c.proofs_attempted}")
        lines.append(
            f"    Proofs successful:  {colorize(c.proofs_successful, Colors.GREEN)} ({c.success_rate * 100:.1f}%)")
        lines.append(f"    Expansions:         {c.expansions}  ({format_rate(c.expansions, elapsed, 'exp')})")

        if c.proofs_successful > 0:
            lines.append(f"    Proofs/sec:         {format_rate(c.proofs_successful, elapsed, 'proof')}")

        # Wait time stats
        wait_min, wait_max, wait_med = c.get_wait_time_stats()
        if wait_med > 0:
            lines.append("")
            lines.append(colorize("  Batch Wait Times", Colors.BOLD))
            lines.append(f"    Min:    {format_duration(wait_min)}")
            lines.append(f"    Median: {format_duration(wait_med)}")
            lines.append(f"    Max:    {format_duration(wait_max)}")

        return lines

    def _make_training_display(self) -> list[str]:
        lines = []
        t = self.training

        lines.append(colorize("  Training Stats", Colors.BOLD))
        lines.append(f"    Loss:       {colorize(f'{t.loss:.6f}', Colors.BRIGHT_WHITE)}")
        if t.num_tokens > 0:
            lines.append(f"    Tokens:     {t.num_tokens:,}")
        if t.learning_rate > 0:
            lines.append(f"    LR:         {t.learning_rate:.2e}")

        return lines

    def _make_buffer_display(self) -> str:
        size = self.replay_buffer_size
        if size >= 1000:
            size_str = f"{size / 1000:.1f}k"
        else:
            size_str = str(size)
        return f"  Replay Buffer: {colorize(size_str, Colors.BRIGHT_MAGENTA)} transitions"

    def _make_eval_history(self) -> list[str]:
        lines = []
        lines.append(colorize("  Evaluation History", Colors.BOLD))

        # Group by dataset
        datasets = {}
        for result in self.eval_history:
            if result.dataset not in datasets:
                datasets[result.dataset] = []
            datasets[result.dataset].append(result)

        for dataset, results in datasets.items():
            # Show last few results
            recent = results[-5:]
            rates = [f"{r.success_rate * 100:.1f}%" for r in recent]
            trend = " → ".join(rates)

            # Color based on trend
            if len(recent) >= 2:
                if recent[-1].success_rate > recent[-2].success_rate:
                    trend_color = Colors.GREEN
                elif recent[-1].success_rate < recent[-2].success_rate:
                    trend_color = Colors.RED
                else:
                    trend_color = Colors.YELLOW
            else:
                trend_color = Colors.WHITE

            lines.append(f"    {dataset}: {colorize(trend, trend_color)}")

        return lines


# Global monitor instance (can be set by rl.py)
_monitor: RLMonitor | None = None
_cleanup_registered = False


def _cleanup_terminal():
    """Reset terminal state on exit."""
    global _monitor
    if _monitor is not None:
        _monitor.cleanup()
    _teardown_scroll_region()


def get_monitor() -> RLMonitor | None:
    """Get the global monitor instance."""
    return _monitor


def set_monitor(monitor: RLMonitor):
    """Set the global monitor instance."""
    global _monitor
    _monitor = monitor


def create_monitor(num_actors: int = 0, enabled: bool = True) -> RLMonitor:
    """Create and set a new global monitor."""
    global _cleanup_registered
    
    monitor = RLMonitor(num_actors=num_actors, enabled=enabled)
    set_monitor(monitor)
    
    # Register cleanup handler on first monitor creation
    if not _cleanup_registered and enabled:
        import atexit
        import signal
        
        atexit.register(_cleanup_terminal)
        
        # Also handle SIGINT gracefully
        original_handler = signal.getsignal(signal.SIGINT)
        def sigint_handler(signum, frame):
            _cleanup_terminal()
            if callable(original_handler):
                original_handler(signum, frame)
            else:
                raise KeyboardInterrupt
        signal.signal(signal.SIGINT, sigint_handler)
        
        _cleanup_registered = True
    
    return monitor
