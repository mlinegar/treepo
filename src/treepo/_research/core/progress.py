"""
Rich Progress Reporting for Pipeline Operations.

Provides live progress bars with token throughput display for batch processing.

Usage:
    from treepo._research.core.progress import PipelineProgress, display_batch_summary

    with PipelineProgress() as progress:
        progress.start_phase("Processing documents", total=100)
        for doc in documents:
            # ... process doc ...
            progress.update("Processing documents", stats=batch_stats)

    display_batch_summary(batch_stats)
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable

# Handle optional rich import gracefully
try:
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TaskProgressColumn, TaskID
    )
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Progress = None
    Console = None
    Table = None
    TaskID = int


# =============================================================================
# Progress Bar Creation
# =============================================================================

def create_progress_bar(console: Optional["Console"] = None) -> Optional["Progress"]:
    """
    Create a rich Progress instance with token throughput display.

    Returns None if rich is not available.
    """
    if not RICH_AVAILABLE:
        return None

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]{task.fields[tokens_per_sec]:.0f} tok/s"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        console=console or Console(),
        transient=False,  # Keep progress after completion
    )


# =============================================================================
# Pipeline Progress Tracker
# =============================================================================

class PipelineProgress:
    """
    Track progress through pipeline phases with live stats.

    Provides per-phase progress bars with rolling token throughput display.

    Example:
        with PipelineProgress() as progress:
            task = progress.start_phase("Chunking", total=50)
            for i, chunk in enumerate(chunks):
                # process chunk
                progress.update("Chunking")
            progress.complete_phase("Chunking")
    """

    def __init__(self, console: Optional["Console"] = None, disable: bool = False):
        """
        Initialize progress tracker.

        Args:
            console: Optional rich Console instance
            disable: Set True to disable progress output
        """
        self.disable = disable or not RICH_AVAILABLE
        self._console = console or (Console() if RICH_AVAILABLE else None)
        self._progress: Optional["Progress"] = None
        self._phases: Dict[str, "TaskID"] = {}
        self._phase_totals: Dict[str, int] = {}

    def __enter__(self):
        """Start progress tracking context."""
        if not self.disable:
            self._progress = create_progress_bar(self._console)
            self._progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress tracking context."""
        if self._progress:
            self._progress.stop()
        return False

    @property
    def progress(self) -> Optional["Progress"]:
        """Get underlying Progress instance (for advanced use)."""
        return self._progress

    @property
    def console(self) -> Optional["Console"]:
        """Get console instance."""
        return self._console

    def start_phase(
        self,
        name: str,
        total: int,
        description: Optional[str] = None,
    ) -> Optional["TaskID"]:
        """
        Start a new progress phase.

        Args:
            name: Phase identifier (used for updates)
            total: Total items in this phase
            description: Optional display description (defaults to name)

        Returns:
            Task ID (or None if disabled)
        """
        if self.disable or not self._progress:
            return None

        display_name = description or name
        task_id = self._progress.add_task(
            display_name,
            total=total,
            tokens_per_sec=0.0,
        )
        self._phases[name] = task_id
        self._phase_totals[name] = total
        return task_id

    def update(
        self,
        name: str,
        advance: int = 1,
        stats: Optional[Any] = None,
        tokens_per_sec: Optional[float] = None,
    ):
        """
        Update phase progress.

        Args:
            name: Phase identifier
            advance: Number of items completed (default 1)
            stats: Optional BatchStats for token throughput
            tokens_per_sec: Optional direct tokens/sec value
        """
        if self.disable or not self._progress:
            return

        task_id = self._phases.get(name)
        if task_id is None:
            return

        # Get tokens per second from stats or direct value
        tps = 0.0
        if stats is not None:
            tps = getattr(stats, 'tokens_per_second', 0.0)
        elif tokens_per_sec is not None:
            tps = tokens_per_sec

        self._progress.update(task_id, advance=advance, tokens_per_sec=tps)

    def complete_phase(self, name: str):
        """
        Mark phase as complete.

        Args:
            name: Phase identifier
        """
        if self.disable or not self._progress:
            return

        task_id = self._phases.get(name)
        total = self._phase_totals.get(name, 0)
        if task_id is not None:
            self._progress.update(task_id, completed=total)

    def set_total(self, name: str, total: int):
        """
        Update the total for a phase (useful when total is discovered later).

        Args:
            name: Phase identifier
            total: New total
        """
        if self.disable or not self._progress:
            return

        task_id = self._phases.get(name)
        if task_id is not None:
            self._progress.update(task_id, total=total)
            self._phase_totals[name] = total

    def print(self, message: str, style: str = ""):
        """
        Print a message without disrupting progress bars.

        Args:
            message: Message to print
            style: Optional rich style
        """
        if self._console:
            self._console.print(message, style=style)
        else:
            print(message)


# =============================================================================
# Batch Summary Display
# =============================================================================

def display_batch_summary(
    stats: Any,
    title: str = "Batch Processing Summary",
    console: Optional["Console"] = None,
):
    """
    Display a formatted summary table of batch processing stats.

    Args:
        stats: BatchStats object with token throughput info
        title: Table title
        console: Optional Console instance
    """
    if not RICH_AVAILABLE:
        # Fallback to plain text
        print(f"\n=== {title} ===")
        print(f"  Total Tokens:     {getattr(stats, 'total_tokens', 0):,}")
        print(f"  Prompt Tokens:    {getattr(stats, 'prompt_tokens', 0):,}")
        print(f"  Completion Tokens:{getattr(stats, 'completion_tokens', 0):,}")
        print(f"  Wall Clock Time:  {getattr(stats, 'wall_clock_seconds', 0):.1f}s")
        print(f"  Overall tok/s:    {getattr(stats, 'tokens_per_second', 0):.0f}")
        print(f"  Read tok/s:       {getattr(stats, 'read_tokens_per_second', 0):.0f}")
        print(f"  Write tok/s:      {getattr(stats, 'write_tokens_per_second', 0):.0f}")
        print()
        return

    console = console or Console()

    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", justify="right", width=15)

    # Token counts
    table.add_row("Total Tokens", f"{getattr(stats, 'total_tokens', 0):,}")
    table.add_row("Prompt Tokens", f"{getattr(stats, 'prompt_tokens', 0):,}")
    table.add_row("Completion Tokens", f"{getattr(stats, 'completion_tokens', 0):,}")
    table.add_row("", "")  # Spacer

    # Timing
    wall_time = getattr(stats, 'wall_clock_seconds', 0)
    table.add_row("Wall Clock Time", f"{wall_time:.1f}s")

    # Throughput
    table.add_row("", "")  # Spacer
    table.add_row("Overall tok/s", f"{getattr(stats, 'tokens_per_second', 0):.0f}")
    table.add_row("Read tok/s", f"{getattr(stats, 'read_tokens_per_second', 0):.0f}")
    table.add_row("Write tok/s", f"{getattr(stats, 'write_tokens_per_second', 0):.0f}")

    # Request stats
    completed = getattr(stats, 'completed_requests', 0)
    failed = getattr(stats, 'failed_requests', 0)
    total = getattr(stats, 'total_requests', 0)
    if total > 0:
        table.add_row("", "")  # Spacer
        table.add_row("Requests", f"{completed}/{total}")
        if failed > 0:
            table.add_row("Failed", f"[red]{failed}[/red]")
        avg_latency = getattr(stats, 'avg_latency_ms', 0)
        table.add_row("Avg Latency", f"{avg_latency:.0f}ms")

    console.print()
    console.print(table)
    console.print()


def display_phase_summary(
    phase_name: str,
    items_processed: int,
    elapsed_seconds: float,
    stats: Optional[Any] = None,
    console: Optional["Console"] = None,
):
    """
    Display a brief summary for a completed phase.

    Args:
        phase_name: Name of the phase
        items_processed: Number of items processed
        elapsed_seconds: Time taken
        stats: Optional BatchStats
        console: Optional Console instance
    """
    if not RICH_AVAILABLE:
        rate = items_processed / max(elapsed_seconds, 0.001)
        print(f"  {phase_name}: {items_processed} items in {elapsed_seconds:.1f}s ({rate:.1f}/s)")
        return

    console = console or Console()

    rate = items_processed / max(elapsed_seconds, 0.001)
    tokens_info = ""
    if stats:
        tps = getattr(stats, 'tokens_per_second', 0)
        if tps > 0:
            tokens_info = f" | [green]{tps:.0f} tok/s[/green]"

    console.print(
        f"  [bold]{phase_name}[/bold]: "
        f"{items_processed} items in {elapsed_seconds:.1f}s "
        f"([cyan]{rate:.1f}/s[/cyan]){tokens_info}"
    )


# =============================================================================
# Context Manager for Simple Progress
# =============================================================================

@contextmanager
def simple_progress(
    description: str,
    total: int,
    disable: bool = False,
):
    """
    Simple context manager for single-phase progress.

    Example:
        with simple_progress("Processing", total=100) as update:
            for item in items:
                # process item
                update()

    Args:
        description: Progress bar description
        total: Total items
        disable: Disable progress output

    Yields:
        Update function that advances the progress bar
    """
    if disable or not RICH_AVAILABLE:
        # Return no-op update function
        def noop(*args, **kwargs):
            pass
        yield noop
        return

    progress = create_progress_bar()
    with progress:
        task_id = progress.add_task(description, total=total, tokens_per_sec=0.0)

        def update(advance: int = 1, tokens_per_sec: float = 0.0):
            progress.update(task_id, advance=advance, tokens_per_sec=tokens_per_sec)

        yield update


# =============================================================================
# Callback Factory for Existing Code
# =============================================================================

def create_progress_callback(
    progress: PipelineProgress,
    phase_name: str,
) -> Callable[[int, int], None]:
    """
    Create a progress callback function for existing APIs.

    Many existing functions expect a callback(completed, total) signature.
    This creates such a callback that updates the PipelineProgress.

    Args:
        progress: PipelineProgress instance
        phase_name: Name of the phase to update

    Returns:
        Callback function(completed, total)
    """
    _last_completed = [0]  # Use list to allow modification in closure

    def callback(completed: int, total: int):
        advance = completed - _last_completed[0]
        if advance > 0:
            progress.update(phase_name, advance=advance)
        _last_completed[0] = completed

    return callback
