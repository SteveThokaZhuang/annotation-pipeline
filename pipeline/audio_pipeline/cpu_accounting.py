"""CPU time in live helpers, including persistent frontend/engine processes."""
import os
from pathlib import Path


def live_child_seconds(pid=None):
    ticks=os.sysconf('SC_CLK_TCK');pending=[pid or os.getpid()];seen=set();total=0.
    while pending:
        parent=pending.pop()
        try:
            children=(Path('/proc')/str(parent)/'task'/str(parent)/'children').read_text().split()
        except OSError:continue
        for value in children:
            if value in seen:continue
            seen.add(value);pending.append(int(value))
            try:
                # The command field may contain spaces or parentheses.
                fields=(Path('/proc')/value/'stat').read_text().rsplit(')',1)[1].split()
                total+=(int(fields[11])+int(fields[12]))/ticks
            except (OSError,ValueError,IndexError):continue
    return total
