# Sample group folders (tests only)

These folders were in `agents/` as `Group1`, `Group2` and `Group997`–`Group999`.
They are here so that their names cannot collide with a real group's folder,
and so that a tournament script that looks for `agents/Group*/cmd.txt` does not
pick them up.

- `Group1/CheatAgent.py`: from turn 10, returns a subclass of `Move` that
  claims to be valid, for a tile that is not empty (the engine must reject
  it: `BAD_MOVE`).
- `Group2/CrashAgent.py`: returns the same kind of `Move` subclass for a
  tile outside the board (`BAD_MOVE`). Despite its name, it does not raise an
  exception; `test/test_Game.py` tests the `CRASH` end state.
- `Group997`–`Group999`: copies of `NaiveAgent` (sample tournament entries).

Each `cmd.txt` now gives the full module path, for example:

    python3 Hex.py -p1 "agents.TestAgents.SampleGroups.Group2.CrashAgent CrashAgent"
