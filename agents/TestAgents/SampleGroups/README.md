# Sample group folders (tests only)

These folders were in `agents/` as `Group1`, `Group2` and `Group997`–`Group999`.
They are here so that their names cannot collide with a real group's folder,
and so that a tournament script that looks for `agents/Group*/cmd.txt` does not
pick them up.

- `Group1/CheatAgent.py`: tries to cheat (tests the engine's checks).
- `Group2/CrashAgent.py`: raises an exception (tests the `CRASH` end state).
- `Group997`–`Group999`: copies of `NaiveAgent` (sample tournament entries).

Each `cmd.txt` now gives the full module path, for example:

    python3 Hex.py -p1 "agents.TestAgents.SampleGroups.Group2.CrashAgent CrashAgent"
