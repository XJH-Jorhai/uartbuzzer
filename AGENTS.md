# Workspace collaboration

Prefer dedicated APIs, SDKs and official CLIs. Use local file/command tools for
code, builds and Git; UI automation is for actual interface verification.

## Active ownership (coordinated 2026-10-29)

- Stable desktop release / Git / publication: primary maintenance chat.
- `优化 UART 8N1 音频编码`: Beam/MPC integration; owns
  `research/beam_integration/`, `src/uart_buzzer/beam*.py`, `_beam_kernels.py`,
  Beam tests and the coordinated minimal desktop app/cache/devices/engine edits.
- `研究负载感知 UART 编码器`: owns `research/generative_uart/` and
  `results/generative_uart/<run>/`; previous `research/differentiable_uart/`
  is frozen reference evidence.

Keep current paths, virtual environments, locks and processes intact. Do not
move research into a worktree without its owner's explicit coordination. Do not
overwrite another owner's files. Coordinate shared-file changes first. Use a
distinct output directory per experiment; frozen inputs are read-only.

## Git discipline

The initial public `main` uses released 1.3.1 source, while this working directory
may already contain unfinished Beam changes. These differences are intentional.
Never use reset/checkout/restore/stash/clean to make this checkout appear clean.
Never use broad `git add .` for publication.

The publication coordinator owns root Git changes during initial upload. Later,
agree explicit file lists and commit boundaries before committing in this shared
checkout. Prefer coordinated worktrees for future independent work. Research
implementation/tests and GUI integration should be separate commits. Do not
change the stable encoder default to an unvalidated candidate.

## Public content

MIT covers project-original code. Preserve third-party notices and licenses.
Do not stage local audio, recordings, personal playlists, credentials, virtual
environments, converted payloads, raw results, papers, downloaded source archives
or research deliverable ZIPs. Accessibility is not redistribution permission.

Review an explicit allowlist and staged content before public pushes. Only
owner-approved, de-identified, independently authored research files may be added
later. Release binaries belong in GitHub Releases, not Git history.

Separate hardware playback from offline tests. Load models predict coil current,
not calibrated SPL. Test an isolated snapshot while shared source is being edited.
