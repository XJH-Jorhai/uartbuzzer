# Workspace collaboration

Prefer dedicated APIs, SDKs and official CLIs. Use local file and command tools
for code, builds and Git; UI automation is for interface verification.

## Ownership and concurrent work

- The project coordinator owns release review, Git policy and shared interfaces.
- The application maintainer owns production source, product tests, build tools,
  runtime assets and desktop integration. Research findings enter production
  through this maintainer after review.
- Researchers write separate research directories and result runs. They do not
  overwrite production files or another researcher's files.
- Frozen measurements, model snapshots and experiment records remain read-only.

Use one writer per mutable file. Coordinate shared changes before writing. Keep
existing paths, environments, locks and running experiments intact. Do not move
research or remove another process's locks as workspace cleanup.

## Git and release discipline

Use explicit reviewed file lists. Never use reset, restore, stash or clean to
discard another contributor's work, or broad `git add .` for a public release.
An active local research checkout can intentionally contain unpublished work.
Prefer isolated feature branches or coordinated worktrees for independent work.

Publish only reviewed original code, portable tests, licensed dependencies and
de-identified documentation. Release binaries belong in GitHub Releases, not
Git history. Keep user music, captures, playlists, caches, converted payloads,
credentials, virtual environments, downloaded papers, model downloads and raw
research results private unless separately reviewed and authorized.

MIT covers project-original code and original fixtures, not third-party files.
Preserve upstream notices, including Numba/LLVM and Qt licensing requirements.
The load model predicts electrical current; it is not calibrated sound pressure.
Separate offline tests from hardware playback, and never open devices as an
automatic consequence of running tests or starting the desktop application.
