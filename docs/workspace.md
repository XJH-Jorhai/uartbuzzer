# Workspace and release boundaries

The stable release contains the reviewed desktop application, B0 and Beam
encoders, single-song calibration, portable original tests, measured load model,
build tools and third-party notices. Local research work is kept separate from
the public release.

| Area | Responsibility | Publication |
|---|---|---|
| `src/uart_buzzer/`, product `tests/` | Application maintainer | Reviewed stable implementation and original fixtures |
| `desktop_assets/`, runtime requirements and build scripts | Application maintainer | Portable assets and complete upstream notices |
| Root Git policy and shared interfaces | Project coordinator | Reviewed generic policy and interfaces |
| `research/` | Individual research owners | Select files only after independent review |
| `results/`, `release/`, local environments and caches | Respective owners | Kept out of Git history |
| User library, original songs and microphone recordings | User | Private; never bundled into source or release fixtures |

Concurrent research must use distinct output directories. Frozen inputs are
read-only, and a shared file has one writer at a time. Do not reset, clean or
move an existing checkout merely to make its status appear clean.

Stable source updates use explicit allowlists and staged-content review.
Application changes, original test fixtures and release tooling should have
clear commit boundaries. Feature development starts from the published stable
commit; experimental code does not silently replace the stable default.

GitHub Releases carry the Windows executables and checksums. Source archives
come from reviewed commits. Old release archives keep their original versions;
they are never relabeled as a newly built version. Hardware I/O stays separate
from offline regression tests and is initiated explicitly by the user.
