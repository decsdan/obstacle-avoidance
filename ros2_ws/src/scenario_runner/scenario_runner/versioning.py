"""Environment-version capture for replay verification."""
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnvVersions:
    """Snapshot of versions pinned to an episode for replay verification."""

    commit_sha: str
    commit_dirty: bool
    ros_distro: str
    gazebo_version: str
    python_version: str
    platform: str

    def as_dict(self) -> dict:
        return asdict(self)


def _git_sha(repo_root: str) -> tuple[str, bool]:
    """Return (short SHA, dirty-flag) for the repo at ``repo_root``."""
    if not shutil.which('git'):
        return ('unknown', False)
    try:
        sha = subprocess.check_output(
            ['git', '-C', repo_root, 'rev-parse', '--short=12', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        status = subprocess.check_output(
            ['git', '-C', repo_root, 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ('unknown', False)
    return (sha, bool(status.strip()))


def _gazebo_version() -> str:
    for exe in ('gz', 'ign'):
        if shutil.which(exe) is None:
            continue
        try:
            out = subprocess.check_output(
                [exe, 'sim', '--version'],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError):
            continue
        return out.splitlines()[0] if out else 'unknown'
    return 'unknown'


def capture(repo_root: str = '.') -> EnvVersions:
    """Collect the full version snapshot in one call."""
    sha, dirty = _git_sha(repo_root)
    return EnvVersions(
        commit_sha=sha,
        commit_dirty=dirty,
        ros_distro=os.environ.get('ROS_DISTRO', 'unknown'),
        gazebo_version=_gazebo_version(),
        python_version=platform.python_version(),
        platform=f'{platform.system()} {platform.release()} {platform.machine()}',
    )
