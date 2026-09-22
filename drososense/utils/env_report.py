"""Compare the environment a result was produced in with the one declared.

The R0 audit (item C7) found a delivery reporting "159 passed" while the same
suite on another machine collected 159 and produced 155 passed / 3 skipped /
1 failed, all of it traceable to a missing ``xgboost``. The count was true in
the declaring environment and false as a general statement, and nothing in the
repository made the difference visible.

This module makes the difference a value rather than a footnote:

    python -m drososense.utils.env_report

prints which declared packages are present, which are missing, and which differ
from the pinned version, so a claim about a test run or a benchmark can be
qualified with the environment it actually applies to.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drososense.utils.paths import PROJECT_ROOT

ENVIRONMENT_FILE = PROJECT_ROOT / "environment.yml"

# Import name -> the string used to probe its version.
_PROBE_MODULES: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "requests": "requests",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
    "xgboost": "xgboost",
    "torch": "torch",
}

# Declaration styles in environment.yml that this parser understands.
_CONDA_TO_PIP = {"pyyaml": "pyyaml", "scikit-learn": "scikit-learn"}


@dataclass(frozen=True)
class PackageStatus:
    """One declared package, compared against the running interpreter.

    Attributes:
        name: Package name as declared.
        declared: The declared version specifier, e.g. ``>=1.26,<3``.
        installed: The version found locally, or ``None`` when absent.
        meets_specifier: Whether the installed version satisfies the specifier.
    """

    name: str
    declared: str
    installed: str | None
    meets_specifier: bool


@dataclass(frozen=True)
class EnvironmentReport:
    """Comparison of the declared environment with the running one.

    Attributes:
        declared_source: Path the declaration was read from.
        python_declared: The declared Python version, if any.
        python_local: The running Python version.
        platform_local: The running platform string.
        packages: One entry per declared package.
    """

    declared_source: str
    python_declared: str | None
    python_local: str
    platform_local: str
    packages: tuple[PackageStatus, ...] = field(default_factory=tuple)

    @property
    def missing(self) -> tuple[str, ...]:
        """Declared packages that are not installed here."""
        return tuple(p.name for p in self.packages if p.installed is None)

    @property
    def mismatched(self) -> tuple[str, ...]:
        """Declared packages that are installed but outside the declared range."""
        return tuple(p.name for p in self.packages if p.installed and not p.meets_specifier)

    @property
    def complete(self) -> bool:
        """Whether this environment satisfies the declaration exactly."""
        return not self.missing and not self.mismatched

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns:
            Mapping of the report.
        """
        return {
            "declared_source": self.declared_source,
            "python_declared": self.python_declared,
            "python_local": self.python_local,
            "platform_local": self.platform_local,
            "complete": self.complete,
            "missing": list(self.missing),
            "mismatched": list(self.mismatched),
            "packages": [
                {
                    "name": p.name,
                    "declared": p.declared,
                    "installed": p.installed,
                    "meets_specifier": p.meets_specifier,
                }
                for p in self.packages
            ],
        }

    def summary(self) -> str:
        """Render a short human-readable summary.

        Returns:
            One line stating the local environment and any gap against the
            declaration.
        """
        head = (
            f"python {self.python_local} on {self.platform_local}; "
            f"{len(self.packages)} declared package(s)"
        )
        if self.complete:
            return head + " — matches environment.yml"
        parts = [head]
        if self.missing:
            parts.append(f"missing: {', '.join(self.missing)}")
        if self.mismatched:
            parts.append(f"outside declared range: {', '.join(self.mismatched)}")
        return "; ".join(parts)

    def qualification(self) -> str:
        """Return the clause a test-count or benchmark claim must carry.

        Returns:
            An empty string in a complete environment; otherwise a phrase naming
            what is absent, so the claim can be stated for the right machine.
        """
        if self.complete:
            return "in the declared environment (environment.yml)"
        return (
            "in THIS environment only, which is NOT the declared one "
            f"(missing: {', '.join(self.missing) or 'none'})"
        )


def parse_environment_yml(path: str | Path | None = None) -> dict[str, str]:
    """Read the declared dependency specifiers from ``environment.yml``.

    The file is parsed by hand rather than with a YAML loader because the
    declaration is a small, stable list and reading it directly keeps the
    function dependency-free and easy to audit.

    Args:
        path: Override for ``environment.yml``.

    Returns:
        Mapping of package name to version specifier (empty string when the
        declaration pins nothing).
    """
    target = Path(path) if path is not None else ENVIRONMENT_FILE
    if not target.is_file():
        return {}

    declarations: dict[str, str] = {}
    in_dependencies = False
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # Only the `dependencies:` block describes what must be installed;
        # `channels:` and `name:` are not packages and are skipped.
        if not line.startswith((" ", "\t")):
            in_dependencies = line.strip().startswith("dependencies:")
            continue
        if not in_dependencies:
            continue
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        item = stripped[2:].strip()
        # `- pip:` introduces a nested list of pip items; it is a block header,
        # not a package.
        if not item or item == "pip" or item.endswith(":"):
            continue
        name = re.split(r"[<>=!~ ]", item, maxsplit=1)[0].strip()
        specifier = item[len(name):].strip()
        if not name:
            continue
        declarations[name] = specifier
    return declarations


def _installed_version(module_name: str) -> str | None:
    """Return the version of an importable module.

    Args:
        module_name: Importable module name.

    Returns:
        The version string, or ``None`` when the module is absent.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def compare_environments(path: str | Path | None = None) -> EnvironmentReport:
    """Compare the declared environment with the running interpreter.

    Args:
        path: Override for ``environment.yml``.

    Returns:
        The :class:`EnvironmentReport`.
    """
    declarations = parse_environment_yml(path)
    local_python = sys.version.split()[0]
    statuses: list[PackageStatus] = []
    for name, specifier in declarations.items():
        if name == "python":
            # The interpreter is not an importable package; compare the running
            # version directly rather than reporting it as missing.
            statuses.append(
                PackageStatus(
                    name="python",
                    declared=specifier or "(unpinned)",
                    installed=local_python,
                    meets_specifier=_meets(local_python, specifier),
                )
            )
            continue
        module_name = _PROBE_MODULES.get(name, name.replace("-", "_"))
        installed = _installed_version(module_name)
        statuses.append(
            PackageStatus(
                name=name,
                declared=specifier or "(unpinned)",
                installed=installed,
                meets_specifier=_meets(installed, specifier),
            )
        )
    python_declared = next(
        (v for k, v in declarations.items() if k == "python"), None
    )
    target = Path(path) if path is not None else ENVIRONMENT_FILE
    return EnvironmentReport(
        declared_source=str(target),
        python_declared=python_declared,
        python_local=sys.version.split()[0],
        platform_local=platform.platform(),
        packages=tuple(statuses),
    )


def _meets(installed: str | None, specifier: str) -> bool:
    """Check an installed version against a specifier.

    Args:
        installed: Installed version, or ``None``.
        specifier: A set of comma-separated PEP 440 clauses.

    Returns:
        ``True`` when the version satisfies every clause, or when there is
        nothing to check.
    """
    if installed is None:
        return False
    if not specifier:
        return True
    clauses = [c.strip() for c in specifier.split(",") if c.strip()]
    if not clauses:
        return True
    try:
        from packaging.version import Version
    except Exception:  # pragma: no cover - packaging ships with pip
        return True
    try:
        version = Version(installed)
    except Exception:
        return True
    for clause in clauses:
        match = re.match(r"^(==|>=|<=|>|<|!=|~=)\s*(.+)$", clause)
        if not match:
            continue
        operator, bound = match.group(1), Version(match.group(2))
        if operator == "==" and not version == bound:
            return False
        if operator == ">=" and not version >= bound:
            return False
        if operator == "<=" and not version <= bound:
            return False
        if operator == ">" and not version > bound:
            return False
        if operator == "<" and not version < bound:
            return False
        if operator == "!=" and not version != bound:
            return False
        if operator == "~=" and not (version >= bound and version.release[: len(bound.release)] == bound.release):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 when the environment matches the declaration,
        1 when it does not.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument("--environment-file", default=None, help="override environment.yml")
    args = parser.parse_args(argv)

    report = compare_environments(args.environment_file)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(report.summary())
        for status in report.packages:
            mark = "ok " if status.meets_specifier else "!! "
            print(f"  {mark}{status.name:14} declared {status.declared:16} installed {status.installed}")
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
