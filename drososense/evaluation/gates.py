"""Evaluable gate and narrative-adjustment rules.

Protocol v1 stated its gates in prose — "R0 is competitive with R4", "R0 > R2",
"R0 ~= R4" — with no threshold, no direction and no alpha. The R0 audit raised
this as CRITICAL (item C2): ``~=`` can only be read as "failed to reject", which
is the classic error of accepting a null, and a branch chosen after seeing the
results is post-hoc storytelling.

Protocol v1.1 therefore declares each gate as a boolean expression over a small,
fixed predicate vocabulary, and this module evaluates those expressions. The
evaluator is a whitelisted AST walker, not :func:`eval`: only a fixed set of
function names and operators is accepted, no attribute access, no imports, no
comprehensions, and every bare name must resolve in the symbol table or the
evaluation fails loudly.

Predicate vocabulary (all of it — anything else is a protocol error):

``delta(A, B, metric, dataset[, condition])``
    Mean paired difference ``A - B``.
``ci_low`` / ``ci_high`` / ``p_holm`` / ``n_pairs``
    Same signature as ``delta``.
``sig(A, B, metric, dataset[, condition])``
    Holm-adjusted p below alpha AND the interval excludes zero on the side that
    favours ``A`` AND the comparison has enough clusters for alpha to be
    arithmetically reachable at all. The third clause is not decoration: a
    dataset with five clusters has an exact two-sided floor of 0.0625, so a
    "significant" result there can only come from a pseudoreplicated p-value
    (protocol v1.2 §13).
``noninferior(A, B, metric, dataset[, condition])``
    ``ci_low(A - B) > -margin(metric)`` — a pre-registered non-inferiority call
    rather than a failure to reject.
``ci_contains_zero(...)``
    The interval is compatible with no difference.
``equiv(A, B, metric, dataset[, condition])``
    TOST at ``margin(metric)``: the interval lies strictly inside the margin.
``equiv_all(A, B, metric, [datasets])``
    ``equiv`` on every listed dataset.
``sig_any(A, B, metric, [datasets], [conditions])``
    ``sig`` on at least one listed (dataset, condition) pair.
``params(model)``
    Trainable parameter count of a model.
``margin(metric)``
    The declared equivalence margin.
``unavailable(dataset)``
    The dataset's data was not obtained.
``sum`` / ``len`` / ``min`` / ``max`` / ``abs``
    Builtins.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from drososense.evaluation.stats import minimum_achievable_p

# Predicates that consume a contrast signature and return a float or a bool.
_CONTRAST_FUNCTIONS = frozenset(
    {
        "delta",
        "ci_low",
        "ci_high",
        "p_holm",
        "n_pairs",
        "sig",
        "noninferior",
        "equiv",
        "ci_contains_zero",
    }
)
_SAFE_BUILTINS: Mapping[str, Callable[..., Any]] = {
    "sum": sum,
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.IfExp,
)


class GateExpressionError(ValueError):
    """Raised when a gate expression is malformed or references unknown names."""


@dataclass(frozen=True)
class GateEvaluation:
    """The outcome of evaluating one gate.

    Attributes:
        gate_id: Gate identifier, e.g. ``Gate_A``.
        expression: The declared expression, verbatim.
        result: The boolean the expression evaluated to.
        detail: Per-term values, for an auditable trace.
    """

    gate_id: str
    expression: str
    result: bool
    detail: dict[str, Any]


class GateEvaluator:
    """Evaluate protocol gate expressions against a contrast table.

    Args:
        contrasts: Maps ``(contrast_id, metric, dataset, condition)`` to a
            mapping with at least ``delta``, ``ci_low``, ``ci_high``,
            ``p_holm`` and ``n_pairs``.
        metrics: Declared metric properties (``direction``, equivalence margin).
        model_params: Trainable parameter counts keyed by model id.
        symbols: Extra bare names the expressions may reference (models, metrics,
            datasets and conditions), each mapping to the string it stands for.
        available_datasets: Datasets whose data was actually obtained. A dataset
            absent from this set makes ``unavailable(dataset)`` true, which is how
            narrative rule N5 fires.
    """

    def __init__(
        self,
        contrasts: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
        metrics: Mapping[str, Mapping[str, Any]],
        model_params: Mapping[str, int] | None = None,
        parameter_provenance: Mapping[str, Any] | None = None,
        symbols: Mapping[str, Any] | None = None,
        available_datasets: Iterable[str] | None = None,
    ) -> None:
        self.contrasts = contrasts
        self.metrics = metrics
        self.model_params = dict(model_params or {})
        #: Protocol v1.5.1: the declared evidence scope behind `model_params`, so
        #: a `params(...)` term that cannot resolve can say WHY (no rows,
        #: conflicting values, unbound condition) instead of blaming the wrong
        #: thing, and so the gate result carries an auditable trail.
        self.parameter_provenance = dict(parameter_provenance or {})
        self.symbols = dict(symbols or {})
        self.available_datasets = {
            str(d) for d in (available_datasets if available_datasets is not None else ())
        }
        self._trace: dict[str, Any] = {}

    # -- public API ---------------------------------------------------------

    def evaluate(self, gate_id: str, expression: str) -> GateEvaluation:
        """Evaluate one gate expression.

        Args:
            gate_id: Identifier used in the trace and error messages.
            expression: The expression text.

        Returns:
            The :class:`GateEvaluation`.

        Raises:
            GateExpressionError: If the expression is malformed, references an
                unknown name, or does not evaluate to a boolean.
        """
        self._trace = {}
        # YAML folded scalars (`>`) end with a newline; strip it rather than
        # making every protocol author remember to.
        source = " ".join(str(expression).split())
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:
            raise GateExpressionError(f"{gate_id}: cannot parse {expression!r}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise GateExpressionError(
                    f"{gate_id}: {type(node).__name__} is not allowed in a gate expression"
                )
        value = self._eval(tree.body, gate_id)
        if not isinstance(value, bool):
            raise GateExpressionError(
                f"{gate_id}: expression {expression!r} evaluated to {type(value).__name__}, "
                f"not a boolean"
            )
        detail = dict(self._trace)
        if self.parameter_provenance:
            # Every gate result carries the parameter scope it was evaluated
            # against, so a published parameter number is auditable from the
            # gate output alone (protocol v1.5.1, item P3).
            detail["parameter_scope"] = self.parameter_provenance
        return GateEvaluation(
            gate_id=gate_id, expression=expression, result=value, detail=detail
        )

    def evaluate_all(self, gates: Mapping[str, Mapping[str, Any]]) -> dict[str, GateEvaluation]:
        """Evaluate every gate in a protocol ``gates`` mapping.

        Args:
            gates: Mapping of gate id to a definition containing
                ``expression``.

        Returns:
            Mapping of gate id to its evaluation. Gates without an
            ``expression`` key (for example a narrative-rule list) are skipped.

        Raises:
            GateExpressionError: If a gate declares no expression at all.
        """
        out: dict[str, GateEvaluation] = {}
        for gate_id, definition in gates.items():
            if not isinstance(definition, Mapping):
                continue
            expression = definition.get("expression")
            if expression is None:
                continue
            out[gate_id] = self.evaluate(gate_id, str(expression))
        return out

    # -- expression walking -------------------------------------------------

    def _eval(self, node: ast.AST, gate_id: str) -> Any:
        """Recursively evaluate a whitelisted AST node.

        Args:
            node: The node to evaluate.
            gate_id: Gate identifier for error messages.

        Returns:
            The node's value.

        Raises:
            GateExpressionError: On an unsupported node or an unknown name.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id, gate_id)
        if isinstance(node, ast.List):
            return [self._eval(e, gate_id) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e, gate_id) for e in node.elts)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, gate_id)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            return +operand
        if isinstance(node, ast.BoolOp):
            values = [self._eval(v, gate_id) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, gate_id)
            right = self._eval(node.right, gate_id)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, gate_id)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, gate_id)
                if not _compare(op, left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._eval(node.body if self._eval(node.test, gate_id) else node.orelse, gate_id)
        if isinstance(node, ast.Call):
            return self._eval_call(node, gate_id)
        raise GateExpressionError(f"{gate_id}: unsupported expression node {type(node).__name__}")

    def _eval_call(self, node: ast.Call, gate_id: str) -> Any:
        """Evaluate a whitelisted function call.

        Args:
            node: The call node.
            gate_id: Gate identifier for error messages.

        Returns:
            The call's value.

        Raises:
            GateExpressionError: If the function is not whitelisted or its
                arguments are wrong.
        """
        if not isinstance(node.func, ast.Name):
            raise GateExpressionError(f"{gate_id}: only plain function names may be called")
        name = node.func.id
        args = [self._eval(a, gate_id) for a in node.args]
        if node.keywords:
            raise GateExpressionError(f"{gate_id}: keyword arguments are not allowed")

        if name in _SAFE_BUILTINS:
            return _SAFE_BUILTINS[name](*args)
        if name in _CONTRAST_FUNCTIONS:
            return self._contrast_predicate(name, args, gate_id)
        if name == "equiv_all":
            return self._equiv_all(args, gate_id)
        if name == "sig_any":
            return self._sig_any(args, gate_id)
        if name == "params":
            return self._params(args, gate_id)
        if name == "margin":
            return self._margin(args, gate_id)
        if name == "unavailable":
            return self._unavailable(args, gate_id)
        raise GateExpressionError(f"{gate_id}: {name!r} is not a known gate predicate")

    def _unavailable(self, args: Sequence[Any], gate_id: str) -> bool:
        """``unavailable(dataset)``.

        Args:
            args: ``(dataset_id,)``.
            gate_id: Gate identifier for error messages.

        Returns:
            Whether the dataset's data was not obtained.
        """
        if len(args) != 1:
            raise GateExpressionError(f"{gate_id}: unavailable takes 1 argument")
        return str(args[0]) not in self.available_datasets

    def _contrast_predicate(self, name: str, args: Sequence[Any], gate_id: str) -> Any:
        """Resolve a contrast predicate.

        Args:
            name: Predicate name.
            args: ``(model_a, model_b, metric, dataset[, condition])``.
            gate_id: Gate identifier for error messages.

        Returns:
            The predicate's float or bool value.

        Raises:
            GateExpressionError: If the contrast is not in the table.
        """
        if len(args) not in (4, 5):
            raise GateExpressionError(
                f"{gate_id}: {name}(A, B, metric, dataset[, condition]) takes 4 or 5 "
                f"arguments, got {len(args)}"
            )
        model_a, model_b = str(args[0]), str(args[1])
        metric, dataset = str(args[2]), str(args[3])
        condition = str(args[4]) if len(args) == 5 else "full"
        contrast_id = f"{model_a}_vs_{model_b}"
        key = (contrast_id, metric, dataset, condition)
        # A protocol may write a contrast in either direction — N3 asks
        # `sig(R4, R0, ...)` where the registry stores `R0_vs_R4`. Treating the
        # reversed form as "no result" would leave the negative-result branch
        # permanently unevaluable, which is precisely the branch that must work.
        reversed_key = (f"{model_b}_vs_{model_a}", metric, dataset, condition)
        if key in self.contrasts:
            row = self.contrasts[key]
            sign = 1.0
        elif reversed_key in self.contrasts:
            row = self.contrasts[reversed_key]
            sign = -1.0
            contrast_id = reversed_key[0]
        else:
            raise GateExpressionError(
                f"{gate_id}: no result for contrast {contrast_id} on {metric}/{dataset}/"
                f"{condition}; the gate cannot be evaluated"
            )
        direction = str(self.metrics.get(metric, {}).get("direction", "maximize"))
        margin = float(self.metrics.get(metric, {}).get("margin", 0.0))
        delta = sign * float(row["delta"])
        # Reversing a contrast negates the estimate and swaps the interval.
        ci_low = sign * float(row["ci_high"] if sign < 0 else row["ci_low"])
        ci_high = sign * float(row["ci_low"] if sign < 0 else row["ci_high"])
        p_holm = float(row["p_holm"])
        alpha = float(self.metrics.get(metric, {}).get("alpha", 0.05))
        favourable = ci_low > 0 if direction == "maximize" else ci_high < 0
        n_clusters = self._cluster_count(row, gate_id, metric, dataset, condition)
        minimum_p = minimum_achievable_p(n_clusters)
        self._trace[f"{contrast_id}|{metric}|{dataset}|{condition}"] = {
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_holm": p_holm,
            "n_pairs": int(row.get("n_pairs", 0)),
            "n_clusters": n_clusters,
            "minimum_achievable_p_over_clusters": minimum_p,
        }

        if name == "delta":
            return delta
        if name == "ci_low":
            return ci_low
        if name == "ci_high":
            return ci_high
        if name == "p_holm":
            return p_holm
        if name == "n_pairs":
            return int(row.get("n_pairs", 0))
        if name == "sig":
            # Three clauses, and the third is the R0.1 review's item C2: a
            # p-value computed over n_folds * n_seeds paired differences is
            # pseudoreplicated, and on a dataset with too few clusters alpha is
            # simply not reachable. Without this clause a gate could be opened by
            # a p that no test over the actual independent units could produce.
            return bool(p_holm < alpha and favourable and minimum_p <= alpha)
        if name == "noninferior":
            return bool(ci_low > -margin) if direction == "maximize" else bool(ci_high < margin)
        if name == "ci_contains_zero":
            return bool(ci_low <= 0 <= ci_high)
        # equiv
        return bool(ci_low > -margin and ci_high < margin)

    def _cluster_count(
        self, row: Mapping[str, Any], gate_id: str, metric: str, dataset: str, condition: str
    ) -> int:
        """Read the number of non-zero clusters behind a contrast.

        ``sig`` refuses to decide without it. Defaulting a missing cluster count
        to "enough" would restore exactly the failure mode the R0.1 review
        removed: a decision taken on an independent-unit count nobody supplied.

        Args:
            row: The contrast row.
            gate_id: Gate identifier for error messages.
            metric: Metric name.
            dataset: Dataset identifier.
            condition: Condition identifier.

        Returns:
            The number of non-zero clusters.

        Raises:
            GateExpressionError: If the row carries no cluster count.
        """
        value = row.get("n_clusters_nonzero", row.get("n_clusters"))
        if value is None:
            raise GateExpressionError(
                f"{gate_id}: contrast {metric}/{dataset}/{condition} carries no cluster count. "
                f"A contrast whose independent-unit count is unknown is not evaluable: `sig` "
                f"needs the count to apply its reachability term, and the interval predicates "
                f"(`equiv`, `noninferior`, `ci_contains_zero`) must disclose how many units the "
                f"verdict rests on (protocol v1.2 §10, §13)."
            )
        return int(value)

    def _equiv_all(self, args: Sequence[Any], gate_id: str) -> bool:
        """``equiv_all(A, B, metric, [datasets])``.

        Args:
            args: ``(model_a, model_b, metric, dataset_list)``.
            gate_id: Gate identifier for error messages.

        Returns:
            Whether every listed dataset is equivalent.
        """
        if len(args) != 4:
            raise GateExpressionError(f"{gate_id}: equiv_all takes 4 arguments")
        model_a, model_b, metric = str(args[0]), str(args[1]), str(args[2])
        datasets = args[3]
        if not isinstance(datasets, (list, tuple)):
            raise GateExpressionError(f"{gate_id}: equiv_all's 4th argument must be a list")
        return all(
            self._contrast_predicate("equiv", (model_a, model_b, metric, ds), gate_id)
            for ds in datasets
        )

    def _sig_any(self, args: Sequence[Any], gate_id: str) -> bool:
        """``sig_any(A, B, metric, [datasets], [conditions])``.

        Args:
            args: ``(model_a, model_b, metric, dataset_list, condition_list)``.
            gate_id: Gate identifier for error messages.

        Returns:
            Whether any listed (dataset, condition) pair is significant.
        """
        if len(args) != 5:
            raise GateExpressionError(f"{gate_id}: sig_any takes 5 arguments")
        model_a, model_b, metric = str(args[0]), str(args[1]), str(args[2])
        datasets, conditions = args[3], args[4]
        if not isinstance(datasets, (list, tuple)) or not isinstance(conditions, (list, tuple)):
            raise GateExpressionError(
                f"{gate_id}: sig_any's 4th and 5th arguments must be lists"
            )
        for dataset in datasets:
            for condition in conditions:
                try:
                    if self._contrast_predicate(
                        "sig", (model_a, model_b, metric, dataset, condition), gate_id
                    ):
                        return True
                except GateExpressionError:
                    continue
        return False

    def _params(self, args: Sequence[Any], gate_id: str) -> int:
        """``params(model)``.

        Args:
            args: ``(model_id,)``.
            gate_id: Gate identifier for error messages.

        Returns:
            The trainable parameter count.

        Raises:
            GateExpressionError: If the model has no recorded parameter count.
        """
        if len(args) != 1:
            raise GateExpressionError(f"{gate_id}: params takes 1 argument")
        model = str(args[0])
        if model not in self.model_params:
            # Protocol v1.5.1: the count is read from matched result rows inside a
            # declared scope, with no fallback. When the scope says why it could
            # not resolve, that reason is the error — it is more specific than
            # "not recorded" and it names what would have to change.
            scope_reason = str(self.parameter_provenance.get("reason") or "")
            if not scope_reason:
                # The analysis pipeline hands the evaluator the aggregate audit
                # shape ({"declaration", "scopes", "unevaluable", ...}) rather
                # than one scope's provenance, so the gate's own entry has to be
                # looked up. Without this the reason silently degraded to
                # "no parameter evidence scope was declared" on the production
                # path while the same scope resolved correctly in isolation.
                scopes = self.parameter_provenance.get("scopes")
                if isinstance(scopes, Mapping):
                    gate_scope = scopes.get(gate_id) or {}
                    if isinstance(gate_scope, Mapping):
                        scope_reason = str(gate_scope.get("reason") or "")
            if scope_reason:
                raise GateExpressionError(
                    f"{gate_id}: params({model}) is unevaluable under the declared parameter "
                    f"evidence scope (protocol v1.5.1). {scope_reason}"
                )
            raise GateExpressionError(
                f"{gate_id}: no trainable-parameter count recorded for {model!r} and no "
                f"parameter evidence scope was declared. A count is never substituted from "
                f"another experiment: it is NOT a missing-data or git-ignore problem."
            )
        return int(self.model_params[model])

    def _margin(self, args: Sequence[Any], gate_id: str) -> float:
        """``margin(metric)``.

        Args:
            args: ``(metric,)``.
            gate_id: Gate identifier for error messages.

        Returns:
            The declared equivalence margin.

        Raises:
            GateExpressionError: If the metric declares no margin.
        """
        if len(args) != 1:
            raise GateExpressionError(f"{gate_id}: margin takes 1 argument")
        metric = str(args[0])
        if metric not in self.metrics or "margin" not in self.metrics[metric]:
            raise GateExpressionError(f"{gate_id}: no equivalence margin declared for {metric!r}")
        return float(self.metrics[metric]["margin"])

    def _resolve_name(self, name: str, gate_id: str) -> Any:
        """Resolve a bare name in an expression.

        Args:
            name: The identifier.
            gate_id: Gate identifier for error messages.

        Returns:
            The value the name stands for.

        Raises:
            GateExpressionError: If the name is unknown.
        """
        if name in self.symbols:
            return self.symbols[name]
        if name in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[name]
        declared = sorted(self.symbols)
        raise GateExpressionError(
            f"{gate_id}: {name!r} is not a declared model, metric, dataset or condition. "
            f"Declared names: {', '.join(declared) if declared else '(none)'}. "
            f"An unknown name is a protocol/code namespace mismatch, NOT a missing contrast: "
            f"a declared dataset that has no result reports 'no result for contrast' instead."
        )


def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    """Apply one comparison operator.

    Args:
        op: The AST comparison operator.
        left: Left operand.
        right: Right operand.

    Returns:
        The comparison result.
    """
    if isinstance(op, ast.Eq):
        return bool(left == right)
    if isinstance(op, ast.NotEq):
        return bool(left != right)
    if isinstance(op, ast.Lt):
        return bool(left < right)
    if isinstance(op, ast.LtE):
        return bool(left <= right)
    if isinstance(op, ast.Gt):
        return bool(left > right)
    return bool(left >= right)


def build_symbols(
    models: Iterable[str],
    metrics: Iterable[str],
    datasets: Iterable[str],
    conditions: Iterable[str] = (),
    aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the symbol table a gate expression resolves bare names against.

    Args:
        models: Model identifiers, e.g. ``R0``.
        metrics: Metric names, e.g. ``macro_f1``.
        datasets: Dataset identifiers, e.g. ``D1``.
        conditions: Condition identifiers, e.g. ``dropout_p0.3``.
        aliases: Extra bare names that resolve to a declared value, e.g.
            ``{"D2": "d2_beef_uncontrolled"}``. This is how the protocol's own
            short dataset names stay usable in a gate expression while the
            contrast table is keyed by the config id. An alias may not shadow a
            declared name with a different meaning.

    Returns:
        Mapping of bare name to the string it stands for.

    Raises:
        ValueError: If two categories declare the same name with different
            meanings, or an alias shadows a declared name with another value,
            which would make an expression ambiguous.
    """
    symbols: dict[str, str] = {}
    for values in (models, metrics, datasets, conditions):
        for value in values:
            existing = symbols.get(str(value))
            if existing is not None and existing != str(value):
                raise ValueError(f"gate symbol {value!r} is declared twice")
            symbols[str(value)] = str(value)
    for name, target in (aliases or {}).items():
        existing = symbols.get(str(name))
        if existing is not None and existing != str(target):
            raise ValueError(
                f"gate alias {name!r} -> {target!r} shadows the declared name "
                f"{existing!r}; an expression using it would be ambiguous"
            )
        if not str(target):
            raise ValueError(f"gate alias {name!r} resolves to an empty target")
        symbols[str(name)] = str(target)
    return symbols


__all__ = [
    "GateEvaluation",
    "GateEvaluator",
    "GateExpressionError",
    "build_symbols",
]
