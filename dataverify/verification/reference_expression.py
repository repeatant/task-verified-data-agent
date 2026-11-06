"""Interpret small, model-authored reference calculations without Python exec.

This is an expression language, not a Python sandbox. The allowed operations
are generic arithmetic, filtering and aggregation, not enumerated business
checks. No attributes, imports, user functions, I/O or access to host objects
are available. A shared operation budget bounds nested comprehensions.
"""

from __future__ import annotations

import ast
import math
import operator
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any


class ExpressionError(ValueError):
    """A reference calculation is invalid or exceeds its resource allowance."""


def _round_half_up(value: int | float, digits: int = 0) -> float:
    """Round decimal ties away from zero without exposing Decimal to programs.

    Precision is checked before Decimal work, keeping allocation bounded. The
    decimal input uses the number's string representation, so values such as
    2.675 retain the stated decimal tie rather than binary-float round behavior.
    This does not make an earlier floating-point computation exact.
    """
    if type(digits) is not int or not 0 <= digits <= 9:
        raise ExpressionError("REFERENCE_ROUND_PRECISION")
    if type(value) not in (int, float) or abs(value) > 10**15 or not math.isfinite(value):
        raise ExpressionError("REFERENCE_ROUND_VALUE")
    with localcontext() as context:
        context.prec = 40
        quantum = Decimal(1).scaleb(-digits)
        result = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return float(result)


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_FUNCTIONS = {
    "sum": sum,
    "len": len,
    "min": min,
    "max": max,
    "sorted": sorted,
    "set": set,
    "abs": abs,
    # Preserve Python's original half-even behavior for old programs/evidence.
    "round": round,
    "round_half_up": _round_half_up,
    "all": all,
    "any": any,
}
_NODES = (
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Subscript,
    ast.BinOp,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    *_BINARY,
    *_COMPARE,
)


def parse_expression(source: str) -> ast.Module:
    if type(source) is not str or not 1 <= len(source) <= 4000:
        raise ExpressionError("REFERENCE_LENGTH")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError):
        raise ExpressionError("REFERENCE_SYNTAX") from None
    nodes = list(ast.walk(tree))
    if not tree.body or len(tree.body) > 20 or not isinstance(tree.body[-1], ast.Expr):
        raise ExpressionError("REFERENCE_FINAL_EXPRESSION_REQUIRED")
    for statement in tree.body[:-1]:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id in {"tables", *_FUNCTIONS}
        ):
            raise ExpressionError("REFERENCE_ONLY_LOCAL_ASSIGNMENTS_ALLOWED")
    if len(nodes) > 500:
        raise ExpressionError("REFERENCE_TOO_COMPLEX")
    for node in nodes:
        if not isinstance(node, _NODES):
            raise ExpressionError("REFERENCE_OPERATION_NOT_ALLOWED_" + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ExpressionError("REFERENCE_PRIVATE_NAME")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTIONS
            or node.keywords
            or not 1 <= len(node.args) <= 2
        ):
            raise ExpressionError("REFERENCE_CALL_NOT_ALLOWED")
        if isinstance(node, ast.comprehension) and (
            not isinstance(node.target, ast.Name) or node.is_async
        ):
            raise ExpressionError("REFERENCE_COMPREHENSION_TARGET")
    return tree


class _Interpreter:
    def __init__(self, budget: int):
        self.remaining = budget

    def charge(self, count: int = 1) -> None:
        self.remaining -= count
        if self.remaining < 0:
            raise ExpressionError("REFERENCE_OPERATION_BUDGET")

    def bounded(self, value: Any) -> Any:
        if type(value) in (int, float):
            if not math.isfinite(value) or abs(value) > 10**15:
                raise ExpressionError("REFERENCE_NUMERIC_LIMIT")
        elif isinstance(value, (str, list, tuple, set, dict)) and len(value) > 1000:
            raise ExpressionError("REFERENCE_COLLECTION_LIMIT")
        return value

    def evaluate(self, node: ast.AST, scope: dict[str, Any]) -> Any:
        self.charge()
        if isinstance(node, ast.Module):
            local = dict(scope)
            for statement in node.body[:-1]:
                local[statement.targets[0].id] = self.evaluate(statement.value, local)
            return self.evaluate(node.body[-1].value, local)
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body, scope)
        if isinstance(node, ast.Constant):
            if type(node.value) not in (str, int, float, bool, type(None)):
                raise ExpressionError("REFERENCE_LITERAL")
            return self.bounded(node.value)
        if isinstance(node, ast.Name):
            return scope[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.evaluate(item, scope) for item in node.elts]
            return self.bounded(set(values) if isinstance(node, ast.Set) else values)
        if isinstance(node, ast.Dict):
            return {
                self.evaluate(k, scope): self.evaluate(v, scope)
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Subscript):
            value, key = self.evaluate(node.value, scope), self.evaluate(node.slice, scope)
            if type(value) not in (dict, list, tuple, str):
                raise ExpressionError("REFERENCE_SUBSCRIPT")
            return value[key]
        if isinstance(node, ast.BinOp):
            left, right = self.evaluate(node.left, scope), self.evaluate(node.right, scope)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ExpressionError("REFERENCE_NUMERIC_OPERANDS_REQUIRED")
            return self.bounded(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand, scope)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) not in (int, float):
                raise ExpressionError("REFERENCE_NUMERIC_OPERANDS_REQUIRED")
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self.evaluate(item, scope)
                if isinstance(node.op, ast.And) and not value:
                    return value
                if isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = self.evaluate(node.left, scope)
            for op, item in zip(node.ops, node.comparators):
                right = self.evaluate(item, scope)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            branch = node.body if self.evaluate(node.test, scope) else node.orelse
            return self.evaluate(branch, scope)
        if isinstance(node, ast.Call):
            args = [self.evaluate(item, scope) for item in node.args]
            self.charge(sum(len(a) for a in args if isinstance(a, (list, dict, set, str))))
            return self.bounded(_FUNCTIONS[node.func.id](*args))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            result = {} if isinstance(node, ast.DictComp) else []

            def visit(index: int, local: dict) -> None:
                if index == len(node.generators):
                    if isinstance(node, ast.DictComp):
                        # Both expressions share this interpreter's operation
                        # budget. Only a newly allocated local dict is mutated.
                        key = self.evaluate(node.key, local)
                        result[key] = self.evaluate(node.value, local)
                    else:
                        result.append(self.evaluate(node.elt, local))
                    self.bounded(result)
                    return
                generator = node.generators[index]
                values = self.evaluate(generator.iter, local)
                if type(values) not in (list, tuple, set, dict):
                    raise ExpressionError("REFERENCE_ITERABLE")
                for value in values:
                    self.charge()
                    nested = {**local, generator.target.id: value}
                    if all(self.evaluate(item, nested) for item in generator.ifs):
                        visit(index + 1, nested)

            visit(0, scope)
            return set(result) if isinstance(node, ast.SetComp) else result
        raise ExpressionError("REFERENCE_OPERATION_NOT_ALLOWED")


def calculate(source: str, tables: dict, *, budget: int = 50_000) -> Any:
    """Only caller-provided JSON data enters the expression's lexical scope."""
    try:
        return _Interpreter(budget).evaluate(parse_expression(source), {"tables": tables})
    except ExpressionError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError, RecursionError):
        raise ExpressionError("REFERENCE_EVALUATION_ERROR") from None
