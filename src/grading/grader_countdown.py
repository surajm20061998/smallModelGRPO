import ast
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.data_bootstrap import ensure_repo_data_path

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
STEP_PREFIX_RE = re.compile(r"^\s*step\s*\d+\s*:\s*", re.IGNORECASE)


def load_countdown_prompt(path: str | Path) -> str:
    return Path(path).read_text()


def load_countdown_split(
    path: str | Path,
    cache_dir: str | Path | None = None,
) -> Dataset:
    path = ensure_repo_data_path(path)
    split_name = path.stem
    ds = load_dataset(
        "parquet",
        data_files={split_name: str(path)},
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return ds[split_name]


def build_countdown_question(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers in the list {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once."
    )


def get_ground_truth(example: dict[str, Any]) -> dict[str, Any]:
    return example["reward_model"]["ground_truth"]


def format_countdown_prompt(example: dict[str, Any], prompt_template: str) -> str:
    ground_truth = get_ground_truth(example)
    question = build_countdown_question(ground_truth["numbers"], ground_truth["target"])
    return prompt_template.format(question=question)


def prepare_countdown_eval(
    dataset: Dataset,
    prompt_template: str,
    max_examples: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if max_examples is not None and max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    prompts = [format_countdown_prompt(example, prompt_template) for example in dataset]
    ground_truths = [get_ground_truth(example) for example in dataset]
    return prompts, ground_truths


def extract_answer_block(response: str) -> str | None:
    match = ANSWER_RE.search(response)
    if match is None:
        return None
    answer = match.group(1).strip()
    return answer or None


def _normalize_expression(expr: str) -> str:
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", expr)
    return expr.strip()


def _strip_step_prefix(line: str) -> str:
    line = STEP_PREFIX_RE.sub("", line)
    return line.strip()


def _to_fraction(value: int | float) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not allowed")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    raise ValueError(f"Unsupported numeric literal: {value!r}")


def _eval_expr(node: ast.AST) -> tuple[Fraction, list[Fraction]]:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        value = _to_fraction(node.value)
        return value, [value]

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value, operands = _eval_expr(node.operand)
        if isinstance(node.op, ast.USub):
            return -value, [-operand for operand in operands] if len(operands) == 1 else [-value]
        return value, operands

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left_value, left_operands = _eval_expr(node.left)
        right_value, right_operands = _eval_expr(node.right)

        if isinstance(node.op, ast.Add):
            return left_value + right_value, left_operands + right_operands
        if isinstance(node.op, ast.Sub):
            return left_value - right_value, left_operands + right_operands
        if isinstance(node.op, ast.Mult):
            return left_value * right_value, left_operands + right_operands
        if right_value == 0:
            raise ZeroDivisionError("Division by zero")
        return left_value / right_value, left_operands + right_operands

    raise ValueError("Unsupported expression")


def evaluate_expression(expr: str) -> tuple[Fraction, list[Fraction]]:
    expr = _normalize_expression(expr)
    parsed = ast.parse(expr, mode="eval")
    return _eval_expr(parsed)


def _consume_available_numbers(
    available: list[Fraction],
    operands: list[Fraction],
) -> list[Fraction] | None:
    remaining = list(available)
    for operand in operands:
        for idx, candidate in enumerate(remaining):
            if candidate == operand:
                remaining.pop(idx)
                break
        else:
            return None
    return remaining


def verify_countdown_solution(
    answer_body: str,
    numbers: list[int],
    target: int,
) -> bool:
    lines = [
        _strip_step_prefix(line)
        for line in answer_body.replace("```", "").splitlines()
        if line.strip()
    ]
    if not lines:
        return False

    available = [Fraction(number) for number in numbers]
    target_value = Fraction(target)
    final_value: Fraction | None = None

    try:
        if len(lines) == 1 and "=" not in lines[0]:
            final_value, operands = evaluate_expression(lines[0])
            return _consume_available_numbers(available, operands) is not None and final_value == target_value

        for line in lines:
            if "=" in line:
                lhs, rhs = line.split("=", maxsplit=1)
                lhs_value, lhs_operands = evaluate_expression(lhs)
                rhs_value, _ = evaluate_expression(rhs)
                if lhs_value != rhs_value:
                    return False
                remaining = _consume_available_numbers(available, lhs_operands)
                if remaining is None:
                    return False
                available = remaining
                available.append(lhs_value)
                final_value = lhs_value
            else:
                final_value, operands = evaluate_expression(line)
                remaining = _consume_available_numbers(available, operands)
                if remaining is None:
                    return False
                available = remaining
                available.append(final_value)

    except (SyntaxError, ValueError, ZeroDivisionError):
        return False

    return final_value == target_value


def countdown_reward_fn(response: str, ground_truth: dict[str, Any]) -> dict[str, float]:
    answer_body = extract_answer_block(response)
    format_reward = 1.0 if answer_body is not None else 0.0
    if answer_body is None:
        return {
            "reward": 0.0,
            "format_reward": format_reward,
            "answer_reward": 0.0,
        }

    is_correct = verify_countdown_solution(
        answer_body=answer_body,
        numbers=ground_truth["numbers"],
        target=ground_truth["target"],
    )
    answer_reward = 1.0 if is_correct else 0.0
    return {
        "reward": answer_reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }
