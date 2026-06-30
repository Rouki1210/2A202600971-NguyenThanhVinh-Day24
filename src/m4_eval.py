from __future__ import annotations

"""Module 4: RAGAS evaluation helpers for the RAG pipeline.

The public API is intentionally small because Day 24 Phase A imports
``evaluate_ragas`` from this file:

    evaluate_ragas(questions, answers, contexts, ground_truths)

It returns aggregate metric scores plus a ``per_question`` list of
``EvalResult`` objects. When RAGAS or an OpenAI key is unavailable, the module
falls back to deterministic lexical metrics so local tests and reports can
still run.
"""

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import TEST_SET_PATH
except ModuleNotFoundError:
    TEST_SET_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_set_50q.json",
    )


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM may be hallucinating or adding unsupported facts.",
        "Tighten the answer prompt, require citation from context, and lower temperature.",
    ),
    "answer_relevancy": (
        "Answer does not directly match the user question.",
        "Improve the prompt template and pass the original question clearly to the LLM.",
    ),
    "context_precision": (
        "Retrieved context contains too many irrelevant chunks.",
        "Add reranking, metadata filtering, or reduce noisy top-k results.",
    ),
    "context_recall": (
        "Retriever missed relevant evidence.",
        "Improve chunking, add BM25/hybrid retrieval, or increase retrieval top-k.",
    ),
}


@dataclass(slots=True)
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return _mean([getattr(self, metric) for metric in METRIC_NAMES])

    @property
    def worst_metric(self) -> str:
        return min(METRIC_NAMES, key=lambda metric: getattr(self, metric))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["avg_score"] = self.avg_score
        data["worst_metric"] = self.worst_metric
        return data


def load_test_set(path: str = TEST_SET_PATH) -> list[dict[str, Any]]:
    """Load the JSON test set used by the pipeline/evaluation scripts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected test set at {path!r} to be a list of objects.")
    return data


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]] | list[Any],
    ground_truths: list[str],
) -> dict[str, Any]:
    """Evaluate RAG outputs with RAGAS, with a deterministic offline fallback.

    Args:
        questions: User questions.
        answers: Generated answers.
        contexts: Retrieved contexts for each question. Each item may be a list
            of strings, a single string, or dict-like chunk objects.
        ground_truths: Reference answers.

    Returns:
        A dictionary with aggregate scores for ``faithfulness``,
        ``answer_relevancy``, ``context_precision``, ``context_recall``, and a
        ``per_question`` list of ``EvalResult`` objects.
    """
    normalized = _normalize_inputs(questions, answers, contexts, ground_truths)
    if not normalized["questions"]:
        return _build_report([])

    if not os.getenv("OPENAI_API_KEY"):
        print("  OPENAI_API_KEY is not set; using lexical fallback for RAGAS metrics.")
        return _fallback_evaluate(**normalized)

    try:
        return _ragas_evaluate(**normalized)
    except Exception as exc:
        print(f"  RAGAS evaluation failed, using lexical fallback: {exc}")
        return _fallback_evaluate(**normalized)


def failure_analysis(eval_results: Iterable[EvalResult | dict[str, Any]], bottom_n: int = 10) -> list[dict[str, Any]]:
    """Analyze the lowest-scoring questions and attach a diagnosis/fix hint."""
    analyzed: list[dict[str, Any]] = []

    for result in eval_results:
        item = _result_to_dict(result)
        metrics = {name: _safe_float(item.get(name, 0.0)) for name in METRIC_NAMES}
        avg_score = _mean(list(metrics.values()))
        worst_metric = min(METRIC_NAMES, key=lambda name: metrics[name])
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]

        analyzed.append(
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "ground_truth": item.get("ground_truth", ""),
                "worst_metric": worst_metric,
                "score": avg_score,
                "metrics": metrics,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    analyzed.sort(key=lambda item: item["score"])
    return analyzed[: max(bottom_n, 0)]


def save_report(results: dict[str, Any], failures: list[dict[str, Any]], path: str = "ragas_report.json") -> None:
    """Save aggregate metrics, per-question scores, and failure analysis."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    per_question = [_result_to_dict(item) for item in results.get("per_question", [])]
    report = {
        "aggregate": {metric: _safe_float(results.get(metric, 0.0)) for metric in METRIC_NAMES},
        "num_questions": len(per_question),
        "per_question": per_question,
        "failures": failures,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


def _ragas_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    try:
        raw_result = evaluate(dataset, metrics=metrics, raise_exceptions=False)
    except TypeError:
        raw_result = evaluate(dataset, metrics=metrics)

    df = raw_result.to_pandas()
    per_question: list[EvalResult] = []

    for index, row in df.iterrows():
        source_index = int(index)
        per_question.append(
            EvalResult(
                question=str(row.get("question", questions[source_index])),
                answer=str(row.get("answer", answers[source_index])),
                contexts=_normalize_contexts(row.get("contexts", contexts[source_index])),
                ground_truth=str(row.get("ground_truth", ground_truths[source_index])),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            )
        )

    # Some RAGAS failures can produce fewer rows. Preserve the input cardinality
    # by scoring missing rows with the fallback implementation.
    if len(per_question) != len(questions):
        fallback = _fallback_evaluate(questions, answers, contexts, ground_truths)
        return fallback

    return _build_report(per_question)


def _fallback_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, Any]:
    """Cheap lexical evaluator used when RAGAS cannot run."""
    per_question: list[EvalResult] = []

    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        joined_context = " ".join(ctxs)

        faithfulness = _precision(answer, joined_context)
        answer_relevancy = max(_f1(question, answer), _f1(ground_truth, answer))
        context_precision = _ranked_context_precision(ctxs, question, ground_truth)
        context_recall = _recall(ground_truth, joined_context)

        per_question.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=ctxs,
                ground_truth=ground_truth,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
            )
        )

    return _build_report(per_question)


def _normalize_inputs(
    questions: list[Any],
    answers: list[Any],
    contexts: list[Any],
    ground_truths: list[Any],
) -> dict[str, list[Any]]:
    lengths = {
        "questions": len(questions),
        "answers": len(answers),
        "contexts": len(contexts),
        "ground_truths": len(ground_truths),
    }
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"RAGAS inputs must have the same length ({detail}).")

    return {
        "questions": [_coerce_text(item) for item in questions],
        "answers": [_coerce_text(item) for item in answers],
        "contexts": [_normalize_contexts(item) for item in contexts],
        "ground_truths": [_coerce_text(item) for item in ground_truths],
    }


def _normalize_contexts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [_coerce_text(value)]

    try:
        items = list(value)
    except TypeError:
        return [_coerce_text(value)]

    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(_coerce_text(item))
        else:
            normalized.append(str(item))
    return normalized


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "page_content", "content", "answer", "ground_truth"):
            if key in value:
                return _coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _build_report(per_question: list[EvalResult]) -> dict[str, Any]:
    report: dict[str, Any] = {
        metric: _mean([getattr(item, metric) for item in per_question])
        for metric in METRIC_NAMES
    }
    report["per_question"] = per_question
    return report


def _result_to_dict(result: EvalResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, EvalResult):
        return result.to_dict()
    return dict(result)


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if token
    ]


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _precision(candidate: str, reference: str) -> float:
    candidate_tokens = _token_set(candidate)
    reference_tokens = _token_set(reference)
    if not candidate_tokens:
        return 0.0
    return _clamp(len(candidate_tokens & reference_tokens) / len(candidate_tokens))


def _recall(reference: str, candidate: str) -> float:
    reference_tokens = _token_set(reference)
    candidate_tokens = _token_set(candidate)
    if not reference_tokens:
        return 0.0
    return _clamp(len(reference_tokens & candidate_tokens) / len(reference_tokens))


def _f1(left: str, right: str) -> float:
    precision = _precision(left, right)
    recall = _recall(left, right)
    if precision + recall == 0:
        return 0.0
    return _clamp((2 * precision * recall) / (precision + recall))


def _ranked_context_precision(contexts: list[str], question: str, ground_truth: str) -> float:
    if not contexts:
        return 0.0

    reference = f"{question} {ground_truth}".strip()
    relevance_scores = [_f1(context, reference) for context in contexts]
    relevant_seen = 0
    precision_at_hits: list[float] = []

    for rank, score in enumerate(relevance_scores, start=1):
        if score > 0:
            relevant_seen += 1
            precision_at_hits.append(relevant_seen / rank)

    if not precision_at_hits:
        return 0.0
    return _clamp(_mean(precision_at_hits))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(number) or math.isinf(number):
        return 0.0
    return _clamp(number)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Generate answers first, then call evaluate_ragas() from pipeline.py or Phase A.")
