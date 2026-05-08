import json
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from guardrails import Guard, OnFailAction
from guardrails.validators import FailResult, PassResult, Validator, register_validator

from config import EVIDENCE_DIR


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    PII_PATTERNS = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "PHONE": re.compile(r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    }

    def validate(self, value: str, metadata: dict):
        redacted = value
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(redacted)
            if matches:
                found.extend((pii_type, match) for match in matches)
                redacted = pattern.sub(f"[{pii_type}_REDACTED]", redacted)

        if found:
            return FailResult(
                error_message=f"Detected PII types: {', '.join(sorted({kind for kind, _ in found}))}",
                fix_value=redacted,
            )
        return PassResult(value_override=value)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    @staticmethod
    def _repair(text: str) -> str:
        repaired = text.strip()
        repaired = re.sub(r"^```(?:json)?\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = repaired.strip()
        repaired = repaired.replace("'", '"')
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            normalized = json.dumps(parsed, indent=2)
            if normalized == value:
                return PassResult(value_override=value)
            return FailResult(
                error_message="JSON normalized for consistent formatting",
                fix_value=normalized,
            )
        except json.JSONDecodeError:
            pass

        repaired = self._repair(value)
        try:
            parsed = json.loads(repaired)
            normalized = json.dumps(parsed, indent=2)
            return FailResult(
                error_message="JSON repaired successfully",
                fix_value=normalized,
            )
        except json.JSONDecodeError as exc:
            fallback = json.dumps(
                {"error": f"Invalid JSON after repair: {exc}", "raw": value},
                indent=2,
            )
            return FailResult(
                error_message="JSON could not be repaired",
                fix_value=fallback,
            )


def demo_pii_guard() -> None:
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Passed: {result.validation_passed}")
        print(f"  Output: {result.validated_output}")


def demo_json_guard() -> None:
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Passed: {result.validation_passed}")
        print(f"  Output: {result.validated_output}")


def _capture_output(fn) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        fn()
    return buffer.getvalue()


def _write_log(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    pii_log = _capture_output(demo_pii_guard)
    json_log = _capture_output(demo_json_guard)

    print(pii_log, end="")
    print(json_log, end="")

    _write_log(EVIDENCE_DIR / "04_pii_demo_log.txt", pii_log)
    _write_log(EVIDENCE_DIR / "04_json_demo_log.txt", json_log)

    print("\nStep 4 complete")


if __name__ == "__main__":
    main()
