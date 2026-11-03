"""Input contract and shared ordering arithmetic for the offline verdict tool.

The document is deliberately small and strict. Unknown fields are rejected so a
misspelt ``measured_latency_ns`` cannot silently turn into a missing value, and
every diagnostic names the exact JSON path that failed.

Ordering convention
-------------------
Latency is lower-is-better. ``order(first, second, tolerance)`` returns -1 when
``first`` is faster by more than ``tolerance``, +1 when it is slower, and 0 when
the two are tied. Ties are decided by the symmetric relative gap
``|a - b| / mean(|a|, |b|)``, so a 2 % tolerance means "within 2 % of each
other". Latency values must be finite and positive; a null (or an absent key)
means "not measured" and is never converted to a number.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "verdict-input-1"

TOP_LEVEL_KEYS = frozenset(("schema", "title", "notes", "conditions"))
CONDITION_KEYS = frozenset(("label", "value", "unit", "notes", "configs"))
CONFIG_KEYS = frozenset(("name", "notes", "predicted_latency_ns", "measured_latency_ns"))
LATENCY_KEYS = ("predicted_latency_ns", "measured_latency_ns")


class ValidationError(ValueError):
    """One or more field-level input errors, each with its JSON path."""

    def __init__(self, errors):
        self.errors = tuple(dict(error) for error in errors)
        super().__init__("\n".join("%s: %s" % (error["path"], error["message"])
                                   for error in self.errors))


def is_number(value):
    """True for finite int/float values; bool is deliberately not a number."""
    return type(value) in (int, float) and math.isfinite(value)


def validate_tolerance(tolerance):
    """Return a float tolerance in [0, 1), or raise ValueError."""
    if not is_number(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must be a finite number in [0, 1), got %r" % (tolerance,))
    return float(tolerance)


def relative_gap(first, second):
    """Symmetric relative difference |a - b| / mean(|a|, |b|)."""
    scale = (abs(first) + abs(second)) / 2.0
    return 0.0 if scale == 0.0 else abs(first - second) / scale


def order(first, second, tolerance=0.02):
    """+1 when first is larger, -1 when smaller, 0 when tied within tolerance."""
    tolerance = validate_tolerance(tolerance)
    if first == second:
        return 0
    if relative_gap(first, second) < tolerance:
        return 0
    return 1 if first > second else -1


def _error(path, message):
    return {"path": path, "message": message}


def _show(value):
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _reject_unknown(value, allowed, path, errors):
    for key in value:
        if key not in allowed:
            location = "%s.%s" % (path, key) if path else key
            errors.append(_error(location, "unknown field; allowed fields: " + ", ".join(sorted(allowed))))


def _optional_string(value, key, path, errors, allow_empty=False):
    if key not in value:
        return
    item = value[key]
    if not isinstance(item, str):
        errors.append(_error("%s.%s" % (path, key), "expected a string, got " + _type_name(item)))
    elif not allow_empty and not item.strip():
        errors.append(_error("%s.%s" % (path, key), "expected a non-empty string"))


def _validate_config(config, path, names, errors):
    if not isinstance(config, dict):
        errors.append(_error(path, "expected an object, got " + _type_name(config)))
        return
    _reject_unknown(config, CONFIG_KEYS, path, errors)
    name = config.get("name")
    if "name" not in config:
        errors.append(_error(path + ".name", "required field is missing"))
    elif not isinstance(name, str) or not name.strip():
        errors.append(_error(path + ".name", "expected a non-empty string, got " + _show(name)))
    elif name in names:
        errors.append(_error(path + ".name", "duplicate configuration name %r (already used at %s)"
                             % (name, names[name])))
    else:
        names[name] = path
    for key in LATENCY_KEYS:
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if not is_number(value) or value <= 0:
            errors.append(_error(path + "." + key,
                                 "expected a finite positive number or null, got " + _show(value)))
    _optional_string(config, "notes", path, errors, allow_empty=True)


def _validate_condition(condition, path, labels, errors):
    if not isinstance(condition, dict):
        errors.append(_error(path, "expected an object, got " + _type_name(condition)))
        return
    _reject_unknown(condition, CONDITION_KEYS, path, errors)
    label = condition.get("label")
    if "label" not in condition:
        errors.append(_error(path + ".label", "required field is missing"))
    elif not isinstance(label, str) or not label.strip():
        errors.append(_error(path + ".label", "expected a non-empty string, got " + _show(label)))
    elif label in labels:
        errors.append(_error(path + ".label", "duplicate condition label %r (already used at %s)"
                             % (label, labels[label])))
    else:
        labels[label] = path
    if "value" in condition and not is_number(condition["value"]):
        errors.append(_error(path + ".value", "expected a finite number, got " + _show(condition["value"])))
    _optional_string(condition, "unit", path, errors)
    _optional_string(condition, "notes", path, errors, allow_empty=True)
    configs = condition.get("configs")
    if "configs" not in condition:
        errors.append(_error(path + ".configs", "required field is missing"))
    elif not isinstance(configs, list):
        errors.append(_error(path + ".configs", "expected an array, got " + _type_name(configs)))
    else:
        names = {}
        for index, config in enumerate(configs):
            _validate_config(config, "%s.configs[%d]" % (path, index), names, errors)


def validate(document):
    """Validate a parsed document in place and return it.

    All field errors are collected, so one run reports every broken field rather
    than only the first. An empty ``configs`` array is allowed: it is counted as
    an empty condition by P2 instead of being silently dropped.
    """
    if not isinstance(document, dict):
        raise ValidationError([_error("<document>", "expected a JSON object, got " + _type_name(document))])
    errors = []
    _reject_unknown(document, TOP_LEVEL_KEYS, "", errors)
    if "schema" in document and document["schema"] != SCHEMA:
        errors.append(_error("schema", "expected %r, got %s" % (SCHEMA, _show(document["schema"]))))
    _optional_string(document, "title", "", errors, allow_empty=True)
    _optional_string(document, "notes", "", errors, allow_empty=True)
    conditions = document.get("conditions")
    if "conditions" not in document:
        errors.append(_error("conditions", "required field is missing"))
    elif not isinstance(conditions, list):
        errors.append(_error("conditions", "expected an array, got " + _type_name(conditions)))
    elif not conditions:
        errors.append(_error("conditions", "at least one condition is required"))
    else:
        labels = {}
        for index, condition in enumerate(conditions):
            _validate_condition(condition, "conditions[%d]" % index, labels, errors)
    if errors:
        raise ValidationError(errors)
    return document


def load(path):
    """Read, parse and validate one JSON input file.

    Duplicate object keys and JSON ``NaN``/``Infinity`` tokens are rejected:
    both would otherwise hide a field or produce non-finite arithmetic.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError([_error("<document>", "cannot read %s: %s" % (path, error))]) from None
    except UnicodeDecodeError as error:
        raise ValidationError([_error("<document>", "cannot decode %s as UTF-8: %s" % (path, error))]) from None

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError([_error("<document>", "duplicate JSON key " + _show(key))])
            result[key] = value
        return result

    def reject_constant(token):
        raise ValidationError([_error("<document>", "non-finite JSON number %s is not allowed" % token)])

    try:
        document = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        raise ValidationError([_error("<document>", "invalid JSON at line %d column %d: %s"
                                      % (error.lineno, error.colno, error.msg))]) from None
    return validate(document)
