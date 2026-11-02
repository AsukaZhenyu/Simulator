"""Audit historical source preservation without weakening campaign validation.

This helper does not rebuild parameters or run an archived evaluator. Archived
mutable sources and unchanged original binary/runtime/input files are checked
against an explicitly pinned historical manifest. Current source edits are
reported, never substituted into a historical measurement or its hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha(value, name):
    _require(isinstance(value, str) and len(value) == 64 and
             set(value) <= set('0123456789abcdef'), name + ' requires a lowercase SHA256')
    return value


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _status(path, expected):
    try:
        actual = _digest(path)
    except FileNotFoundError:
        return dict(status='missing', sha256=None)
    except OSError as error:
        return dict(status='unreadable', sha256=None, error=type(error).__name__)
    return dict(status='unchanged' if actual == expected else 'changed', sha256=actual)


def audit_snapshot(manifest_path, *, expected_manifest_sha256, archive_index_path=None):
    """Verify preserved bytes, not execution semantics or scientific accuracy.

    The caller pins the old manifest's file digest from a pre-existing receipt.
    Archived entries must remain beneath the index directory. An archive is
    selected for each snapshotted source even if its current original changed.
    Non-archived inputs are verified at their original, explicitly named paths.
    Unknown or mismatched bytes cannot produce a successful audit.
    """
    manifest_path = Path(manifest_path).resolve()
    expected_manifest_sha256 = _sha(expected_manifest_sha256, 'expected manifest hash')
    _require(_digest(manifest_path) == expected_manifest_sha256, 'historical manifest bytes differ')
    manifest = _json(manifest_path)
    _require(manifest.get('schema') == 'state-response-campaign-1', 'unsupported historical campaign schema')
    sources = manifest.get('sources')
    _require(isinstance(sources, dict) and sources, 'nonempty historical source inventory required')
    canonical = json.dumps(sources, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    inventory_sha = hashlib.sha256(canonical).hexdigest()
    _require(inventory_sha == manifest.get('source_inventory_sha256'), 'historical source inventory digest differs')
    archive_index_path = (manifest_path.parent / 'recorded-sources' / 'archive.json'
                          if archive_index_path is None else Path(archive_index_path)).resolve()
    archive = _json(archive_index_path)
    _require(set(archive) == {'schema', 'created_at_utc', 'manifest_sha256', 'entries'} and
             archive['schema'] == 'service-response-source-snapshot-1', 'unsupported source archive index')
    _require(archive['manifest_sha256'] == expected_manifest_sha256, 'source archive belongs to another manifest')
    _require(isinstance(archive['entries'], list) and archive['entries'], 'nonempty source archive entries required')
    archive_root = archive_index_path.parent
    archived = {}
    archive_paths = set()
    for entry in archive['entries']:
        _require(isinstance(entry, dict) and set(entry) == {'original_path', 'archive_path', 'sha256'},
                 'malformed source archive entry')
        original, retained = entry['original_path'], entry['archive_path']
        _require(isinstance(original, str) and original in sources, 'archive original is absent from historical inventory')
        _require(isinstance(retained, str) and Path(retained).is_absolute(), 'archive source path must be absolute')
        retained_path = Path(retained).resolve()
        _require(retained_path != archive_root and retained_path.is_relative_to(archive_root),
                 'archive source escapes its snapshot directory')
        _require(original not in archived and retained_path not in archive_paths, 'duplicate source archive binding')
        _require(_sha(entry['sha256'], 'archive source hash') == sources[original],
                 'archive source hash differs from historical inventory')
        archived[original] = retained_path
        archive_paths.add(retained_path)
    rows = []
    for original, expected in sources.items():
        _require(isinstance(original, str) and Path(original).is_absolute(), 'historical source paths must be absolute')
        _sha(expected, 'historical source hash')
        original_path = Path(original)
        original_observation = _status(original_path, expected)
        retained_path = archived.get(original)
        selected = original_path if retained_path is None else retained_path
        observed = original_observation if retained_path is None else _status(retained_path, expected)
        rows.append(dict(original_path=original, expected_sha256=expected,
                         lookup='original' if retained_path is None else 'archive',
                         consumed_path=str(selected), consumed=observed,
                         current_original=original_observation,
                         preserved=observed['status'] == 'unchanged'))
    summary = dict(source_count=len(rows), archived_source_count=len(archived),
                   original_source_count=len(rows) - len(archived),
                   failed_source_count=sum(not row['preserved'] for row in rows),
                   current_original_status_counts=dict(Counter(row['current_original']['status'] for row in rows)),
                   source_preservation_verified=all(row['preserved'] for row in rows))
    return dict(schema='service-response-source-preservation-audit-1',
                manifest_path=str(manifest_path), manifest_sha256=expected_manifest_sha256,
                source_inventory_sha256=inventory_sha, archive_index_path=str(archive_index_path),
                archive_index_sha256=_digest(archive_index_path), summary=summary, sources=rows,
                parameter_rebuild_verified=False, archived_evaluator_executed=False,
                raw_measurement_files_verified=False,
                current_code_matches_recorded_version=all(row['current_original']['status'] == 'unchanged' for row in rows),
                hardware_accuracy_qualified=False,
                scope='Historical source bytes only; no source-validation bypass, parameter rebuild, code replay or accuracy claim')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--archive-index', type=Path)
    parser.add_argument('--output', type=Path, help='New output file; existing paths are never overwritten')
    parser.add_argument('--details', action='store_true', help='Print the per-source archive/original lookup inventory')
    args = parser.parse_args(argv)
    result = audit_snapshot(args.manifest, expected_manifest_sha256=args.expected_manifest_sha256,
                            archive_index_path=args.archive_index)
    if args.output is not None:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result if args.details else {k:v for k,v in result.items() if k != 'sources'},
                     ensure_ascii=False, allow_nan=False))
    return 0 if result['summary']['source_preservation_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
