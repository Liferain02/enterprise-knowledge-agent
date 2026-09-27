"""Offline reproducible routing acceptance report; never calls a model or service.

Authored regression data is not a held-out accuracy benchmark. Latencies cover
only the warmed pure policy function, excluding import/network/model costs.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent.routing.policy import POLICY_VERSION, rules_route


def summarize(rows):
    labels = ('general_agent', 'knowledge_agent', 'operation_agent')
    matrix = {expected: {actual: 0 for actual in labels} for expected in labels}
    for row in rows:
        matrix[row['expected']][row['decision']['handler']] += 1
    f1s = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return {'cases': len(rows), 'correct': sum(row['correct'] for row in rows),
            'macro_f1': sum(f1s) / len(f1s), 'confusion_matrix': matrix,
            'capability_contracts_passed': sum(row['skills_correct'] for row in rows)}


def evaluate(fixture, rounds=100):
    raw = fixture.read_bytes()
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not cases:
        raise ValueError('Empty fixture')
    rows = []
    for case in cases:
        decision = rules_route(case['query'], case.get('previous')).to_dict()
        rows.append(case | {'decision': decision, 'correct': decision['handler'] == case['expected'],
            'skills_correct': set(case.get('required_skills', ())).issubset(decision['skills'])})
    latencies = []
    for _ in range(rounds):
        for case in cases:
            start = time.perf_counter_ns()
            rules_route(case['query'], case.get('previous'))
            latencies.append((time.perf_counter_ns() - start) / 1e6)
    latencies.sort()
    # Include source hashes: HEAD alone cannot identify an uncommitted implementation.
    paths = [Path(__file__), ROOT/'src/agent/routing/policy.py', ROOT/'src/agent/routing/jev.py']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    baseline = json.loads((ROOT/'tests/eval/fixtures/router_baseline_20260926.json').read_text())
    baseline_queries = {case['query'] for case in baseline['cases']}
    comparable = [row for row in rows if row['query'] in baseline_queries]
    report = {
        'kind': 'authored_regression_not_independent_benchmark', 'policy_version': POLICY_VERSION,
        'fixture_sha256': hashlib.sha256(raw).hexdigest(), 'source_sha256': hashes,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
        'external_requests': 0, 'jev_live_evaluated': False,
        'metrics': summarize(rows),
        'groups': {group: summarize([row for row in rows if row['group'] == group]) for group in sorted({r['group'] for r in rows})},
        'old_diagnostic': {'correct': baseline['correct'], 'total': baseline['total'],
                           'new_correct_on_same_queries': sum(row['correct'] for row in comparable)},
        'latency': {'scope': 'warm rules_route only; no imports, LLM, retrieval or HTTP',
            'calls': len(latencies), 'p50_ms': latencies[math.ceil(len(latencies)*.5)-1],
            'p95_ms': latencies[math.ceil(len(latencies)*.95)-1]},
        'rule_counts': dict(Counter(row['decision']['rule_id'] for row in rows)), 'cases': rows,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT/'tests/eval/fixtures/research_routing_v1.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'.run/research-routing-v1.json')
    parser.add_argument('--rounds', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10000:
        parser.error('--rounds must be 1..10000')
    result = evaluate(args.fixture, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('metrics', 'old_diagnostic', 'latency')}, ensure_ascii=False, indent=2))
    failures = [row['id'] for row in result['cases'] if not row['correct'] or not row['skills_correct']]
    if failures:
        print('FAILED: ' + ', '.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
