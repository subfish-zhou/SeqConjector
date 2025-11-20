#!/usr/bin/env python3
"""
Analyze results.jsonl to:
1. Calculate general statistics (success rates, primitives usage, etc.)
2. Find and enrich one-sided derivations (where A->B works but B->A fails, or vice versa)
"""

import json
import argparse
from collections import Counter, defaultdict
import os

class CompactJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder to keep lists on one line"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = 2
        self.current_indent = 0
        
    def encode(self, obj):
        if isinstance(obj, list):
            return '[' + ', '.join(json.dumps(item) for item in obj) + ']'
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        return self._iterencode(obj, 0)
    
    def _iterencode(self, obj, indent_level):
        if isinstance(obj, dict):
            yield '{\n'
            items = list(obj.items())
            for i, (key, value) in enumerate(items):
                yield '  ' * (indent_level + 1)
                yield json.dumps(key) + ': '
                
                if isinstance(value, list):
                    yield '[' + ', '.join(json.dumps(item, ensure_ascii=False) for item in value) + ']'
                elif isinstance(value, dict):
                    for chunk in self._iterencode(value, indent_level + 1):
                        yield chunk
                else:
                    yield json.dumps(value, ensure_ascii=False)
                
                if i < len(items) - 1:
                    yield ','
                yield '\n'
            yield '  ' * indent_level + '}'
        elif isinstance(obj, list):
            yield '[\n'
            for i, item in enumerate(obj):
                yield '  ' * (indent_level + 1)
                if isinstance(item, dict):
                    for chunk in self._iterencode(item, indent_level + 1):
                        yield chunk
                else:
                    yield json.dumps(item, ensure_ascii=False)
                
                if i < len(obj) - 1:
                    yield ','
                yield '\n'
            yield '  ' * indent_level + ']'
        else:
            yield json.dumps(obj, ensure_ascii=False)

def save_compact_json(data, filename):
    """Save JSON with lists on single lines"""
    with open(filename, 'w', encoding='utf-8') as f:
        for chunk in CompactJSONEncoder().iterencode(data):
            f.write(chunk)

def calculate_general_stats(results, output_file):
    """Calculate and save general statistics"""
    print("Calculating general statistics...")
    stats = {
        "total": 0,
        "exact": {"total": 0, "success": 0, "fail": 0, "error": 0},
        "moonshine": {"total": 0, "success": 0, "fail": 0, "error": 0},
        "program_length_dist": Counter(),
        "primitive_usage": Counter(),
        "primitive_total_count": 0
    }

    for data in results:
        stats["total"] += 1
        mode = data.get("mode", "unknown")
        status = data.get("status", "unknown")
        
        if mode not in stats:
            stats[mode] = {"total": 0, "success": 0, "fail": 0, "error": 0}

        stats[mode]["total"] += 1
        if status == "success":
            stats[mode]["success"] += 1
            
            # Analyze program
            prog = data.get("program", "")
            if prog:
                tokens = prog.split()
                length = len(tokens)
                stats["program_length_dist"][length] += 1
                
                for tok in tokens:
                    stats["primitive_usage"][tok] += 1
                    stats["primitive_total_count"] += 1
                    
        elif status == "fail":
            stats[mode]["fail"] += 1
        else:
            stats[mode]["error"] += 1

    # Post-process stats
    primitive_proportions = {}
    total_prims = stats["primitive_total_count"]
    if total_prims > 0:
        for prim, count in stats["primitive_usage"].items():
            primitive_proportions[prim] = count / total_prims

    final_stats = {
        "summary": {
            "total_tests": stats["total"],
            "exact_mode": stats.get("exact", {}),
            "moonshine_mode": stats.get("moonshine", {})
        },
        "program_length_distribution": dict(stats["program_length_dist"]),
        "primitive_usage_counts": dict(stats["primitive_usage"]),
        "primitive_usage_proportions": primitive_proportions
    }

    print(f"Writing stats to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2)
    
    print(json.dumps(final_stats["summary"], indent=2))

def find_one_sided(results):
    """Find pairs where derivation succeeds in only one direction"""
    print("\nFinding one-sided derivations...")
    
    # Organize data: key = (A_id, B_id, mode)
    derivation_map = {}
    for item in results:
        key = (item['A_id'], item['B_id'], item['mode'])
        derivation_map[key] = item
    
    one_sided_cases = []
    checked_pairs = set()
    
    for (a_id, b_id, mode), item in derivation_map.items():
        # Avoid duplicate checks
        if (a_id, b_id) in checked_pairs or (b_id, a_id) in checked_pairs:
            continue
        
        # Look for reverse derivation
        reverse_key = (b_id, a_id, mode)
        if reverse_key in derivation_map:
            reverse_item = derivation_map[reverse_key]
            
            # Check if only one side succeeded
            forward_success = item['status'] == 'success'
            reverse_success = reverse_item['status'] == 'success'
            
            if forward_success != reverse_success:
                one_sided_cases.append({
                    'seq1': a_id,
                    'seq2': b_id,
                    'mode': mode,
                    'forward': {
                        'direction': f"{a_id} -> {b_id}",
                        'status': item['status'],
                        'program': item['program'],
                        'time': item['time']
                    },
                    'reverse': {
                        'direction': f"{b_id} -> {a_id}",
                        'status': reverse_item['status'],
                        'program': reverse_item['program'],
                        'time': reverse_item['time']
                    }
                })
                checked_pairs.add((a_id, b_id))
    
    print(f"Found {len(one_sided_cases)} one-sided cases")
    
    # Print stats about modes
    mode_stats = defaultdict(int)
    for case in one_sided_cases:
        mode_stats[case['mode']] += 1
    
    print("By mode:")
    for mode, count in sorted(mode_stats.items()):
        print(f"  {mode}: {count}")
        
    return one_sided_cases

def enrich_cases(cases, db_path):
    """Enrich cases with sequence descriptions"""
    print(f"\nEnriching cases using data from {db_path}...")
    
    if not os.path.exists(db_path):
        print(f"Warning: Database file {db_path} not found. Skipping enrichment.")
        return cases

    seq_dict = {}
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    seq_dict[data['id']] = {
                        'seq': data['seq'],
                        'description': data.get('description', ''),
                        'primary_domain': data.get('primary_domain', ''),
                        'secondary_domain': data.get('secondary_domain', []),
                        'related_seq': data.get('related_seq', [])
                    }
        print(f"Loaded {len(seq_dict)} sequences from database")
    except Exception as e:
        print(f"Error reading database: {e}")
        return cases

    found_count = 0
    missing_seqs = set()
    
    for item in cases:
        seq1_id = item['seq1']
        seq2_id = item['seq2']
        
        if seq1_id in seq_dict:
            item['seq1_data'] = seq_dict[seq1_id]
            found_count += 1
        else:
            missing_seqs.add(seq1_id)
        
        if seq2_id in seq_dict:
            item['seq2_data'] = seq_dict[seq2_id]
            found_count += 1
        else:
            missing_seqs.add(seq2_id)
            
    print(f"Enriched {found_count} sequence entries")
    if missing_seqs:
        print(f"Warning: {len(missing_seqs)} sequences not found in database")
        
    return cases

def main():
    parser = argparse.ArgumentParser(description="Analyze derivation results")
    parser.add_argument("--input", default="results.jsonl", help="Input results file")
    parser.add_argument("--stats-output", default="stats.json", help="Output file for general stats")
    parser.add_argument("--onesided-output", default="one_sided_derivations.json", help="Output file for one-sided derivations")
    parser.add_argument("--db-path", default="oeis_seq_labeled/formula_true/trivial.jsonl", help="Path to sequence database for enrichment")
    args = parser.parse_args()

    # 1. Read results
    print(f"Reading {args.input}...")
    results = []
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        print(f"Read {len(results)} records")
    except FileNotFoundError:
        print("Input file not found.")
        return

    # 2. General Stats
    calculate_general_stats(results, args.stats_output)

    # 3. Find One-Sided
    one_sided_cases = find_one_sided(results)

    # 4. Enrich One-Sided
    if one_sided_cases:
        enriched_cases = enrich_cases(one_sided_cases, args.db_path)
        
        # 5. Save One-Sided
        print(f"Saving enriched one-sided cases to {args.onesided_output}...")
        save_compact_json(enriched_cases, args.onesided_output)
        print(f"Saved {len(enriched_cases)} cases")
        
        # Print example
        if enriched_cases:
            print("\n" + "=" * 80)
            print("Example Case:")
            example = enriched_cases[0]
            print(f"Pair: {example['seq1']} <-> {example['seq2']} ({example['mode']})")
            if 'seq1_data' in example:
                 print(f"  {example['seq1']}: {example['seq1_data'].get('description', 'No desc')}")
            if 'seq2_data' in example:
                 print(f"  {example['seq2']}: {example['seq2_data'].get('description', 'No desc')}")
            print(f"  Forward ({example['forward']['direction']}): {example['forward']['status']}")
            print(f"  Reverse ({example['reverse']['direction']}): {example['reverse']['status']}")
            print("=" * 80)

if __name__ == "__main__":
    main()
