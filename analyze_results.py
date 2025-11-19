import json
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.jsonl")
    parser.add_argument("--output", default="stats.json")
    args = parser.parse_args()

    stats = {
        "total": 0,
        "exact": {"total": 0, "success": 0, "fail": 0, "error": 0},
        "moonshine": {"total": 0, "success": 0, "fail": 0, "error": 0},
        "program_length_dist": Counter(),
        "primitive_usage": Counter(),
        "primitive_total_count": 0
    }

    print(f"Reading {args.input}...")
    try:
        with open(args.input, 'r') as f:
            for line in f:
                data = json.loads(line)
                stats["total"] += 1
                mode = data["mode"]
                status = data["status"]
                
                stats[mode]["total"] += 1
                if status == "success":
                    stats[mode]["success"] += 1
                    
                    # Analyze program
                    prog = data["program"]
                    if prog:
                        tokens = prog.split()
                        length = len(tokens)
                        stats["program_length_dist"][length] += 1
                        
                        for tok in tokens:
                            # Count everything as a primitive/token usage
                            stats["primitive_usage"][tok] += 1
                            stats["primitive_total_count"] += 1
                            
                elif status == "fail":
                    stats[mode]["fail"] += 1
                else:
                    stats[mode]["error"] += 1
                    
    except FileNotFoundError:
        print("Input file not found.")
        return

    # Post-process stats
    # Calculate proportions for primitives
    primitive_proportions = {}
    total_prims = stats["primitive_total_count"]
    if total_prims > 0:
        for prim, count in stats["primitive_usage"].items():
            primitive_proportions[prim] = count / total_prims

    final_stats = {
        "summary": {
            "total_tests": stats["total"],
            "exact_mode": stats["exact"],
            "moonshine_mode": stats["moonshine"]
        },
        "program_length_distribution": dict(stats["program_length_dist"]),
        "primitive_usage_counts": dict(stats["primitive_usage"]),
        "primitive_usage_proportions": primitive_proportions
    }

    print("Writing stats...")
    with open(args.output, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(json.dumps(final_stats["summary"], indent=2))

if __name__ == "__main__":
    main()

