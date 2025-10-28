#!/usr/bin/env python3
"""
Filter a MEME motif file to only include motifs significantly enriched in SEA results (qvalue <= 0.05).

Usage:
    python filter_significant_motifs.py --meme_file <motifs.meme> --sea_tsv <sea.tsv> --output_file <significantly_enriched_motifs.meme>
"""

import argparse
import pandas as pd
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Filter MEME file for significantly enriched motifs from one or more SEA results.")
    parser.add_argument('--meme_file', required=True, help='Input MEME motif file')
    parser.add_argument('--sea_tsv', nargs='+', help='One or more SEA results TSV file(s)')
    parser.add_argument('--sea_dir', help='Directory containing per-seed SEA results; the script will find all sea.tsv in subfolders')
    parser.add_argument('--output_file', required=True, help='Output MEME file for significant motifs')
    parser.add_argument('--ratio', type=float, default=2.5, help='Enrichment ratio threshold (default: 2.5)')
    parser.add_argument('--score_thresh', type=float, default=10, help='Score threshold for significant matches (default: 10)')
    parser.add_argument('--min_support', type=int, default=None, help='Minimum number of SEA runs a motif must pass thresholds in to be kept (default: majority of provided runs)')
    parser.add_argument('--aggregate', choices=['any','majority','all'], default='majority', help='Consensus rule across runs: any/majority/all (default: majority)')
    return parser.parse_args()

def parse_meme_file(meme_path):
    # Parse MEME file into header and motif_blocks (motif_id -> list of lines)
    from collections import defaultdict
    motifs = dict()
    with open(meme_path) as f:
        lines = f.readlines()
    header = []
    motif_blocks = defaultdict(list)
    current = None
    for line in lines:
        if line.startswith('MOTIF'):
            current = line.split()[1]
            motif_blocks[current].append(line)
        elif current:
            motif_blocks[current].append(line)
        else:
            header.append(line)
    return header, motif_blocks

def write_meme_file(header, motif_blocks, keep_ids, out_path):
    with open(out_path, 'w') as f:
        for line in header:
            f.write(line)
        for motif_id in keep_ids:
            for line in motif_blocks[motif_id]:
                f.write(line)

def main():
    args = parse_args()

    # Determine SEA files source: explicit list or directory discovery
    sea_files = []
    if args.sea_tsv:
        sea_files = list(args.sea_tsv)
        print(f"Using {len(sea_files)} SEA file(s) from --sea_tsv")
    elif args.sea_dir:
        import os
        for root, _, files in os.walk(args.sea_dir):
            for fname in files:
                if fname == 'sea.tsv':
                    sea_files.append(os.path.join(root, fname))
        sea_files.sort()
        print(f"Discovered {len(sea_files)} SEA file(s) under {args.sea_dir}")
    else:
        raise SystemExit("Error: provide either --sea_tsv files or --sea_dir directory")

    # Load SEA TSVs and compute per-run significant sets
    per_run_sig = []
    for tsv in sea_files:
        sea = pd.read_csv(tsv, sep='\t', comment='#')
        sig = sea[(sea['ENR_RATIO'] >= args.ratio) & (sea['SCORE_THR'] >= args.score_thresh)]
        per_run_sig.append(set(sig['ID']))
        print(f"{tsv}: {len(per_run_sig[-1])} significant motifs")

    n_runs = len(per_run_sig)
    if args.aggregate == 'any':
        sig_ids = set().union(*per_run_sig) if per_run_sig else set()
    elif args.aggregate == 'all':
        sig_ids = set.intersection(*per_run_sig) if per_run_sig else set()
    else:
        # majority or explicit min_support
        from collections import Counter
        counts = Counter()
        for s in per_run_sig:
            counts.update(s)
        min_support = args.min_support if args.min_support is not None else (n_runs // 2 + 1)
        sig_ids = {m for m, c in counts.items() if c >= min_support}
        print(f"Consensus rule: >= {min_support} of {n_runs} runs")

    print(f"Consensus significant motifs: {len(sig_ids)}")

    # Parse MEME file and write the subset
    header, motif_blocks = parse_meme_file(args.meme_file)
    keep_ids = [mid for mid in motif_blocks if mid in sig_ids]
    print(f"Writing {len(keep_ids)} motifs to {args.output_file}")
    write_meme_file(header, motif_blocks, keep_ids, args.output_file)

if __name__ == "__main__":
    main()
