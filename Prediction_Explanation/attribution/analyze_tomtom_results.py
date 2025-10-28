#!/usr/bin/env python3
"""
Analyze Tomtom motif comparison results and motif file statistics.

Usage:
    python analyze_tomtom_results.py --meme_file <motif_file.meme> --tomtom_tsv <tomtom.tsv> --output_dir <output_dir>

- Computes and prints genetic statistics for all motifs in the MEME file (GC content, motif length, base composition).
- Separates motifs into three categories:
  1. Significant matches: q-value < threshold (default 0.5)
  2. Non-significant matches: q-value >= threshold but matches found
  3. Novel motifs: no matches found in the database at all
- Prints and writes all results to a single text file (tomtom_analysis_report.txt) in the output directory.
"""
import os
import argparse
import pandas as pd
from collections import defaultdict, Counter
import requests


def parse_meme_file(meme_path):
    # Parse MEME file into header and motif_blocks (motif_id -> list of lines)
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

import re
def parse_pwm_from_block(block):
    # Extract PWM from a motif block (list of lines)
    pwm_lines = []
    in_matrix = False
    for line in block:
        if line.strip().startswith('letter-probability matrix'):
            in_matrix = True
            continue
        if in_matrix:
            if line.strip() == '' or line.startswith('URL'):
                break
            pwm_lines.append(line.strip())
    pwm = []
    for l in pwm_lines:
        row = [float(x) for x in l.split()]
        pwm.append(row)
    return pwm

def compute_motif_stats_from_blocks(motif_blocks):
    stats = []
    for name, block in motif_blocks.items():
        pwm = parse_pwm_from_block(block)
        arr = pd.DataFrame(pwm, columns=['A','C','G','T'])
        L = arr.shape[0]
        gc_content = arr[['G','C']].sum().sum() / arr.sum().sum() if arr.sum().sum() > 0 else 0
        base_means = arr.mean()
        stats.append({
            'motif': name,
            'length': L,
            'GC_content': gc_content,
            'A_mean': base_means['A'],
            'C_mean': base_means['C'],
            'G_mean': base_means['G'],
            'T_mean': base_means['T']
        })
    return pd.DataFrame(stats)

def analyze_tomtom_tsv(tomtom_tsv, motif_names, qval_thresh=0.5):
    # Read the Tomtom TSV file
    df = pd.read_csv(tomtom_tsv, sep='\t', comment='#')
    # For each query, get the best (lowest q-value) match
    best_per_query = df.loc[df.groupby('Query_ID')['q-value'].idxmin()]
    significant = best_per_query[best_per_query['q-value'] < qval_thresh]
    non_significant = best_per_query[best_per_query['q-value'] >= qval_thresh]
    # Motifs in MEME file but not in summary (no matches at all) - these are truly novel
    novel = set(motif_names) - set(best_per_query['Query_ID'])
    return significant, non_significant, novel

def fetch_jaspar_info(matrix_id):
    url = f"https://jaspar2024.elixir.no/api/v1/matrix/{matrix_id}/"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {
                'matrix_id': matrix_id,
                'name': data.get('name', ''),
                'collection': data.get('collection', ''),
                'class': data.get('class', ''),
                'family': data.get('family', ''),
                'type': data.get('type', ''),
                'description': data.get('description', '')
            }
        else:
            return {'matrix_id': matrix_id, 'name': '', 'collection': '', 'class': '', 'family': '', 'type': '', 'description': ''}
    except Exception as e:
        return {'matrix_id': matrix_id, 'name': '', 'collection': '', 'class': '', 'family': '', 'type': '', 'description': ''}

def main():
    parser = argparse.ArgumentParser(description="Analyze Tomtom results and motif statistics.")
    parser.add_argument('--meme_file', required=True, help='MEME motif file (query motifs)')
    parser.add_argument('--tomtom_tsv', required=True, help='Tomtom results TSV file (e.g. tomtom.tsv)')
    parser.add_argument('--output_dir', required=True, help='Output directory for report')
    parser.add_argument('--qval_thresh', type=float, default=0.05, help='Q-value threshold for significant matches')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'tomtom_analysis_report.txt')
    with open(report_path, 'w') as f:
        def write(s):
            print(s)
            f.write(s + '\n')

        write('=== Motif File Statistics ===')
        header, motif_blocks = parse_meme_file(args.meme_file)
        stats_df = compute_motif_stats_from_blocks(motif_blocks)
        # Calculate mean values across all motifs
        mean_gc = stats_df['GC_content'].mean()
        mean_a = stats_df['A_mean'].mean()
        mean_c = stats_df['C_mean'].mean()
        mean_g = stats_df['G_mean'].mean()
        mean_t = stats_df['T_mean'].mean()
        write(f"Mean GC content: {mean_gc:.4f}")
        write(f"Mean A: {mean_a:.4f}")
        write(f"Mean C: {mean_c:.4f}")
        write(f"Mean G: {mean_g:.4f}")
        write(f"Mean T: {mean_t:.4f}")
        write('\n=== Tomtom Motif Match Summary ===')
        motif_names = list(motif_blocks.keys())
        significant, non_significant, novel = analyze_tomtom_tsv(args.tomtom_tsv, motif_names, args.qval_thresh)
        write(f"\nMotifs with significant matches (q < {args.qval_thresh}): {len(significant)}")
        if not significant.empty:
            # Fetch JASPAR info for each unique target
            targets = significant['Target_ID'].unique()
            jaspar_info = {mid: fetch_jaspar_info(mid) for mid in targets}
            # Add class/family columns to significant
            significant = significant.copy()
            significant['jaspar_class'] = significant['Target_ID'].map(lambda x: jaspar_info[x]['class'])
            significant['jaspar_family'] = significant['Target_ID'].map(lambda x: jaspar_info[x]['family'])
            significant['jaspar_name'] = significant['Target_ID'].map(lambda x: jaspar_info[x]['name'])
            write(significant.sort_values('p-value')[['Query_ID','Target_ID','p-value','jaspar_name','jaspar_class','jaspar_family']].to_string(index=False))
            # Count how many query motifs matched to each JASPAR motif
            target_counts = significant['Target_ID'].value_counts()
            write("\nCount of matched query motifs per JASPAR motif:")
            for target, count in target_counts.items():
                name = jaspar_info[target]['name'] if target in jaspar_info else ''
                write(f"  {target} ({name}): {count}")
            # Print class/family distribution, matching families to their classes
            # Build mapping: class -> Counter(families)
            class_family_counter = {}
            class_counter = Counter()
            # For each target, get class and family (may be list or str)
            for x in targets:
                cls = jaspar_info[x]['class']
                fam = jaspar_info[x]['family']
                # Normalize to list
                if not isinstance(cls, list):
                    cls = [cls] if cls else []
                if not isinstance(fam, list):
                    fam = [fam] if fam else []
                for c in cls:
                    if not c:
                        continue
                    class_counter[c] += 1
                    if c not in class_family_counter:
                        class_family_counter[c] = Counter()
                    for fa in fam:
                        if fa:
                            class_family_counter[c][fa] += 1
            write("\nDistribution of motif classes:")
            for c, v in class_counter.most_common():
                fams = class_family_counter[c]
                if fams:
                    fam_str = ', '.join(f"{fam}: {count}" for fam, count in fams.most_common())
                    write(f"  {c}: {v} ({fam_str})")
                else:
                    write(f"  {c}: {v}")
        write(f"\nMotifs with non-significant matches (q >= {args.qval_thresh}): {len(non_significant)}")
        if not non_significant.empty:
            write(non_significant[['Query_ID','p-value']].to_string(index=False))
        if novel:
            write(f"\nNovel motifs (no matches found in database): {', '.join(novel)}")
        write('\n=== End of Report ===')
    print(f"\nReport written to: {report_path}")

if __name__ == "__main__":
    main()
