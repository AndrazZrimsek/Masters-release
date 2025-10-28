#!/usr/bin/env python3
"""
Compare motif windows for top vs bottom accessions for a selected variable.

Steps:
 1. Load phenotype table and select top/bottom N validation accessions by variable value.
 2. Load weights_explanation.csv, pick one influential weight (abs / positive / negative mode).
 3. Load gradient_peaks_summary.csv, choose peak (best_zscore or first) for that (variable, snp_index, embedding_dim).
 4. Open the SNP alignment FASTA (snp{SNP_INDEX}_aligned.fasta) and extract a window of columns around the peak position.
     - Use max_grad_pos directly as the ungapped (0-based) index within the SNP window sequence.
    - Map ungapped base index to alignment column (accounting for gaps).
    - Extract +/- window_half columns around that alignment column.
 5. Build per-position base frequency matrices (A/C/G/T/N; '-' ignored) for top and bottom groups.
 6. Output FASTA of extracted motif windows, PFMs, consensus sequences, comparison deltas, and JSON metadata.

Outputs (prefix = --output-prefix):
  {prefix}_selected_weight.json
  {prefix}_selected_peak.json
  {prefix}_top_group_motifs.fasta
  {prefix}_bottom_group_motifs.fasta
  {prefix}_top_group_pfm.tsv
  {prefix}_bottom_group_pfm.tsv
  {prefix}_motif_comparison.tsv
  {prefix}_consensus_top.txt
  {prefix}_consensus_bottom.txt
  {prefix}_summary.txt

Assumptions:
 - Alignment FASTA headers exactly match accession IDs (string form). If not found, accession skipped with warning.
 - Alignment FASTA resides at: <align_dir>/snp{snp_index}_aligned.fasta
 - gradient_peaks_summary.csv has columns used: variable,snp_index,embedding_dim,region_start,max_grad_pos,zscore,peak_index
 - weights_explanation.csv has columns: variable,snp_index,embedding_dim,weight
 - Phenotype CSV contains target variable column and an ID column (specified by --phenotype-id-column)

Limitations:
 - If mapping base index to alignment column fails (e.g., due to offset beyond sequence), accession motif skipped.
 - Consensus: picks base with highest count (ties -> N). Gaps ignored in counts.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

BASES = ['A','C','G','T','N']


def parse_args():
    ap = argparse.ArgumentParser(description="Compare motif windows between top and bottom accessions for a variable.")
    ap.add_argument('--variable', required=True, help='Target variable name')
    ap.add_argument('--phenotypes', required=True, help='Phenotype CSV file')
    ap.add_argument('--phenotype-id-column', required=True, help='Column name for accession IDs in phenotype table')
    ap.add_argument('--val-ids', required=True, help='Validation accession IDs file (whitespace or newline separated)')
    ap.add_argument('--weights', required=True, help='weights_explanation.csv path')
    ap.add_argument('--peaks', required=True, help='gradient_peaks_summary.csv path')
    ap.add_argument('--align-dir', required=True, help='Directory with snp{index}_aligned.fasta files')
    ap.add_argument('--weight-mode', choices=['abs','positive','negative'], default='abs', help='How to select influential weight (default: abs)')
    ap.add_argument('--peak-mode', choices=['best_zscore','first','all'], default='best_zscore', help='Peak selection: best_zscore|first|all (all = process every peak)')
    ap.add_argument('--window-half', type=int, default=5, help='Half window size (total window = 2*half+1)')
    ap.add_argument('--top-n', type=int, default=10, help='Number of top accessions (default 10)')
    ap.add_argument('--bottom-n', type=int, default=10, help='Number of bottom accessions (default 10)')
    ap.add_argument('--output-prefix', required=True, help='Output file prefix (directory created if needed)')
    ap.add_argument('--num-weights', type=int, default=1, help='Number of top weights to analyze (default 1)')
    return ap.parse_args()


def load_validation_ids(path: str) -> List[str]:
    with open(path, "r") as f:
        val_ids = f.read().split()
        # Keep original tokens; attempt int casting later only if needed.
    return [v.strip() for v in val_ids if v.strip()]


def select_accessions(phenos: pd.DataFrame, var: str, id_col: str, val_ids: List[str], top_n: int, bottom_n: int):
    # Normalize both phenotype IDs and validation IDs to strings for consistent matching.
    phenos['_tmp_id_str'] = phenos[id_col].astype(str)
    val_ids_str = set(str(v) for v in val_ids)
    sub = phenos[phenos['_tmp_id_str'].isin(val_ids_str)].copy()
    if var not in sub.columns:
        raise ValueError(f"Variable '{var}' not in phenotype table columns: {list(sub.columns)}")
    sub = sub[[id_col, '_tmp_id_str', var]].dropna()
    if sub.empty:
        raise ValueError("No phenotype rows after filtering validation IDs.")
    sub_sorted = sub.sort_values(var)
    bottom = sub_sorted.head(bottom_n)
    top = sub_sorted.tail(top_n)
    # ensure uniqueness
    bottom_ids = bottom['_tmp_id_str'].tolist()
    top_ids = top['_tmp_id_str'].tolist()
    return top_ids, bottom_ids, sub.shape[0]
    
def pick_weights(weights_df: pd.DataFrame, variable: str, mode: str, n: int) -> List[Dict]:
    sub = weights_df[weights_df['variable'] == variable].copy()
    if sub.empty:
        raise ValueError(f"No weights for variable {variable}")
    if mode == 'abs':
        sub['score'] = sub['weight'].abs()
        ordered = sub.sort_values('score', ascending=False)
    elif mode == 'positive':
        sub = sub[sub['weight'] > 0]
        if sub.empty:
            raise ValueError('No positive weights available')
        ordered = sub.sort_values('weight', ascending=False)
    else:  # negative
        sub = sub[sub['weight'] < 0]
        if sub.empty:
            raise ValueError('No negative weights available')
        ordered = sub.sort_values('weight', ascending=True)
    ordered = ordered.head(n)
    metas = []
    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        metas.append({
            'rank': rank,
            'variable': row['variable'],
            'snp_index': int(row['snp_index']),
            'embedding_dim': int(row['embedding_dim']),
            'weight': float(row['weight'])
        })
    return metas

def pick_peak(peaks_df: pd.DataFrame, variable: str, snp_index: int, embedding_dim: int, mode: str) -> Dict:
    sub = peaks_df[(peaks_df['variable'] == variable) &
                   (peaks_df['snp_index'] == snp_index) &
                   (peaks_df['embedding_dim'] == embedding_dim)].copy()
    if sub.empty:
        raise ValueError('No peaks match selected weight')
    if mode == 'best_zscore':
        if 'zscore' not in sub.columns:
            raise ValueError('zscore column missing for best_zscore mode')
        row = sub.sort_values('zscore', ascending=False).iloc[0]
    elif mode == 'first':
        row = sub.iloc[0]
    else:
        # for 'all' we won't call pick_peak, but keep for completeness
        row = sub.iloc[0]
    return {k: (int(row[k]) if k in ['snp_index','embedding_dim','region_start','region_end','max_grad_pos'] else row[k])
            for k in row.index if k in ['variable','snp_index','embedding_dim','region_start','region_end','max_grad_pos','zscore','peak_index','chromosome']}

def parse_alignment_fasta(path: str) -> Dict[str,str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Alignment FASTA not found: {path}")
    seqs = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                current = line[1:].strip()
                seqs[current] = []
            else:
                if current is None:
                    continue
                seqs[current].append(line.upper())
    return {k: ''.join(v) for k,v in seqs.items()}

def ungapped_index_to_alignment_col(seq: str, ungapped_index: int) -> int:
    count = -1
    for i, ch in enumerate(seq):
        if ch != '-':
            count += 1
        if count == ungapped_index:
            return i
    return -1

def extract_window(seq: str, center_col: int, half: int) -> str:
    start = center_col - half
    end = center_col + half
    if start < 0 or center_col < 0 or end >= len(seq):
        # allow partial clipping instead of empty? Keep strict for now: clip
        start = max(0, start)
        end = min(len(seq)-1, end)
    return seq[start:end+1]

def build_pfm(seqs: List[str]) -> Tuple[pd.DataFrame,str]:
    if not seqs:
        return pd.DataFrame(), ''
    max_len = max(len(s) for s in seqs)
    counts = {b: [0]*max_len for b in BASES}
    for s in seqs:
        for i, ch in enumerate(s):
            base = ch.upper()
            if base == '-':
                continue
            if base not in BASES:
                base = 'N'
            counts[base][i] += 1
    rows = []
    consensus_chars = []
    for i in range(max_len):
        row = {'position': i}
        max_count = 0
        col_counts = {}
        for b in BASES:
            c = counts[b][i]
            row[b] = c
            col_counts[b] = c
            if c > max_count:
                max_count = c
        rows.append(row)
        tops = [b for b,c in col_counts.items() if c == max_count and max_count > 0]
        if len(tops) == 1:
            consensus_chars.append(tops[0])
        elif len(tops) == 0:
            consensus_chars.append('N')
        else:
            consensus_chars.append('N')
    df = pd.DataFrame(rows)
    consensus = ''.join(consensus_chars)
    return df, consensus


def write_fasta(path: str, records: Dict[str,str]):
    with open(path, 'w') as f:
        for rid, seq in records.items():
            f.write(f'>{rid}\n')
            # wrap at 80
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')


def compare_top_bottom(top_seqs: List[str], bottom_seqs: List[str]) -> pd.DataFrame:
    if not top_seqs or not bottom_seqs:
        return pd.DataFrame()
    max_len = max(max(len(s) for s in top_seqs), max(len(s) for s in bottom_seqs))
    def freq_matrix(group):
        fm = {b: [0]*max_len for b in BASES}
        for s in group:
            for i in range(max_len):
                if i >= len(s):
                    continue
                ch = s[i]
                if ch == '-':
                    continue
                base = ch.upper()
                if base not in BASES:
                    base = 'N'
                fm[base][i] += 1
        return fm
    top_fm = freq_matrix(top_seqs)
    bottom_fm = freq_matrix(bottom_seqs)
    rows = []
    for i in range(max_len):
        row = {'position': i}
        for b in BASES:
            row[f'top_{b}'] = top_fm[b][i]
            row[f'bottom_{b}'] = bottom_fm[b][i]
            row[f'delta_{b}'] = top_fm[b][i] - bottom_fm[b][i]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    phenos = pd.read_csv(args.phenotypes)
    val_ids = load_validation_ids(args.val_ids)
    weights_df = pd.read_csv(args.weights)
    peaks_df = pd.read_csv(args.peaks)

    top_ids, bottom_ids, _ = select_accessions(phenos, args.variable, args.phenotype_id_column, val_ids, args.top_n, args.bottom_n)
    weights_list = pick_weights(weights_df, args.variable, args.weight_mode, args.num_weights)

    # Cache alignments per SNP index
    alignment_cache: Dict[int, Dict[str,str]] = {}

    aggregated_peaks_rows = []
    selected_weights_rows = []

    multi = len(weights_list) > 1
    if multi:
        base_folder = args.output_prefix
        os.makedirs(base_folder, exist_ok=True)
        # write accession lists once at top-level
        with open(os.path.join(base_folder, 'top_accessions.txt'), 'w') as f:
            for aid in top_ids:
                f.write(str(aid) + '\n')
        with open(os.path.join(base_folder, 'bottom_accessions.txt'), 'w') as f:
            for aid in bottom_ids:
                f.write(str(aid) + '\n')
    else:
        base_folder = os.path.dirname(args.output_prefix) if os.path.dirname(args.output_prefix) else ''

    def load_alignment_for_snp(snp_index: int):
        if snp_index in alignment_cache:
            return alignment_cache[snp_index]
        path = os.path.join(args.align_dir, f"snp{snp_index}_aligned.fasta")
        try:
            aln = parse_alignment_fasta(path)
        except FileNotFoundError as e:
            print(str(e))
            aln = {}
        alignment_cache[snp_index] = aln
        return aln

    def map_center_col(alignment: Dict[str,str], base_index: int, probe_ids):
        center_col = -1
        probe_used = None
        for pid in probe_ids:
            pid_key = str(pid)
            if pid_key not in alignment:
                continue
            seq = alignment[pid_key]
            ungapped_len = sum(1 for c in seq if c != '-')
            if ungapped_len == 0:
                continue
            adj_index = base_index if base_index < ungapped_len else ungapped_len - 1
            col = ungapped_index_to_alignment_col(seq, adj_index)
            if col != -1:
                center_col = col
                probe_used = pid_key
                break
        return center_col, probe_used

    def collect_windows(alignment: Dict[str,str], center_col: int, group_ids):
        motifs = {}
        for gid in group_ids:
            gid_key = str(gid)
            if gid_key not in alignment:
                continue
            seq = alignment[gid_key]
            window = extract_window(seq, center_col, args.window_half)
            if window:
                motifs[gid_key] = window
        return motifs

    for wmeta in weights_list:
        selected_weights_rows.append({k: wmeta[k] for k in ['rank','variable','snp_index','embedding_dim','weight']})
        # Set output prefix for this weight
        if multi:
            weight_prefix_dir = os.path.join(base_folder, f"w{wmeta['rank']}_snp{wmeta['snp_index']}_dim{wmeta['embedding_dim']}")
            os.makedirs(weight_prefix_dir, exist_ok=True)
            oprefix = os.path.join(weight_prefix_dir, 'result') if args.peak_mode != 'all' else weight_prefix_dir
        else:
            oprefix = args.output_prefix

        alignment = load_alignment_for_snp(wmeta['snp_index'])
        if not alignment:
            print(f"Warning: alignment missing for SNP {wmeta['snp_index']}; skipping weight rank {wmeta['rank']}")
            continue

        if args.peak_mode != 'all':
            peak_meta = pick_peak(peaks_df, wmeta['variable'], wmeta['snp_index'], wmeta['embedding_dim'], args.peak_mode)
            base_index = peak_meta['max_grad_pos']
            if base_index < 0:
                base_index = 0
            probe_ids = top_ids + bottom_ids
            center_col, probe_used = map_center_col(alignment, base_index, probe_ids)
            mapped = center_col != -1
            top_motifs = collect_windows(alignment, center_col, top_ids) if mapped else {}
            bottom_motifs = collect_windows(alignment, center_col, bottom_ids) if mapped else {}
            top_pfm_df, top_consensus = build_pfm(list(top_motifs.values()))
            bottom_pfm_df, bottom_consensus = build_pfm(list(bottom_motifs.values()))
            comparison_df = compare_top_bottom(list(top_motifs.values()), list(bottom_motifs.values())) if mapped else pd.DataFrame()
            # Outputs
            def save_json(obj, suffix):
                with open(f"{oprefix}_{suffix}.json", 'w') as f:
                    json.dump(obj, f, indent=2)
            save_json(wmeta, 'selected_weight')
            save_json(peak_meta, 'selected_peak')
            if mapped:
                write_fasta(f"{oprefix}_top_group_motifs.fasta", top_motifs)
                write_fasta(f"{oprefix}_bottom_group_motifs.fasta", bottom_motifs)
                if not top_pfm_df.empty:
                    top_pfm_df.to_csv(f"{oprefix}_top_group_pfm.tsv", sep='\t', index=False)
                if not bottom_pfm_df.empty:
                    bottom_pfm_df.to_csv(f"{oprefix}_bottom_group_pfm.tsv", sep='\t', index=False)
                if not comparison_df.empty:
                    comparison_df.to_csv(f"{oprefix}_motif_comparison.tsv", sep='\t', index=False)
                with open(f"{oprefix}_consensus_top.txt", 'w') as f:
                    f.write(top_consensus + '\n')
                with open(f"{oprefix}_consensus_bottom.txt", 'w') as f:
                    f.write(bottom_consensus + '\n')
            # Accessions (single-weight non-multi kept at top-level previously; here always per-weight if multi)
            if not multi:
                with open(f"{oprefix}_top_accessions.txt", 'w') as f:
                    for aid in top_ids:
                        f.write(str(aid) + '\n')
                with open(f"{oprefix}_bottom_accessions.txt", 'w') as f:
                    for aid in bottom_ids:
                        f.write(str(aid) + '\n')
            peak_row = {
                'rank': wmeta['rank'],
                'variable': wmeta['variable'],
                'snp_index': wmeta['snp_index'],
                'embedding_dim': wmeta['embedding_dim'],
                'weight': wmeta['weight'],
                'peak_index': peak_meta.get('peak_index', 1),
                'max_grad_pos': peak_meta['max_grad_pos'],
                'zscore': peak_meta.get('zscore', np.nan),
                'mapped': mapped
            }
            aggregated_peaks_rows.append(peak_row)
            # Summary
            summary_lines = [
                f"Weight rank: {wmeta['rank']}",
                f"Variable: {wmeta['variable']}",
                f"Weight mode: {args.weight_mode}",
                f"Peak mode: {args.peak_mode}",
                f"SNP index: {wmeta['snp_index']} embedding_dim: {wmeta['embedding_dim']} weight: {wmeta['weight']}",
                f"Peak max_grad_pos: {peak_meta['max_grad_pos']} alignment_col: {center_col if mapped else 'NA'}",
                f"Mapped: {mapped}",
                f"Top group: {len(top_motifs)} Bottom group: {len(bottom_motifs)}",
                f"Window half-size: {args.window_half} total length (expected): {2*args.window_half + 1}",
                f"Consensus top: {top_consensus if mapped else ''}",
                f"Consensus bottom: {bottom_consensus if mapped else ''}",
            ]
            with open(f"{oprefix}_summary.txt", 'w') as f:
                f.write('\n'.join(summary_lines) + '\n')
        else:
            # All peaks for this weight
            peak_folder = oprefix  # already a directory if multi; if single, treat prefix as directory
            if not multi:
                os.makedirs(peak_folder, exist_ok=True)
                # accessions
                with open(os.path.join(peak_folder, 'top_accessions.txt'), 'w') as f:
                    for aid in top_ids:
                        f.write(str(aid) + '\n')
                with open(os.path.join(peak_folder, 'bottom_accessions.txt'), 'w') as f:
                    for aid in bottom_ids:
                        f.write(str(aid) + '\n')
            subset = peaks_df[(peaks_df['variable'] == wmeta['variable']) &
                              (peaks_df['snp_index'] == wmeta['snp_index']) &
                              (peaks_df['embedding_dim'] == wmeta['embedding_dim'])].copy()
            if subset.empty:
                print(f"WARNING: No peaks for weight rank {wmeta['rank']}")
                continue
            if 'peak_index' in subset.columns:
                subset['peak_index'] = pd.to_numeric(subset['peak_index'], errors='coerce')
                subset = subset.sort_values('peak_index')
            summary_accum = []
            with open(os.path.join(peak_folder, 'selected_weight.json'), 'w') as f:
                json.dump(wmeta, f, indent=2)
            probe_ids = top_ids + bottom_ids
            processed = 0
            for _, row in subset.iterrows():
                peak_idx = int(row.get('peak_index', processed + 1) if not pd.isna(row.get('peak_index', np.nan)) else processed + 1)
                base_index = int(row['max_grad_pos'])
                center_col, probe_used = map_center_col(alignment, base_index, probe_ids)
                mapped = center_col != -1
                if not mapped:
                    summary_accum.append(f"peak{peak_idx}: FAILED mapping (base_index={base_index})")
                    aggregated_peaks_rows.append({
                        'rank': wmeta['rank'],
                        'variable': wmeta['variable'],
                        'snp_index': wmeta['snp_index'],
                        'embedding_dim': wmeta['embedding_dim'],
                        'weight': wmeta['weight'],
                        'peak_index': peak_idx,
                        'max_grad_pos': base_index,
                        'zscore': row.get('zscore', np.nan),
                        'mapped': False
                    })
                    continue
                top_motifs = collect_windows(alignment, center_col, top_ids)
                bottom_motifs = collect_windows(alignment, center_col, bottom_ids)
                top_pfm_df, top_consensus = build_pfm(list(top_motifs.values()))
                bottom_pfm_df, bottom_consensus = build_pfm(list(bottom_motifs.values()))
                comparison_df = compare_top_bottom(list(top_motifs.values()), list(bottom_motifs.values()))
                peak_meta = {
                    'peak_index': peak_idx,
                    'max_grad_pos': int(row['max_grad_pos']),
                    'zscore': float(row.get('zscore', np.nan))
                }
                with open(os.path.join(peak_folder, f"peak{peak_idx}_selected_peak.json"), 'w') as f:
                    json.dump(peak_meta, f, indent=2)
                write_fasta(os.path.join(peak_folder, f"peak{peak_idx}_top_group_motifs.fasta"), top_motifs)
                write_fasta(os.path.join(peak_folder, f"peak{peak_idx}_bottom_group_motifs.fasta"), bottom_motifs)
                if not top_pfm_df.empty:
                    top_pfm_df.to_csv(os.path.join(peak_folder, f"peak{peak_idx}_top_group_pfm.tsv"), sep='\t', index=False)
                if not bottom_pfm_df.empty:
                    bottom_pfm_df.to_csv(os.path.join(peak_folder, f"peak{peak_idx}_bottom_group_pfm.tsv"), sep='\t', index=False)
                if not comparison_df.empty:
                    comparison_df.to_csv(os.path.join(peak_folder, f"peak{peak_idx}_motif_comparison.tsv"), sep='\t', index=False)
                with open(os.path.join(peak_folder, f"peak{peak_idx}_consensus_top.txt"), 'w') as f:
                    f.write(top_consensus + '\n')
                with open(os.path.join(peak_folder, f"peak{peak_idx}_consensus_bottom.txt"), 'w') as f:
                    f.write(bottom_consensus + '\n')
                summary_accum.append(
                    f"peak{peak_idx}: base_index={base_index} center_col={center_col} top={len(top_motifs)} bottom={len(bottom_motifs)} consensus_top={top_consensus} consensus_bottom={bottom_consensus}")
                aggregated_peaks_rows.append({
                    'rank': wmeta['rank'],
                    'variable': wmeta['variable'],
                    'snp_index': wmeta['snp_index'],
                    'embedding_dim': wmeta['embedding_dim'],
                    'weight': wmeta['weight'],
                    'peak_index': peak_idx,
                    'max_grad_pos': base_index,
                    'zscore': row.get('zscore', np.nan),
                    'mapped': True
                })
                processed += 1
            summary_path = os.path.join(peak_folder, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Weight rank: {wmeta['rank']} Variable: {wmeta['variable']}\n")
                f.write(f"Weight mode: {args.weight_mode} Peak mode: all\n")
                f.write(f"SNP index: {wmeta['snp_index']} embedding_dim: {wmeta['embedding_dim']} weight: {wmeta['weight']}\n")
                f.write(f"Window half-size: {args.window_half} total length (expected): {2*args.window_half + 1}\n")
                f.write(f"Top group requested: {args.top_n} Bottom group requested: {args.bottom_n}\n")
                f.write("\n".join(summary_accum) + '\n')
            print(f"Completed weight rank {wmeta['rank']} (all peaks). Summary: {summary_path}")

    # Write aggregated CSVs
    if selected_weights_rows:
        if multi:
            sel_path = os.path.join(base_folder, 'selected_weights.csv')
        else:
            sel_path = f"{args.output_prefix}_selected_weights.csv"
        pd.DataFrame(selected_weights_rows).to_csv(sel_path, index=False)
    if aggregated_peaks_rows:
        if multi:
            peaks_path = os.path.join(base_folder, 'relevant_peaks.csv')
        else:
            peaks_path = f"{args.output_prefix}_relevant_peaks.csv"  # already created earlier for single but this will overwrite identically
        pd.DataFrame(aggregated_peaks_rows).to_csv(peaks_path, index=False)
    print('Done processing weights.')


if __name__ == '__main__':
    main()
