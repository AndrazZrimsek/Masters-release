#!/usr/bin/env python3
"""
Filter non-redundant motifs from a MEME file using Tomtom self-comparison results.
Clusters motifs with q-value < 0.01 and outputs a new MEME file with one representative per cluster.

Usage:
  python filter_nonredundant_motifs.py \
    --tomtom_tsv Results/Enrichment/tomtom_self_512/tomtom.tsv \
    --meme_in Results/GradientPeaks/512/gradient_peak_motifs.meme \
    --meme_out Results/GradientPeaks/512/gradient_peak_motifs_nonredundant.meme

Requires: pandas, networkx, Biopython
"""
import argparse
import pandas as pd
import networkx as nx
import re
import sys
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Filter non-redundant motifs using Tomtom results.")
    parser.add_argument('--tomtom_tsv', required=True, help='Tomtom self-comparison tsv file')
    parser.add_argument('--meme_in', required=True, help='Input MEME motif file')
    parser.add_argument('--meme_out', required=True, help='Output MEME motif file (non-redundant)')
    parser.add_argument('--qval_thresh', type=float, default=0.01, help='Q-value threshold (default: 0.01)')
    return parser.parse_args()

def build_motif_graph(tomtom_tsv, qval_thresh):
    df = pd.read_csv(tomtom_tsv, sep='\t', comment='#')
    # Tomtom self-comparison: query_id, target_id, q-value
    G = nx.Graph()
    for _, row in df.iterrows():
        qval = float(row['q-value']) if 'q-value' in row else float(row['qvalue'])
        if qval < qval_thresh and row['Query_ID'] != row['Target_ID']:
            G.add_edge(row['Query_ID'], row['Target_ID'], qval=qval)
        G.add_node(row['Query_ID'])
        G.add_node(row['Target_ID'])
    return G

def select_representatives(G, qval_dict):
    # For each cluster, pick the motif with the lowest sum of q-values to others
    representatives = set()
    for component in nx.connected_components(G):
        if len(component) == 1:
            representatives.add(next(iter(component)))
        else:
            # For each motif, sum q-values to others in the cluster
            min_sum = float('inf')
            best = None
            for motif in component:
                s = sum(qval_dict.get((motif, other), 1.0) for other in component if other != motif)
                if s < min_sum:
                    min_sum = s
                    best = motif
            representatives.add(best)
    return representatives

def parse_meme_file(meme_path):
    # Parse MEME file into a dict: motif_id -> motif_text
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

if __name__ == "__main__":
    args = parse_args()
    # Read Tomtom tsv and build graph
    df = pd.read_csv(args.tomtom_tsv, sep='\t', comment='#')
    # Build q-value lookup
    qval_dict = {}
    for _, row in df.iterrows():
        qval = float(row['q-value']) if 'q-value' in row else float(row['qvalue'])
        qval_dict[(row['Query_ID'], row['Target_ID'])] = qval
    G = build_motif_graph(args.tomtom_tsv, args.qval_thresh)
    # Select representatives
    keep_ids = select_representatives(G, qval_dict)
    print(f"Keeping {len(keep_ids)} non-redundant motifs out of {len(G.nodes)} total.")
    # Parse MEME file
    header, motif_blocks = parse_meme_file(args.meme_in)
    # Write output MEME file
    write_meme_file(header, motif_blocks, keep_ids, args.meme_out)
    print(f"Wrote non-redundant motifs to {args.meme_out}")
