#!/usr/bin/env python3
import argparse
import warnings  # still used for any future notices (kept, though per-accession variants/deltas removed)
import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
MUTATION_FILE_RE = re.compile(r"w(?P<rank>\d+)_snp(?P<snp>\d+)_dim(?P<dim>\d+)")


def load_embeddings(embeddings_pkl: str) -> Dict[str, np.ndarray]:
    with open(embeddings_pkl, 'rb') as f:
        data = pickle.load(f)
    # Expect accession -> matrix (num_snps x embed_dim) OR list convertible
    cleaned = {}
    for acc, val in data.items():
        arr = np.array(val)
        if arr.ndim == 1:
            raise ValueError("Embedding per accession must be 2D (num_snps x embed_dim)")
        cleaned[str(acc)] = arr
    return cleaned


def flatten_embedding(mat: np.ndarray) -> np.ndarray:
    return mat.reshape(-1)


def load_model(model_pkl: str):
    with open(model_pkl, 'rb') as f:
        obj = pickle.load(f)
    required_keys = ['model', 'scaler', 'env_vars']
    for k in required_keys:
        if k not in obj:
            raise KeyError(f"Model pickle missing key '{k}'")
    return obj


def list_snp_dirs(mutations_root: str) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = {}
    for entry in Path(mutations_root).iterdir():
        if not entry.is_dir():
            continue
        m = MUTATION_FILE_RE.match(entry.name)
        if not m:
            continue
        snp_index = int(m.group('snp'))
        mapping.setdefault(snp_index, []).append(entry)
    return mapping


def load_mutation_npz(path: Path):
    return np.load(path, allow_pickle=True)


def enumerate_single_mutations(npz_obj) -> List[Tuple[int, str, str, np.ndarray]]:
    positions = npz_obj['positions']
    bases = npz_obj['bases']
    alts_list = npz_obj['alts']
    muts = npz_obj['mutations']
    out = []
    for i in range(len(positions)):
        pos = int(positions[i])
        ref = bases[i].item() if hasattr(bases[i], 'item') else bases[i]
        alts = alts_list[i]
        row = muts[i]
        for j, alt in enumerate(alts):
            emb = row[j]
            if np.isnan(emb).any():
                continue
            out.append((pos, ref, alt, emb))
    return out


def enumerate_combo_mutations(npz_obj) -> List[Tuple[List[int], List[str], List[str], np.ndarray]]:
    if 'combo_mutations' not in npz_obj.files:
        return []
    combo_mutations = npz_obj['combo_mutations']
    starts = npz_obj['combo_variant_starts']
    lengths = npz_obj['combo_variant_lengths']
    flat_pos = npz_obj['combo_positions_flat']
    flat_ref = npz_obj['combo_ref_bases_flat']
    flat_alt = npz_obj['combo_alt_bases_flat']
    out = []
    for i in range(len(combo_mutations)):
        l = int(lengths[i])
        s = int(starts[i])
        pos_slice = flat_pos[s:s+l].tolist()
        ref_slice = [b.item() if hasattr(b, 'item') else b for b in flat_ref[s:s+l]]
        alt_slice = [b.item() if hasattr(b, 'item') else b for b in flat_alt[s:s+l]]
        out.append((pos_slice, ref_slice, alt_slice, combo_mutations[i]))
    return out


def apply_snp_mutation(original_matrix: np.ndarray, snp_index: int, new_row: np.ndarray) -> np.ndarray:
    mutated = original_matrix.copy()
    if snp_index >= mutated.shape[0]:
        raise IndexError(f"snp_index {snp_index} out of range for embedding shape {mutated.shape}")
    mutated[snp_index] = new_row
    return mutated


def predict_target(model_obj, X: np.ndarray, target_idx: int) -> np.ndarray:
    scaler = model_obj['scaler']
    models = model_obj['model']  # list of estimators per variable
    Xs = scaler.transform(X)
    # If models is list, pick target_idx model
    if isinstance(models, list):
        return models[target_idx].predict(Xs)
    # else assume multioutput wrapper
    preds = models.predict(Xs)
    return preds[:, target_idx]


def batch_predict(model_obj, rows: np.ndarray, target_idx: int):
    scaler = model_obj['scaler']
    models = model_obj['model']
    Xs = scaler.transform(rows)
    if isinstance(models, list):
        return models[target_idx].predict(Xs)
    preds = models.predict(Xs)
    return preds[:, target_idx]


def main():
    parser = argparse.ArgumentParser(description="Analyze SNP mutation embedding effects per accession (best single/combo + optional cumulative multi-SNP application)")
    parser.add_argument('--embeddings-pkl', required=True, help='Pickle with accession -> (num_snps x embed_dim)')
    parser.add_argument('--mutations-root', required=True, help='Root directory containing mutation result folders w*_snp*_dim*')
    parser.add_argument('--model-pkl', required=True, help='Trained model pickle path (with scaler, model, env_vars)')
    parser.add_argument('--target-variable', required=True, help='Target variable name to analyze (must be in env_vars)')
    parser.add_argument('--output-dir', required=True, help='Directory to store analysis outputs')
    parser.add_argument('--limit-snps', type=int, default=0, help='Optional limit of SNPs processed (0 = all)')
    parser.add_argument('--accessions-file', help='Optional file listing accessions (one per line) to restrict analysis to (must match keys in embeddings)')
    parser.add_argument('--top-n-snps', type=int, nargs='+', default=[0], help='List of top-N values for cumulative single-SNP analysis (e.g., --top-n-snps 5 10 15). Each value creates a subdirectory. 0 = use all positive singles.')
    parser.add_argument('--top-n-combined', type=int, nargs='+', default=[0], help='List of top-N values for combined cumulative (singles+combos) analysis. 0 = use all positive variants. Must match length of --top-n-snps or be a single value to apply to all.')
    parser.add_argument('--effect-direction', choices=['positive', 'negative'], default='positive', help='Direction of effect to analyze: "positive" for mutations that increase the target variable (default), "negative" for mutations that decrease it.')
    parser.add_argument('--no-save-intermediates', action='store_true', help='Disable writing per-accession intermediate NPZ files (enabled by default).')
    parser.add_argument('--intermediates-dir', help='Directory for per-accession intermediate NPZ files (default: <output-dir>/intermediates)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Validate and process top-n arguments
    top_n_snps_list = args.top_n_snps
    top_n_combined_list = args.top_n_combined
    
    # If top_n_combined is a single value, replicate it for all top_n_snps values
    if len(top_n_combined_list) == 1:
        top_n_combined_list = top_n_combined_list * len(top_n_snps_list)
    elif len(top_n_combined_list) != len(top_n_snps_list):
        raise ValueError(f"--top-n-combined must be either a single value or match length of --top-n-snps. "
                        f"Got {len(top_n_combined_list)} values for --top-n-combined and {len(top_n_snps_list)} for --top-n-snps")
    
    print(f"[Config] Will compute results for {len(top_n_snps_list)} top-N scenario(s):")
    for i, (tn_snps, tn_comb) in enumerate(zip(top_n_snps_list, top_n_combined_list)):
        label_snps = "all" if tn_snps == 0 else str(tn_snps)
        label_comb = "all" if tn_comb == 0 else str(tn_comb)
        print(f"  Scenario {i+1}: top_n_snps={label_snps}, top_n_combined={label_comb}")

    # Determine effect direction: positive means delta > 0, negative means delta < 0
    seeking_positive = (args.effect_direction == 'positive')
    effect_label = 'positive' if seeking_positive else 'negative'
    print(f"[Config] Analyzing mutations with {effect_label} effect on {args.target_variable}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    print('[Load] Embeddings...')
    embeddings = load_embeddings(args.embeddings_pkl)
    print(f"[Load] Accessions (raw): {len(embeddings)}")

    if args.accessions_file:
        with open(args.accessions_file) as f:
            wanted = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
        before = len(embeddings)
        embeddings = {k: v for k, v in embeddings.items() if k in wanted}
        missing = sorted(set(wanted) - set(embeddings.keys()))
        print(f"[Filter] Accessions file specifies {len(wanted)}; kept {len(embeddings)} present (dropped {before - len(embeddings)})")
        if missing:
            print(f"[Filter] Warning: {len(missing)} missing (showing up to 10): {missing[:10]}")

    # Load model
    print('[Load] Model...')
    model_obj = load_model(args.model_pkl)
    env_vars = model_obj['env_vars']
    if args.target_variable not in env_vars:
        raise ValueError(f"Target variable {args.target_variable} not in model env_vars: {env_vars}")
    target_idx = env_vars.index(args.target_variable)
    print(f"[Info] Target '{args.target_variable}' index {target_idx}")

    accession_list = sorted(embeddings.keys())
    if not accession_list:
        print('[Abort] No accessions after filtering.')
        return

    # Baseline flattened embeddings & predictions
    baseline_flat = np.stack([flatten_embedding(embeddings[a]) for a in accession_list])
    baseline_preds = predict_target(model_obj, baseline_flat, target_idx)
    baseline_pred_map = {acc: pred for acc, pred in zip(accession_list, baseline_preds)}

    # Discover SNP dirs
    snp_dirs = list_snp_dirs(args.mutations_root)
    snp_indices = sorted(snp_dirs.keys())
    if args.limit_snps > 0:
        snp_indices = snp_indices[:args.limit_snps]
    print(f"[Scan] Processing {len(snp_indices)} SNPs")

    # Tracking structures
    best_single_by_accession: Dict[str, dict] = {}
    best_combo_by_accession: Dict[str, dict] = {}

    # For cumulative multi-SNP simulation (store all variants with desired effect direction with their mutated row)
    filtered_variants_by_accession: Dict[str, List[dict]] = {}
    # For combined cumulative simulation including combos
    filtered_combo_variants_by_accession: Dict[str, List[dict]] = {}

    single_records_all: List[dict] = []  # optional detailed per-mutation (single) if needed later
    combo_records_all: List[dict] = []   # optional detailed per-mutation (combo)

    # Iterate SNPs
    for snp_index in tqdm(snp_indices, desc='SNPs', unit='snp'):
        dirs = snp_dirs[snp_index]
        # Map accession -> mutation npz file(s)
        acc_to_npz = {}
        for d in dirs:
            for file in d.glob('accession_*_mutations.npz'):
                # Derive accession id from filename
                acc = file.stem.split('_', 1)[1].rsplit('_', 1)[0].replace('_mutations', '')
                acc_to_npz.setdefault(acc, []).append(file)
        # Restrict to accessions we have embeddings for
        accessions_available = [a for a in acc_to_npz if a in embeddings]
        if not accessions_available:
            continue

        row_start = snp_index * model_obj['scaler'].n_features_in_
        row_end = row_start + model_obj['scaler'].n_features_in_

        # Process singles per accession with batching
        singles_batch_entries = []  # (acc, pos, ref, alt, embedding_row)
        combos_batch_entries = []   # (acc, positions_list, ref_list, alt_list, combo_emb)

        for acc in accessions_available:
            npz_path = acc_to_npz[acc][0]  # first file
            npz_obj = load_mutation_npz(npz_path)
            emb_matrix = embeddings[acc]
            base_pred = baseline_pred_map[acc]
            base_flat_row = flatten_embedding(emb_matrix)

            # SINGLE mutations
            singles = enumerate_single_mutations(npz_obj)
            for (pos, ref_base, alt_base, new_row) in singles:
                mutated = apply_snp_mutation(emb_matrix, snp_index, new_row)
                mut_flat = flatten_embedding(mutated)
                mut_pred = predict_target(model_obj, np.stack([mut_flat,]), target_idx)[0]
                delta = mut_pred - base_pred
                # Record only if delta > 0 or if no positive exists yet (we keep max even if negative)
                prev = best_single_by_accession.get(acc)
                accept = False
                if prev is None:
                    accept = True
                else:
                    # Prefer positive improvements; if existing best is negative and this is positive -> accept
                    if prev['delta_pred'] <= 0 < delta:
                        accept = True
                    elif (delta > 0 and prev['delta_pred'] > 0 and delta > prev['delta_pred']):
                        accept = True
                    elif (prev['delta_pred'] <= 0 and delta <= 0 and delta > prev['delta_pred']):
                        # both negative: keep the less negative (closer to zero)
                        accept = True
                if accept:
                    best_single_by_accession[acc] = {
                        'accession': acc,
                        'snp_index': snp_index,
                        'position': pos,
                        'ref_base': ref_base,
                        'alt_base': alt_base,
                        'delta_pred': delta
                    }
                # Keep variants with desired effect direction for cumulative simulation
                if (seeking_positive and delta > 0) or (not seeking_positive and delta < 0):
                    filtered_variants_by_accession.setdefault(acc, []).append({
                        'accession': acc,
                        'snp_index': snp_index,
                        'position': pos,
                        'ref_base': ref_base,
                        'alt_base': alt_base,
                        'delta_pred': delta,
                        'new_row': new_row  # store embedding row for later application
                    })
                # Store detailed record (optional downstream)
                single_records_all.append({
                    'accession': acc,
                    'snp_index': snp_index,
                    'position': pos,
                    'ref_base': ref_base,
                    'alt_base': alt_base,
                    'delta_pred': delta
                })

            # COMBO mutations (size > 1)
            if 'combo_mutations' in npz_obj.files:
                combos = enumerate_combo_mutations(npz_obj)
                for (pos_list, ref_list, alt_list, combo_emb) in combos:
                    if len(pos_list) <= 1:
                        continue  # treat as single above
                    mutated = apply_snp_mutation(emb_matrix, snp_index, combo_emb)
                    mut_flat = flatten_embedding(mutated)
                    mut_pred = predict_target(model_obj, np.stack([mut_flat,]), target_idx)[0]
                    delta = mut_pred - base_pred
                    prev = best_combo_by_accession.get(acc)
                    accept = False
                    if prev is None:
                        accept = True
                    else:
                        if prev['delta_pred'] <= 0 < delta:
                            accept = True
                        elif (delta > 0 and prev['delta_pred'] > 0 and delta > prev['delta_pred']):
                            accept = True
                        elif (prev['delta_pred'] <= 0 and delta <= 0 and delta > prev['delta_pred']):
                            accept = True
                    if accept:
                        best_combo_by_accession[acc] = {
                            'accession': acc,
                            'snp_index': snp_index,
                            'positions': ';'.join(map(str, pos_list)),
                            'ref_bases': ''.join(ref_list),
                            'alt_bases': ''.join(alt_list),
                            'size': len(pos_list),
                            'delta_pred': delta
                        }
                    if (seeking_positive and delta > 0) or (not seeking_positive and delta < 0):
                        filtered_combo_variants_by_accession.setdefault(acc, []).append({
                            'accession': acc,
                            'snp_index': snp_index,
                            'positions': pos_list,
                            'ref_bases': ref_list,
                            'alt_bases': alt_list,
                            'size': len(pos_list),
                            'delta_pred': delta,
                            'new_row': combo_emb,
                            'variant_type': 'combo'
                        })
                    combo_records_all.append({
                        'accession': acc,
                        'snp_index': snp_index,
                        'positions': ';'.join(map(str, pos_list)),
                        'ref_bases': ''.join(ref_list),
                        'alt_bases': ''.join(alt_list),
                        'size': len(pos_list),
                        'delta_pred': delta
                    })

        # Batch evaluate singles
        if singles_batch_entries:
            # Group by accession for baseline reuse
            # Build batch mutated matrix
            mutated_rows = []
            meta = []  # parallel metadata
            # We'll build from baseline flat for each accession to avoid re-flatten
            for (acc, pos, refb, altb, new_row) in singles_batch_entries:
                base_flat = baseline_flat[row_start:row_end].copy()
                base_flat = apply_snp_mutation(base_flat, snp_index, new_row)
                mutated_rows.append(base_flat)
                meta.append((acc, pos, refb, altb))
            mutated_rows = np.vstack(mutated_rows)
            preds = batch_predict(model_obj, mutated_rows, target_idx)
            # Convert to deltas
            for k, pred in enumerate(preds):
                acc, pos, refb, altb = meta[k]
                delta = pred - baseline_pred_map[acc]
                # Accept if we move from non-positive to positive, or positive to bigger positive, or improve among non-positives
                prev = best_single_by_accession.get(acc)
                if prev is None:
                    best_single_by_accession[acc] = {
                        'accession': acc,
                        'snp_index': snp_index,
                        'position': pos,
                        'ref_base': refb,
                        'alt_base': altb,
                        'delta_pred': delta
                    }
                else:
                    pdv = prev['delta_pred']
                    if pdv <= 0 < delta or (delta > 0 and pdv > 0 and delta > pdv) or (pdv <= 0 and delta <= 0 and delta > pdv):
                        best_single_by_accession[acc] = {
                            'accession': acc,
                            'snp_index': snp_index,
                            'position': pos,
                            'ref_base': refb,
                            'alt_base': altb,
                            'delta_pred': delta
                        }

        # Batch evaluate combos
        if combos_batch_entries:
            mutated_rows = []
            meta = []
            for (acc, pos_list, ref_list, alt_list, combo_emb) in combos_batch_entries:
                base_flat = baseline_flat[row_start:row_end].copy()
                base_flat = apply_snp_mutation(base_flat, snp_index, combo_emb)
                mutated_rows.append(base_flat)
                meta.append((acc, pos_list, ref_list, alt_list))
            mutated_rows = np.vstack(mutated_rows)
            preds = batch_predict(model_obj, mutated_rows, target_idx)
            for k, pred in enumerate(preds):
                acc, pos_list, ref_list, alt_list = meta[k]
                delta = pred - baseline_pred_map[acc]
                # Accept if we move from non-positive to positive, or positive to bigger positive, or improve among non-positives
                prev = best_combo_by_accession.get(acc)
                if prev is None:
                    best_combo_by_accession[acc] = {
                        'accession': acc,
                        'snp_index': snp_index,
                        'positions': ';'.join(map(str, pos_list)),
                        'ref_bases': ''.join(ref_list),
                        'alt_bases': ''.join(alt_list),
                        'size': len(pos_list),
                        'delta_pred': delta
                    }
                else:
                    pdv = prev['delta_pred']
                    if pdv <= 0 < delta or (delta > 0 and pdv > 0 and delta > pdv) or (pdv <= 0 and delta <= 0 and delta > pdv):
                        best_combo_by_accession[acc] = {
                            'accession': acc,
                            'snp_index': snp_index,
                            'positions': ';'.join(map(str, pos_list)),
                            'ref_bases': ''.join(ref_list),
                            'alt_bases': ''.join(alt_list),
                            'size': len(pos_list),
                            'delta_pred': delta
                        }

    # Prepare outputs
    best_single_df = pd.DataFrame(best_single_by_accession.values()).sort_values('delta_pred', ascending=False)
    best_combo_df = pd.DataFrame(best_combo_by_accession.values()).sort_values('delta_pred', ascending=False) if best_combo_by_accession else pd.DataFrame(columns=['accession','snp_index','positions','ref_bases','alt_bases','size','delta_pred'])

    # Filter based on desired effect direction (positive or negative)
    if seeking_positive:
        filtered_single_df = best_single_df[best_single_df['delta_pred'] > 0].copy()
        filtered_combo_df = best_combo_df[best_combo_df['delta_pred'] > 0].copy()
        other_single_df = best_single_df[best_single_df['delta_pred'] <= 0]
        other_combo_df = best_combo_df[best_combo_df['delta_pred'] <= 0]
    else:  # seeking negative
        filtered_single_df = best_single_df[best_single_df['delta_pred'] < 0].copy()
        filtered_combo_df = best_combo_df[best_combo_df['delta_pred'] < 0].copy()
        other_single_df = best_single_df[best_single_df['delta_pred'] >= 0]
        other_combo_df = best_combo_df[best_combo_df['delta_pred'] >= 0]

    # Write main outputs
    single_out = os.path.join(args.output_dir, 'best_single_mutation_per_accession.csv')
    combo_out = os.path.join(args.output_dir, 'best_combo_mutation_per_accession.csv')
    filtered_single_df.to_csv(single_out, index=False)
    filtered_combo_df.to_csv(combo_out, index=False)

    # Supplemental (opposite direction accessions) for completeness
    neg_single_out = os.path.join(args.output_dir, f'best_single_mutation_per_accession_non_{effect_label}.csv')
    neg_combo_out = os.path.join(args.output_dir, f'best_combo_mutation_per_accession_non_{effect_label}.csv')
    other_single_df.to_csv(neg_single_out, index=False)
    other_combo_df.to_csv(neg_combo_out, index=False)

    # --------------------------------------------------
    # Variant overlap summary (shared variants across accessions)
    # --------------------------------------------------
    total_accessions_with_target = len(set(filtered_single_df['accession']).union(set(filtered_combo_df['accession'])))

    overlap_rows: List[dict] = []
    # Singles aggregation
    if not filtered_single_df.empty:
        grp = filtered_single_df.groupby(['snp_index','position','alt_base'])
        for (snp_idx, pos, alt), sub in grp:
            accs = sorted(sub['accession'].tolist())
            overlap_rows.append({
                'variant_type': 'single',
                'snp_index': snp_idx,
                'positions': str(pos),
                'size': 1,
                'alt_bases': alt,
                'n_accessions': len(accs),
                'accession_fraction': len(accs)/total_accessions_with_target if total_accessions_with_target else 0.0,
                'accession_ids': ';'.join(accs[:50]) + (';...' if len(accs) > 50 else ''),
                'mean_delta': sub['delta_pred'].mean(),
                'median_delta': sub['delta_pred'].median(),
                'max_delta': sub['delta_pred'].max(),
                'min_delta': sub['delta_pred'].min()
            })
    # Combos aggregation
    if not filtered_combo_df.empty:
        grp = filtered_combo_df.groupby(['snp_index','positions','alt_bases'])
        for (snp_idx, positions, alt_bases), sub in grp:
            pos_list = positions.split(';') if positions else []
            accs = sorted(sub['accession'].tolist())
            overlap_rows.append({
                'variant_type': 'combo',
                'snp_index': snp_idx,
                'positions': positions,
                'size': len(pos_list),
                'alt_bases': alt_bases,
                'n_accessions': len(accs),
                'accession_fraction': len(accs)/total_accessions_with_target if total_accessions_with_target else 0.0,
                'accession_ids': ';'.join(accs[:50]) + (';...' if len(accs) > 50 else ''),
                'mean_delta': sub['delta_pred'].mean(),
                'median_delta': sub['delta_pred'].median(),
                'max_delta': sub['delta_pred'].max(),
                'min_delta': sub['delta_pred'].min()
            })

    overlap_df = pd.DataFrame(overlap_rows)
    if not overlap_df.empty:
        overlap_df.sort_values(['n_accessions','mean_delta'], ascending=[False, False], inplace=True)

    overlap_out = os.path.join(args.output_dir, 'variant_overlap_summary.csv')
    overlap_df.to_csv(overlap_out, index=False)

    # --------------------------------------------------
    # Per-SNP overlap summary
    # --------------------------------------------------
    per_snp_rows: List[dict] = []
    if not overlap_df.empty:
        for snp_idx, sub in overlap_df.groupby('snp_index'):
            singles_sub = sub[sub['variant_type'] == 'single']
            combos_sub = sub[sub['variant_type'] == 'combo']
            top_single = singles_sub.iloc[0] if not singles_sub.empty else None
            top_combo = combos_sub.iloc[0] if not combos_sub.empty else None
            per_snp_rows.append({
                'snp_index': snp_idx,
                'n_distinct_single_variants': singles_sub.shape[0],
                'n_distinct_combo_variants': combos_sub.shape[0],
                'top_single_positions': top_single['positions'] if top_single is not None else '',
                'top_single_alt_bases': top_single['alt_bases'] if top_single is not None else '',
                'top_single_n_accessions': int(top_single['n_accessions']) if top_single is not None else 0,
                'top_combo_positions': top_combo['positions'] if top_combo is not None else '',
                'top_combo_alt_bases': top_combo['alt_bases'] if top_combo is not None else '',
                'top_combo_n_accessions': int(top_combo['n_accessions']) if top_combo is not None else 0
            })
    per_snp_df = pd.DataFrame(per_snp_rows)
    per_snp_out = os.path.join(args.output_dir, 'per_snp_overlap_summary.csv')
    per_snp_df.to_csv(per_snp_out, index=False)

    # --------------------------------------------------
    # Cumulative multi-SNP application for multiple top-N scenarios
    # --------------------------------------------------
    # Loop over all top-N scenarios and save results in subdirectories
    for scenario_idx, (top_n_snps_val, top_n_combined_val) in enumerate(zip(top_n_snps_list, top_n_combined_list)):
        # Create subdirectory for this scenario
        if top_n_snps_val == 0:
            scenario_dir = os.path.join(args.output_dir, 'all')
        else:
            scenario_dir = os.path.join(args.output_dir, str(top_n_snps_val))
        os.makedirs(scenario_dir, exist_ok=True)
        
        scenario_label_snps = "all" if top_n_snps_val == 0 else str(top_n_snps_val)
        scenario_label_comb = "all" if top_n_combined_val == 0 else str(top_n_combined_val)
        print(f"\n[Scenario {scenario_idx + 1}/{len(top_n_snps_list)}] Computing cumulative analysis with top_n_snps={scenario_label_snps}, top_n_combined={scenario_label_comb}")
        
        # --------------------------------------------------
        # Single-only cumulative analysis
        # --------------------------------------------------
        cum_variants_rows: List[dict] = []
        cum_summary_rows: List[dict] = []
        top_n = top_n_snps_val if top_n_snps_val > 0 else None
        for acc in accession_list:
            variants = filtered_variants_by_accession.get(acc, [])
            if not variants:
                cum_summary_rows.append({
                    'accession': acc,
                    'baseline_pred': baseline_pred_map[acc],
                    'final_pred': baseline_pred_map[acc],
                    'cumulative_delta': 0.0,
                    'n_steps': 0
                })
                continue
            # Sort by delta: descending for positive (largest improvements first), ascending for negative (largest decreases first)
            variants_sorted = sorted(variants, key=lambda d: d['delta_pred'], reverse=seeking_positive)
            # Don't truncate before dedup - apply limit based on unique SNPs selected
            used_snp_indices = set()
            emb_mat = embeddings[acc].copy()
            baseline_pred = baseline_pred_map[acc]
            prev_pred = baseline_pred
            step = 0
            for var in variants_sorted:
                si = var['snp_index']
                if si in used_snp_indices:
                    continue
                # Stop when we've applied top_n unique SNP indices
                if top_n is not None and step >= top_n:
                    break
                used_snp_indices.add(si)
                emb_mat[si] = var['new_row']
                mut_flat = flatten_embedding(emb_mat)
                pred = predict_target(model_obj, mut_flat.reshape(1, -1), target_idx)[0]
                step_delta = pred - prev_pred
                cumulative_delta = pred - baseline_pred
                step += 1
                cum_variants_rows.append({
                    'accession': acc,
                    'step': step,
                    'snp_index': si,
                    'position': var['position'],
                    'ref_base': var['ref_base'],
                    'alt_base': var['alt_base'],
                    'original_single_delta': var['delta_pred'],
                    'step_delta': step_delta,
                    'cumulative_delta': cumulative_delta,
                    'prediction_after_step': pred
                })
                prev_pred = pred
            cum_summary_rows.append({
                'accession': acc,
                'baseline_pred': baseline_pred,
                'final_pred': prev_pred,
                'cumulative_delta': prev_pred - baseline_pred,
                'n_steps': step
            })
        variants_out = os.path.join(scenario_dir, 'per_accession_cumulative_variants.csv')
        pd.DataFrame(cum_variants_rows).to_csv(variants_out, index=False)
        deltas_out = os.path.join(scenario_dir, 'per_accession_cumulative_deltas.csv')
        pd.DataFrame(cum_summary_rows).sort_values('cumulative_delta', ascending=False).to_csv(deltas_out, index=False)
        print(f"  [Output] Single cumulative variant steps -> {variants_out} ({len(cum_variants_rows)} rows)")
        print(f"  [Output] Single cumulative summary -> {deltas_out} ({len(cum_summary_rows)} accessions)")

        # --------------------------------------------------
        # Combined cumulative application including combos (single + combo pool)
        # Strategy: merge all filtered single variants and filtered combo variants, sort by delta (desc for positive, asc for negative), enforce one applied variant per SNP index.
        # --------------------------------------------------
        combo_cum_variants_rows: List[dict] = []
        combo_cum_summary_rows: List[dict] = []
        top_n_combined = top_n_combined_val if top_n_combined_val > 0 else None

        # Diagnostic: log pool sizes for first few accessions
        if args.verbose and accession_list and scenario_idx == 0:
            for acc in accession_list[:3]:
                n_singles = len(filtered_variants_by_accession.get(acc, []))
                n_combos = len(filtered_combo_variants_by_accession.get(acc, []))
                singles_snps = set(v['snp_index'] for v in filtered_variants_by_accession.get(acc, []))
                combos_snps = set(v['snp_index'] for v in filtered_combo_variants_by_accession.get(acc, []))
                print(f"  [Diag] {acc}: {n_singles} {effect_label} singles at {len(singles_snps)} SNP indices, {n_combos} {effect_label} combos at {len(combos_snps)} SNP indices")

        for acc in accession_list:
            singles = filtered_variants_by_accession.get(acc, [])
            # convert singles to a unified format with variant_type
            singles_u = [dict(v, variant_type='single', positions=[v['position']], ref_bases=[v['ref_base']], alt_bases=[v['alt_base']], size=1) for v in singles]
            combos_u = filtered_combo_variants_by_accession.get(acc, [])
            merged = singles_u + combos_u
            if not merged:
                combo_cum_summary_rows.append({
                    'accession': acc,
                    'baseline_pred': baseline_pred_map[acc],
                    'final_pred': baseline_pred_map[acc],
                    'cumulative_delta': 0.0,
                    'n_steps': 0
                })
                continue
            # Sort by delta: descending for positive (largest improvements first), ascending for negative (largest decreases first)
            merged_sorted = sorted(merged, key=lambda d: d['delta_pred'], reverse=seeking_positive)
            # Don't truncate before dedup - apply limit based on unique SNPs selected
            used_snp_indices = set()
            emb_mat = embeddings[acc].copy()
            baseline_pred = baseline_pred_map[acc]
            prev_pred = baseline_pred
            step = 0
            for var in merged_sorted:
                si = var['snp_index']
                if si in used_snp_indices:
                    continue
                # Stop when we've applied top_n_combined unique SNP indices
                if top_n_combined is not None and step >= top_n_combined:
                    break
                used_snp_indices.add(si)
                emb_mat[si] = var['new_row']
                mut_flat = flatten_embedding(emb_mat)
                pred = predict_target(model_obj, mut_flat.reshape(1, -1), target_idx)[0]
                step_delta = pred - prev_pred
                cumulative_delta = pred - baseline_pred
                step += 1
                combo_cum_variants_rows.append({
                    'accession': acc,
                    'step': step,
                    'variant_type': var['variant_type'],
                    'snp_index': si,
                    'positions': ';'.join(map(str, var['positions'])),
                    'ref_bases': ''.join(var['ref_bases']),
                    'alt_bases': ''.join(var['alt_bases']),
                    'size': var['size'],
                    'original_delta': var['delta_pred'],
                    'step_delta': step_delta,
                    'cumulative_delta': cumulative_delta,
                    'prediction_after_step': pred
                })
                prev_pred = pred
            combo_cum_summary_rows.append({
                'accession': acc,
                'baseline_pred': baseline_pred,
                'final_pred': prev_pred,
                'cumulative_delta': prev_pred - baseline_pred,
                'n_steps': step
            })
        combo_variants_out = os.path.join(scenario_dir, 'per_accession_cumulative_combo_variants.csv')
        pd.DataFrame(combo_cum_variants_rows).to_csv(combo_variants_out, index=False)
        combo_deltas_out = os.path.join(scenario_dir, 'per_accession_cumulative_combo_deltas.csv')
        pd.DataFrame(combo_cum_summary_rows).sort_values('cumulative_delta', ascending=False).to_csv(combo_deltas_out, index=False)
        print(f"  [Output] Combined cumulative variant steps -> {combo_variants_out} ({len(combo_cum_variants_rows)} rows)")
        print(f"  [Output] Combined cumulative summary -> {combo_deltas_out} ({len(combo_cum_summary_rows)} accessions)")
        
        # Store results for this scenario for intermediates (last scenario only)
        if scenario_idx == len(top_n_snps_list) - 1:
            final_cum_variants_rows = cum_variants_rows
            final_cum_summary_rows = cum_summary_rows
            final_combo_cum_variants_rows = combo_cum_variants_rows
            final_combo_cum_summary_rows = combo_cum_summary_rows

    # --------------------------------------------------
    # Persist intermediates (on by default unless disabled)
    # Uses results from the LAST scenario
    # --------------------------------------------------
    if not args.no_save_intermediates:
        intermediates_dir = args.intermediates_dir or os.path.join(args.output_dir, 'intermediates')
        os.makedirs(intermediates_dir, exist_ok=True)
        index_rows = []

        # Pre-build quick lookups for cumulative steps by accession
        from collections import defaultdict
        single_steps_by_acc = defaultdict(list)
        for row in final_cum_variants_rows:
            single_steps_by_acc[row['accession']].append(row)
        combo_steps_by_acc = defaultdict(list)
        for row in final_combo_cum_variants_rows:
            combo_steps_by_acc[row['accession']].append(row)

        # Helper to build structured numpy arrays
        def build_struct_array(records, dtype):
            arr = np.zeros(len(records), dtype=dtype)
            for i, r in enumerate(records):
                for name in arr.dtype.names:
                    arr[i][name] = r.get(name, 0 if arr.dtype[name].kind in 'if' else '')
            return arr

        for acc in accession_list:
            baseline_pred = baseline_pred_map[acc]

            # Filtered singles table
            singles_list = filtered_variants_by_accession.get(acc, [])
            single_records = []
            for s in singles_list:
                pred_val = baseline_pred + s['delta_pred']
                single_records.append({
                    'snp_index': s['snp_index'],
                    'position': s['position'],
                    'ref_base': s['ref_base'],
                    'alt_base': s['alt_base'],
                    'delta_pred': s['delta_pred'],
                    'pred': pred_val
                })
            singles_arr = build_struct_array(single_records, np.dtype([
                ('snp_index','i4'),('position','i4'),('ref_base','U4'),('alt_base','U4'),('delta_pred','f8'),('pred','f8')
            ])) if single_records else np.zeros(0, dtype=[('snp_index','i4'),('position','i4'),('ref_base','U4'),('alt_base','U4'),('delta_pred','f8'),('pred','f8')])

            # Filtered combos table
            combos_list = filtered_combo_variants_by_accession.get(acc, [])
            combo_records = []
            for c in combos_list:
                pred_val = baseline_pred + c['delta_pred']
                combo_records.append({
                    'snp_index': c['snp_index'],
                    'size': c['size'],
                    'positions': ';'.join(map(str, c['positions'])),
                    'ref_bases': ''.join(c['ref_bases']),
                    'alt_bases': ''.join(c['alt_bases']),
                    'delta_pred': c['delta_pred'],
                    'pred': pred_val
                })
            combos_arr = build_struct_array(combo_records, np.dtype([
                ('snp_index','i4'),('size','i4'),('positions','U256'),('ref_bases','U256'),('alt_bases','U256'),('delta_pred','f8'),('pred','f8')
            ])) if combo_records else np.zeros(0, dtype=[('snp_index','i4'),('size','i4'),('positions','U256'),('ref_bases','U256'),('alt_bases','U256'),('delta_pred','f8'),('pred','f8')])

            # Single cumulative steps
            sc_steps = single_steps_by_acc.get(acc, [])
            sc_records = []
            for r in sc_steps:
                sc_records.append({
                    'step': r['step'],
                    'snp_index': r['snp_index'],
                    'position': r['position'],
                    'ref_base': r['ref_base'],
                    'alt_base': r['alt_base'],
                    'original_single_delta': r['original_single_delta'],
                    'step_delta': r['step_delta'],
                    'cumulative_delta': r['cumulative_delta'],
                    'prediction_after_step': r['prediction_after_step']
                })
            sc_arr = build_struct_array(sc_records, np.dtype([
                ('step','i4'),('snp_index','i4'),('position','i4'),('ref_base','U4'),('alt_base','U4'),('original_single_delta','f8'),('step_delta','f8'),('cumulative_delta','f8'),('prediction_after_step','f8')
            ])) if sc_records else np.zeros(0, dtype=[('step','i4'),('snp_index','i4'),('position','i4'),('ref_base','U4'),('alt_base','U4'),('original_single_delta','f8'),('step_delta','f8'),('cumulative_delta','f8'),('prediction_after_step','f8')])

            # Combined cumulative steps
            cc_steps = combo_steps_by_acc.get(acc, [])
            cc_records = []
            for r in cc_steps:
                cc_records.append({
                    'step': r['step'],
                    'variant_type': 0 if r['variant_type'] == 'single' else 1,
                    'snp_index': r['snp_index'],
                    'size': r['size'],
                    'positions': r['positions'],
                    'ref_bases': r['ref_bases'],
                    'alt_bases': r['alt_bases'],
                    'original_delta': r['original_delta'],
                    'step_delta': r['step_delta'],
                    'cumulative_delta': r['cumulative_delta'],
                    'prediction_after_step': r['prediction_after_step']
                })
            cc_arr = build_struct_array(cc_records, np.dtype([
                ('step','i4'),('variant_type','i1'),('snp_index','i4'),('size','i4'),('positions','U256'),('ref_bases','U256'),('alt_bases','U256'),('original_delta','f8'),('step_delta','f8'),('cumulative_delta','f8'),('prediction_after_step','f8')
            ])) if cc_records else np.zeros(0, dtype=[('step','i4'),('variant_type','i1'),('snp_index','i4'),('size','i4'),('positions','U256'),('ref_bases','U256'),('alt_bases','U256'),('original_delta','f8'),('step_delta','f8'),('cumulative_delta','f8'),('prediction_after_step','f8')])

            # Best single & combo summaries
            bs = best_single_by_accession.get(acc)
            bc = best_combo_by_accession.get(acc)
            best_single_delta = bs['delta_pred'] if bs and bs['delta_pred'] > 0 else 0.0
            best_combo_delta = bc['delta_pred'] if bc and bc['delta_pred'] > 0 else 0.0

            # Final cumulative predictions (single-only & combined) from LAST scenario
            final_single_only_row = next((r for r in final_cum_summary_rows if r['accession'] == acc), None)
            final_combined_row = next((r for r in final_combo_cum_summary_rows if r['accession'] == acc), None)
            final_single_only_pred = final_single_only_row['final_pred'] if final_single_only_row else baseline_pred
            final_single_only_delta = final_single_only_row['cumulative_delta'] if final_single_only_row else 0.0
            final_combined_pred = final_combined_row['final_pred'] if final_combined_row else baseline_pred
            final_combined_delta = final_combined_row['cumulative_delta'] if final_combined_row else 0.0

            out_path = os.path.join(intermediates_dir, f"{acc}.npz")
            np.savez_compressed(
                out_path,
                baseline_pred=baseline_pred,
                positive_single_variants=singles_arr,
                positive_combo_variants=combos_arr,
                single_cumulative_steps=sc_arr,
                combined_cumulative_steps=cc_arr,
                best_single_delta=best_single_delta,
                best_combo_delta=best_combo_delta,
                final_single_only_pred=final_single_only_pred,
                final_single_only_delta=final_single_only_delta,
                final_combined_pred=final_combined_pred,
                final_combined_delta=final_combined_delta
            )

            index_rows.append({
                'accession': acc,
                'intermediate_file': out_path,
                'n_positive_singles': singles_arr.shape[0],
                'n_positive_combos': combos_arr.shape[0],
                'single_only_steps': sc_arr.shape[0],
                'combined_steps': cc_arr.shape[0],
                'final_single_only_delta': final_single_only_delta,
                'final_combined_delta': final_combined_delta
            })

        index_df = pd.DataFrame(index_rows)
        index_csv = os.path.join(intermediates_dir, 'intermediates_index.csv')
        index_df.to_csv(index_csv, index=False)
        print(f"[Intermediates] Saved per-accession NPZ files to {intermediates_dir} (index: {index_csv})")

    print('[Done] Per-accession mutation analysis complete.')
    print(f"  {effect_label.capitalize()} single variants: {filtered_single_df.shape[0]}")
    print(f"  {effect_label.capitalize()} combo variants: {filtered_combo_df.shape[0]}")
    print(f"  Overlap summary: {overlap_out} ({overlap_df.shape[0]} rows)")
    print(f"  Per-SNP overlap summary: {per_snp_out} ({per_snp_df.shape[0]} rows)")

if __name__ == '__main__':
    main()
