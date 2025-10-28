import argparse
import os
from typing import Dict, Iterable, Optional, List, Tuple
import sys

import enformer_pytorch
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import pandas as pd

# from src.dataloaders.utils.rc import string_reverse_complement
# from src.utils.train import get_logger

STRING_COMPLEMENT_MAP = {
    "A": "T", "C": "G", "G": "C", "T": "A", "a": "t", "c": "g", "g": "c", "t": "a",
    "N": "N", "n": "n",
}

def string_reverse_complement(seq, sep_token='[SEP]'):
    """
    Reverse complement a DNA sequence, treating [SEP] as an atomic separator.
    Segments separated by [SEP] are reverse-complemented individually and their order is reversed.
    """
    segments = seq.split(sep_token)
    # Remove trailing empty segment if seq ends with [SEP]
    if segments and segments[-1] == '':
        segments = segments[:-1]
    rc_segments = []
    for segment in segments:
        rc = ''
        for base in segment[::-1]:
            if base in STRING_COMPLEMENT_MAP:
                rc += STRING_COMPLEMENT_MAP[base]
            else:
                rc += base
        rc_segments.append(rc)
    # Reverse the order and join with [SEP]
    return sep_token.join(rc_segments[::-1])

WINDOW_SIZE_BP = 1536
# log = get_logger(__name__)


class DNAEmbeddingModel(nn.Module):
    """Wrapper around HF model.

    Args:
        model_name_or_path: str, path to HF model.
    """
    def __init__(
            self,
            model_name_or_path: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        # Enformer uses different library for loading
        if "enformer" in model_name_or_path.lower():
            self.backbone = enformer_pytorch.from_pretrained(
                model_name_or_path,
                use_tf_gamma=False,
                use_checkpointing=True
            )
        # NT model is not compatible with AutoModel class
        elif "nucleotide-transformer" in model_name_or_path.lower():
            # NT LM `backbone` is under the `.esm` attribute
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).esm
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    def forward(self, input_ids):
        """Backbone forward pass to retrieve last_hidden_state."""
        if "enformer" in self.model_name_or_path.lower():
            # Enformer forward pass has different signature
            return self.backbone(input_ids, return_embeddings=True)[1]
        return self.backbone(input_ids).last_hidden_state

class EnformerTokenizer:
    """Enformer tokenizer."""
    # Order is important here! (See: https://github.com/lucidrains/enformer-pytorch?tab=readme-ov-file#usage)
    pad_token = "P"  # Padding token should be a character to avoid issues with tokenization
    encode_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, pad_token: -1}

    @classmethod
    def encode(
            cls, seq: str, max_length: Optional[int] = None, truncation: Optional[bool] = False
    ) -> Iterable[int]:
        """Convert bp to token ids."""
        if max_length is not None:
            assert max_length >= 0, "max_length should be a positive integer."
            if len(seq) < max_length:
                seq = seq + cls.pad_token * (max_length - len(seq))
            elif truncation:
                seq = seq[:max_length]
        return [cls.encode_map[bp] for bp in seq.upper()]

    @classmethod
    def batch_encode_plus(
            cls, seqs: Iterable[str], max_length: Optional[int] = None, truncation: Optional[bool] = False,
            **kwargs,  # ensures compatibility with HF tokenizer-like API
    ) -> Dict[str, Iterable[Iterable[int]]]:
        """Batch encode sequences using HF tokenizer-like API."""
        input_ids = [cls.encode(seq, max_length=max_length, truncation=truncation) for seq in seqs]
        return {"input_ids": input_ids}


def setup_distributed():
    """Set environment variables for distributed runs."""
    dist.init_process_group("nccl")


def cleanup_distributed():
    """Clean up processes from distributed runs."""
    dist.destroy_process_group()


## Removed fsspec utilities (not needed in mutation-only script)


# Processing functions
def recast_chromosome_tissue_dist2TSS(examples):
    """Recast chromosome to int."""
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }


def tokenize_variants(examples, tokenizer, max_length: int):
    """Tokenize sequence.

    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
        max_length: int.
    Returns:
        dict with values as list of token ids.
    """

    ref_tokenized = tokenizer.batch_encode_plus(
        examples["ref_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_tokenized = tokenizer.batch_encode_plus(
        examples["alt_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    ref_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["ref_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["alt_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )

    return {
        "ref_input_ids": ref_tokenized["input_ids"],
        "alt_input_ids": alt_tokenized["input_ids"],
        "ref_rc_input_ids": ref_rc_tokenized["input_ids"],
        "alt_rc_input_ids": alt_rc_tokenized["input_ids"],
    }


def find_variant_idx(examples):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = len(examples["ref_input_ids"]) // 2
    if examples["ref_input_ids"][idx] == examples["alt_input_ids"][idx]:
        # If no, loop through sequence and find variant location
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_input_ids"], examples["alt_input_ids"])):
            if ref != alt:
                idx = i
    # Same as above, but for reverse complement
    rc_idx = len(examples["ref_rc_input_ids"]) // 2 - 1
    if examples["ref_rc_input_ids"][rc_idx] == examples["alt_rc_input_ids"][rc_idx]:
        rc_idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_rc_input_ids"], examples["alt_rc_input_ids"])):
            if ref != alt:
                rc_idx = i
    return {"variant_idx": idx, "rc_variant_idx": rc_idx}


def get_backbone_model(args, device):
    """Get the backbone model; wrap with DDP unless --no-ddp specified."""
    model = DNAEmbeddingModel(model_name_or_path=args.model_name_or_path)
    model.eval()
    model = model.to(device)
    if not args.no_ddp:
        model = DDP(model)
    return model


def concat_storage_dict_values(storage_dict):
    """Helper method that combines lists of tensors in storage_dict into a single torch.Tensor."""
    return {key: torch.cat(storage_dict[key], dim=0) for key in storage_dict.keys()}

def embed_custom_sequences(sequences_dict, model, tokenizer, device, args):
    """
    Embed custom sequences per accession using the model, with optional joining of all sequences for the accession.
    If --join_regions is set, join all sequences for the accession (across all chromosomes) with [SEP],
    split into chunks if needed, and process as if they were separate sequences (pooled as usual).
    Otherwise, process each sequence individually as before.
    """
    embeddings = {}
    join_regions = getattr(args, 'join_regions', False)
    max_tokens = args.seq_len // args.bp_per_token
    for accession_id, chromosomes in tqdm(sequences_dict.items(), desc="Processing accessions"):
        print(f"Processing {accession_id}")
        single_embeddings = []
        if join_regions:
            # Gather all sequences for this accession (across all chromosomes)
            all_seqs = []
            for seqs in chromosomes.values():
                all_seqs.extend(seqs)
            if not all_seqs:
                continue
            sep_token = '[SEP]'
            # Split into chunks at [SEP] boundaries, ensuring no sequence is split
            chunks = []
            current_chunk = []
            current_len = 0
            for i, seq in enumerate(all_seqs):
                # Only add [SEP] if not the last sequence
                seq_with_sep = seq + (sep_token if i < len(all_seqs) - 1 else "")
                seq_tokens = tokenizer.encode(seq_with_sep, add_special_tokens=False)
                if current_len + len(seq_tokens) > max_tokens and current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(seq_with_sep)
                current_len += len(seq_tokens)
            if current_chunk:
                chunks.append(''.join(current_chunk))
            # Process each chunk as a sequence
            for chunk_seq in chunks:
                rc_seq = string_reverse_complement(chunk_seq)
                tokenized = tokenizer.batch_encode_plus([chunk_seq], add_special_tokens=False, return_attention_mask=False, max_length=max_tokens, truncation=True)
                rc_tokenized = tokenizer.batch_encode_plus([rc_seq], add_special_tokens=False, return_attention_mask=False, max_length=max_tokens, truncation=True)
                # Check for length mismatch
                if len(tokenized["input_ids"][0]) != len(rc_tokenized["input_ids"][0]):
                    print(f"Warning: input_ids and rc_input_ids have different lengths for accession {accession_id}. Skipping this chunk.")
                    continue
                input_ids = torch.tensor(tokenized["input_ids"]).to(device)
                rc_input_ids = torch.tensor(rc_tokenized["input_ids"]).to(device)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        embedding = model(input_ids)
                        rc_embedding = model(rc_input_ids)
                        rc_embedding = rc_embedding.flip(dims=[1])
                        full_combined_embedding = (embedding + rc_embedding) / 2.0
                        if args.pool_method == "avg":
                            full_combined_embedding = torch.mean(full_combined_embedding, dim=1)
                        elif args.pool_method == "max":
                            full_combined_embedding = torch.max(full_combined_embedding, 1)[0]
                        else:
                            raise ValueError(f"Unsupported pool_method: {args.pool_method}")
                        # SNP context window is not supported in join_regions mode
                        if args.snp_context_window is not None:
                            print("Warning: SNP context window is not supported in join_regions mode. Skipping.")
                        combined_embedding = full_combined_embedding
                single_embeddings.append(combined_embedding.cpu().detach().numpy()[0])
        else:
            # Process each chromosome
            for chromosome, seqs in chromosomes.items():
                for seq in seqs:
                    # Tokenize the sequence and its reverse complement
                    tokenized = tokenizer.batch_encode_plus(
                        [seq],
                        add_special_tokens=False,
                        return_attention_mask=False,
                        max_length=max_tokens,
                        truncation=True,
                    )
                    rc_seq = string_reverse_complement(seq)
                    rc_tokenized = tokenizer.batch_encode_plus(
                        [rc_seq],
                        add_special_tokens=False,
                        return_attention_mask=False,
                        max_length=max_tokens,
                        truncation=True,
                    )

                    input_ids = torch.tensor(tokenized["input_ids"]).to(device)
                    rc_input_ids = torch.tensor(rc_tokenized["input_ids"]).to(device)

                    # Get embeddings for both forward and RC sequences
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            embedding = model(input_ids)
                            rc_embedding = model(rc_input_ids)
                            rc_embedding = rc_embedding.flip(dims=[1])
                            # Average the embeddings
                            full_combined_embedding = (embedding + rc_embedding) / 2.0
                            if args.pool_method == "avg":
                                full_combined_embedding = torch.mean(full_combined_embedding, dim=1)
                            elif args.pool_method == "max":
                                full_combined_embedding = torch.max(full_combined_embedding, 1)[0]
                            else:
                                raise ValueError(f"Unsupported pool_method: {args.pool_method}")
                            # If SNP context window is set, compute embedding for the sub-sequence
                            if args.snp_context_window is not None:
                                # Center window on the middle of the sequence
                                seq_len = full_combined_embedding.shape[1] if len(full_combined_embedding.shape) > 1 else input_ids.shape[1]
                                center = seq_len // 2
                                half_window = args.snp_context_window // 2
                                start = max(center - half_window, 0)
                                end = min(center + half_window, seq_len)
                                window_embedding = full_combined_embedding[:, start:end]
                                pooled_window = torch.mean(window_embedding, dim=1) if args.pool_method == "avg" else torch.max(window_embedding, 1)[0]
                                single_embeddings.append(pooled_window.cpu().detach().numpy()[0])
                            else:
                                single_embeddings.append(full_combined_embedding.cpu().detach().numpy()[0])

        if single_embeddings:
            embeddings[accession_id] = single_embeddings
    return embeddings

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_len", type=int, default=131072,
                        help="Sequence length (in bp)..")
    parser.add_argument("--bp_per_token", type=int, default=1,
                        help="Number of base pairs per token.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--downstream_save_dir", type=str, default="./outputs/downstream/vep_embeddings",
                        help="Directory to save downstream task.")
    parser.add_argument("--name", type=str, default=None, help="Embeddings model name.")
    parser.add_argument("--rcps", default=False, action="store_true", help="Use RCPS.")
    parser.add_argument("--no-rcps", dest="rcps", action="store_false", help="Do not use RCPS.")
    parser.add_argument("--embed_dump_batch_size", type=int, default=1,
                        help="Batch size for embedding dump.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--accessions_dir", type=str, default=None,
                        help="Input directory for sequences.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for embeddings.")
    parser.add_argument("--snp_context_window", type=int, default=None,
                        help="SNP context window size.")
    parser.add_argument("--pool_method", type=str, default="max",
                        help="Pooling for embeddings.")
    parser.add_argument("--join_regions", default=False, action="store_true",
                        help="Join all sequences for an accession with [SEP] and process as separate sequences.")
    parser.add_argument("--no-join_regions", dest="join_regions", action="store_false",
                        help="Do not join sequences for an accession.")
    # Mutation embedding specific arguments
    parser.add_argument("--selected_weights", type=str, default=None, help="CSV with selected weights (rank, variable, snp_index, embedding_dim, weight)")
    parser.add_argument("--relevant_peaks", type=str, default=None, help="CSV with relevant peaks (optional, not strictly needed here)")
    parser.add_argument("--gradients_dir", type=str, default=None, help="Directory containing integrated gradients CSV files named integrated_gradients_averaged_SNP{snp_index}_dim{dim}.csv")
    parser.add_argument("--accessions_list", type=str, default=None, help="Text file with accession IDs to process (one per line)")
    parser.add_argument("--alignment_dir", type=str, default=None, help="Directory with SNP alignment FASTAs (snp{snp_index}_aligned.fasta)")
    parser.add_argument("--num-weights", type=int, default=1, help="Top N weights (by existing selected_weights rank order) to process")
    parser.add_argument("--top-grad-positions", type=int, default=5, help="Top N gradient positions (by absolute mean_saliency) to mutate per weight")
    parser.add_argument("--mutation-output-dir", type=str, default=None, help="Output directory for mutated embeddings (NPZ files)")
    parser.add_argument("--mutation-window-context", type=int, default=0, help="Optional +/- context (ungapped) bases to include around mutated position (0 = mutate full ungapped sequence)")
    parser.add_argument("--mutate-combinations", action="store_true", help="Also generate multi-position mutation combinations across selected positions")
    parser.add_argument("--max-combination-size", type=int, default=None, help="Maximum size of position subsets to mutate together (default = all selected positions)")
    parser.add_argument("--combination-variant-limit", type=int, default=50000, help="Safety cap: maximum total combination variants per accession-weight; skip combos if exceeded")
    parser.add_argument("--no-ddp", action="store_true", help="Disable DistributedDataParallel / NCCL init (single process)")
    parser.add_argument("--heartbeat-seconds", type=int, default=300, help="Heartbeat logging interval in seconds (0 disables)")
    # Original non-mutation mode removed; always mutation embedding
    opts, _ = parser.parse_known_args()

    print("*** Args ************************")
    for k, v in vars(opts).items():
        print(f"  - {k}: {v}")
    print("******************************\n")

    def init_model_and_tokenizer():
        if opts.no_ddp:
            print("[Init] --no-ddp set; skipping distributed init", flush=True)
            device_local = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            print("[Init] Initializing distributed (nccl)...", flush=True)
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            device_local = torch.device(f"cuda:{dist.get_rank()}")
        print("[Init] Initializing tokenizer...", flush=True)
        if "enformer" in opts.model_name_or_path.lower():
            tok = EnformerTokenizer
        else:
            tok = AutoTokenizer.from_pretrained(opts.model_name_or_path, trust_remote_code=True)
        print(f"[Init] Loading model from {opts.model_name_or_path}...", flush=True)
        mdl = get_backbone_model(opts, device_local)
        print("[Init] Model ready", flush=True)
        return mdl, tok, device_local

    def read_alignment_fasta(path: str) -> Dict[str,str]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        seqs = {}
        with open(path) as f:
            current = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    current = line[1:].strip()
                    seqs[current] = []
                else:
                    if current is not None:
                        seqs[current].append(line.upper())
        return {k: ''.join(v) for k,v in seqs.items()}

    def ungap_sequence(seq: str) -> Tuple[str, List[int]]:
        ungapped = []
        mapping = []  # ungapped index -> alignment column
        for i,ch in enumerate(seq):
            if ch != '-':
                mapping.append(i)
                ungapped.append(ch)
        return ''.join(ungapped), mapping

    def mutate_sequence(base_seq: str, pos: int, new_base: str) -> str:
        return base_seq[:pos] + new_base + base_seq[pos+1:]

    def embed_sequences_list(seqs: List[str], model, tokenizer, device, pool_method: str, max_tokens: int) -> np.ndarray:
        if not seqs:
            return np.zeros((0,))
        embeddings = []
        for seq in seqs:
            tokenized = tokenizer.batch_encode_plus([seq], add_special_tokens=False, return_attention_mask=False, max_length=max_tokens, truncation=True)
            rc_seq = string_reverse_complement(seq)
            rc_tokenized = tokenizer.batch_encode_plus([rc_seq], add_special_tokens=False, return_attention_mask=False, max_length=max_tokens, truncation=True)
            input_ids = torch.tensor(tokenized["input_ids"]).to(device)
            rc_input_ids = torch.tensor(rc_tokenized["input_ids"]).to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = model(input_ids)
                    rc_emb = model(rc_input_ids)
                    rc_emb = rc_emb.flip(dims=[1])
                    full = (emb + rc_emb) / 2.0
                    if pool_method == 'avg':
                        full = torch.mean(full, dim=1)
                    elif pool_method == 'max':
                        full = torch.max(full, 1)[0]
                    else:
                        raise ValueError(f"Unsupported pool_method: {pool_method}")
            embeddings.append(full.detach().cpu().numpy()[0])
        return np.stack(embeddings, axis=0)

    def run_mutation_mode():
        required = [opts.selected_weights, opts.gradients_dir, opts.accessions_list, opts.alignment_dir, opts.mutation_output_dir]
        if any(r is None for r in required):
            raise ValueError("Mutation mode requires --selected_weights, --gradients_dir, --accessions_list, --alignment_dir, --mutation-output-dir")
        os.makedirs(opts.mutation_output_dir, exist_ok=True)
        weights_df = pd.read_csv(opts.selected_weights)
        weights_df = weights_df.sort_values('rank').head(opts.num_weights)
        with open(opts.accessions_list) as f:
            accessions = [line.strip() for line in f if line.strip()]
        model_mut, tokenizer_mut, device_mut = init_model_and_tokenizer()
        max_tokens = opts.seq_len // opts.bp_per_token
        all_meta_rows = []
        import time
        next_heartbeat = (time.time() + opts.heartbeat_seconds) if opts.heartbeat_seconds > 0 else None
        def heartbeat(context):
            nonlocal next_heartbeat
            if next_heartbeat is None:
                return
            now = time.time()
            if now >= next_heartbeat:
                print(f"[heartbeat] {context} t={int(now)}", flush=True)
                next_heartbeat = now + opts.heartbeat_seconds
        for _, wrow in weights_df.iterrows():
            snp_index = int(wrow['snp_index'])
            emb_dim = int(wrow['embedding_dim'])
            rank = int(wrow['rank'])
            weight_val = float(wrow['weight'])
            # Track total combination variants across all accessions for this SNP/embedding
            total_combo_variants_for_weight = 0
            grad_path = os.path.join(opts.gradients_dir, f"integrated_gradients_averaged_SNP{snp_index}_dim{emb_dim}.csv")
            if not os.path.exists(grad_path):
                print(f"Warning: gradients file missing {grad_path}; skipping weight rank {rank}")
                continue
            gdf = pd.read_csv(grad_path)
            if 'position' not in gdf.columns or 'mean_saliency' not in gdf.columns:
                print(f"Warning: gradients file {grad_path} missing required columns; skipping")
                continue
            gdf = gdf[gdf['position'] >= 0]  # drop possible header row with -1
            gdf['abs_sal'] = gdf['mean_saliency'].abs()
            # Z-score filter: retain only positions with |z| > 1.96
            mean_val = gdf['mean_saliency'].mean()
            std_val = gdf['mean_saliency'].std(ddof=0)
            if std_val == 0 or np.isnan(std_val):
                print(f"Warning: zero or NaN std for gradients (SNP {snp_index} dim {emb_dim}); no positions pass z-score filter.")
                continue
            gdf['z'] = (gdf['mean_saliency'] - mean_val) / std_val
            gdf_filt = gdf[gdf['z'].abs() > 1.96]
            if gdf_filt.empty:
                print(f"Warning: no positions with |z|>1.96 for SNP {snp_index} dim {emb_dim}; skipping weight rank {rank}")
                continue
            top_pos_df = gdf_filt.sort_values('abs_sal', ascending=False).head(opts.top_grad_positions)
            top_positions = top_pos_df['position'].tolist()
            alignment_file = os.path.join(opts.alignment_dir, f"snp{snp_index}_aligned.fasta")
            try:
                alignment = read_alignment_fasta(alignment_file)
            except FileNotFoundError:
                print(f"Warning: alignment missing {alignment_file}; skipping weight rank {rank}")
                continue
            weight_out_dir = os.path.join(opts.mutation_output_dir, f"w{rank}_snp{snp_index}_dim{emb_dim}")
            os.makedirs(weight_out_dir, exist_ok=True)
            for acc in accessions:
                heartbeat(f"SNP {snp_index} dim {emb_dim} accession {acc}")
                if acc not in alignment:
                    # try numeric vs string mismatch
                    if str(acc) in alignment:
                        acc_key = str(acc)
                    else:
                        print(f"Accession {acc} not in alignment for SNP {snp_index}; skipping accession")
                        continue
                else:
                    acc_key = acc
                aln_seq = alignment[acc_key]
                ungapped_seq, mapping = ungap_sequence(aln_seq)
                # Build reverse map: alignment column -> ungapped index (only where this accession has a base)
                align_to_ungapped = {aln_idx: u_idx for u_idx, aln_idx in enumerate(mapping)}
                if not ungapped_seq:
                    print(f"Accession {acc_key} ungapped sequence empty; skipping")
                    continue
                # Determine mutation sequences
                orig_embedding = embed_sequences_list([ungapped_seq], model_mut, tokenizer_mut, device_mut, opts.pool_method, max_tokens)[0]
                mut_embeddings_per_position = []
                position_meta = []
                # Interpret top_positions as alignment (gapped) coordinates; map to ungapped. Skip gaps for this accession.
                for aln_pos in top_positions:
                    if aln_pos not in align_to_ungapped:
                        # Gap for this accession at this alignment column; skip
                        print(f"Skipping aln pos {aln_pos} (SNP {snp_index}) for accession {acc_key} (gap in ungapped seq)", flush=True)
                        continue
                    ug_pos = align_to_ungapped[aln_pos]
                    if ug_pos >= len(ungapped_seq):
                        continue
                    base = ungapped_seq[ug_pos]
                    if base in 'ACGT':
                        alts = [b for b in 'ACGT' if b != base]
                    else:
                        alts = list('ACGT')
                    if opts.mutation_window_context > 0:
                        half = opts.mutation_window_context
                        start = max(0, ug_pos - half)
                        end = min(len(ungapped_seq), ug_pos + half + 1)
                        base_seq_window = ungapped_seq[start:end]
                        window_pos = ug_pos - start
                        mut_seqs = []
                        for alt in alts:
                            mut_seq = base_seq_window[:window_pos] + alt + base_seq_window[window_pos+1:]
                            mut_seqs.append(mut_seq)
                    else:
                        mut_seqs = [mutate_sequence(ungapped_seq, ug_pos, alt) for alt in alts]
                    mut_embs = embed_sequences_list(mut_seqs, model_mut, tokenizer_mut, device_mut, opts.pool_method, max_tokens)
                    mut_embeddings_per_position.append(mut_embs)
                    position_meta.append({'alignment_position': aln_pos, 'position': ug_pos, 'base': base, 'alts': ''.join(alts)})
                if not mut_embeddings_per_position:
                    continue
                # Stack to (num_positions, num_alts, embed_dim)
                embed_dim_model = mut_embeddings_per_position[0].shape[1]
                max_alts = max(m.shape[0] for m in mut_embeddings_per_position)
                # Pad with NaNs for heterogeneous alt counts
                stacked = np.full((len(mut_embeddings_per_position), max_alts, embed_dim_model), np.nan, dtype=np.float32)
                for i, m in enumerate(mut_embeddings_per_position):
                    stacked[i, :m.shape[0], :] = m

                # ----------------------------------------------------------
                # Optional multi-position combination mutations
                # ----------------------------------------------------------
                combo_mutations = None
                combo_variant_starts = None
                combo_variant_lengths = None
                combo_positions_flat = None
                combo_ref_bases_flat = None
                combo_alt_bases_flat = None
                num_combo_variants = 0
                max_combo_size_used = 0
                if opts.mutate_combinations and len(position_meta) > 1:
                    from itertools import combinations, product
                    # Use ungapped positions actually present for this accession
                    all_positions_list = [pm['position'] for pm in position_meta]
                    # Build per-position alt base lists (reuse earlier logic)
                    per_pos_alts = []
                    per_pos_ref = []
                    for ug_pos in all_positions_list:
                        ref_base = ungapped_seq[ug_pos] if ug_pos < len(ungapped_seq) else 'N'
                        per_pos_ref.append(ref_base)
                        if ref_base in 'ACGT':
                            alts_here = [b for b in 'ACGT' if b != ref_base]
                        else:
                            alts_here = list('ACGT')
                        per_pos_alts.append(alts_here)
                    max_size = opts.max_combination_size if opts.max_combination_size is not None else len(all_positions_list)
                    max_size = min(max_size, len(all_positions_list))
                    combo_seqs = []
                    combo_positions_records = []  # list of lists
                    combo_ref_bases_records = []  # list of lists
                    combo_alt_bases_records = []  # list of lists (mutated bases)
                    for size in range(2, max_size + 1):
                        for pos_indices in combinations(range(len(all_positions_list)), size):
                            chosen_positions = [all_positions_list[i] for i in pos_indices]
                            alt_lists = [per_pos_alts[i] for i in pos_indices]
                            ref_bases = [per_pos_ref[i] for i in pos_indices]
                            # Skip if any position had no alts (shouldn't happen)
                            if any(len(al) == 0 for al in alt_lists):
                                continue
                            for alt_combo in product(*alt_lists):
                                # Safety cap check
                                if len(combo_seqs) >= opts.combination_variant_limit:
                                    break
                                if opts.mutation_window_context > 0:
                                    half = opts.mutation_window_context
                                    start = max(0, min(chosen_positions) - half)
                                    end = min(len(ungapped_seq), max(chosen_positions) + half + 1)
                                    base_seq_window = ungapped_seq[start:end]
                                    # mutate relative positions
                                    rel_positions = [p - start for p in chosen_positions]
                                    seq_list = list(base_seq_window)
                                    for rp, new_b in zip(rel_positions, alt_combo):
                                        seq_list[rp] = new_b
                                    mutated_seq = ''.join(seq_list)
                                else:
                                    seq_list = list(ungapped_seq)
                                    for p, new_b in zip(chosen_positions, alt_combo):
                                        if p < len(seq_list):
                                            seq_list[p] = new_b
                                    mutated_seq = ''.join(seq_list)
                                combo_seqs.append(mutated_seq)
                                combo_positions_records.append(chosen_positions)
                                combo_ref_bases_records.append(ref_bases)
                                combo_alt_bases_records.append(list(alt_combo))
                            if len(combo_seqs) >= opts.combination_variant_limit:
                                break
                        if len(combo_seqs) >= opts.combination_variant_limit:
                            print(f"Combination variant cap ({opts.combination_variant_limit}) reached; stopping further combinations.")
                            break
                    if combo_seqs:
                        max_combo_size_used = max(len(r) for r in combo_positions_records)
                        # Embed all combination mutants in a single batch
                        combo_embs = embed_sequences_list(combo_seqs, model_mut, tokenizer_mut, device_mut, opts.pool_method, max_tokens)
                        num_combo_variants = combo_embs.shape[0]
                        # Build ragged representation
                        lengths = [len(r) for r in combo_positions_records]
                        starts = np.cumsum([0] + lengths[:-1])
                        flat_positions = [p for rec in combo_positions_records for p in rec]
                        flat_ref = [b for rec in combo_ref_bases_records for b in rec]
                        flat_alt = [b for rec in combo_alt_bases_records for b in rec]
                        combo_mutations = combo_embs.astype(np.float32)
                        combo_variant_starts = np.array(starts, dtype=np.int32)
                        combo_variant_lengths = np.array(lengths, dtype=np.int32)
                        combo_positions_flat = np.array(flat_positions, dtype=np.int32)
                        combo_ref_bases_flat = np.array(flat_ref)
                        combo_alt_bases_flat = np.array(flat_alt)

                out_npz = os.path.join(weight_out_dir, f"accession_{acc_key}_mutations.npz")
                meta_simple = {
                    'rank': rank,
                    'snp_index': snp_index,
                    'embedding_dim': emb_dim,
                    'weight': weight_val,
                    'accession': acc_key,
                    'num_positions': len(position_meta),
                    'num_combo_variants': num_combo_variants,
                    'max_combo_size_used': max_combo_size_used
                }
                save_kwargs = dict(
                    original=orig_embedding.astype(np.float32),
                    mutations=stacked,
                    positions=np.array([pm['position'] for pm in position_meta], dtype=np.int32),  # ungapped positions
                    alignment_positions=np.array([pm['alignment_position'] for pm in position_meta], dtype=np.int32),
                    bases=np.array([pm['base'] for pm in position_meta]),
                    alts=np.array([pm['alts'] for pm in position_meta]),
                    metadata=np.string_(str(meta_simple))
                )
                if combo_mutations is not None:
                    save_kwargs.update(dict(
                        combo_mutations=combo_mutations,
                        combo_variant_starts=combo_variant_starts,
                        combo_variant_lengths=combo_variant_lengths,
                        combo_positions_flat=combo_positions_flat,
                        combo_ref_bases_flat=combo_ref_bases_flat,
                        combo_alt_bases_flat=combo_alt_bases_flat
                    ))
                np.savez_compressed(out_npz, **save_kwargs)
                # Print per-accession combination variant count (if combinations enabled)
                if opts.mutate_combinations:
                    print(f"[Mutation] SNP {snp_index} (rank {rank}, dim {emb_dim}) accession {acc_key}: combo_variants={num_combo_variants}")
                    total_combo_variants_for_weight += num_combo_variants
                all_meta_rows.append({**meta_simple, 'file': out_npz})
            # After all accessions for this SNP/embedding dimension, print total combos
            if opts.mutate_combinations:
                print(f"[Mutation] SNP {snp_index} (rank {rank}, dim {emb_dim}) total combo variants across accessions: {total_combo_variants_for_weight}")
        if all_meta_rows:
            pd.DataFrame(all_meta_rows).to_csv(os.path.join(opts.mutation_output_dir, 'mutation_embeddings_manifest.csv'), index=False)
        print("Mutation mode completed.")

    # Always run mutation embedding
    run_mutation_mode()
    if (not opts.no_ddp) and dist.is_initialized():
        cleanup_distributed()
    print("Done!")