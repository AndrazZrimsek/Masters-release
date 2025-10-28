import argparse
import os
from functools import partial
from os import path as osp
from typing import Dict, Iterable, Optional
import sys

import enformer_pytorch
import fsspec
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn import preprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator

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


def fsspec_exists(filename):
    """Check if file exists in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


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
    """Get the backbone model."""

    model = DNAEmbeddingModel(
        model_name_or_path=args.model_name_or_path,
    )
    model.eval()
    return DDP(model.to(device))


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
    opts, _ = parser.parse_known_args()

    print("*** Args ************************")
    for k, v in vars(opts).items():
        print(f"  - {k}: {v}")
    print("******************************\n")

    column_ids_file = "/d/hpc/projects/arabidopsis_fri/Masters/src/columnIDs.txt"
    # accessions_dir = "/d/hpc/projects/arabidopsis_fri/Masters/Data/Pseudogenomes/SparSNP_New_100"
    opts.column_ids_file = column_ids_file
    # opts.accessions_dir = opts.input_dir if opts.input_dir else accessions_dir
    # opts.output_file = "/d/hpc/projects/arabidopsis_fri/Masters/Embeddings/CaduceusEmbeddings.pkl"
    if opts.accessions_dir is None:
        raise ValueError("--accessions_dir must be specified.")

    if opts.output_file is None:
        opts.output_file = os.path.join(opts.downstream_save_dir, f"{opts.name}_embeddings.pkl")

    print(f"Output file will be saved to: {opts.output_file}")
    print(f"Accessions directory: {opts.accessions_dir}")

    # Initialize distributed environment
    print("Initializing distributed...")
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{dist.get_rank()}")

    # Initialize tokenizer
    print("Initializing tokenizer...")
    if "enformer" in opts.model_name_or_path.lower():
        tokenizer = EnformerTokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path, trust_remote_code=True)

    # Load model
    print(f"Loading model from {opts.model_name_or_path}...")
    model = get_backbone_model(opts, device)

    # Load accession IDs
    print(f"Reading accession IDs from {opts.column_ids_file}...")
    try:
        with open(opts.column_ids_file, "r") as f:
            accession_ids = f.read().split()
    except FileNotFoundError:
        print(f"File not found: {opts.column_ids_file}")
        sys.exit(1)

    # Load sequences from FASTA files
    print(f"Loading sequences from {opts.accessions_dir}...")
    from Bio import SeqIO
    accession_sequences = {}
    
    for accession in tqdm(accession_ids):
        accession_file = f"{opts.accessions_dir}/{accession}.fa"
        try:
            sequences = {1: [], 2: [], 3: [], 4: [], 5: []}
            with open(accession_file) as f:
                for record in SeqIO.parse(f, 'fasta'):
                    id_parts = record.id.split('|')
                    if len(id_parts) >= 5:
                        region = id_parts[4]
                        if region.startswith('Chr'):
                            try:
                                chromosome = int(region[3])
                                if chromosome in sequences:
                                    sequences[chromosome].append(str(record.seq))
                            except ValueError:
                                # Skip non-numeric chromosomes
                                pass
                    else:
                        print(f"Warning: Record ID {record.id} doesn't match expected format")
            
            # Only include accession if we found sequences
            if any(seqs for seqs in sequences.values()):
                accession_sequences[accession] = sequences
                seq_counts = {chr_num: len(seqs) for chr_num, seqs in sequences.items()}
                print(f"Loaded {sum(seq_counts.values())} sequences for {accession}: {seq_counts}")
            else:
                print(f"Warning: No valid sequences found for {accession}")
        except FileNotFoundError:
            print(f"Warning: Could not find FASTA file for {accession}")

    # Embed sequences
    if accession_sequences:
        print(f"Embedding sequences for {len(accession_sequences)} accessions...")
        embeddings = embed_custom_sequences(accession_sequences, model, tokenizer, device, opts)

        # all_concat = {}
        # for id, chromosomes in embeddings.items():
        #     embed_per_chrom = []
        #     max_per_chrom = []
        #     for chromosome, seqs in enumerate(chromosomes):
        #         embed_per_chrom.append(np.hstack(seqs))
        #         max_per_chrom.append(np.max(seqs, 0))
        #     all_concat[id] = np.hstack(embed_per_chrom)
        
        # Save embeddings
        print(f"Saving embeddings to {opts.output_file}...")
        import os
        import pickle
        os.makedirs(os.path.dirname(opts.output_file), exist_ok=True)
        with open(opts.output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Successfully saved embeddings for {len(embeddings)} accessions")
    else:
        print("No sequences to embed")
    
    # Clean up
    print("Cleaning up...")
    cleanup_distributed()
    print("Done!")