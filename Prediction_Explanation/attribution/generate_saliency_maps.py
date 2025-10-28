#!/usr/bin/env python3
"""
Generate integrated gradient saliency maps for DNA sequences using Caduceus model.

This script computes saliency maps using integrated gradients, which works well with
Mamba/state-space models like Caduceus. The method computes gradients of the model output
with respect to input embeddings along a path from a baseline (zero embedding) to the
actual input, providing attribution scores for each position.

The script computes AVERAGED saliency maps across multiple accessions/sequences for each
SNP/dimension pair, providing more robust and representative attribution patterns.

Usage:
    python generate_saliency_maps.py --caduceus_model_path /path/to/caduceus \
           --fasta_dir /path/to/sequences --val_ids val_ids.txt \
           --target_weights "1:161,8:101,9:226" --output_dir saliency_results
    
    # Using a weights file from explain_model_weights.py:
    python generate_saliency_maps.py --caduceus_model_path /path/to/caduceus \
           --fasta_dir /path/to/sequences --val_ids val_ids.txt \
           --weights_file weights_explanation.csv --top_n_weights 50 --output_dir saliency_results
    
    # For specific sequences only:
    python generate_saliency_maps.py --caduceus_model_path /path/to/caduceus \
           --fasta_file sequences.fa --target_weights "1:161,8:101,9:226" \
           --output_dir saliency_results
    
    # With visualization and max length constraint:
    python generate_saliency_maps.py --caduceus_model_path /path/to/caduceus \
           --fasta_dir /path/to/sequences --val_ids val_ids.txt \
           --target_weights "1:161,8:101,9:226" --output_dir saliency_results \
           --generate_plots --max_length 1024

Output:
    - Creates subfolders for each variable in the output directory
    - Saves one CSV file per SNP/dimension pair: {variable}/integrated_gradients_averaged_SNP{snp}_dim{dim}.csv
    - Each CSV contains averaged saliency scores across all accessions/sequences
    - Includes mean saliency, standard deviation, and sample size information
    - Optional visualization plots showing averaged attribution patterns

Key Features:
- Integrated gradient saliency compatible with Mamba/state-space models
- Bidirectional analysis (forward + reverse complement)
- Averaged results across multiple accessions for robust patterns
- CSV output with position-by-position statistics (mean, std, n_samples)
- Optional visualization plots
- Batch processing for multiple sequences and embedding dimensions
- Memory-efficient processing with proper gradient handling
- Configurable sequence length constraints for consistent processing
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from tqdm import tqdm
import warnings
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

STRING_COMPLEMENT_MAP = {
    "A": "T", "C": "G", "G": "C", "T": "A", "a": "t", "c": "g", "g": "c", "t": "a",
    "N": "N", "n": "n",
}

def string_reverse_complement(seq):
    """Reverse complement a DNA sequence."""
    rev_comp = ""
    for base in seq[::-1]:
        if base in STRING_COMPLEMENT_MAP:
            rev_comp += STRING_COMPLEMENT_MAP[base]
        else:
            rev_comp += base
    return rev_comp

def compute_integrated_gradients_bidirectional(model, tokenizer, sequence, embedding_dim, device='cuda', max_length=None, n_token_id=None, pooling='max'):
    """
    Compute integrated gradient saliency for a specific embedding dimension using bidirectional analysis.
    
    This method computes integrated gradients of the target embedding output with respect to the input
    embeddings, providing proper attribution scores that work well with Mamba/state-space models.
    
    Args:
        model: Caduceus model
        tokenizer: Tokenizer for the model
        sequence: DNA sequence string
        embedding_dim: Embedding dimension to analyze
        device: Device to run on
        max_length: Maximum length for sequences (if None, uses full sequence length)
        n_token_id: Token ID for 'N' nucleotide (for baseline embedding)
        pooling: 'max' or 'average' (default: 'max')
    
    Returns:
        saliency_scores: numpy array of saliency scores for each position
        max_position: position that produces the maximum embedding value
        max_value: the maximum embedding value
        forward_saliency: saliency scores for forward direction only
        rc_saliency: saliency scores for reverse complement only
    """
    import torch
    
    try:
        # Trim sequence to max_length if specified
        if max_length is not None and len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        # Prepare sequences
        rc_sequence = string_reverse_complement(sequence)
        
        # Tokenize sequences with appropriate max_length
        tokenize_max_length = max_length if max_length is not None else len(sequence)
        
        tokens = tokenizer.batch_encode_plus([sequence], add_special_tokens=False, 
                                           return_attention_mask=False, max_length=tokenize_max_length, 
                                           truncation=True)
        input_ids = torch.tensor(tokens['input_ids']).to(device)
        
        rc_tokens = tokenizer.batch_encode_plus([rc_sequence], add_special_tokens=False, 
                                              return_attention_mask=False, max_length=tokenize_max_length, 
                                              truncation=True)
        rc_input_ids = torch.tensor(rc_tokens['input_ids']).to(device)
        
        # Ensure input sequences have the same length as original sequence
        if input_ids.shape[1] != len(sequence) or rc_input_ids.shape[1] != len(sequence):
            print(f"Warning: Tokenization length mismatch. Original: {len(sequence)}, "
                  f"Forward: {input_ids.shape[1]}, RC: {rc_input_ids.shape[1]}")
            # Fallback to zero saliency and zero attributions
            zero_saliency = np.zeros(len(sequence))
            zero_attr = np.zeros((len(sequence), 128))  # Assume 128 input embedding dims as fallback
            return zero_saliency, 0, 0.0, zero_saliency, zero_saliency, zero_attr

        # Get max position and value from combined embedding (no gradients needed)
        model.eval()
        with torch.no_grad():
            forward_embedding = model(input_ids)  # (1, seq_len, emb_dim)
            rc_embedding = model(rc_input_ids)  # (1, seq_len, emb_dim)
            rc_embedding_flipped = rc_embedding.flip(dims=[1])
            combined_embedding = (forward_embedding + rc_embedding_flipped) / 2.0
            if pooling == 'average':
                max_position = None
                max_value = combined_embedding[0, :, embedding_dim].mean().item()
            else:
                max_position = combined_embedding[0, :, embedding_dim].argmax().item()
                max_value = combined_embedding[0, max_position, embedding_dim].item()
        
        # Forward direction saliency
        forward_saliency, forward_attr = compute_integrated_gradients_single_direction(
            model, input_ids, embedding_dim, device, max_position, n_token_id=n_token_id, pooling=pooling
        )

        # Reverse complement saliency - flip max position for RC
        if pooling == 'average':
            rc_max_position = None
        else:
            rc_max_position = len(sequence) - 1 - max_position
        rc_saliency, rc_attr = compute_integrated_gradients_single_direction(
            model, rc_input_ids, embedding_dim, device, rc_max_position, n_token_id=n_token_id, pooling=pooling
        )

        # Flip RC saliency and attributions to match forward orientation
        rc_saliency_flipped = np.flip(rc_saliency)
        rc_attr_flipped = np.flip(rc_attr, axis=0)

        # Combine bidirectional saliency and attributions
        combined_saliency = (forward_saliency + rc_saliency_flipped) / 2.0
        combined_attr = (forward_attr + rc_attr_flipped) / 2.0

        return combined_saliency, (max_position if max_position is not None else -1), max_value, forward_saliency, rc_saliency_flipped, combined_attr
        
    except Exception as e:
        print(f"Error in integrated gradient saliency computation: {e}")
        # Return zero saliency and zero attributions as fallback
        fallback_len = len(sequence)
        zero_saliency = np.zeros(fallback_len)
        zero_attr = np.zeros((fallback_len, 128))  # Assume 128 input embedding dims as fallback
        return zero_saliency, 0, 0.0, zero_saliency, zero_saliency, zero_attr

def compute_integrated_gradients_single_direction(model, input_ids, embedding_dim, device, max_position, n_token_id, steps=25, pooling='max'):
    """
    Compute integrated gradients saliency for a single direction using proper attribution.
    
    This method computes gradients along a straight path from a baseline (zero embedding)
    to the actual input embedding, which provides better attribution than simple gradients
    for discrete inputs like DNA tokens in Mamba/state-space models.
    
    Args:
        model: Caduceus model
        input_ids: Tokenized input sequence
        embedding_dim: Target embedding dimension
        device: Device to run on
        max_position: Position that produces the maximum embedding value (used if pooling='max')
        n_token_id: Token ID for 'N' nucleotide (for baseline embedding)
        steps: Number of integration steps (default: 50)
        pooling: 'max' or 'average' (default: 'max')
    
    Returns:
        saliency_scores: numpy array of saliency scores
    """
    import torch
    
    # Convert input_ids to tensor if not already
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids).to(device)
    
    # Get the embedding layer from the model
    embedding_layer = None
    
    # Special handling for Enformer models (they don't have traditional embeddings)
    if hasattr(model, 'model_name_or_path') and 'enformer' in model.model_name_or_path.lower():
        print("Warning: Enformer models don't use traditional token embeddings. Integrated gradients not applicable.")
        return np.zeros(input_ids.shape[1])
    
    # Try different possible locations for the embedding layer
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        
        # Check for Caduceus-style embeddings
        if hasattr(backbone, 'embeddings'):
            if hasattr(backbone.embeddings, 'word_embeddings'):
                embedding_layer = backbone.embeddings.word_embeddings
        # Check for other transformer-style embeddings
        elif hasattr(backbone, 'embed_tokens'):
            embedding_layer = backbone.embed_tokens
        # Check if backbone itself has embeddings directly
        elif hasattr(backbone, 'word_embeddings'):
            embedding_layer = backbone.word_embeddings
        # Check for double-wrapped HuggingFace models (backbone.backbone.embeddings)
        elif hasattr(backbone, 'backbone'):
            inner_backbone = backbone.backbone
            if hasattr(inner_backbone, 'embeddings'):
                if hasattr(inner_backbone.embeddings, 'word_embeddings'):
                    embedding_layer = inner_backbone.embeddings.word_embeddings
            elif hasattr(inner_backbone, 'embed_tokens'):
                embedding_layer = inner_backbone.embed_tokens
    # Direct model access (if not wrapped)
    elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
        embedding_layer = model.embeddings.word_embeddings
    elif hasattr(model, 'embed_tokens'):
        embedding_layer = model.embed_tokens
    
    if embedding_layer is None:
        print("ERROR: Cannot find embedding layer for gradient computation")
        raise RuntimeError("Cannot find embedding layer for integrated gradient computation. "
                          "Model structure is not supported. Please check model compatibility.")
    
    # Get input embeddings
    input_embeddings = embedding_layer(input_ids.long())  # (1, seq_len, embed_dim)
    n_token_id_tensor = torch.full_like(input_ids, n_token_id)
    baseline_embeddings = embedding_layer(n_token_id_tensor)
    
    # Prepare to accumulate gradients
    integrated_gradients = torch.zeros_like(input_embeddings)
    
    # For models that expect input_ids, we need to monkey-patch the embedding layer temporarily
    original_forward = embedding_layer.forward
    current_embeddings = None
    
    def patched_forward(input_ids_arg):
        # Return the current embeddings we're working with
        if current_embeddings is not None:
            return current_embeddings
        return original_forward(input_ids_arg)
    
    # Patch the embedding layer
    embedding_layer.forward = patched_forward
    
    try:
        # Integrate gradients along the path
        for step in range(steps):
            # Interpolate between baseline and input
            alpha = float(step) / (steps - 1)
            interpolated_embeddings = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
            interpolated_embeddings.requires_grad_(True)
            interpolated_embeddings.retain_grad()
            
            # Set current embeddings for the patched forward method
            current_embeddings = interpolated_embeddings
            
            # Forward pass through the model
            model.zero_grad()
            # Forward pass using monkey-patched embeddings
            outputs = model(input_ids)
            # Robust pooling/indexing for 2D or 3D outputs
            output_tensor = outputs[0]
            if output_tensor.dim() == 3:
                # (batch, seq_len, embed_dim)
                if pooling == 'average':
                    pooled = output_tensor[0, :, embedding_dim].mean()
                else:
                    pooled = output_tensor[0, max_position, embedding_dim]
            elif output_tensor.dim() == 2:
                # (seq_len, embed_dim)
                if pooling == 'average':
                    pooled = output_tensor[:, embedding_dim].mean()
                else:
                    pooled = output_tensor[max_position, embedding_dim]
            else:
                raise RuntimeError(f"Unexpected output tensor shape: {output_tensor.shape}")
            
            
            # Backward pass
            pooled.backward(retain_graph=True)
            
            # Accumulate gradients
            if interpolated_embeddings.grad is not None:
                # Check for inf/nan gradients before accumulating
                if torch.any(torch.isinf(interpolated_embeddings.grad)) or torch.any(torch.isnan(interpolated_embeddings.grad)):
                    print(f"WARNING: Found inf/nan gradients at step {step}, skipping this step")
                else:
                    integrated_gradients += interpolated_embeddings.grad / steps
            else:
                print(f"Warning: No gradients computed at step {step}")
                raise RuntimeError("No gradients computed at step {}. Stopping execution.".format(step))
    
        # Multiply by input difference to get integrated gradients
        input_diff = input_embeddings - baseline_embeddings
        attribution = integrated_gradients * input_diff
        
        # Compute saliency as L2 norm across embedding dimensions
        saliency = torch.norm(attribution[0], dim=1)  # (seq_len,)

        attribution_raw = attribution[0]

        # Check for inf/nan values in saliency and replace with zeros
        if torch.any(torch.isinf(saliency)) or torch.any(torch.isnan(saliency)):
            print(f"WARNING: Found inf/nan in saliency computation, replacing with zeros")
            saliency = torch.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.any(torch.isinf(attribution_raw)) or torch.any(torch.isnan(attribution_raw)):
            print(f"WARNING: Found inf/nan in attribution computation, replacing with zeros")
            attribution_raw = torch.nan_to_num(attribution_raw, nan=0.0, posinf=0.0, neginf=0.0)

        return saliency.detach().cpu().numpy(), attribution_raw.detach().cpu().numpy()

    except Exception as e:
        print(f"Error in integrated gradients computation: {e}")
        zero_saliency = np.zeros(input_ids.shape[1])
        zero_attr = np.zeros((input_ids.shape[1], 128))  # Assume 128 input embedding dims as fallback
        return zero_saliency, zero_attr
    
    finally:
        # Restore original embedding layer
        embedding_layer.forward = original_forward

def generate_integrated_gradients_visualization(sequence, gradient_scores, output_path, 
                                  embedding_dim, accession=None, snp_index=None):
    """
    Generate a visualization plot for integrated gradient scores.
    
    Args:
        sequence: DNA sequence string
        gradient_scores: Integrated gradient scores for each position
        output_path: Path to save the plot
        embedding_dim: Embedding dimension being analyzed
        accession: Optional accession ID
        snp_index: Optional SNP index
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the plot
        plt.figure(figsize=(16, 6))
        
        # Color mapping for nucleotides (robust: always uppercase, only ATGCN get color)
        color_map = {'A': 'red', 'T': 'blue', 'G': 'green', 'C': 'orange', 'N': 'gray'}
        # Ensure sequence is uppercased and gaps/ambiguous bases are gray
        colors = []
        for nt in sequence:
            nt_up = nt.upper()
            if nt_up in color_map:
                colors.append(color_map[nt_up])
            else:
                colors.append('gray')
        
        # Create bar plot
        positions = np.arange(len(sequence))
        bars = plt.bar(positions, gradient_scores, color=colors, alpha=0.7, width=1.0)
        
        # Customize plot
        plt.xlabel('Sequence Position')
        plt.ylabel('Integrated Gradient Attribution Score')
        
        title = f'Integrated Gradients Attribution Map - Embedding Dim {embedding_dim}'
        # if accession:
        #     title += f' - Accession {accession}'
        if snp_index is not None:
            title += f' - SNP {snp_index}'
        plt.title(title)
        
        # Add position labels on x-axis (sample every nth position for readability)
        step = max(1, len(sequence) // 20)  # Show ~50 labels max
        plt.xticks(positions[::step], positions[::step], 
                  rotation=0, fontsize=8)
        
        # Add legend for nucleotide colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[nt], label=nt) 
                          for nt in ['A', 'T', 'G', 'C'] if nt in sequence]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping visualization.")
        return False
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate integrated gradient saliency maps for DNA sequences or perform sequence alignment only.")
    parser.add_argument('--caduceus_model_path', required=False, default=None, help='Path to Caduceus model (not required for --align-only mode)')
    parser.add_argument('--fasta_dir', required=False, default=None, 
                       help='Directory containing per-accession FASTA files named <id>.fa')
    parser.add_argument('--fasta_file', required=False, default=None, 
                       help='Single FASTA file with multiple sequences (not supported with --align-only)')
    parser.add_argument('--val_ids', required=False, default=None, 
                       help='File with validation accession IDs (one per line)')
    parser.add_argument('--target_weights', required=False, default=None,
                       help='Comma-separated list of SNP:dimension pairs to analyze (e.g., "1:161,8:101,9:226")')
    parser.add_argument('--weights_file', required=False, default=None,
                       help='CSV file with important weights (variable,snp_index,embedding_dim,weight columns). If provided, uses top weights from this file instead of --target_weights')
    parser.add_argument('--target_variable', required=False, default=None,
                       help='Target variable to filter weights by (when using --weights_file)')
    parser.add_argument('--top_n_weights', required=False, default=100, type=int,
                       help='Number of top weights (by absolute magnitude) to process when using --weights_file (default: 100)')
    parser.add_argument('--output_dir', required=False, default='saliency_results', 
                       help='Output directory for results (default: saliency_results)')
    parser.add_argument('--generate_plots', action='store_true', 
                       help='Generate visualization plots for saliency maps (ignored in --align-only mode)')
    parser.add_argument('--force', action='store_true', 
                       help='Force recomputation even if output files already exist')
    parser.add_argument('--max_length', required=False, default=None, type=int,
                       help='Maximum length for sequences (sequences will be trimmed to this length if longer)')
    parser.add_argument('--pooling', required=False, default='max', choices=['max', 'average'],
                       help='Pooling method for embedding: max (default) or average (ignored in --align-only mode)')
    parser.add_argument('--align-only', action='store_true',
                       help='Only perform sequence alignment without computing saliency maps (useful for creating alignments for later use)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.fasta_dir and not args.fasta_file:
        print("Error: Must specify either --fasta_dir or --fasta_file")
        return
    
    if args.fasta_dir and not args.val_ids:
        print("Error: --val_ids is required when using --fasta_dir")
        return
    
    # For align-only mode, we don't need target weights or model loading
    if not args.align_only:
        if not args.target_weights and not args.weights_file:
            print("Error: Must specify either --target_weights or --weights_file (unless using --align-only)")
            return
        
        if not args.caduceus_model_path:
            print("Error: --caduceus_model_path is required (unless using --align-only)")
            return
    
    # For align-only mode, only fasta_dir is supported (not single fasta file)
    if args.align_only and not args.fasta_dir:
        print("Error: --align-only mode requires --fasta_dir (single fasta file not supported for alignment)")
        return
    
    # Parse target weight pairs (with variable information if available)
    target_weight_pairs = []  # List of tuples: (snp_index, embedding_dim, variable)
    
    if args.align_only:
        # For align-only mode, we need to determine which SNPs to align
        # We can either use all sequences (if no weights specified) or specific SNPs from weights
        if args.weights_file:
            # Use SNPs from weights file
            try:
                weights_df = pd.read_csv(args.weights_file)
                if 'snp_index' in weights_df.columns:
                    # Get unique SNP indices for alignment
                    unique_snps = weights_df['snp_index'].unique()
                    for snp_idx in unique_snps:
                        target_weight_pairs.append((int(snp_idx), 0, 'alignment_only'))  # dummy dim and variable
                    print(f"Align-only mode: Will align {len(unique_snps)} unique SNPs from weights file")
                else:
                    print("Error: Weights file must contain 'snp_index' column for align-only mode")
                    return
            except Exception as e:
                print(f"Error reading weights file {args.weights_file}: {e}")
                return
        elif args.target_weights:
            # Parse SNP indices from target_weights
            try:
                for pair_str in args.target_weights.split(','):
                    snp_str, _ = pair_str.strip().split(':')
                    snp_idx = int(snp_str)
                    if (snp_idx, 0, 'alignment_only') not in target_weight_pairs:
                        target_weight_pairs.append((snp_idx, 0, 'alignment_only'))  # dummy dim and variable
                print(f"Align-only mode: Will align {len(target_weight_pairs)} SNPs from target_weights")
            except ValueError:
                print("Error: Invalid format for --target_weights. Use format 'snp:dim,snp:dim' (e.g., '1:161,8:101')")
                return
        else:
            print("Error: For --align-only mode, must specify either --weights_file or --target_weights to determine which SNPs to align")
            return
    elif args.weights_file:
        # Load weights from CSV file
        try:
            weights_df = pd.read_csv(args.weights_file)
            required_cols = ['snp_index', 'embedding_dim']
            if not all(col in weights_df.columns for col in required_cols):
                print(f"Error: Weights file must contain columns: {required_cols}")
                return
            
            # Filter by target variable if specified
            if args.target_variable:
                if 'variable' not in weights_df.columns:
                    print("Error: --target_variable specified but weights file has no 'variable' column")
                    return
                
                original_len = len(weights_df)
                weights_df = weights_df[weights_df['variable'] == args.target_variable]
                
                if len(weights_df) == 0:
                    print(f"Error: No weights found for variable '{args.target_variable}'")
                    available_vars = pd.read_csv(args.weights_file)['variable'].unique()
                    print("Available variables:")
                    for var in sorted(available_vars):
                        print(f"  {var}")
                    return
                
                print(f"Filtered weights from {original_len} to {len(weights_df)} rows for variable '{args.target_variable}'")
                
                # Filter by top N weights for the specific variable
                if 'weight' in weights_df.columns:
                    weights_df['abs_weight'] = weights_df['weight'].abs()
                    weights_df = weights_df.nlargest(args.top_n_weights, 'abs_weight')
                    weights_df = weights_df.drop('abs_weight', axis=1)
                    print(f"Selected top {len(weights_df)} weights for variable '{args.target_variable}' (by absolute magnitude)")
                else:
                    print("Warning: No 'weight' column found - using all weight pairs without filtering by magnitude")
                    # If no weight column, just take first N unique pairs
                    unique_pairs = weights_df[['snp_index', 'embedding_dim']].drop_duplicates()
                    if len(unique_pairs) > args.top_n_weights:
                        selected_indices = unique_pairs.index[:args.top_n_weights]
                        weights_df = weights_df.loc[weights_df['snp_index'].isin(unique_pairs.loc[selected_indices, 'snp_index']) & 
                                                  weights_df['embedding_dim'].isin(unique_pairs.loc[selected_indices, 'embedding_dim'])]
                        print(f"Selected first {args.top_n_weights} unique weight pairs")
            else:
                # No target variable specified - select top N weights per variable
                if 'variable' not in weights_df.columns:
                    print("Warning: No 'variable' column found - treating all weights as one group")
                    # Filter by top N weights globally if no variable column
                    if 'weight' in weights_df.columns:
                        weights_df['abs_weight'] = weights_df['weight'].abs()
                        weights_df = weights_df.nlargest(args.top_n_weights, 'abs_weight')
                        weights_df = weights_df.drop('abs_weight', axis=1)
                        print(f"Selected top {len(weights_df)} weights globally (by absolute magnitude)")
                    else:
                        print("Warning: No 'weight' column found - using all weight pairs without filtering by magnitude")
                        unique_pairs = weights_df[['snp_index', 'embedding_dim']].drop_duplicates()
                        if len(unique_pairs) > args.top_n_weights:
                            selected_indices = unique_pairs.index[:args.top_n_weights]
                            weights_df = weights_df.loc[weights_df['snp_index'].isin(unique_pairs.loc[selected_indices, 'snp_index']) & 
                                                      weights_df['embedding_dim'].isin(unique_pairs.loc[selected_indices, 'embedding_dim'])]
                        print(f"Selected first {args.top_n_weights} unique weight pairs")
                else:
                    # Select top N weights per variable
                    if 'weight' in weights_df.columns:
                        filtered_dfs = []
                        variables = weights_df['variable'].unique()
                        print(f"Selecting top {args.top_n_weights} weights per variable for {len(variables)} variables:")
                        
                        for variable in variables:
                            var_df = weights_df[weights_df['variable'] == variable].copy()
                            var_df['abs_weight'] = var_df['weight'].abs()
                            var_df_top = var_df.nlargest(args.top_n_weights, 'abs_weight')
                            var_df_top = var_df_top.drop('abs_weight', axis=1)
                            filtered_dfs.append(var_df_top)
                            print(f"  - {variable}: {len(var_df_top)} weights")
                        
                        weights_df = pd.concat(filtered_dfs, ignore_index=True)
                        print(f"Total selected weights: {len(weights_df)}")
                    else:
                        print("Warning: No 'weight' column found - using all weight pairs without filtering by magnitude")
                        # Group by variable and take top N pairs per variable
                        filtered_dfs = []
                        variables = weights_df['variable'].unique()
                        print(f"Selecting first {args.top_n_weights} weight pairs per variable for {len(variables)} variables:")
                        
                        for variable in variables:
                            var_df = weights_df[weights_df['variable'] == variable]
                            unique_pairs = var_df[['snp_index', 'embedding_dim']].drop_duplicates()
                            if len(unique_pairs) > args.top_n_weights:
                                selected_indices = unique_pairs.index[:args.top_n_weights]
                                var_df_filtered = var_df.loc[var_df['snp_index'].isin(unique_pairs.loc[selected_indices, 'snp_index']) & 
                                                            var_df['embedding_dim'].isin(unique_pairs.loc[selected_indices, 'embedding_dim'])]
                                filtered_dfs.append(var_df_filtered)
                                print(f"  - {variable}: {args.top_n_weights} pairs")
                            else:
                                filtered_dfs.append(var_df)
                                print(f"  - {variable}: {len(unique_pairs)} pairs (all available)")
                        
                        weights_df = pd.concat(filtered_dfs, ignore_index=True)
                        print(f"Total selected weight pairs: {len(weights_df)}")
            
            # Get unique SNP:dimension pairs from filtered data with variable information
            for _, row in weights_df.iterrows():
                variable = row.get('variable', 'unknown') if 'variable' in weights_df.columns else 'unknown'
                pair = (int(row['snp_index']), int(row['embedding_dim']), variable)
                if pair not in target_weight_pairs:
                    target_weight_pairs.append(pair)
            
            filter_msg = f" (filtered for {args.target_variable})" if args.target_variable else ""
            print(f"Loaded {len(target_weight_pairs)} unique weight pairs from {args.weights_file}{filter_msg}")
            
        except Exception as e:
            print(f"Error reading weights file {args.weights_file}: {e}")
            return
    
    else:
        # Parse from command line (no variable information available)
        try:
            for pair_str in args.target_weights.split(','):
                snp_str, dim_str = pair_str.strip().split(':')
                target_weight_pairs.append((int(snp_str), int(dim_str), 'unknown'))
        except ValueError:
            print("Error: Invalid format for --target_weights. Use format 'snp:dim,snp:dim' (e.g., '1:161,8:101')")
            return
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load validation IDs if using fasta_dir
    val_id_set = None
    if args.val_ids:
        with open(args.val_ids, 'r') as f:
            content = f.read().strip()
            val_id_set = content.split()
        print(f"Loaded {len(val_id_set)} validation accession IDs")
    
    # Set up GPU (only needed for saliency computation)
    device = None
    n_token_id = None
    caduceus_model = None
    tokenizer = None
    
    if not args.align_only:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Using CPU (will be slow).")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load Caduceus model and tokenizer
        try:
            from Masters.caduceus.embed_sequences import DNAEmbeddingModel, EnformerTokenizer
            print("Successfully imported Caduceus modules")
        except Exception as e:
            print(f"Error importing Caduceus modules: {e}")
            return
        
        try:
            print("Loading Caduceus model...")
            caduceus_model = DNAEmbeddingModel(args.caduceus_model_path)
            caduceus_model = caduceus_model.to(device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading Caduceus model: {e}")
            return
        
        try:
            print("Loading tokenizer...")
            if 'enformer' in args.caduceus_model_path.lower():
                tokenizer = EnformerTokenizer()
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.caduceus_model_path, trust_remote_code=True)
            print("Tokenizer loaded successfully")
            n_token_id = tokenizer.convert_tokens_to_ids("N")
            print(f"Token ID for 'N': {n_token_id}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return
    else:
        print("Align-only mode: Skipping model and tokenizer loading")
    
    # Process sequences
    if args.fasta_dir:
        process_fasta_directory(caduceus_model, tokenizer, args, target_weight_pairs, val_id_set, device, n_token_id)
    else:
        process_fasta_file(caduceus_model, tokenizer, args, target_weight_pairs, device, n_token_id)

def load_sequences_from_fasta(fasta_file):
    """Load sequences from a FASTA file."""
    try:
        from Bio import SeqIO
        sequences = []
        for rec in SeqIO.parse(fasta_file, 'fasta'):
            sequences.append(str(rec.seq))
        return sequences
    except Exception as e:
        print(f"Error loading sequences from {fasta_file}: {e}")
        return []

def save_snp_dimension_results(args, snp_index, embedding_dim, variable, data):
    """Save averaged results for a single SNP/dimension pair immediately after accumulating data."""
    
    if len(data['saliencies']) == 0:
        return
        
    # Create subfolder for the variable
    variable_dir = os.path.join(args.output_dir, variable)
    os.makedirs(variable_dir, exist_ok=True)
        
    # Check if output file already exists
    output_filename = f"integrated_gradients_averaged_SNP{snp_index}_dim{embedding_dim}.csv"
    output_path = os.path.join(variable_dir, output_filename)
    
    if not args.force and os.path.exists(output_path):
        return
        
    try:
        # Find minimum length across all sequences first
        min_len = min(len(sal) for sal in data['saliencies'])
        min_len = min(min_len, min(len(sal) for sal in data['forward_saliencies']))
        min_len = min(min_len, min(len(sal) for sal in data['rc_saliencies']))
        
        # Trim all sequences to minimum length before converting to arrays
        trimmed_saliencies = [sal[:min_len] for sal in data['saliencies']]
        trimmed_forward = [sal[:min_len] for sal in data['forward_saliencies']]
        trimmed_rc = [sal[:min_len] for sal in data['rc_saliencies']]
        
        # Convert to numpy arrays (now all have same length)
        saliencies = np.array(trimmed_saliencies)
        forward_saliencies = np.array(trimmed_forward)
        rc_saliencies = np.array(trimmed_rc)
        
        # Average across sequences
        mean_saliency = np.mean(saliencies, axis=0)
        mean_forward = np.mean(forward_saliencies, axis=0)
        mean_rc = np.mean(rc_saliencies, axis=0)
        
        # Compute metadata
        mean_max_val = np.mean(data['max_values'])
        mean_max_pos = np.mean(data['max_positions'])
        
        # Handle both accessions (from directory processing) and sequence_ids (from file processing)
        if 'accessions' in data and data['accessions']:
            n_samples = len(data['accessions'])
            sample_type = 'accessions'
        elif 'sequence_ids' in data and data['sequence_ids']:
            n_samples = len(data['sequence_ids'])
            sample_type = 'sequences'
        else:
            n_samples = len(data['saliencies'])
            sample_type = 'samples'
        
        
        # Trim all sequences to minimum length for nucleotide frequency calculation
        trimmed_sequences = [seq[:min_len] for seq in data['sequences']]
        
        # Calculate nucleotide frequencies for each position
        nucleotide_frequencies = []
        for pos in range(min_len):
            nucleotides = [seq[pos] if pos < len(seq) else 'N' for seq in trimmed_sequences]
            total = len(nucleotides)
            frequencies = {
                'pct_A': round((nucleotides.count('A') / total) * 100, 2),
                'pct_T': round((nucleotides.count('T') / total) * 100, 2),
                'pct_G': round((nucleotides.count('G') / total) * 100, 2),
                'pct_C': round((nucleotides.count('C') / total) * 100, 2),
                'pct_N': round((nucleotides.count('N') / total) * 100, 2)
            }
            nucleotide_frequencies.append(frequencies)
        
        # Build representative sequence from majority nucleotide at each position
        representative_sequence = ''
        for pos_freqs in nucleotide_frequencies:
            # Find the nucleotide with the highest percentage (A, T, G, C, N)
            max_nt = max(['A', 'T', 'G', 'C', 'N'], key=lambda nt: pos_freqs[f'pct_{nt}'])
            representative_sequence += max_nt
        
        # Trim representative sequence to minimum length
        representative_sequence = representative_sequence[:min_len]
        
        # Create output data
        saliency_data = []
        for pos in range(min_len):
            row_data = {
                'snp_index': snp_index,
                'embedding_dim': embedding_dim,
                'position': pos,
                'mean_saliency': mean_saliency[pos],
                'std_saliency': np.std([sal[pos] for sal in saliencies]),
                'mean_forward_saliency': mean_forward[pos],
                'mean_rc_saliency': mean_rc[pos],
                'n_samples': n_samples
            }
            # Add nucleotide frequency percentages
            row_data.update(nucleotide_frequencies[pos])
            saliency_data.append(row_data)
        
        df = pd.DataFrame(saliency_data)
        
        # Add metadata
        metadata_row = {
            'snp_index': snp_index,
            'embedding_dim': embedding_dim,
            'position': -1,  # Special marker for metadata
            'mean_saliency': mean_max_val,
            'std_saliency': mean_max_pos,
            'mean_forward_saliency': n_samples,
            'mean_rc_saliency': 0,
            'n_samples': n_samples,
            'pct_A': 0, 'pct_T': 0, 'pct_G': 0, 'pct_C': 0, 'pct_N': 0,
            'pooling': getattr(args, 'pooling', 'max')
        }
        df = pd.concat([pd.DataFrame([metadata_row]), df], ignore_index=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Generate visualization if requested
        if args.generate_plots:
            plot_filename = f"integrated_gradients_averaged_SNP{snp_index}_dim{embedding_dim}.png"
            plot_path = os.path.join(variable_dir, plot_filename)
            generate_integrated_gradients_visualization(
                representative_sequence, mean_saliency, plot_path, embedding_dim, 
                f"Averaged_across_{n_samples}_{sample_type}", snp_index
            )
            
    except Exception as e:
        print(f"Error saving averaged results for SNP{snp_index} dim{embedding_dim}: {e}")

def process_fasta_directory(model, tokenizer, args, target_weight_pairs, val_id_set, device, n_token_id):
    """Process sequences from a directory of per-accession FASTA files and compute averaged integrated gradients."""
    from Bio import SeqIO
    from collections import defaultdict
    import tempfile
    import shutil
    import subprocess
    from Bio import AlignIO
    from Bio.Align.Applications import MafftCommandline, ClustalwCommandline
    import numpy as np

    print(f"Processing {len(target_weight_pairs)} weight pairs across {len(val_id_set)} accessions...")
    print(f"Alignments will be saved to: {os.path.abspath(args.output_dir)}/alignments/")

    # 1. Identify all unique SNP indices to process
    snp_indices = sorted(set(pair[0] for pair in target_weight_pairs))
    print(f"Unique SNP indices to process: {snp_indices}")

    # 2. For each SNP index, collect all sequences and perform MSA (cache results)
    # Create alignments directory within output directory
    alignments_dir = os.path.join(args.output_dir, "alignments")
    os.makedirs(alignments_dir, exist_ok=True)
    
    snp_alignment_cache = {}
    snp_accession_order = {}
    snp_seqid_to_acc = {}
    for snp_index in tqdm(snp_indices, desc="Aligning SNPs"):
        # Check if alignment already exists
        saved_alignment_path = os.path.join(alignments_dir, f"snp{snp_index}_aligned.fasta")
        
        if os.path.exists(saved_alignment_path) and not args.force:
            print(f"Loading existing alignment for SNP {snp_index}: {saved_alignment_path}")
            try:
                # Read existing alignment
                alignment = AlignIO.read(saved_alignment_path, "fasta")
                acc_order = [rec.id for rec in alignment]
                snp_alignment_cache[snp_index] = alignment
                snp_accession_order[snp_index] = acc_order
                snp_seqid_to_acc[snp_index] = {rec.id: acc for rec, acc in zip(alignment, acc_order)}
                continue
            except Exception as e:
                print(f"Error loading existing alignment for SNP {snp_index}: {e}")
                # Fall through to create new alignment
        
        seq_records = []
        acc_order = []
        for acc_id in val_id_set:
            fasta_path = os.path.join(args.fasta_dir, f"{acc_id}.fa")
            if not os.path.exists(fasta_path):
                continue
            for seq_idx, rec in enumerate(SeqIO.parse(fasta_path, 'fasta')):
                if seq_idx == snp_index:
                    # Use accession as record id for mapping
                    rec.id = acc_id
                    rec.description = ""
                    seq_records.append(rec)
                    acc_order.append(acc_id)
                    break
        if not seq_records:
            print(f"Warning: No sequences found for SNP index {snp_index}")
            continue
        
        print(f"Creating alignment for SNP {snp_index} with {len(seq_records)} sequences...")
        
        # Write to temp fasta
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_tmp = os.path.join(tmpdir, f"snp{snp_index}_msa.fasta")
            SeqIO.write(seq_records, fasta_tmp, "fasta")
            # Try MAFFT first
            aligned_fasta = os.path.join(tmpdir, f"snp{snp_index}_aligned.fasta")
            mafft_ok = False
            try:
                mafft_cline = MafftCommandline(input=fasta_tmp)
                mafft_cline.set_parameter('--auto', True)
                stdout, stderr = mafft_cline()
                with open(aligned_fasta, "w") as af:
                    af.write(stdout)
                mafft_ok = True
            except Exception as e:
                print(f"MAFFT failed for SNP {snp_index}: {e}")
            if not mafft_ok:
                # Try ClustalW
                try:
                    clustalw_exe = shutil.which("clustalw") or shutil.which("clustalw2")
                    if clustalw_exe is None:
                        raise RuntimeError("ClustalW not found in PATH")
                    clustalw_cline = ClustalwCommandline(clustalw_exe, infile=fasta_tmp)
                    stdout, stderr = clustalw_cline()
                    aln_file = fasta_tmp.replace(".fasta", ".aln")
                    aligned_fasta = aln_file
                except Exception as e:
                    print(f"ClustalW failed for SNP {snp_index}: {e}")
                    continue
            # Read alignment
            alignment = AlignIO.read(aligned_fasta, "fasta")
            snp_alignment_cache[snp_index] = alignment
            snp_accession_order[snp_index] = acc_order
            snp_seqid_to_acc[snp_index] = {rec.id: acc for rec, acc in zip(alignment, acc_order)}
            
            # Save alignment to permanent location
            try:
                with open(saved_alignment_path, "w") as out_f:
                    AlignIO.write(alignment, out_f, "fasta")
                print(f"Saved alignment for SNP {snp_index}: {saved_alignment_path}")
                
                # Also save alignment statistics
                stats_path = os.path.join(alignments_dir, f"snp{snp_index}_alignment_stats.txt")
                with open(stats_path, "w") as stats_f:
                    stats_f.write(f"SNP Index: {snp_index}\n")
                    stats_f.write(f"Number of sequences: {len(alignment)}\n")
                    stats_f.write(f"Alignment length: {len(alignment[0].seq)}\n")
                    stats_f.write(f"Accession order: {', '.join(acc_order)}\n")
                    
                    # Calculate gap statistics
                    total_positions = len(alignment) * len(alignment[0].seq)
                    gap_count = sum(str(rec.seq).count('-') for rec in alignment)
                    gap_percentage = (gap_count / total_positions) * 100
                    stats_f.write(f"Gap percentage: {gap_percentage:.2f}%\n")
                    
                    # Calculate conservation statistics per position
                    conservation_scores = []
                    for pos in range(len(alignment[0].seq)):
                        column = [str(rec.seq)[pos] for rec in alignment]
                        non_gap_column = [c for c in column if c != '-']
                        if non_gap_column:
                            # Most frequent non-gap character
                            from collections import Counter
                            counter = Counter(non_gap_column)
                            most_common_count = counter.most_common(1)[0][1]
                            conservation = most_common_count / len(non_gap_column)
                            conservation_scores.append(conservation)
                        else:
                            conservation_scores.append(0.0)
                    
                    avg_conservation = sum(conservation_scores) / len(conservation_scores)
                    stats_f.write(f"Average conservation: {avg_conservation:.3f}\n")
                    
                print(f"Saved alignment statistics: {stats_path}")
                
            except Exception as e:
                print(f"Warning: Could not save alignment for SNP {snp_index}: {e}")
                # Continue with cached alignment even if saving failed

    # If align-only mode, stop here after creating all alignments
    if args.align_only:
        print(f"Align-only mode: Completed alignment of {len(snp_indices)} SNPs.")
        print(f"Alignments saved to: {alignments_dir}")
        return

    # 3. For each SNP/dim/variable, map gradients to alignment and average
    for pair_idx, (snp_index, embedding_dim, variable) in enumerate(tqdm(target_weight_pairs, desc="Processing SNP/dimension pairs")):
        # Check if output file already exists and skip if not forcing recomputation
        variable_dir = os.path.join(args.output_dir, variable)
        output_filename = f"integrated_gradients_averaged_SNP{snp_index}_dim{embedding_dim}.csv"
        output_path = os.path.join(variable_dir, output_filename)
        
        if not args.force and os.path.exists(output_path):
            print(f"Skipping SNP{snp_index} dim{embedding_dim} - output file already exists: {output_path}")
            continue
        
        # Check if alignment exists for this SNP
        if snp_index not in snp_alignment_cache:
            print(f"Skipping SNP{ snp_index } (no alignment)")
            continue
        alignment = snp_alignment_cache[snp_index]
        acc_order = snp_accession_order[snp_index]
        aln_length = alignment[0].seq.__len__()
        # Prepare to collect gradients for each aligned sequence
        acc_to_alnseq = {rec.id: str(rec.seq) for rec in alignment}
        # For each accession, get the original sequence and compute gradients
        gradients_aligned = []
        forward_aligned = []
        rc_aligned = []
        raw_attrs_aligned = []
        seqs_aligned = []
        max_positions = []
        max_values = []
        used_accs = []
        for acc_id in acc_order:
            fasta_path = os.path.join(args.fasta_dir, f"{acc_id}.fa")
            if not os.path.exists(fasta_path):
                continue
            # Get original sequence for this SNP
            orig_seq = None
            for seq_idx, rec in enumerate(SeqIO.parse(fasta_path, 'fasta')):
                if seq_idx == snp_index:
                    orig_seq = str(rec.seq)
                    break
            if orig_seq is None:
                continue
            # Compute gradients (no trimming/max_length)
            try:
                saliency, max_pos, max_val, forward_sal, rc_sal, raw_attr = compute_integrated_gradients_bidirectional(
                    model, tokenizer, orig_seq, embedding_dim, device, None, n_token_id, pooling=args.pooling
                )
            except Exception as e:
                print(f"Error computing gradients for {acc_id} SNP{snp_index} dim{embedding_dim}: {e}")
                continue
            # Map gradients to alignment (insert NaN for gaps)
            aln_seq = acc_to_alnseq[acc_id]
            grad_aln = []
            fwd_aln = []
            rc_aln = []
            raw_attr_aln = []
            seq_aln = []
            seq_pos = 0
            for c in aln_seq:
                if c == "-":
                    grad_aln.append(np.nan)
                    fwd_aln.append(np.nan)
                    rc_aln.append(np.nan)
                    raw_attr_aln.append(np.full(raw_attr.shape[1], np.nan))  # NaN for all input embedding dims
                    seq_aln.append("-")
                else:
                    grad_aln.append(saliency[seq_pos])
                    fwd_aln.append(forward_sal[seq_pos])
                    rc_aln.append(rc_sal[seq_pos])
                    raw_attr_aln.append(raw_attr[seq_pos])  # All input embedding dims for this position
                    seq_aln.append(c)
                    seq_pos += 1
            gradients_aligned.append(grad_aln)
            forward_aligned.append(fwd_aln)
            rc_aligned.append(rc_aln)
            raw_attrs_aligned.append(raw_attr_aln)
            seqs_aligned.append(seq_aln)
            max_positions.append(max_pos)
            max_values.append(max_val)
            used_accs.append(acc_id)
        if not gradients_aligned:
            print(f"No gradients for SNP{snp_index} dim{embedding_dim}")
            continue
        # Transpose to get per-position arrays
        gradients_aligned = np.array(gradients_aligned)
        forward_aligned = np.array(forward_aligned)
        rc_aligned = np.array(rc_aligned)
        # Ensure seqs_aligned is a numpy array of str type for robust comparison
        seqs_aligned = np.array(seqs_aligned, dtype=str)
        # Compute mean/std ignoring NaN
        mean_saliency = np.nanmean(gradients_aligned, axis=0)
        std_saliency = np.nanstd(gradients_aligned, axis=0)
        mean_forward = np.nanmean(forward_aligned, axis=0)
        mean_rc = np.nanmean(rc_aligned, axis=0)
        n_samples = np.sum(~np.isnan(gradients_aligned), axis=0)
        # Nucleotide frequencies and gap fraction
        nucleotide_frequencies = []
        for pos in range(aln_length):
            nts = seqs_aligned[:, pos]
            # Flatten and ensure string type, then uppercase for robust comparison
            nts = np.array(nts, dtype=str).flatten()
            nts = np.char.upper(nts)
            total_seqs = len(nts)
            n_gap = np.sum(nts == "-")
            nts_no_gap = nts[nts != "-"]
            total = len(nts_no_gap)
            # Debug print to help diagnose content
            if pos == 0 or pos == aln_length - 1:
                print(f"DEBUG: aln pos {pos} nts: {nts.tolist()} nts_no_gap: {nts_no_gap.tolist()} n_gap: {n_gap} total: {total}")
            pct_gap = round(n_gap / total_seqs * 100, 2) if total_seqs > 0 else 0.0
            if total == 0:
                freqs = {nt: 0.0 for nt in ["A", "T", "G", "C", "N"]}
            else:
                freqs = {
                    "A": round(np.sum(nts_no_gap == "A") / total * 100, 2),
                    "T": round(np.sum(nts_no_gap == "T") / total * 100, 2),
                    "G": round(np.sum(nts_no_gap == "G") / total * 100, 2),
                    "C": round(np.sum(nts_no_gap == "C") / total * 100, 2),
                    "N": round(np.sum(nts_no_gap == "N") / total * 100, 2),
                }
            freq_row = {f"pct_{nt}": freqs[nt] for nt in ["A", "T", "G", "C", "N"]}
            freq_row["pct_gap"] = pct_gap
            nucleotide_frequencies.append(freq_row)
        # Build representative sequence (majority base, ignoring gaps)
        representative_sequence = ""
        for pos in range(aln_length):
            nts = seqs_aligned[:, pos]
            nts = np.array(nts, dtype=str)
            nts = nts[nts != "-"]
            if len(nts) == 0:
                representative_sequence += "N"
            else:
                unique, counts = np.unique(nts, return_counts=True)
                representative_sequence += unique[np.argmax(counts)]
        # Prepare output data
        saliency_data = []
        for pos in range(aln_length):
            row_data = {
                'snp_index': snp_index,
                'embedding_dim': embedding_dim,
                'position': pos,
                'mean_saliency': mean_saliency[pos],
                'std_saliency': std_saliency[pos],
                'mean_forward_saliency': mean_forward[pos],
                'mean_rc_saliency': mean_rc[pos],
                'n_samples': int(n_samples[pos])
            }
            row_data.update(nucleotide_frequencies[pos])
            saliency_data.append(row_data)
        df = pd.DataFrame(saliency_data)
        # Add metadata
        mean_max_val = np.nanmean(max_values)
        mean_max_pos = np.nanmean(max_positions)
        metadata_row = {
            'snp_index': snp_index,
            'embedding_dim': embedding_dim,
            'position': -1,
            'mean_saliency': mean_max_val,
            'std_saliency': mean_max_pos,
            'mean_forward_saliency': len(used_accs),
            'mean_rc_saliency': 0,
            'n_samples': len(used_accs),
            'pct_A': 0, 'pct_T': 0, 'pct_G': 0, 'pct_C': 0, 'pct_N': 0,
            'pooling': getattr(args, 'pooling', 'max')
        }
        df = pd.concat([pd.DataFrame([metadata_row]), df], ignore_index=True)
        # Save to CSV
        variable_dir = os.path.join(args.output_dir, variable)
        os.makedirs(variable_dir, exist_ok=True)
        output_filename = f"integrated_gradients_averaged_SNP{snp_index}_dim{embedding_dim}.csv"
        output_path = os.path.join(variable_dir, output_filename)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        # Generate visualization if requested
        if args.generate_plots:
            plot_filename = f"integrated_gradients_averaged_SNP{snp_index}_dim{embedding_dim}.png"
            plot_path = os.path.join(variable_dir, plot_filename)
            # Use the consensus/representative sequence for coloring
            generate_integrated_gradients_visualization(
                representative_sequence, mean_saliency, plot_path, embedding_dim,
                f"Averaged_across_{len(used_accs)}_accessions", snp_index
            )
        
        # Process and save raw attributions
        if raw_attrs_aligned:
            try:
                # Check if raw attributions file already exists
                raw_output_filename = f"raw_attributions_averaged_SNP{snp_index}_dim{embedding_dim}.csv"
                raw_output_path = os.path.join(variable_dir, raw_output_filename)
                
                if not args.force and os.path.exists(raw_output_path):
                    print(f"Skipping raw attributions for SNP{snp_index} dim{embedding_dim} - file already exists: {raw_output_path}")
                else:
                    # Convert to numpy array and average across accessions, ignoring NaN
                    raw_attrs_array = np.array(raw_attrs_aligned)  # (n_accs, aln_length, input_embedding_dims)
                    mean_raw_attrs = np.nanmean(raw_attrs_array, axis=0)  # (aln_length, input_embedding_dims)
                    
                    # Save as long-format CSV: position, input_embedding_dim, mean_attribution
                    rows = []
                    for pos in range(mean_raw_attrs.shape[0]):
                        for input_dim in range(mean_raw_attrs.shape[1]):
                            rows.append({
                                'position': pos, 
                                'input_embedding_dim': input_dim, 
                                'mean_attribution': mean_raw_attrs[pos, input_dim]
                            })
                    pd.DataFrame(rows).to_csv(raw_output_path, index=False)
                    print(f"Saved: {raw_output_path}")
            except Exception as e:
                print(f"Error saving raw attributions for SNP{snp_index} dim{embedding_dim}: {e}")
    print(f"Completed processing all {len(target_weight_pairs)} SNP/dimension pairs.")

def process_fasta_file(model, tokenizer, args, target_weight_pairs, device, n_token_id):
    """Process sequences from a single FASTA file and compute averaged integrated gradients."""
    from Bio import SeqIO
    from collections import defaultdict
    
    # Load all sequences
    sequences = {}
    for seq_idx, rec in enumerate(SeqIO.parse(args.fasta_file, 'fasta')):
        sequences[seq_idx] = (rec.id, str(rec.seq))
    
    print(f"Processing {len(target_weight_pairs)} weight pairs for {len(sequences)} sequences...")
    
    # Process each SNP/dimension pair
    for snp_index, embedding_dim, variable in tqdm(target_weight_pairs, desc="Processing pairs"):
        # Check if output file already exists and skip if not forcing recomputation
        variable_dir = os.path.join(args.output_dir, variable)
        output_filename = f"integrated_gradients_averaged_SNP{snp_index}_dim{embedding_dim}.csv"
        output_path = os.path.join(variable_dir, output_filename)
        
        if not args.force and os.path.exists(output_path):
            print(f"Skipping SNP{snp_index} dim{embedding_dim} - output file already exists: {output_path}")
            continue
        
        pair_data = defaultdict(list)
        
        if snp_index in sequences:
            # Process specific sequence
            seq_id, sequence = sequences[snp_index]
            try:
                saliency, max_pos, max_val, forward_sal, rc_sal, raw_attr = compute_integrated_gradients_bidirectional(
                    model, tokenizer, sequence, embedding_dim, device, args.max_length, n_token_id, pooling=args.pooling
                )
                pair_data['saliencies'].append(saliency)
                pair_data['forward_saliencies'].append(forward_sal)
                pair_data['rc_saliencies'].append(rc_sal)
                pair_data['raw_attributions'].append(raw_attr)
                pair_data['sequences'].append(sequence)
                pair_data['max_positions'].append(max_pos)
                pair_data['max_values'].append(max_val)
                pair_data['sequence_ids'].append(seq_id)
            except Exception as e:
                print(f"Error processing {seq_id}: {e}")
        else:
            # Process all sequences for this dimension
            for seq_idx, (seq_id, sequence) in sequences.items():
                try:
                    saliency, max_pos, max_val, forward_sal, rc_sal, raw_attr = compute_integrated_gradients_bidirectional(
                        model, tokenizer, sequence, embedding_dim, device, args.max_length, n_token_id, pooling=args.pooling
                    )
                    pair_data['saliencies'].append(saliency)
                    pair_data['forward_saliencies'].append(forward_sal)
                    pair_data['rc_saliencies'].append(rc_sal)
                    pair_data['raw_attributions'].append(raw_attr)
                    pair_data['sequences'].append(sequence)
                    pair_data['max_positions'].append(max_pos)
                    pair_data['max_values'].append(max_val)
                    pair_data['sequence_ids'].append(seq_id)
                except Exception as e:
                    print(f"Error processing {seq_id}: {e}")
        
        # Save results immediately
        if pair_data['saliencies']:
            save_snp_dimension_results(args, snp_index, embedding_dim, variable, pair_data)
            
            # Save raw attributions if available
            if 'raw_attributions' in pair_data and pair_data['raw_attributions']:
                try:
                    # Check if raw attributions file already exists
                    variable_dir = os.path.join(args.output_dir, variable)
                    os.makedirs(variable_dir, exist_ok=True)
                    raw_output_filename = f"raw_attributions_averaged_SNP{snp_index}_dim{embedding_dim}.csv"
                    raw_output_path = os.path.join(variable_dir, raw_output_filename)
                    
                    if not args.force and os.path.exists(raw_output_path):
                        print(f"Skipping raw attributions for SNP{snp_index} dim{embedding_dim} - file already exists: {raw_output_path}")
                        continue
                    
                    # Find minimum length across all sequences
                    min_len = min(len(attr) for attr in pair_data['raw_attributions'])
                    
                    # Trim all attributions to minimum length and stack
                    trimmed_attrs = [attr[:min_len] for attr in pair_data['raw_attributions']]
                    raw_attrs_array = np.stack(trimmed_attrs, axis=0)  # (n_samples, seq_len, input_embedding_dims)
                    
                    # Average across samples
                    mean_raw_attrs = np.mean(raw_attrs_array, axis=0)  # (seq_len, input_embedding_dims)
                    
                    # Save as long-format CSV
                    variable_dir = os.path.join(args.output_dir, variable)
                    os.makedirs(variable_dir, exist_ok=True)
                    raw_output_filename = f"raw_attributions_averaged_SNP{snp_index}_dim{embedding_dim}.csv"
                    raw_output_path = os.path.join(variable_dir, raw_output_filename)
                    
                    rows = []
                    for pos in range(mean_raw_attrs.shape[0]):
                        for input_dim in range(mean_raw_attrs.shape[1]):
                            rows.append({
                                'position': pos, 
                                'input_embedding_dim': input_dim, 
                                'mean_attribution': mean_raw_attrs[pos, input_dim]
                            })
                    pd.DataFrame(rows).to_csv(raw_output_path, index=False)
                    print(f"Saved: {raw_output_path}")
                except Exception as e:
                    print(f"Error saving raw attributions for SNP{snp_index} dim{embedding_dim}: {e}")
    
    print(f"Completed processing all {len(target_weight_pairs)} SNP/dimension pairs.")
if __name__ == "__main__":
    main()
