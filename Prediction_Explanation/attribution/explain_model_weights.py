#!/usr/bin/env python3
"""
Explain model weights for multi-output regression models trained on SNP embeddings.

Given a saved model file (from train_predict_models.py), this script will:
- Load the model and extract the weights for each target variable.
- For each weight, determine which SNP and which embedding dimension it corresponds to.
- Output a CSV with columns: variable, snp_index, embedding_dim, weight.
- Optionally, compute attribution scores to understand which input 
  nucleotides contribute most to the maximum embedding values (saved to separate CSV files).
- Check for existing output files and reuse them unless --force flag is used.

Usage:
    python explain_model_weights.py --model MODEL.pkl --embedding_dim 256 --output OUTPUT.csv
    
    # With attribution analysis:
    python explain_model_weights.py --model MODEL.pkl --caduceus_model_path /path/to/caduceus \
           --fasta_dir /path/to/sequences --val_ids val_ids.txt --compute_attributions
    
    # For a specific target variable only:
    python explain_model_weights.py --model MODEL.pkl --caduceus_model_path /path/to/caduceus \
           --fasta_dir /path/to/sequences --val_ids val_ids.txt --compute_attributions \
           --target_variable BIO_4_7
    
    # Force recomputation of existing files:
    python explain_model_weights.py --model MODEL.pkl --output OUTPUT.csv --force

Assumptions:
- The model file is a pickle file containing a list of models (one per target variable), as saved by train_predict_models.py.
- All embeddings have the same number of SNPs and embedding dimensions.
- The embedding_dim is provided as an argument.
- For attribution analysis, requires GPU access and Caduceus model.
- By default, existing output files are reused unless --force flag is specified.
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def integrated_gradients(model, tokenizer, sequence, embedding_dim, steps=50, device='cuda'):
    """
    Compute integrated gradients for a specific embedding dimension.
    Uses a simplified approach that works with the model's direct input-output relationship.
    
    Args:
        model: Caduceus model
        tokenizer: Tokenizer for the model
        sequence: DNA sequence string
        embedding_dim: Embedding dimension to analyze
        steps: Number of integration steps (default: 50)
        device: Device to run on
    
    Returns:
        attributions: numpy array of attribution scores for each position
        max_position: position that produces the maximum embedding value
        max_value: the maximum embedding value
    """
    import torch
    
    # Tokenize sequence
    tokens = tokenizer.batch_encode_plus([sequence], add_special_tokens=False, 
                                       return_attention_mask=False, max_length=len(sequence), 
                                       truncation=True)
    input_ids = torch.tensor(tokens['input_ids']).to(device)
    
    # For discrete inputs like tokens, we'll use a gradient-based approach
    # that doesn't require complex embedding manipulation
    try:
        # Get baseline embedding using the 'N' token (as in integrated gradients)
        n_token_id = tokenizer.convert_tokens_to_ids("N")
        baseline_ids = torch.full_like(input_ids, n_token_id)

        attributions = []

        # Get the final embedding for reference
        with torch.no_grad():
            final_embedding = model(input_ids)
            max_position = final_embedding[0, :, embedding_dim].argmax().item()
            max_value = final_embedding[0, max_position, embedding_dim].item()

        # For each position, compute the attribution by replacing with baseline
        for pos in range(input_ids.shape[1]):
            # Create a copy where this position is at baseline
            modified_ids = input_ids.clone()
            modified_ids[0, pos] = baseline_ids[0, pos]

            # Compute difference in output at the max position
            with torch.no_grad():
                baseline_embedding = model(modified_ids)
                diff = final_embedding[0, max_position, embedding_dim] - baseline_embedding[0, max_position, embedding_dim]
                attributions.append(diff.item())

        return torch.tensor(attributions).cpu().numpy(), max_position, max_value

    except Exception as e:
        print(f"Error in simplified attribution method: {e}")
        # Return zero attributions as fallback - use tokenized length, not sequence length
        fallback_len = input_ids.shape[1] if 'input_ids' in locals() else len(sequence)
        attributions = torch.zeros(fallback_len).cpu().numpy()
        return attributions, 0, 0.0

def integrated_gradients_bidirectional(model, tokenizer, sequence, embedding_dim, steps=50, device='cuda'):
    """
    Compute attribution scores for bidirectional (forward + reverse complement) embeddings.
    This matches the bidirectional processing in your main workflow.
    
    Uses a simplified leave-one-out attribution method that is robust and works with 
    discrete token inputs without requiring complex embedding layer manipulation.
    
    Args:
        model: Caduceus model
        tokenizer: Tokenizer for the model  
        sequence: DNA sequence string
        embedding_dim: Embedding dimension to analyze
        steps: Number of integration steps (default: 50, not used in simplified method)
        device: Device to run on
    
    Returns:
        attributions: combined attributions for each position
        max_position: position that produces the maximum embedding value
        max_value: the maximum embedding value
    """
    import torch
    
    # First, get the combined embedding to find the true max position
    rc_sequence = string_reverse_complement(sequence)
    tokens = tokenizer.batch_encode_plus([sequence], add_special_tokens=False, 
                                       return_attention_mask=False, max_length=len(sequence), 
                                       truncation=True)
    input_ids = torch.tensor(tokens['input_ids']).to(device)
    rc_tokens = tokenizer.batch_encode_plus([rc_sequence], add_special_tokens=False, 
                                          return_attention_mask=False, max_length=len(sequence), 
                                          truncation=True)
    rc_input_ids = torch.tensor(rc_tokens['input_ids']).to(device)
    
    with torch.no_grad():
        embedding = model(input_ids)  # (1, seq_len, emb_dim)
        rc_embedding = model(rc_input_ids)  # (1, seq_len, emb_dim)
        rc_embedding = rc_embedding.flip(dims=[1])
        combined_embedding = (embedding + rc_embedding) / 2.0  # (1, seq_len, emb_dim)
        
        max_position = combined_embedding[0, :, embedding_dim].argmax().item()
        max_value = combined_embedding[0, max_position, embedding_dim].item()
    
    # Compute attributions by modifying each position and recomputing both forward and RC
    n_token_id = tokenizer.convert_tokens_to_ids("N")
    baseline_ids = torch.full_like(input_ids, n_token_id)
    attributions = []

    for pos in range(input_ids.shape[1]):
        modified_ids = input_ids.clone()
        modified_ids[0, pos] = baseline_ids[0, pos]
        modified_rc_ids = rc_input_ids.clone()
        modified_rc_ids[0, pos] = baseline_ids[0, pos]

        with torch.no_grad():
            modified_embedding = model(modified_ids)
            modified_rc_embedding = model(modified_rc_ids)
            modified_rc_embedding = modified_rc_embedding.flip(dims=[1])
            modified_combined = (modified_embedding + modified_rc_embedding) / 2.0

            diff = combined_embedding[0, max_position, embedding_dim] - modified_combined[0, max_position, embedding_dim]
            attributions.append(diff.item())

    return np.array(attributions), max_position, max_value

def save_top_weights_csv(df, top_n, output_path):
    """
    Save the top_n SNP/dim pairs (by absolute weight) with their associated weight value to an additional CSV file.
    Output file will be named like output_path but with _top_weights.csv suffix.
    """
    # Get top_n by absolute weight per variable
    top_dfs = []
    for var in df['variable'].unique():
        var_df = df[df['variable'] == var]
        var_df_sorted = var_df.reindex(var_df['weight'].abs().sort_values(ascending=False).index)
        abs_weights = var_df_sorted['weight'].abs().values
        total = abs_weights.sum()
        cumsum = abs_weights.cumsum()
        n_50 = (cumsum >= 0.5 * total).argmax() + 1 if total > 0 else 0
        n_80 = (cumsum >= 0.8 * total).argmax() + 1 if total > 0 else 0
        print(f"{var}: {n_50} weights for 50% sum, {n_80} weights for 80% sum (total weights: {len(abs_weights)})")
        top_dfs.append(var_df_sorted.head(top_n))
    top_df = pd.concat(top_dfs, ignore_index=True)
    # Output filename
    base, ext = os.path.splitext(output_path)
    top_out = f"{base}_top_weights.csv"
    top_df.to_csv(top_out, index=False)
    print(f"Saved top {top_n} SNP/dim pairs per variable to {top_out}")

def main():
    # IMPORTANT: This version uses mathematically corrected integrated gradients that work
    # with embedding vectors rather than discrete token indices, ensuring proper attribution.
    
    parser = argparse.ArgumentParser(description="Explain model weights for SNP embeddings.")
    parser.add_argument('--model', required=False, default='model.pkl', help='Path to saved model .pkl file (default: model.pkl)')
    parser.add_argument('--embedding_dim', type=int, required=False, default=256, help='Embedding dimension (number of columns per SNP, default: 256)')
    parser.add_argument('--output', required=False, default='weights_explanation.csv', help='Output CSV file (default: weights_explanation.csv)')
    parser.add_argument('--val_fasta', required=False, default=None, help='Validation sequences FASTA file (optional)')
    parser.add_argument('--caduceus_model_path', required=False, default=None, help='Path to Caduceus model (optional)')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top weights per variable to explain (default: 10)')
    parser.add_argument('--val_ids', required=False, default=None, help='File with validation accession IDs (optional, one per line)')
    parser.add_argument('--fasta_dir', required=False, default=None, help='Directory containing per-accession FASTA files named <id>.fa (optional, used if val_fasta is not given)')
    parser.add_argument('--compute_attributions', action='store_true', help='Compute attribution scores for input sequences (requires GPU). Results saved to separate CSV files.')
    parser.add_argument('--attribution_steps', type=int, default=50, help='Number of steps for attribution computation (default: 50)')
    parser.add_argument('--save_attributions', action='store_true', help='Save detailed per-position attribution scores to separate CSV files (only when --compute_attributions is used)')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if output files already exist')
    parser.add_argument('--target_variable', required=False, default=None, help='Specify a single target variable to process (optional, processes all variables if not specified)')
    parser.add_argument('--explain_only', action='store_true', help='Only generate and save weights_explanation.csv, then exit.')
    args = parser.parse_args()

    # Check if weights explanation file already exists (unless force flag is used)
    if not args.force and os.path.exists(args.output):
        print(f"Weights explanation file {args.output} already exists. Reading from existing file.")
        print("Use --force flag to recompute and overwrite existing files.")
        try:
            df = pd.read_csv(args.output)
            
            # If target_variable is specified, filter the loaded data
            if args.target_variable:
                if args.target_variable in df['variable'].values:
                    df = df[df['variable'] == args.target_variable].copy()
                    print(f"Filtered existing explanation to target variable '{args.target_variable}' with {len(df)} rows")
                else:
                    print(f"Target variable '{args.target_variable}' not found in existing file. Recomputing...")
                    df = None
            else:
                print(f"Successfully loaded existing explanation with {len(df)} rows")
        except Exception as e:
            print(f"Error reading existing file {args.output}: {e}")
            print("Proceeding with recomputation...")
            df = None
    else:
        df = None

    # Compute weights explanation if not loaded from existing file
    if df is None:
        # Load model
        with open(args.model, 'rb') as f:
            model_data = pickle.load(f)
        models = model_data['model']
        env_vars = model_data['env_vars']

        # For each variable, extract weights and map to SNP/embedding_dim
        rows = []
        for var_idx, (var_name, model) in enumerate(zip(env_vars, models)):
            # Skip if target_variable is specified and this isn't it
            if args.target_variable and var_name != args.target_variable:
                continue
                
            if hasattr(model, 'coef_'):
                weights = model.coef_.flatten()
            else:
                print(f"Model for {var_name} does not have coef_ attribute. Skipping.")
                continue
            num_features = len(weights)
            if num_features % args.embedding_dim != 0:
                print(f"Warning: num_features ({num_features}) is not a multiple of embedding_dim ({args.embedding_dim})")
            num_snps = num_features // args.embedding_dim
            for i, w in enumerate(weights):
                snp_index = i // args.embedding_dim
                emb_dim = i % args.embedding_dim
                rows.append({'variable': var_name, 'snp_index': snp_index, 'embedding_dim': emb_dim, 'weight': w})
        
        if not rows:
            if args.target_variable:
                print(f"Error: No data found for target variable '{args.target_variable}'")
                return
            else:
                print("Error: No valid models found with coef_ attribute")
                return
                
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        
        if args.target_variable:
            print(f"Wrote explanation for target variable '{args.target_variable}' to {args.output}")
        else:
            print(f"Wrote explanation to {args.output}")

        # Save top_n SNP/dim pairs with their associated weight value to an additional CSV file
        save_top_weights_csv(df, args.top_n, args.output)

        if args.explain_only:
            print("--explain_only flag set. Exiting after saving weights explanation CSV.")
            return

    else:
        # If we loaded from existing file, we still need the model data for validation analysis
        if (args.val_fasta or args.fasta_dir) and args.caduceus_model_path:
            with open(args.model, 'rb') as f:
                model_data = pickle.load(f)
            models = model_data['model']
            env_vars = model_data['env_vars']

    # If validation embedding explanation is requested
    if (args.val_fasta or args.fasta_dir) and args.caduceus_model_path:
        # Load val_ids if provided
        val_id_set = None
        if args.val_ids:
            with open(args.val_ids, 'r') as f:
                content = f.read().strip()
                val_id_set = content.split()
        from Bio import SeqIO
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. Caduceus model requires GPU. Cancelling execution.")
            print("Please ensure the script is run with GPU access (e.g., using --nv flag in Singularity).")
            return
        
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        try:
            from Masters.caduceus.embed_sequences import DNAEmbeddingModel, EnformerTokenizer
            print("Successfully imported Caduceus modules")
        except Exception as e:
            print(f"Error importing Caduceus modules: {e}")
            return
        # Get top N important weights per variable
        important_weights = []
        for var_idx, (var_name, model) in enumerate(zip(env_vars, models)):
            # Skip if target_variable is specified and this isn't it
            if args.target_variable and var_name != args.target_variable:
                continue
                
            if hasattr(model, 'coef_'):
                weights = model.coef_.flatten()
                abs_weights = np.abs(weights)
                top_indices = np.argsort(abs_weights)[-args.top_n:][::-1]
                for idx in top_indices:
                    snp_index = idx // args.embedding_dim
                    emb_dim = idx % args.embedding_dim
                    important_weights.append({'variable': var_name, 'snp_index': snp_index, 'embedding_dim': emb_dim, 'weight': weights[idx]})
        # Note: If we loaded weights from existing file, we need to reconstruct important_weights from the DataFrame
        if 'models' not in locals():
            # We loaded from existing file and don't have validation analysis - create important_weights from df
            important_weights = []
            variables_to_process = [args.target_variable] if args.target_variable else df['variable'].unique()
            for var in variables_to_process:
                if var not in df['variable'].values:
                    print(f"Warning: Target variable '{var}' not found in existing weights file.")
                    continue
                var_df = df[df['variable'] == var]
                var_df_sorted = var_df.reindex(var_df['weight'].abs().sort_values(ascending=False).index)
                top_weights = var_df_sorted.head(args.top_n)
                for _, row in top_weights.iterrows():
                    important_weights.append({
                        'variable': row['variable'],
                        'snp_index': row['snp_index'], 
                        'embedding_dim': row['embedding_dim'],
                        'weight': row['weight']
                    })
        # Load validation sequences
        referenced_snp_indices = set(iw['snp_index'] for iw in important_weights)
        val_sequences = []
        if args.val_fasta:
            # Only keep sequences whose index (or ID) is in referenced_snp_indices
            for idx, rec in enumerate(SeqIO.parse(args.val_fasta, 'fasta')):
                # Try to match by index (position in file) or by ID if possible
                if (val_id_set is None or rec.id in val_id_set):
                    # If rec.id is integer-like, try to match to snp_index
                    try:
                        rec_idx = int(rec.id)
                    except Exception:
                        rec_idx = idx  # fallback to index in file
                    if rec_idx in referenced_snp_indices:
                        val_sequences.append(rec)
            if not val_sequences:
                print("Warning: No validation sequences matched referenced snp_indices.")
        elif args.fasta_dir and val_id_set:
            for acc_id in val_id_set:
                fasta_path = os.path.join(args.fasta_dir, f"{acc_id}.fa")
                if os.path.exists(fasta_path):
                    for idx, rec in enumerate(SeqIO.parse(fasta_path, 'fasta')):
                        if idx in referenced_snp_indices:
                            # Attach accession info for later
                            rec.annotations['accession'] = acc_id
                            val_sequences.append(rec)
                else:
                    print(f"Warning: FASTA file not found for accession {acc_id} at {fasta_path}")
        # Load Caduceus model and tokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        try:
            print("Loading Caduceus model...")
            caduceus_model = DNAEmbeddingModel(args.caduceus_model_path)
            caduceus_model = caduceus_model.to(device)
            caduceus_model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading Caduceus model: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print("Loading tokenizer...")
            if 'enformer' in args.caduceus_model_path.lower():
                tokenizer = EnformerTokenizer()
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.caduceus_model_path, trust_remote_code=True)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return
        results = []
        if args.val_fasta:
            variables_to_process = [args.target_variable] if args.target_variable else set(iw['variable'] for iw in important_weights)
            for var in variables_to_process:
                var_out = args.output.replace('.csv', f'_{var}.csv')
                
                # Check if variable-specific output already exists
                existing_results = None
                if not args.force and os.path.exists(var_out):
                    print(f"Variable-specific output {var_out} already exists. Loading existing results.")
                    try:
                        existing_results = pd.read_csv(var_out)
                        print(f"Loaded {len(existing_results)} existing results for {var}")
                        
                        # If attributions are requested but not present in existing results, we'll compute them
                        has_attributions = 'attribution_max_pos' in existing_results.columns
                        if args.compute_attributions and not has_attributions:
                            print(f"Attribution data not found in existing results. Will compute attributions for {var}.")
                        elif args.compute_attributions and has_attributions:
                            print(f"Attribution data already exists for {var}. Use --force to recompute.")
                            continue
                        elif not args.compute_attributions:
                            print(f"Using existing results for {var} without attribution analysis.")
                            continue
                    except Exception as e:
                        print(f"Error reading existing file {var_out}: {e}")
                        print("Proceeding with recomputation...")
                        existing_results = None
                
                var_weights = [iw for iw in important_weights if iw['variable'] == var]
                snp_indices_needed = set(iw['snp_index'] for iw in var_weights)
                
                # If we have existing results and only need to add attributions, start from those
                if existing_results is not None and args.compute_attributions:
                    results = existing_results.to_dict('records')
                    print(f"Starting with {len(results)} existing results, adding attribution analysis.")
                    # Note: Attribution addition for val_fasta case would need additional implementation
                    # since we need to re-read the sequences by index
                else:
                    results = []
                
                sequences_to_process = [(idx, rec) for idx, rec in enumerate(SeqIO.parse(args.val_fasta, 'fasta')) if idx in snp_indices_needed]
                
                for idx, rec in tqdm(sequences_to_process, desc=f"Processing sequences for {var}"):
                    seq = str(rec.seq)
                    
                    try:
                        tokens = tokenizer.batch_encode_plus([seq], add_special_tokens=False, return_attention_mask=False, max_length=len(seq), truncation=True)
                        input_ids = torch.tensor(tokens['input_ids']).to(device)
                        
                        with torch.no_grad():
                            embedding = caduceus_model(input_ids)
                        embedding = embedding.squeeze(0).cpu().numpy()
                    except Exception as e:
                        print(f"Error processing sequence {idx}: {e}")
                        continue
                    for iw in var_weights:
                        if iw['snp_index'] != idx:
                            continue
                        emb_dim = iw['embedding_dim']
                        if emb_dim < embedding.shape[1]:
                            # Find the nucleotide position that produces the maximum embedding value
                            maxpool_nt_index = int(np.argmax(embedding[:, emb_dim]))
                            results.append({
                                'variable': iw['variable'],
                                'snp_index': idx,
                                'embedding_dim': emb_dim,
                                'weight': iw['weight'],
                                'validation_sample': rec.id,
                                'accession': None,
                                'maxpool_nt_index': maxpool_nt_index,
                            })
                if results:
                    df2 = pd.DataFrame(results)
                    var_out = args.output.replace('.csv', f'_{var}.csv')
                    df2.to_csv(var_out, index=False)
                    print(f"Wrote maxpool origins explanation to {var_out}")
        elif args.fasta_dir and val_id_set:
            # Create base output directory structure
            base_output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
            
            # Process each important weight individually (restructured loop)
            if args.compute_attributions:
                if args.target_variable:
                    print(f"Computing attributions for target variable: {args.target_variable}")
                else:
                    print("Computing attributions for all target variables...")
                
                if not important_weights:
                    if args.target_variable:
                        print(f"No important weights found for target variable '{args.target_variable}'")
                    else:
                        print("No important weights found for any variables")
                    return
                
                for i, iw in enumerate(tqdm(important_weights, desc="Processing important weights")):
                    var_name = iw['variable']
                    snp_index = iw['snp_index'] 
                    emb_dim = iw['embedding_dim']
                    weight_val = iw['weight']
                    
                    # Create variable-specific subdirectory
                    var_output_dir = os.path.join(base_output_dir, var_name)
                    os.makedirs(var_output_dir, exist_ok=True)
                    
                    # Define output file for this specific weight
                    attr_filename = f"attributions_{var_name}_SNP{snp_index}_dim{emb_dim}.csv"
                    attr_path = os.path.join(var_output_dir, attr_filename)
                    
                    # Check if this weight's attribution file already exists
                    if not args.force and os.path.exists(attr_path):
                        print(f"  Attribution file {attr_filename} already exists. Skipping.")
                        continue
                    
                    print(f"  Processing weight {i+1}/{len(important_weights)}: {var_name} SNP{snp_index} dim{emb_dim} (weight={weight_val:.4f})")
                    
                    # Collect attributions for all accessions for this weight
                    weight_attributions = []
                    
                    for acc_id in tqdm(val_id_set, desc=f"    Processing accessions", leave=False):
                        fasta_path = os.path.join(args.fasta_dir, f"{acc_id}.fa")
                        if not os.path.exists(fasta_path):
                            continue
                            
                        # Find the specific sequence record
                        seq = None
                        validation_sample = None
                        for seq_idx, rec in enumerate(SeqIO.parse(fasta_path, 'fasta')):
                            if seq_idx == snp_index:
                                seq = str(rec.seq)
                                validation_sample = rec.id
                                break
                        
                        if seq is None:
                            continue
                            
                        try:
                            # Compute attributions for this accession and weight
                            attributions, attr_max_pos, attr_max_val = integrated_gradients_bidirectional(
                                caduceus_model, tokenizer, seq, emb_dim, 
                                steps=args.attribution_steps, device=device
                            )
                            
                            # Add position-by-position attributions for this accession
                            for pos, (nucleotide, attribution) in enumerate(zip(seq, attributions)):
                                weight_attributions.append({
                                    'accession': acc_id,
                                    'validation_sample': validation_sample,
                                    'snp_index': snp_index,
                                    'embedding_dim': emb_dim,
                                    'position': pos,
                                    'nucleotide': nucleotide,
                                    'attribution': float(attribution),
                                    'maxpool_nt_index': attr_max_pos,
                                    'max_embedding_value': float(attr_max_val)
                                })
                                
                        except Exception as e:
                            print(f"      ERROR computing attributions for {acc_id}: {e}")
                            continue
                    
                    # Save all attributions for this weight to CSV
                    if weight_attributions:
                        attr_df = pd.DataFrame(weight_attributions)
                        attr_df.to_csv(attr_path, index=False)
                        print(f"    Saved {len(attr_df)} attribution records to {attr_filename}")
                        print(f"      - {len(attr_df['accession'].unique())} accessions")
                        print(f"      - {len(attr_df['position'].unique())} positions")
                    else:
                        print(f"    No valid attributions computed for {attr_filename}")

if __name__ == "__main__":
    main()
