def find_non_overlapping_peaks(data_rows, window_size=10, max_peaks=3, zscore_thresh=1.95):
    """
    Find up to max_peaks non-overlapping peaks in the mean_saliency sequence using z-score thresholding.
    Returns a list of dicts: {position, value, zscore}
    """
    if len(data_rows) == 0:
        return []
    values = np.abs(data_rows['mean_saliency'].values)
    positions = data_rows['position'].values
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []
    zscores = (values - mean) / std
    # abs_zscores = np.abs(zscores)
    mask = np.ones(len(values), dtype=bool)
    peaks = []
    half_window = window_size // 2
    for _ in range(max_peaks):
        masked_z = zscores * mask
        idx = masked_z.argmax()
        if not mask[idx] or masked_z[idx] < zscore_thresh:
            break
        peak = {
            'position': int(positions[idx]),
            'value': float(values[idx]),
            'zscore': float(zscores[idx])
        }
        peaks.append(peak)
        # Mask out window around this peak
        left = max(0, idx - half_window)
        right = min(len(values), idx + half_window + 1)
        mask[left:right] = False
    return peaks
#!/usr/bin/env python3
"""
Extract gradient peak positions and nucleotide patterns from integrated gradients results.

This script processes integrated gradients CSV files to find maximum gradient positions,
extract absolute genomic coordinates from FASTA headers, and analyze nucleotide patterns
at those positions. Each SNP/dimension pair is processed individually since they represent
different genomic locations.

Usage:
    python extract_gradient_peaks.py \
           --gradients_dir Results/Saliency/512/BIO_11_6 \
           --fasta_dir Data/Pseudogenomes/SparSNP_Joint_512 \
           --output_dir Results/GradientPeaks
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
from glob import glob
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_fasta_header(header):
    """
    Parse FASTA header to extract genomic coordinates.
    
    Example header: >MPI-GMI|Ath-1001-Genomes|pseudo-genome|88|Chr1:10048252..10048764|V0.2
    
    Returns:
        tuple: (chromosome, start_pos, end_pos) or (None, None, None) if parsing fails
    """
    try:
        # Look for pattern like Chr1:10048252..10048764
        coord_pattern = r'(Chr\w+):(\d+)\.\.(\d+)'
        match = re.search(coord_pattern, header)
        
        if match:
            chromosome = match.group(1)
            start_pos = int(match.group(2))
            end_pos = int(match.group(3))
            return chromosome, start_pos, end_pos
        else:
            print(f"Warning: Could not parse coordinates from header: {header}")
            return None, None, None
            
    except Exception as e:
        print(f"Error parsing header '{header}': {e}")
        return None, None, None

def get_nucleotide_composition_from_csv(gradients_csv, window_size=10):
    """
    Extract nucleotide composition around the peak position from integrated gradients CSV.
    
    Args:
        gradients_csv: Path to integrated gradients CSV file
        window_size: Size of window around peak (default: 10)
        
    Returns:
        tuple: (window_data, peak_position) where window_data is list of dicts with nucleotide percentages
    """
    try:
        df = pd.read_csv(gradients_csv)
        
        # Filter out metadata row (position = -1)
        data_rows = df[df['position'] >= 0].copy()
        
        if len(data_rows) == 0:
            print(f"Warning: No data rows found in {gradients_csv}")
            return None, None
        
        # Find position with maximum absolute gradient
        max_idx = data_rows['mean_saliency'].abs().idxmax()
        peak_position = int(data_rows.loc[max_idx, 'position'])
        
        # Calculate window boundaries with shifting to maintain window size
        half_window = window_size // 2
        ideal_start = peak_position - half_window
        ideal_end = peak_position + half_window + 1
        
        # Adjust window boundaries to maintain window_size
        if ideal_start < 0:
            # Peak too close to start, shift window right
            start_pos = 0
            end_pos = min(len(data_rows), window_size)
        elif ideal_end > len(data_rows):
            # Peak too close to end, shift window left
            end_pos = len(data_rows)
            start_pos = max(0, len(data_rows) - window_size)
        else:
            # Peak in middle, use ideal boundaries
            start_pos = ideal_start
            end_pos = ideal_end
        
        # Extract window data
        window_data = []
        for pos in range(start_pos, end_pos):
            if pos < len(data_rows):
                row = data_rows.iloc[pos]
                nucleotide_data = {
                    'position': int(row['position']),
                    'pct_A': float(row['pct_A']),
                    'pct_T': float(row['pct_T']),
                    'pct_G': float(row['pct_G']),
                    'pct_C': float(row['pct_C']),
                    'pct_N': float(row['pct_N'])
                }
                window_data.append(nucleotide_data)
        
        return window_data, peak_position
        
    except Exception as e:
        print(f"Error extracting nucleotide composition from {gradients_csv}: {e}")
        return None, None

def get_sequence_coordinates(fasta_file, snp_index):
    """
    Get genomic coordinates for a specific SNP index from FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        snp_index: Index of the sequence (0-based)
    
    Returns:
        tuple: (chromosome, start_pos, end_pos, actual_seq_length)
    """
    try:
        for seq_idx, rec in enumerate(SeqIO.parse(fasta_file, 'fasta')):
            if seq_idx == snp_index:
                chromosome, start_pos, end_pos = parse_fasta_header(rec.description)
                actual_seq_length = len(str(rec.seq))
                return chromosome, start_pos, end_pos, actual_seq_length
        
        print(f"Warning: SNP index {snp_index} not found in {fasta_file}")
        return None, None, None, None
        
    except Exception as e:
        print(f"Error reading {fasta_file}: {e}")
        return None, None, None, None

def find_max_gradient_position(gradients_csv):
    """
    Find the position with maximum absolute gradient value from integrated gradients CSV.
    
    Args:
        gradients_csv: Path to integrated gradients CSV file
    
    Returns:
        tuple: (max_position, max_value, nucleotide_freqs) or (None, None, None)
    """
    try:
        df = pd.read_csv(gradients_csv)
        # Filter out metadata row (position = -1)
        data_rows = df[df['position'] >= 0].copy()
        if len(data_rows) == 0:
            print(f"Warning: No data rows found in {gradients_csv}")
            return None, None, None
        # Find up to 3 non-overlapping peaks by z-score
        peaks = find_non_overlapping_peaks(data_rows, window_size=10, max_peaks=3, zscore_thresh=1.96)
        if not peaks:
            return None, None, None
        # Use the highest peak for compatibility with old code
        main_peak = peaks[0]
        max_position = main_peak['position']
        max_value = main_peak['value']
        max_row = data_rows[data_rows['position'] == max_position].iloc[0]
        nucleotide_freqs = {
            'pct_A': float(max_row['pct_A']),
            'pct_T': float(max_row['pct_T']),
            'pct_G': float(max_row['pct_G']),
            'pct_C': float(max_row['pct_C']),
            'pct_N': float(max_row['pct_N'])
        }
        # Optionally, you can return all peaks if you want to use them elsewhere
        return max_position, max_value, nucleotide_freqs
    except Exception as e:
        print(f"Error processing {gradients_csv}: {e}")
        return None, None, None

def create_consensus_logo(ax, window_data, peak_position, snp_index, embedding_dim, max_grad_value):
    """
    Create a consensus logo visualization for nucleotide composition.
    
    Args:
        ax: Matplotlib axis object
        window_data: List of dicts with nucleotide percentages for each position
        peak_position: Position of the gradient peak
        snp_index: SNP index
        embedding_dim: Embedding dimension
        max_grad_value: Maximum gradient value
    """
    try:
        if not window_data:
            ax.text(0.5, 0.5, 'No data\navailable', ha='center', va='center')
            ax.set_title(f"SNP{snp_index}_dim{embedding_dim}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Color mapping for nucleotides
        color_map = {'A': '#FF0000', 'T': '#0000FF', 'G': '#00AA00', 'C': '#FFA500', 'N': '#808080'}
        
        # Calculate information content and relative positions
        positions = []
        heights_data = []
        
        for pos_data in window_data:
            relative_pos = pos_data['position'] - peak_position
            positions.append(relative_pos)
            
            # Get nucleotide frequencies (convert from percentage to fraction)
            freqs = {
                'A': pos_data['pct_A'] / 100.0,
                'T': pos_data['pct_T'] / 100.0,
                'G': pos_data['pct_G'] / 100.0,
                'C': pos_data['pct_C'] / 100.0,
                'N': pos_data['pct_N'] / 100.0
            }
            
            # Use frequencies directly for letter heights
            # This gives a clearer representation of nucleotide composition
            heights = {}
            for nuc, freq in freqs.items():
                heights[nuc] = freq  # Use frequency directly as height
            
            heights_data.append(heights)
        
        # Set up the plot
        max_height = max([sum(h.values()) for h in heights_data]) if heights_data else 1.0
        ax.set_xlim(-0.5, len(positions) - 0.5)
        ax.set_ylim(0, max(max_height * 1.1, 0.1))
        
        # Draw letters for each position
        for i, (pos, heights) in enumerate(zip(positions, heights_data)):
            # Sort nucleotides by frequency (smallest first, so largest is on top)
            sorted_nucs = sorted(heights.items(), key=lambda x: x[1])
            
            current_bottom = 0
            for nuc, height in sorted_nucs:
                if height > 0.01:  # Only draw if height is significant
                    # Draw a rectangle representing the frequency
                    rect = patches.Rectangle((i-0.4, current_bottom), 0.8, height,
                                           facecolor=color_map[nuc], alpha=0.8,
                                           edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
                    
                    # Add the nucleotide letter on top of the rectangle
                    # Scale font size to fit within the rectangle height
                    fontsize = max(8, min(24, height * 30))
                    
                    # Position text at center of the rectangle
                    ax.text(i, current_bottom + height/2, nuc, 
                           ha='center', va='center',
                           fontsize=fontsize, fontweight='bold',
                           color='white' if nuc != 'N' else 'black')  # White text for visibility
                    
                    current_bottom += height
            
            # Highlight peak position
            if pos == 0:  # Peak position (relative position 0)
                rect = patches.Rectangle((i-0.4, 0), 0.8, max_height * 1.05,
                                       linewidth=2, edgecolor='yellow', 
                                       facecolor='yellow', alpha=0.2)
                ax.add_patch(rect)
        
        # Customize the plot
        ax.set_title(f"SNP{snp_index}_dim{embedding_dim}\nGrad: {max_grad_value:.4f}", fontsize=12)
        ax.set_xlabel('Position relative to peak')
        ax.set_ylabel('Nucleotide Frequency')
        
        # Set x-axis labels
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([f"{pos:+d}" if pos != 0 else "0" for pos in positions])
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to show frequency scale (0 to 1.0)
        ax.set_ylim(0, max(max_height * 1.1, 0.1))
        
    except Exception as e:
        print(f"Error creating consensus logo: {e}")
        # Fallback to simple text
        ax.text(0.5, 0.5, f'Error creating\nconsensus logo\n{str(e)[:50]}', 
               ha='center', va='center', fontsize=8)
        ax.set_title(f"SNP{snp_index}_dim{embedding_dim}")

def create_sequence_window_visualization(results_df, gradients_dir, output_path, window_size=10):
    """
    Create consensus logo visualization showing nucleotide composition around gradient peaks.
    
    Args:
        results_df: DataFrame with gradient peak results
        gradients_dir: Directory containing gradient CSV files
        output_path: Path to save the plot
        window_size: Size of sequence window around peak
    """
    try:
        n_results = len(results_df)
        if n_results == 0:
            print("No results to visualize")
            return
        
        # Calculate subplot layout
        cols = min(4, n_results)  # Max 4 columns
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        
        # Handle different subplot configurations
        if rows == 1 and cols == 1:
            # Single subplot
            axes = [axes]
        elif rows == 1 or cols == 1:
            # Single row or single column - already a 1D array
            if not hasattr(axes, '__len__'):
                axes = [axes]
        else:
            # 2D array of subplots - flatten to 1D
            axes = axes.flatten()
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            snp_index = int(row['snp_index'])
            embedding_dim = int(row['embedding_dim'])
            
            # Get the gradient CSV file for this SNP/dim pair
            gradients_csv = os.path.join(gradients_dir, 
                                       f"integrated_gradients_averaged_SNP{snp_index}_dim{embedding_dim}.csv")
            
            if not os.path.exists(gradients_csv):
                ax.text(0.5, 0.5, 'Gradient file\nnot found', ha='center', va='center')
                ax.set_title(f"SNP{snp_index}_dim{embedding_dim}")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Extract nucleotide composition window
            window_data, peak_position = get_nucleotide_composition_from_csv(gradients_csv, window_size)
            
            # Create consensus logo
            create_consensus_logo(ax, window_data, peak_position, snp_index, embedding_dim, row['max_grad_value'])
        
        # Hide unused subplots
        for i in range(n_results, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Consensus Logos Around Gradient Peaks (window size: {window_size})', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved consensus logo visualization: {output_path}")
        
    except Exception as e:
        print(f"Error creating consensus logo visualization: {e}")

def write_meme_motif_file(motif_list, output_path):
    """
    Write a list of motifs (as PFMs) to a MEME format file for use with Tomtom.
    motif_list: list of dicts with keys: 'name', 'pfm' (list of [A,C,G,T] counts per position), 'width'
    """
    with open(output_path, 'w') as f:
        f.write('MEME version 4\n\n')
        f.write('ALPHABET= ACGT\n\n')
        f.write('strands: + -\n\n')
        for motif in motif_list:
            f.write(f'MOTIF {motif["name"]}\n')
            f.write(f'letter-probability matrix: alength= 4 w= {motif["width"]}\n')
            for row in motif['pfm']:
                f.write(' '.join(f'{x:.4f}' for x in row) + '\n')
            f.write('\n')

def extract_motif_from_window(window_data):
    """
    Convert window_data (list of dicts with pct_A, pct_C, pct_G, pct_T, pct_N) to a PFM (A,C,G,T order), renormalizing so each row sums to 1.
    Returns a list of lists (rows: positions, cols: A,C,G,T frequencies as fractions).
    """
    pfm = []
    for pos in window_data:
        # Use only A,C,G,T, ignore N
        a = pos['pct_A']
        c = pos['pct_C']
        g = pos['pct_G']
        t = pos['pct_T']
        total = a + c + g + t
        if total == 0:
            pfm.append([0.25, 0.25, 0.25, 0.25])
        else:
            pfm.append([
                a / total,
                c / total,
                g / total,
                t / total
            ])
    return pfm

def process_gradient_files(parent_gradients_dir, fasta_dir, output_dir, window_size=10):
    """
    Iterate over variable subfolders in the parent gradients directory, process all integrated gradients CSV files,
    and save a joint CSV with a 'variable' column.
    """
    all_results = []
    variable_folders = [d for d in os.listdir(parent_gradients_dir) if os.path.isdir(os.path.join(parent_gradients_dir, d))]
    if not variable_folders:
        print(f"No variable subfolders found in {parent_gradients_dir}")
        return
    print(f"Found {len(variable_folders)} variable subfolders: {variable_folders}")
    motif_list = []
    for variable in variable_folders:
        gradients_dir = os.path.join(parent_gradients_dir, variable)
        csv_pattern = os.path.join(gradients_dir, "integrated_gradients_averaged_SNP*_dim*.csv")
        csv_files = glob(csv_pattern)
        if not csv_files:
            print(f"No integrated gradients CSV files found in {gradients_dir}")
            continue
        print(f"Processing variable '{variable}' with {len(csv_files)} CSV files")
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            match = re.search(r'SNP(\d+)_dim(\d+)\.csv', filename)
            if not match:
                print(f"Warning: Could not parse SNP/dim from filename: {filename}")
                continue
            snp_index = int(match.group(1))
            embedding_dim = int(match.group(2))
            print(f"Processing {variable} SNP{snp_index} dim{embedding_dim}...")
            # Find all peaks for this SNP/dim
            try:
                df_temp = pd.read_csv(csv_file)
                data_rows = df_temp[df_temp['position'] >= 0].copy()
                analyzed_seq_len = len(data_rows)
                peaks = find_non_overlapping_peaks(data_rows, window_size=10, max_peaks=3, zscore_thresh=1.96)
                if not peaks:
                    continue
                sample_fasta = None
                for fasta_file in os.listdir(fasta_dir):
                    if fasta_file.endswith('.fa'):
                        sample_fasta = os.path.join(fasta_dir, fasta_file)
                        break
                if sample_fasta is None:
                    print(f"Warning: No FASTA files found in {fasta_dir}")
                    continue
                chromosome, start_pos, end_pos, actual_seq_len = get_sequence_coordinates(
                    sample_fasta, snp_index)
                if chromosome is None:
                    continue
                trim_offset = 0
                if actual_seq_len > analyzed_seq_len:
                    effective_seq_len = analyzed_seq_len
                else:
                    effective_seq_len = actual_seq_len
                effective_end_pos = start_pos + effective_seq_len - 1
                for peak_idx, peak in enumerate(peaks):
                    max_pos = peak['position']
                    max_val = peak['value']
                    zscore = peak['zscore']
                    if max_pos >= analyzed_seq_len:
                        print(f"Warning: Max position {max_pos} exceeds analyzed sequence length {analyzed_seq_len}")
                        continue
                    abs_max_pos = start_pos + trim_offset + max_pos
                    max_row = data_rows[data_rows['position'] == max_pos].iloc[0]
                    nucleotide_freqs = {
                        'pct_A': float(max_row['pct_A']),
                        'pct_T': float(max_row['pct_T']),
                        'pct_G': float(max_row['pct_G']),
                        'pct_C': float(max_row['pct_C']),
                        'pct_N': float(max_row['pct_N'])
                    }
                    result = {
                        'variable': variable,
                        'snp_index': snp_index,
                        'embedding_dim': embedding_dim,
                        'peak_index': peak_idx + 1,
                        'chromosome': chromosome,
                        'region_start': start_pos,
                        'region_end': end_pos,
                        'effective_end': effective_end_pos,
                        'sequence_length': actual_seq_len,
                        'effective_length': effective_seq_len,
                        'analyzed_length': analyzed_seq_len,
                        'trim_offset': trim_offset,
                        'max_grad_pos': max_pos,
                        'abs_max_pos': abs_max_pos,
                        'max_grad_value': max_val,
                        'zscore': zscore,
                        **nucleotide_freqs
                    }
                    all_results.append(result)
                    # Extract motif window for MEME file for every peak
                    # Shift window to maintain consistent size even when peak is near edges
                    half_window = window_size // 2
                    ideal_start = max_pos - half_window
                    ideal_end = max_pos + half_window + 1
                    
                    # Adjust window boundaries to maintain window_size
                    if ideal_start < 0:
                        # Peak too close to start, shift window right
                        win_start = 0
                        win_end = min(len(data_rows), window_size)
                    elif ideal_end > len(data_rows):
                        # Peak too close to end, shift window left
                        win_end = len(data_rows)
                        win_start = max(0, len(data_rows) - window_size)
                    else:
                        # Peak in middle, use ideal boundaries
                        win_start = ideal_start
                        win_end = ideal_end
                    
                    window_data = []
                    for pos in range(win_start, win_end):
                        if pos < len(data_rows):
                            row = data_rows.iloc[pos]
                            window_data.append({
                                'pct_A': float(row['pct_A']),
                                'pct_C': float(row['pct_C']) if 'pct_C' in row else 0.0,
                                'pct_G': float(row['pct_G']),
                                'pct_T': float(row['pct_T']),
                                'pct_N': float(row['pct_N'])
                            })
                    if len(window_data) > 0:
                        motif_name = f"{variable}_SNP{snp_index}_dim{embedding_dim}_center{max_pos}"
                        pfm = extract_motif_from_window(window_data)
                        motif_list.append({'name': motif_name, 'pfm': pfm, 'width': len(window_data)})
                    else:
                        print(f"Warning: Skipped motif {variable}_SNP{snp_index}_dim{embedding_dim}_center{max_pos} (empty window)")
            except Exception as e:
                print(f"Warning: Could not process peaks for {csv_file}: {e}")
    if not all_results:
        print("No valid results found across all variable folders")
        return
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(output_dir, 'gradient_peaks_summary.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Saved results to: {results_csv}")
    # Write MEME motif file
    meme_path = os.path.join(output_dir, 'gradient_peak_motifs.meme')
    write_meme_motif_file(motif_list, meme_path)
    print(f"Saved motifs to: {meme_path}")
    # Visualization is still disabled
    # viz_path = os.path.join(output_dir, 'consensus_logos_at_peaks.png')
    # create_sequence_window_visualization(results_df, gradients_dir, viz_path, window_size)
    print(f"\nSummary:")
    print(f"Processed {len(all_results)} SNP/dimension pairs across {len(variable_folders)} variables")
    print(f"Chromosomes found: {sorted(results_df['chromosome'].unique())}")
    print(f"Gradient values range: {results_df['max_grad_value'].min():.6f} - {results_df['max_grad_value'].max():.6f}")
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Extract gradient peak positions and nucleotide patterns.")
    parser.add_argument('--gradients_dir', required=True, 
                       help='Directory containing integrated gradients CSV files')
    parser.add_argument('--fasta_dir', required=True,
                       help='Directory containing FASTA files')
    parser.add_argument('--output_dir', default='gradient_peaks',
                       help='Output directory for results')
    parser.add_argument('--window_size', type=int, default=10,
                       help='Size of nucleotide window around peak (default: 10)')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.gradients_dir):
        print(f"Error: Gradients directory not found: {args.gradients_dir}")
        return
    
    if not os.path.exists(args.fasta_dir):
        print(f"Error: FASTA directory not found: {args.fasta_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    results_df = process_gradient_files(
        args.gradients_dir, args.fasta_dir, args.output_dir, args.window_size
    )
    
    if results_df is not None:
        print(f"\nResults saved to: {args.output_dir}")
        print("Files created:")
        print("- gradient_peaks_summary.csv: Summary of all gradient peaks")
        print("- consensus_logos_at_peaks.png: Consensus logo visualization of nucleotide patterns")

if __name__ == "__main__":
    main()
