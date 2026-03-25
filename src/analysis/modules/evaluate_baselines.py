#!/usr/bin/env python3
"""
extract_and_cluster_embeddings.py
==================================
Extract embeddings from MERT and CLAP, apply clustering analysis
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import torch
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Import your existing clustering methods
from analysis.modules.clustering_algorithms import EDMClusterer, NaturalClusterFinder

# For analysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    normalized_mutual_info_score, 
    adjusted_rand_score,
    mutual_info_score,
    confusion_matrix
)
from scipy.stats import entropy
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

# ==================== PROGRESS TRACKING ====================

class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.stage_start = time.time()
        
    def print_stage(self, message):
        elapsed = time.time() - self.stage_start
        total_elapsed = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        print(f"Total elapsed: {total_elapsed/60:.1f} min | Stage time: {elapsed:.1f} sec")
        print('='*80)
        self.stage_start = time.time()
        
    def print_substage(self, message):
        print(f"\n>>> {message}")

progress = ProgressTracker()

# ==================== DATA LOADING ====================

def load_beatport_dataset(root_dir='dataset/BeatportTop100'):
    """Load Beatport dataset structure"""
    progress.print_substage("Scanning dataset structure...")
    
    tracks = []
    labels = []
    genre_mapping = {}
    
    genre_folders = sorted([d for d in Path(root_dir).iterdir() if d.is_dir()])
    print(f"Found {len(genre_folders)} genre folders")
    
    for genre_idx, genre_folder in enumerate(tqdm(genre_folders, desc="Loading genres", unit="genre")):
        folder_name = genre_folder.name
        
        # Format: "Afro House-Top100-19_03_2025"
        if '-Top100' in folder_name:
            genre_name = folder_name.split('-Top100')[0].strip()
        else:
            genre_name = folder_name
        
        genre_id = genre_idx
        genre_mapping[genre_id] = genre_name
        
        track_folders = sorted([d for d in genre_folder.iterdir() if d.is_dir()])
        
        for track_folder in track_folders:
            preview_path = track_folder / 'preview.mp3'
            if preview_path.exists():
                tracks.append(str(preview_path))
                labels.append(genre_id)
    
    print(f"\n✓ Loaded {len(tracks)} tracks across {len(genre_mapping)} genres")
    print(f"✓ Genres found: {', '.join(list(genre_mapping.values())[:5])}...")
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"✓ Tracks per genre: min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}")
    
    return tracks, np.array(labels), genre_mapping

# ==================== EMBEDDING EXTRACTORS ====================

def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print(f"⚠ Using CPU (this will be slower)")
    return device

def extract_mert_embeddings(audio_paths, batch_size=16):
    """Extract MERT-95M embeddings"""
    from transformers import AutoModel, AutoProcessor
    
    progress.print_substage("Loading MERT-95M model...")
    device = get_device()
    
    model_name = "m-a-p/MERT-v1-95M"
    print(f"Downloading/loading from HuggingFace: {model_name}")
    
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print(f"✓ Model loaded successfully")
    
    embeddings = []
    failed_count = 0
    
    progress.print_substage(f"Extracting embeddings from {len(audio_paths)} audio files...")
    
    for i in tqdm(range(0, len(audio_paths), batch_size), 
                  desc="MERT-95M extraction", 
                  unit="batch"):
        batch_paths = audio_paths[i:i+batch_size]
        
        for path in batch_paths:
            try:
                audio, sr = librosa.load(path, sr=24000, duration=120, mono=True)
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(emb.squeeze())
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:
                    print(f"\n⚠ Error processing {Path(path).name}: {e}")
                embeddings.append(np.zeros(768))
        
        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"  Processed: {min(i + batch_size, len(audio_paths))}/{len(audio_paths)} files")
    
    if failed_count > 0:
        print(f"⚠ Failed to process {failed_count} files (using zero vectors)")
    
    print(f"✓ Extracted {len(embeddings)} embeddings, shape: {embeddings[0].shape}")
    return np.array(embeddings)

def extract_clap_embeddings(audio_paths, batch_size=32):
    """Extract CLAP embeddings"""
    import laion_clap
    
    progress.print_substage("Loading CLAP model...")
    
    print("Downloading/loading LAION CLAP model...")
    
    # Try different model configurations
    try:
        # First try HTSAT-tiny which is more stable
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        model.load_ckpt()
        model_type = 'HTSAT-tiny'
        embedding_dim = 512
        print(f"✓ Loaded CLAP with {model_type}")
    except Exception as e:
        print(f"Failed to load HTSAT-tiny: {e}")
        try:
            # Fallback to base if needed
            model = laion_clap.CLAP_Module(enable_fusion=False, device='cpu')
            model.load_ckpt()
            model_type = 'default'
            embedding_dim = 512
            print(f"✓ Loaded CLAP with default config")
        except Exception as e2:
            print(f"Failed to load CLAP: {e2}")
            raise
    
    print(f"✓ Model loaded successfully")
    
    embeddings = []
    failed_count = 0
    
    progress.print_substage(f"Extracting embeddings from {len(audio_paths)} audio files...")
    
    for idx, path in enumerate(tqdm(audio_paths, 
                                   desc=f"CLAP-{model_type} extraction", 
                                   unit="file")):
        try:
            audio, sr = librosa.load(path, sr=48000, duration=120, mono=True)
            audio = audio.reshape(1, -1)
            emb = model.get_audio_embedding_from_data(audio, use_tensor=False)
            embeddings.append(emb.squeeze())
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                print(f"\n⚠ Error processing {Path(path).name}: {e}")
            embeddings.append(np.zeros(embedding_dim))
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed: {idx + 1}/{len(audio_paths)} files")
    
    if failed_count > 0:
        print(f"⚠ Failed to process {failed_count} files (using zero vectors)")
    
    print(f"✓ Extracted {len(embeddings)} embeddings, shape: {embeddings[0].shape}")
    return np.array(embeddings)

# ==================== EVALUATION METRICS ====================

def compute_purity(y_true, y_pred):
    """Compute purity score"""
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def compute_cluster_balance(labels):
    """Compute cluster balance (entropy) and normalized balance"""
    unique, counts = np.unique(labels, return_counts=True)
    
    # Cluster balance (entropy)
    cluster_balance = entropy(counts / counts.sum())
    
    # Normalized cluster balance (0-1, where 1 is perfectly balanced)
    n_clusters = len(unique)
    max_entropy = np.log(n_clusters)
    norm_balance = cluster_balance / max_entropy if max_entropy > 0 else 0
    
    return cluster_balance, norm_balance

def compute_cophenetic_correlation(X, linkage_matrix):
    """Compute cophenetic correlation coefficient"""
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    
    # cophenet returns (correlation_coefficient, cophenetic_distance_matrix)
    corr_coeff, cophenetic_dists = cophenet(linkage_matrix, pdist(X))
    
    return corr_coeff

# ==================== CLUSTERING PIPELINE ====================

def run_clustering_analysis(embeddings, true_labels, model_name):
    """Run clustering using existing methods with all metrics"""
    
    progress.print_stage(f"CLUSTERING ANALYSIS: {model_name}")
    
    # Normalize embeddings
    progress.print_substage("Normalizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    print(f"✓ Normalized to zero mean and unit variance")
    
    # Compute linkage matrix for hierarchical metrics
    linkage_matrix = linkage(X, method='ward')
    
    # Initialize clusterer
    clusterer = EDMClusterer(random_state=42)
    results = {}
    
    # 1. Forced clustering (k=35)
    progress.print_substage("Testing forced clustering (k=35)...")
    labels_35 = clusterer.fit_forced_clusters(X, n_clusters=35, method='kmeans')
    
    # Calculate all metrics for k=35
    metrics_35 = clusterer.evaluate_clustering(X, labels_35, y_true=true_labels)
    
    # Add missing metrics
    metrics_35['purity'] = compute_purity(true_labels, labels_35)
    metrics_35['mi_score'] = mutual_info_score(true_labels, labels_35)
    metrics_35['cophenetic_corr'] = compute_cophenetic_correlation(X, linkage_matrix)
    balance, norm_balance = compute_cluster_balance(labels_35)
    metrics_35['cluster_balance'] = balance
    metrics_35['norm_balance'] = norm_balance
    
    results['k35'] = metrics_35
    
    print(f"\n=== Results for k=35 ===")
    print(f"External Metrics:")
    print(f"  NMI:              {metrics_35.get('nmi', 0):.4f}")
    print(f"  ARI:              {metrics_35.get('ari', 0):.4f}")
    print(f"  Purity:           {metrics_35.get('purity', 0):.4f}")
    print(f"  MI Score:         {metrics_35.get('mi_score', 0):.4f}")
    print(f"Internal Metrics:")
    print(f"  Silhouette:       {metrics_35.get('silhouette', 0):.4f}")
    print(f"  Davies-Bouldin:   {metrics_35.get('davies_bouldin', 0):.4f}")
    print(f"  Cophenetic Corr:  {metrics_35.get('cophenetic_corr', 0):.4f}")
    print(f"Distribution Metrics:")
    print(f"  Cluster Balance:  {metrics_35.get('cluster_balance', 0):.4f}")
    print(f"  Norm. Balance:    {metrics_35.get('norm_balance', 0):.4f}")
    
    # 2. Natural clustering
    progress.print_substage("Finding natural clusters...")
    print("This may take a few minutes as we test k=15 to k=40...")
    
    labels_natural, optimal_k = clusterer.find_natural_clusters(X, min_k=15, max_k=40)
    metrics_natural = clusterer.evaluate_clustering(X, labels_natural, y_true=true_labels)
    
    # Add missing metrics for natural clustering
    metrics_natural['purity'] = compute_purity(true_labels, labels_natural)
    metrics_natural['mi_score'] = mutual_info_score(true_labels, labels_natural)
    metrics_natural['cophenetic_corr'] = compute_cophenetic_correlation(X, linkage_matrix)
    balance_nat, norm_balance_nat = compute_cluster_balance(labels_natural)
    metrics_natural['cluster_balance'] = balance_nat
    metrics_natural['norm_balance'] = norm_balance_nat
    
    results['natural'] = metrics_natural
    
    print(f"\n=== Natural clustering (k={optimal_k}) ===")
    print(f"External Metrics:")
    print(f"  NMI:              {metrics_natural.get('nmi', 0):.4f}")
    print(f"  ARI:              {metrics_natural.get('ari', 0):.4f}")
    print(f"  Purity:           {metrics_natural.get('purity', 0):.4f}")
    print(f"  MI Score:         {metrics_natural.get('mi_score', 0):.4f}")
    print(f"Internal Metrics:")
    print(f"  Silhouette:       {metrics_natural.get('silhouette', 0):.4f}")
    print(f"  Davies-Bouldin:   {metrics_natural.get('davies_bouldin', 0):.4f}")
    print(f"  Cophenetic Corr:  {metrics_natural.get('cophenetic_corr', 0):.4f}")
    print(f"Distribution Metrics:")
    print(f"  Cluster Balance:  {metrics_natural.get('cluster_balance', 0):.4f}")
    print(f"  Norm. Balance:    {metrics_natural.get('norm_balance', 0):.4f}")
    
    if optimal_k < 30:
        print(f"\n→ Suggests overspecification (natural {optimal_k} < commercial 35)")
    else:
        print(f"→ Close to commercial taxonomy")
    
    return results

# ==================== MAIN PIPELINE ====================

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("EDM CLUSTERING VALIDATION WITH PRE-TRAINED EMBEDDINGS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Models to test
    models_to_test = {
        'MERT-95M': extract_mert_embeddings,
    }
    
    # Try to add CLAP
    try:
        import laion_clap
        models_to_test['CLAP-tiny'] = extract_clap_embeddings
        print("✓ CLAP available")
    except ImportError:
        print("⚠ CLAP not available, continuing with MERT only")
    
    # Step 1: Load dataset
    progress.print_stage("STEP 1: LOADING DATASET")
    audio_paths, true_labels, genre_mapping = load_beatport_dataset()
    
    # Results storage
    all_results = {}
    
    # Step 2: Process each model
    for idx, (model_name, extractor_func) in enumerate(models_to_test.items(), 1):
        progress.print_stage(f"STEP {idx+1}: PROCESSING {model_name}")
        
        # Check for saved embeddings
        embedding_file = f"embeddings_{model_name.replace('-', '_')}.npy"
        
        if os.path.exists(embedding_file):
            print(f"✓ Found cached embeddings: {embedding_file}")
            print(f"  Loading...")
            embeddings = np.load(embedding_file)
            print(f"✓ Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        else:
            print(f"✗ No cached embeddings found")
            print(f"  Extracting fresh embeddings...")
            try:
                embeddings = extractor_func(audio_paths)
                
                print(f"  Saving embeddings to {embedding_file}...")
                np.save(embedding_file, embeddings)
                print(f"✓ Saved successfully")
            except Exception as e:
                print(f"✗ Failed to extract {model_name} embeddings: {e}")
                print(f"  Skipping {model_name}")
                continue
        
        # Run clustering analysis
        results = run_clustering_analysis(embeddings, true_labels, model_name)
        all_results[model_name] = results

if __name__ == "__main__":
    main()