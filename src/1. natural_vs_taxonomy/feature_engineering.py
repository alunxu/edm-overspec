"""
feature_engineering.py
======================
Feature engineering module for EDM analysis
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import warnings
warnings.filterwarnings('ignore')


class EDMFeatureEngineer:
    """
    Advanced feature engineering for EDM genre analysis.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.feature_metadata = {}
        self.original_features = None
        self.engineered_features = None
        
    def fit_transform(self, X, feature_names=None):
        """
        Apply complete feature engineering pipeline.
        
        Parameters:
        -----------
        X : DataFrame or array-like
            Original features
        feature_names : list, optional
            Feature names
            
        Returns:
        --------
        X_engineered : DataFrame
            Engineered features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        self.original_features = X.columns.tolist()
        print(f"Original features: {len(self.original_features)}")
        
        # 1. Create multi-scale features
        X_multi = self._create_multi_scale_features(X)
        
        # 2. Apply advanced scaling
        X_scaled = self._apply_advanced_scaling(X_multi)
        
        self.engineered_features = X_scaled.columns.tolist()
        print(f"Engineered features: {len(self.engineered_features)}")
        print(f"New features created: {len(self.engineered_features) - len(self.original_features)}")
        
        return X_scaled
    
    def _create_multi_scale_features(self, X):
        """Create features at multiple scales."""
        print("\nCreating multi-scale features...")
        X_multi = X.copy()
        new_features = []
        
        # 1. Log-transformed features for skewed distributions
        for col in X.columns:
            if X[col].min() > 0:  # Only for positive features
                X_multi[f'{col}_log'] = np.log1p(X[col])
                new_features.append(f'{col}_log')
        
        # 2. Polynomial features for key characteristics
        key_features = ['meta.Bpm', 'danceability', '77-danceability', 'onset_rate',
                       'energy', 'valence', 'speechiness', 'acousticness']
        
        for feat in key_features:
            if feat in X.columns:
                X_multi[f'{feat}_squared'] = X[feat] ** 2
                X_multi[f'{feat}_sqrt'] = np.sqrt(np.abs(X[feat]))
                new_features.extend([f'{feat}_squared', f'{feat}_sqrt'])
        
        # 3. Interaction features between rhythm and energy
        rhythm_features = [col for col in X.columns if any(term in col.lower() 
                          for term in ['bpm', 'beat', 'tempo', 'rhythm'])]
        energy_features = [col for col in X.columns if any(term in col.lower() 
                          for term in ['energy', 'power', 'strength'])]
        
        # Limit features to avoid explosion
        rhythm_features = list(set(rhythm_features))[:3]
        energy_features = list(set(energy_features))[:3]
        
        for r_feat in rhythm_features:
            for e_feat in energy_features:
                if r_feat in X.columns and e_feat in X.columns and r_feat != e_feat:
                    interaction_name = f'{r_feat}_x_{e_feat}'
                    if interaction_name not in X_multi.columns:
                        X_multi[interaction_name] = X[r_feat] * X[e_feat]
                        new_features.append(interaction_name)
        
        # 4. Ratio features
        ratio_pairs = [
            ('meta.Bpm', 'onset_rate', 'bpm_onset_ratio'),
            ('energy', 'acousticness', 'energy_acoustic_ratio'),
            ('danceability', 'valence', 'dance_valence_ratio')
        ]
        
        for num, den, name in ratio_pairs:
            if num in X.columns and den in X.columns:
                X_multi[name] = X[num] / (X[den] + 1e-6)
                new_features.append(name)
        
        # 5. Statistical aggregations for tempogram features
        tempogram_cols = [col for col in X.columns if 'tempogram' in col.lower()]
        if tempogram_cols:
            X_multi['tempogram_mean'] = X[tempogram_cols].mean(axis=1)
            X_multi['tempogram_std'] = X[tempogram_cols].std(axis=1)
            X_multi['tempogram_max'] = X[tempogram_cols].max(axis=1)
            new_features.extend(['tempogram_mean', 'tempogram_std', 'tempogram_max'])
        
        self.feature_metadata['new_features'] = new_features
        print(f"  Created {len(new_features)} new features")
        
        return X_multi
    
    def _apply_advanced_scaling(self, X, method='ensemble'):
        """Apply advanced scaling techniques."""
        print(f"\nApplying {method} scaling...")
        
        if method == 'ensemble':
            # Combine multiple scaling approaches
            X_robust = RobustScaler().fit_transform(X)
            X_power = PowerTransformer(method='yeo-johnson').fit_transform(X)
            X_standard = StandardScaler().fit_transform(X)
            
            # Weighted average
            X_scaled = 0.5 * X_robust + 0.3 * X_power + 0.2 * X_standard
            
        elif method == 'robust':
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['robust'] = scaler
            
        elif method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
            X_scaled = scaler.fit_transform(X)
            self.scalers['power'] = scaler
            
        else:  # standard
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['standard'] = scaler
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def get_feature_info(self):
        """Get information about feature engineering."""
        return {
            'original_features': self.original_features,
            'engineered_features': self.engineered_features,
            'new_features': self.feature_metadata.get('new_features', []),
            'n_original': len(self.original_features) if self.original_features else 0,
            'n_engineered': len(self.engineered_features) if self.engineered_features else 0,
            'n_new': len(self.feature_metadata.get('new_features', []))
        }