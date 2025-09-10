# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_selector = None
        self.imputers = {}
        self.selected_features = None
        self.temporal_cutoff = None
        self.pca_transformer = None
        
    def safe_data_conversion(self, df):
        """안전한 데이터 변환"""
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
        
        return df_clean
        
    def apply_temporal_filtering(self, train_df, temporal_info=None):
        """시간적 필터링"""
        if temporal_info is None or 'temporal_threshold' not in temporal_info:
            return train_df
        
        threshold = temporal_info['temporal_threshold']
        
        if 'temporal_id' not in train_df.columns:
            return train_df
        
        safe_mask = train_df['temporal_id'] <= threshold
        safe_data = train_df[safe_mask].copy()
        
        removal_ratio = 1 - (len(safe_data) / len(train_df))
        if removal_ratio > 0.12:
            percentile_92 = np.percentile(train_df['temporal_id'], 92)
            safe_mask = train_df['temporal_id'] <= percentile_92
            safe_data = train_df[safe_mask].copy()
        
        return safe_data if len(safe_data) > 6000 else train_df
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        if 'temporal_position' in train_clean.columns:
            numeric_cols.append('temporal_position')
        
        for col in train_clean.columns:
            if 'after_interaction' in col:
                numeric_cols.append(col)
        
        common_numeric = [col for col in numeric_cols if col in train_clean.columns and col in test_clean.columns]
        
        if common_numeric:
            missing_ratios = {}
            for col in common_numeric:
                missing_ratio = train_clean[col].isnull().mean()
                missing_ratios[col] = missing_ratio
            
            high_missing_cols = [col for col, ratio in missing_ratios.items() if ratio > 0.25]
            medium_missing_cols = [col for col, ratio in missing_ratios.items() if 0.05 < ratio <= 0.25]
            low_missing_cols = [col for col, ratio in missing_ratios.items() if ratio <= 0.05]
            
            if high_missing_cols:
                if 'knn' not in self.imputers:
                    self.imputers['knn'] = KNNImputer(n_neighbors=7, weights='distance')
                    train_imputed_knn = self.imputers['knn'].fit_transform(train_clean[high_missing_cols])
                else:
                    train_imputed_knn = self.imputers['knn'].transform(train_clean[high_missing_cols])
                
                test_imputed_knn = self.imputers['knn'].transform(test_clean[high_missing_cols])
                
                train_clean[high_missing_cols] = train_imputed_knn
                test_clean[high_missing_cols] = test_imputed_knn
            
            if medium_missing_cols:
                if 'iterative' not in self.imputers:
                    self.imputers['iterative'] = IterativeImputer(
                        max_iter=10, 
                        random_state=42,
                        initial_strategy='median'
                    )
                    train_imputed_iter = self.imputers['iterative'].fit_transform(train_clean[medium_missing_cols])
                else:
                    train_imputed_iter = self.imputers['iterative'].transform(train_clean[medium_missing_cols])
                
                test_imputed_iter = self.imputers['iterative'].transform(test_clean[medium_missing_cols])
                
                train_clean[medium_missing_cols] = train_imputed_iter
                test_clean[medium_missing_cols] = test_imputed_iter
            
            for col in low_missing_cols:
                if col not in self.imputers:
                    self.imputers[col] = train_clean[col].median()
                
                train_clean[col].fillna(self.imputers[col], inplace=True)
                test_clean[col].fillna(self.imputers[col], inplace=True)
        
        categorical_cols = ['gender', 'subscription_type']
        
        for col in categorical_cols:
            if col in train_clean.columns and col in test_clean.columns:
                if col not in self.imputers:
                    if 'support_needs' in train_clean.columns:
                        target_modes = {}
                        for target in [0, 1, 2]:
                            subset = train_clean[train_clean['support_needs'] == target]
                            if len(subset) > 0 and not subset[col].mode().empty:
                                target_modes[target] = subset[col].mode().iloc[0]
                        
                        overall_mode = train_clean[col].mode()
                        default_mode = overall_mode.iloc[0] if len(overall_mode) > 0 else 'Unknown'
                        
                        self.imputers[col] = {'target_modes': target_modes, 'default': default_mode}
                    else:
                        mode_val = train_clean[col].mode()
                        self.imputers[col] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                
                if isinstance(self.imputers[col], dict):
                    for idx in train_clean[train_clean[col].isnull()].index:
                        if 'support_needs' in train_clean.columns:
                            target = train_clean.loc[idx, 'support_needs']
                            if target in self.imputers[col]['target_modes']:
                                train_clean.loc[idx, col] = self.imputers[col]['target_modes'][target]
                            else:
                                train_clean.loc[idx, col] = self.imputers[col]['default']
                        else:
                            train_clean.loc[idx, col] = self.imputers[col]['default']
                    
                    test_clean[col].fillna(self.imputers[col]['default'], inplace=True)
                else:
                    train_clean[col].fillna(self.imputers[col], inplace=True)
                    test_clean[col].fillna(self.imputers[col], inplace=True)
        
        return train_clean, test_clean
    
    def remove_outliers_robust(self, train_df):
        """이상치 제거"""
        train_clean = self.safe_data_conversion(train_df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for col in train_clean.columns:
            if any(keyword in col for keyword in ['_log', '_sqrt', '_rank', 'customer_value', 'payment_stability', 'usage_intensity']):
                if col not in numeric_cols:
                    numeric_cols.append(col)
        
        available_cols = [col for col in numeric_cols if col in train_clean.columns]
        
        for col in available_cols:
            values = train_clean[col].dropna()
            
            if len(values) > 100:
                lower_bound = values.quantile(0.002)
                upper_bound = values.quantile(0.998)
                
                train_clean[col] = np.clip(train_clean[col], lower_bound, upper_bound)
        
        return train_clean
    
    def select_features_optimized(self, train_df, max_features=78):
        """피처 선택"""
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        
        variance_selector = VarianceThreshold(threshold=0.00005)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        correlation_matrix = X_var_df.corr().abs()
        
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        reduced_features = [f for f in variance_features if f not in high_corr_features]
        
        if len(reduced_features) <= max_features:
            self.selected_features = reduced_features
            return reduced_features
        
        X_reduced = train_df[reduced_features].fillna(0)
        mi_scores = mutual_info_classif(X_reduced, y, random_state=42)
        
        feature_scores = list(zip(reduced_features, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [f[0] for f in feature_scores[:max_features]]
        
        self.selected_features = selected_features
        return selected_features
    
    def apply_scaling(self, train_df, test_df):
        """스케일링"""
        train_clean = self.safe_data_conversion(train_df)
        test_clean = self.safe_data_conversion(test_df)
        
        numeric_cols = [col for col in train_clean.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        
        common_numeric = [col for col in numeric_cols if col in test_clean.columns]
        
        if not common_numeric:
            return train_clean, test_clean
        
        if 'main' not in self.scalers:
            self.scalers['main'] = RobustScaler()
            
            train_scaled = self.scalers['main'].fit_transform(train_clean[common_numeric])
        else:
            train_scaled = self.scalers['main'].transform(train_clean[common_numeric])
        
        test_scaled = self.scalers['main'].transform(test_clean[common_numeric])
        
        train_clean[common_numeric] = train_scaled
        test_clean[common_numeric] = test_scaled
        
        return train_clean, test_clean
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        issues = []
        
        train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        test_numeric_cols = test_df.select_dtypes(include=[np.number]).columns
        
        train_inf = np.isinf(train_df[train_numeric_cols]).sum().sum()
        test_inf = np.isinf(test_df[test_numeric_cols]).sum().sum()
        
        if train_inf > 0:
            issues.append(f"train_infinite_values: {train_inf}")
        if test_inf > 0:
            issues.append(f"test_infinite_values: {test_inf}")
        
        train_nan = train_df[train_numeric_cols].isnull().sum().sum()
        test_nan = test_df[test_numeric_cols].isnull().sum().sum()
        
        if train_nan > 0:
            issues.append(f"train_missing_values: {train_nan}")
        if test_nan > 0:
            issues.append(f"test_missing_values: {test_nan}")
        
        if 'support_needs' in train_df.columns:
            invalid_count = (~train_df['support_needs'].isin([0, 1, 2])).sum()
            if invalid_count > 0:
                issues.append(f"invalid_targets: {invalid_count}")
        
        return len(issues) == 0, issues
    
    def process_data(self, train_df, test_df, temporal_info=None):
        """전처리 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        if temporal_info is not None:
            train_df = self.apply_temporal_filtering(train_df, temporal_info)
        
        train_df, test_df = self.handle_missing_values(train_df, test_df)
        
        train_df = self.remove_outliers_robust(train_df)
        
        if 'support_needs' in train_df.columns:
            selected_features = self.select_features_optimized(train_df, max_features=72)
            
            keep_cols_train = ['ID', 'support_needs'] + selected_features
            keep_cols_test = ['ID'] + [f for f in selected_features if f in test_df.columns]
            
            available_train_cols = [col for col in keep_cols_train if col in train_df.columns]
            available_test_cols = [col for col in keep_cols_test if col in test_df.columns]
            
            train_df = train_df[available_train_cols]
            test_df = test_df[available_test_cols]
        
        train_df, test_df = self.apply_scaling(train_df, test_df)
        
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        
        return train_df, test_df
    
    def prepare_data_temporal_optimized(self, train_df, test_df, val_size=0.20, gap_size=0.015):
        """시간 기반 데이터 준비"""
        if train_df is None or test_df is None:
            return None, None, None, None, None, None
            
        if 'support_needs' not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            common_features = [col for col in feature_cols if col in test_df.columns]
            
            if not common_features:
                raise ValueError("공통 피처 없음")
                
            X_train = train_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            X_test = test_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
            test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))
            
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_dummy, y_val_dummy = train_test_split(
                X_train, np.zeros(len(X_train)), test_size=val_size, random_state=42
            )
            
            return X_train_split, X_val_split, y_dummy[:len(X_train_split)], y_val_dummy, X_test, test_ids
        
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
        
        if not common_features:
            raise ValueError("공통 피처 없음")
        
        common_features = sorted(common_features)
        
        X = train_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df['support_needs'], 0, 2)
        X_test = test_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
        test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))
        
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            total_samples = len(sorted_indices)
            gap_samples = int(total_samples * gap_size)
            val_samples = int(total_samples * val_size)
            train_samples = total_samples - val_samples - gap_samples
            
            if train_samples < 500 or val_samples < 200:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=val_size, random_state=42, stratify=y
                )
            else:
                train_indices = sorted_indices[:train_samples]
                val_indices = sorted_indices[train_samples + gap_samples:]
                
                X_train = X.iloc[train_indices]
                X_val = X.iloc[val_indices]
                y_train = y.iloc[train_indices]
                y_val = y.iloc[val_indices]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=42, stratify=y
            )
        
        return X_train, X_val, y_train, y_val, X_test, test_ids

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        preprocessor = DataPreprocessor()
        train_processed, test_processed = preprocessor.process_data(train_df, test_df)
        
        if train_processed is not None and test_processed is not None:
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal_optimized(
                train_processed, test_processed
            )
            
            return preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids
        else:
            return preprocessor, None, None, None, None, None, None
            
    except Exception as e:
        print(f"전처리 오류: {e}")
        return None, None, None, None, None, None, None

if __name__ == "__main__":
    main()