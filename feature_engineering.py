# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.target_encoders = {}
        self.kmeans_model = None
        self.selected_features = None
        self.scaler = None
        self.can_use_after_interaction = False
        
    def safe_data_conversion(self, df):
        """안전한 데이터 변환"""
        df_clean = df.copy()
        
        categorical_cols = ['gender', 'subscription_type']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
        
        return df_clean
        
    def remove_leakage_features(self, train_df, test_df):
        """누수 피처 제거"""
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        if not self.can_use_after_interaction:
            if 'after_interaction' in train_clean.columns:
                train_clean = train_clean.drop('after_interaction', axis=1)
            if 'after_interaction' in test_clean.columns:
                test_clean = test_clean.drop('after_interaction', axis=1)
        
        return train_clean, test_clean
    
    def determine_after_interaction_usage(self, train_df, test_df):
        """after_interaction 사용 가능성 판단"""
        try:
            def extract_id_numbers(id_series):
                numbers = []
                for id_val in id_series:
                    try:
                        if '_' in str(id_val):
                            num = int(str(id_val).split('_')[1])
                            numbers.append(num)
                    except:
                        continue
                return numbers
            
            train_id_nums = extract_id_numbers(train_df['ID'])
            test_id_nums = extract_id_numbers(test_df['ID'])
            
            if train_id_nums and test_id_nums:
                train_max = max(train_id_nums)
                test_min = min(test_id_nums)
                
                overlap_samples = len([x for x in train_id_nums if x >= test_min])
                overlap_ratio = overlap_samples / len(train_id_nums)
                
                self.can_use_after_interaction = overlap_ratio <= 0.001
            else:
                self.can_use_after_interaction = False
                
        except Exception:
            self.can_use_after_interaction = False
    
    def create_basic_features(self, df):
        """기본 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        if 'age' in df.columns:
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            df_new['age_group'] = pd.cut(age_safe, 
                                       bins=[0, 25, 35, 50, 65, 100], 
                                       labels=[0, 1, 2, 3, 4])
            df_new['age_group'] = df_new['age_group'].fillna(2).astype(int)
            
            df_new['age_normalized'] = (age_safe - age_safe.mean()) / age_safe.std()
            df_new['age_squared'] = age_safe ** 2
        
        if all(col in df.columns for col in ['tenure', 'frequent']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            
            df_new['activity_score'] = frequent_safe / np.sqrt(tenure_safe + 1)
            df_new['activity_score'] = np.clip(df_new['activity_score'], 0, 10)
            
            df_new['usage_intensity'] = frequent_safe / (tenure_safe / 30 + 1)
            df_new['tenure_log'] = np.log1p(tenure_safe)
            df_new['frequent_log'] = np.log1p(frequent_safe)
        
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['payment_stability'] = contract_safe / payment_safe
            df_new['payment_stability'] = np.clip(df_new['payment_stability'], 0, 20)
            
            df_new['contract_payment_ratio'] = contract_safe / (payment_safe + 1)
            df_new['payment_frequency'] = 365 / payment_safe
        
        return df_new
    
    def create_ratio_features(self, df):
        """비율 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        if all(col in df.columns for col in ['age', 'tenure']):
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            df_new['age_tenure_ratio'] = age_safe / np.sqrt(tenure_safe + 1)
            df_new['age_tenure_ratio'] = np.clip(df_new['age_tenure_ratio'], 0, 20)
            
            df_new['maturity_score'] = (age_safe - 18) * np.log1p(tenure_safe)
        
        if all(col in df.columns for col in ['frequent', 'payment_interval']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            df_new['usage_payment_ratio'] = frequent_safe / payment_safe
            df_new['usage_payment_ratio'] = np.clip(df_new['usage_payment_ratio'], 0, 5)
        
        if all(col in df.columns for col in ['age', 'frequent']):
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            df_new['age_usage_interaction'] = (age_safe / 10) * np.log1p(frequent_safe)
        
        return df_new
    
    def create_target_encoding(self, train_df, test_df):
        """타겟 인코딩"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        categorical_cols = ['gender', 'subscription_type']
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                train_encoded = np.full(len(train_df), np.nan)
                
                global_mean = train_df['support_needs'].mean()
                
                for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    category_stats = fold_train.groupby(col)['support_needs'].agg(['mean', 'count'])
                    
                    for category in category_stats.index:
                        mean_val = category_stats.loc[category, 'mean']
                        count_val = category_stats.loc[category, 'count']
                        
                        smoothing_factor = max(5, min(15, count_val / 10))
                        
                        smoothed_mean = (mean_val * count_val + global_mean * smoothing_factor) / (count_val + smoothing_factor)
                        
                        mask = fold_val[col] == category
                        train_encoded[fold_val.index[mask]] = smoothed_mean
                
                train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
                
                train_encoded = 0.85 * train_encoded + 0.15 * global_mean
                train_new[f'{col}_encoded'] = train_encoded
                
                category_stats_all = train_df.groupby(col)['support_needs'].agg(['mean', 'count'])
                
                test_encoded = np.full(len(test_df), global_mean)
                for category in category_stats_all.index:
                    if category in test_df[col].values:
                        mean_val = category_stats_all.loc[category, 'mean']
                        count_val = category_stats_all.loc[category, 'count']
                        
                        smoothing_factor = max(5, min(15, count_val / 10))
                        smoothed_mean = (mean_val * count_val + global_mean * smoothing_factor) / (count_val + smoothing_factor)
                        smoothed_mean = 0.85 * smoothed_mean + 0.15 * global_mean
                        
                        mask = test_df[col] == category
                        test_encoded[mask] = smoothed_mean
                
                test_new[f'{col}_encoded'] = test_encoded
                
                train_new[col] = train_df[col]
                test_new[col] = test_df[col]
        
        return train_new, test_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        if all(col in df.columns for col in ['age', 'tenure']):
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            df_new['age_tenure_product'] = np.log1p(age_safe * tenure_safe / 1000)
            
            df_new['age_tenure_diff'] = age_safe - (tenure_safe / 30)
            df_new['seniority_index'] = (age_safe - 18) * np.log1p(tenure_safe / 365)
        
        if all(col in df.columns for col in ['frequent', 'contract_length']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            df_new['usage_contract_product'] = np.log1p(frequent_safe * contract_safe / 100)
            
            df_new['commitment_usage_ratio'] = contract_safe / (frequent_safe + 1)
        
        if all(col in df.columns for col in ['payment_interval', 'frequent']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            df_new['payment_usage_sync'] = frequent_safe / payment_safe
            
        if all(col in df.columns for col in ['age', 'payment_interval', 'contract_length']):
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['financial_maturity'] = (age_safe / 10) * np.log1p(contract_safe / payment_safe)
        
        return df_new
    
    def create_clustering_features(self, train_df, test_df):
        """클러스터링 피처 생성"""
        try:
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            if self.can_use_after_interaction and 'after_interaction' in train_df.columns:
                numeric_cols.append('after_interaction')
            
            available_cols = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
            
            if len(available_cols) < 3:
                return train_df, test_df
            
            train_cluster_data = train_df[available_cols].fillna(0)
            test_cluster_data = test_df[available_cols].fillna(0)
            
            scaler = RobustScaler()
            train_scaled = scaler.fit_transform(train_cluster_data)
            test_scaled = scaler.transform(test_cluster_data)
            
            self.kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10)
            train_clusters = self.kmeans_model.fit_predict(train_scaled)
            test_clusters = self.kmeans_model.predict(test_scaled)
            
            train_df['cluster_id'] = train_clusters
            test_df['cluster_id'] = test_clusters
            
            cluster_centers = self.kmeans_model.cluster_centers_
            
            train_distances = []
            test_distances = []
            
            for i, center in enumerate(cluster_centers):
                train_dist = np.linalg.norm(train_scaled - center, axis=1)
                test_dist = np.linalg.norm(test_scaled - center, axis=1)
                
                train_df[f'dist_cluster_{i}'] = train_dist
                test_df[f'dist_cluster_{i}'] = test_dist
            
            return train_df, test_df
            
        except Exception as e:
            return train_df, test_df
    
    def create_polynomial_features(self, df):
        """다항식 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        numeric_base = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        if self.can_use_after_interaction and 'after_interaction' in df.columns:
            numeric_base.append('after_interaction')
        
        available_numeric = [col for col in numeric_base if col in df.columns]
        
        for col in available_numeric:
            values = df_new[col].fillna(df_new[col].median())
            values_normalized = (values - values.mean()) / (values.std() + 1e-8)
            
            df_new[f'{col}_squared'] = values_normalized ** 2
            df_new[f'{col}_sqrt'] = np.sqrt(np.abs(values_normalized))
            
            if col not in ['after_interaction']:
                df_new[f'{col}_log'] = np.log1p(np.abs(values))
        
        return df_new
    
    def select_features_optimized(self, train_df, target_col='support_needs', max_features=25):
        """피처 선택 최적화"""
        if target_col not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', target_col]]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df[target_col], 0, 2)
        
        variance_selector = VarianceThreshold(threshold=0.005)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        mi_selector.fit(X_var_df, y)
        
        selected_features = [variance_features[i] for i, selected in enumerate(mi_selector.get_support()) if selected]
        
        self.selected_features = selected_features
        return selected_features
    
    def create_statistical_features(self, df):
        """통계적 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        if self.can_use_after_interaction and 'after_interaction' in df.columns:
            numeric_cols.append('after_interaction')
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            numeric_data = df_new[available_cols].fillna(0)
            
            df_new['feature_mean'] = numeric_data.mean(axis=1)
            df_new['feature_std'] = numeric_data.std(axis=1)
            df_new['feature_median'] = numeric_data.median(axis=1)
            df_new['feature_max'] = numeric_data.max(axis=1)
            df_new['feature_min'] = numeric_data.min(axis=1)
            
            df_new['feature_range'] = df_new['feature_max'] - df_new['feature_min']
            df_new['feature_cv'] = df_new['feature_std'] / (df_new['feature_mean'] + 1e-8)
        
        return df_new
    
    def create_features(self, train_df, test_df):
        """피처 생성 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        try:
            self.determine_after_interaction_usage(train_df, test_df)
            
            train_df, test_df = self.remove_leakage_features(train_df, test_df)
            
            train_df = self.create_basic_features(train_df)
            test_df = self.create_basic_features(test_df)
            
            train_df = self.create_ratio_features(train_df)
            test_df = self.create_ratio_features(test_df)
            
            train_df = self.create_interaction_features(train_df)
            test_df = self.create_interaction_features(test_df)
            
            train_df = self.create_polynomial_features(train_df)
            test_df = self.create_polynomial_features(test_df)
            
            train_df = self.create_statistical_features(train_df)
            test_df = self.create_statistical_features(test_df)
            
            train_df, test_df = self.create_clustering_features(train_df, test_df)
            
            train_df, test_df = self.create_target_encoding(train_df, test_df)
            
            train_df = self.safe_data_conversion(train_df)
            test_df = self.safe_data_conversion(test_df)
            
            return train_df, test_df
            
        except Exception as e:
            print(f"피처 생성 중 오류: {e}")
            return None, None

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        engineer = FeatureEngineer()
        train_processed, test_processed = engineer.create_features(train_df, test_df)
        
        return engineer, train_processed, test_processed
        
    except Exception as e:
        return None, None, None

if __name__ == "__main__":
    main()