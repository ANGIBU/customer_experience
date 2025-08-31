# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.kmeans_model = None
        self.pca_model = None
        self.poly_features = None
        self.feature_names_order = None
        self.is_fitted = False
        
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
        
    def create_temporal_features(self, df):
        """시간 기반 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        if 'ID' in df.columns:
            id_numbers = []
            for id_val in df['ID']:
                try:
                    if '_' in str(id_val):
                        num = int(str(id_val).split('_')[1])
                        id_numbers.append(num)
                    else:
                        id_numbers.append(0)
                except:
                    id_numbers.append(0)
            
            df_new['temporal_id'] = id_numbers
            
            if len(id_numbers) > 0:
                id_min = min(id_numbers)
                id_max = max(id_numbers)
                if id_max > id_min:
                    df_new['temporal_position'] = [(x - id_min) / (id_max - id_min) for x in id_numbers]
                else:
                    df_new['temporal_position'] = [0.5] * len(id_numbers)
        
        return df_new
    
    def create_customer_segments(self, df):
        """고객 세분화 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        if all(col in df.columns for col in ['age', 'tenure', 'frequent']):
            df_new['customer_segment'] = 0
            
            age_val = df_new['age'].fillna(35)
            tenure_val = df_new['tenure'].fillna(100)
            frequent_val = df_new['frequent'].fillna(10)
            
            high_value_mask = (
                (age_val >= 35) & 
                (tenure_val >= 180) & 
                (frequent_val >= 15)
            )
            df_new.loc[high_value_mask, 'customer_segment'] = 2
            
            medium_value_mask = (
                (age_val >= 25) & 
                (tenure_val >= 60) & 
                (frequent_val >= 5) &
                (~high_value_mask)
            )
            df_new.loc[medium_value_mask, 'customer_segment'] = 1
            
            df_new['segment_stability'] = df_new['tenure'].fillna(0) * df_new['customer_segment']
        
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            df_new['payment_behavior'] = 0
            
            payment_val = df_new['payment_interval'].fillna(30)
            
            regular_mask = payment_val <= 30
            df_new.loc[regular_mask, 'payment_behavior'] = 1
            
            irregular_mask = payment_val > 90
            df_new.loc[irregular_mask, 'payment_behavior'] = 2
            
            contract_val = df_new['contract_length'].fillna(90)
            df_new['contract_payment_ratio'] = contract_val / (payment_val + 1)
            df_new['contract_payment_ratio'] = df_new['contract_payment_ratio'].replace([np.inf, -np.inf], 0)
        
        return df_new
    
    def create_mathematical_features(self, df):
        """수학적 변환 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        for col in available_cols:
            try:
                values = df_new[col].fillna(0)
                values = values.replace([np.inf, -np.inf], 0)
                
                df_new[f'{col}_log'] = np.log1p(np.abs(values))
                df_new[f'{col}_sqrt'] = np.sqrt(np.abs(values))
                df_new[f'{col}_square'] = values ** 2
                
                if values.std() > 0:
                    df_new[f'{col}_zscore'] = (values - values.mean()) / values.std()
                else:
                    df_new[f'{col}_zscore'] = 0
                    
                df_new[f'{col}_log'] = df_new[f'{col}_log'].replace([np.inf, -np.inf], 0)
                df_new[f'{col}_sqrt'] = df_new[f'{col}_sqrt'].replace([np.inf, -np.inf], 0)
                df_new[f'{col}_square'] = df_new[f'{col}_square'].replace([np.inf, -np.inf], 0)
                df_new[f'{col}_zscore'] = df_new[f'{col}_zscore'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"수학적 변환 오류 {col}: {e}")
                continue
        
        if len(available_cols) >= 2:
            try:
                numeric_data = df_new[available_cols].fillna(0)
                numeric_data = numeric_data.replace([np.inf, -np.inf], 0)
                
                df_new['numeric_sum'] = numeric_data.sum(axis=1)
                df_new['numeric_product'] = numeric_data.prod(axis=1)
                df_new['numeric_variance'] = numeric_data.var(axis=1)
                
                df_new['numeric_sum'] = df_new['numeric_sum'].replace([np.inf, -np.inf], 0)
                df_new['numeric_product'] = df_new['numeric_product'].replace([np.inf, -np.inf], 0)
                df_new['numeric_variance'] = df_new['numeric_variance'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"집계 피처 생성 오류: {e}")
        
        return df_new
    
    def create_business_logic_features(self, df):
        """비즈니스 로직 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        if all(col in df.columns for col in ['frequent', 'tenure', 'contract_length']):
            try:
                frequent_val = df_new['frequent'].fillna(10)
                tenure_val = df_new['tenure'].fillna(100)
                contract_val = df_new['contract_length'].fillna(90)
                
                df_new['engagement_score'] = (
                    frequent_val * np.log1p(tenure_val) * 
                    np.sqrt(contract_val)
                ) / 1000
                
                df_new['loyalty_index'] = (
                    tenure_val * contract_val
                ) / (frequent_val + 1)
                
                df_new['engagement_score'] = df_new['engagement_score'].replace([np.inf, -np.inf], 0)
                df_new['loyalty_index'] = df_new['loyalty_index'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"비즈니스 로직 피처 오류: {e}")
        
        if all(col in df.columns for col in ['age', 'payment_interval', 'contract_length']):
            try:
                age_val = df_new['age'].fillna(35)
                payment_val = df_new['payment_interval'].fillna(30)
                contract_val = df_new['contract_length'].fillna(90)
                
                df_new['financial_stability'] = (
                    contract_val / (payment_val + 1) * 
                    (1 + np.exp(-age_val / 50))
                )
                
                df_new['financial_stability'] = df_new['financial_stability'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"금융 안정성 피처 오류: {e}")
        
        if 'after_interaction' in df.columns and 'frequent' in df.columns:
            try:
                after_val = df_new['after_interaction'].fillna(0)
                frequent_val = df_new['frequent'].fillna(10)
                
                df_new['interaction_ratio'] = after_val / (frequent_val + 1)
                df_new['support_intensity'] = np.log1p(after_val)
                
                df_new['interaction_ratio'] = df_new['interaction_ratio'].replace([np.inf, -np.inf], 0)
                df_new['support_intensity'] = df_new['support_intensity'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"상호작용 피처 오류: {e}")
        
        if all(col in df.columns for col in ['age', 'tenure']):
            try:
                age_val = df_new['age'].fillna(35)
                tenure_val = df_new['tenure'].fillna(100)
                
                df_new['experience_factor'] = age_val * np.log1p(tenure_val)
                df_new['experience_factor'] = df_new['experience_factor'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"경험 팩터 피처 오류: {e}")
        
        return df_new
    
    def create_polynomial_features(self, train_df, test_df):
        """다항식 피처 생성"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols 
                         if col in train_df.columns and col in test_df.columns]
        
        if len(available_cols) < 2:
            return train_df, test_df
        
        try:
            train_numeric = train_df[available_cols].fillna(0)
            test_numeric = test_df[available_cols].fillna(0)
            
            train_numeric = train_numeric.replace([np.inf, -np.inf], 0)
            test_numeric = test_numeric.replace([np.inf, -np.inf], 0)
            
            if self.poly_features is None:
                self.poly_features = PolynomialFeatures(
                    degree=2, 
                    interaction_only=True, 
                    include_bias=False
                )
                poly_train = self.poly_features.fit_transform(train_numeric)
            else:
                poly_train = self.poly_features.transform(train_numeric)
            
            poly_test = self.poly_features.transform(test_numeric)
            
            poly_feature_names = self.poly_features.get_feature_names_out(available_cols)
            
            train_new = train_df.copy()
            test_new = test_df.copy()
            
            for i, name in enumerate(poly_feature_names):
                if name not in available_cols:
                    train_new[f'poly_{name}'] = poly_train[:, i]
                    test_new[f'poly_{name}'] = poly_test[:, i]
                    
                    train_new[f'poly_{name}'] = train_new[f'poly_{name}'].replace([np.inf, -np.inf], 0)
                    test_new[f'poly_{name}'] = test_new[f'poly_{name}'].replace([np.inf, -np.inf], 0)
            
            return train_new, test_new
            
        except Exception as e:
            print(f"다항식 피처 생성 오류: {e}")
            return train_df, test_df
    
    def create_target_encoding(self, train_df, test_df):
        """타겟 인코딩 생성"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        categorical_cols = ['gender', 'subscription_type']
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for col in categorical_cols:
                if col in train_df.columns and col in test_df.columns:
                    train_encoded = np.zeros(len(train_df))
                    
                    for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                        fold_train = train_df.iloc[train_idx]
                        fold_val = train_df.iloc[val_idx]
                        
                        target_mean = fold_train.groupby(col)['support_needs'].mean()
                        global_mean = fold_train['support_needs'].mean()
                        
                        encoded_vals = fold_val[col].map(target_mean).fillna(global_mean)
                        train_encoded[val_idx] = encoded_vals
                    
                    train_new[f'{col}_target_mean'] = train_encoded
                    train_new[f'{col}_target_mean'] = train_new[f'{col}_target_mean'].replace([np.inf, -np.inf], 0)
                    
                    target_mean_all = train_df.groupby(col)['support_needs'].mean()
                    global_mean_all = train_df['support_needs'].mean()
                    
                    test_encoded = test_df[col].map(target_mean_all).fillna(global_mean_all)
                    test_new[f'{col}_target_mean'] = test_encoded
                    test_new[f'{col}_target_mean'] = test_new[f'{col}_target_mean'].replace([np.inf, -np.inf], 0)
                    
                    for cls in [0, 1, 2]:
                        try:
                            class_mean = train_df[train_df['support_needs'] == cls].groupby(col)['support_needs'].size()
                            class_total = train_df.groupby(col)['support_needs'].size()
                            class_ratio = (class_mean / class_total).fillna(0)
                            class_ratio = class_ratio.replace([np.inf, -np.inf], 0)
                            
                            test_new[f'{col}_class_{cls}_ratio'] = test_df[col].map(class_ratio).fillna(0)
                            test_new[f'{col}_class_{cls}_ratio'] = test_new[f'{col}_class_{cls}_ratio'].replace([np.inf, -np.inf], 0)
                            
                            train_class_encoded = np.zeros(len(train_df))
                            for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                                fold_train = train_df.iloc[train_idx]
                                fold_val = train_df.iloc[val_idx]
                                
                                fold_class_mean = fold_train[fold_train['support_needs'] == cls].groupby(col)['support_needs'].size()
                                fold_class_total = fold_train.groupby(col)['support_needs'].size()
                                fold_class_ratio = (fold_class_mean / fold_class_total).fillna(0)
                                fold_class_ratio = fold_class_ratio.replace([np.inf, -np.inf], 0)
                                
                                encoded_vals = fold_val[col].map(fold_class_ratio).fillna(0)
                                train_class_encoded[val_idx] = encoded_vals
                            
                            train_new[f'{col}_class_{cls}_ratio'] = train_class_encoded
                            train_new[f'{col}_class_{cls}_ratio'] = train_new[f'{col}_class_{cls}_ratio'].replace([np.inf, -np.inf], 0)
                            
                        except Exception as cls_e:
                            print(f"클래스 {cls} 인코딩 오류: {cls_e}")
                            continue
                            
        except Exception as e:
            print(f"타겟 인코딩 오류: {e}")
        
        return train_new, test_new
    
    def create_clustering_features(self, train_df, test_df):
        """클러스터링 피처 생성"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols 
                         if col in train_df.columns and col in test_df.columns]
        
        if len(available_cols) < 2:
            return train_df, test_df
        
        try:
            train_numeric = train_df[available_cols].fillna(0)
            test_numeric = test_df[available_cols].fillna(0)
            
            train_numeric = train_numeric.replace([np.inf, -np.inf], 0)
            test_numeric = test_numeric.replace([np.inf, -np.inf], 0)
            
            if self.kmeans_model is None:
                self.kmeans_model = KMeans(n_clusters=12, random_state=42, n_init=10)
                train_clusters = self.kmeans_model.fit_predict(train_numeric)
            else:
                train_clusters = self.kmeans_model.predict(train_numeric)
                
            test_clusters = self.kmeans_model.predict(test_numeric)
            
            train_new = train_df.copy()
            test_new = test_df.copy()
            
            train_new['cluster'] = train_clusters
            test_new['cluster'] = test_clusters
            
            train_distances = self.kmeans_model.transform(train_numeric)
            test_distances = self.kmeans_model.transform(test_numeric)
            
            train_new['cluster_distance'] = train_distances.min(axis=1)
            test_new['cluster_distance'] = test_distances.min(axis=1)
            
            train_new['cluster_density'] = 1 / (train_distances.min(axis=1) + 1)
            test_new['cluster_density'] = 1 / (test_distances.min(axis=1) + 1)
            
            train_new['cluster_distance'] = train_new['cluster_distance'].replace([np.inf, -np.inf], 0)
            test_new['cluster_distance'] = test_new['cluster_distance'].replace([np.inf, -np.inf], 0)
            train_new['cluster_density'] = train_new['cluster_density'].replace([np.inf, -np.inf], 0)
            test_new['cluster_density'] = test_new['cluster_density'].replace([np.inf, -np.inf], 0)
            
            return train_new, test_new
            
        except Exception as e:
            print(f"클러스터링 피처 오류: {e}")
            return train_df, test_df
    
    def create_pca_features(self, train_df, test_df):
        """PCA 피처 생성"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols 
                         if col in train_df.columns and col in test_df.columns]
        
        if len(available_cols) < 2:
            return train_df, test_df
        
        try:
            train_numeric = train_df[available_cols].fillna(0)
            test_numeric = test_df[available_cols].fillna(0)
            
            train_numeric = train_numeric.replace([np.inf, -np.inf], 0)
            test_numeric = test_numeric.replace([np.inf, -np.inf], 0)
            
            if self.pca_model is None:
                self.pca_model = PCA(n_components=min(5, len(available_cols)), random_state=42)
                train_pca = self.pca_model.fit_transform(train_numeric)
            else:
                train_pca = self.pca_model.transform(train_numeric)
                
            test_pca = self.pca_model.transform(test_numeric)
            
            train_new = train_df.copy()
            test_new = test_df.copy()
            
            for i in range(train_pca.shape[1]):
                train_new[f'pca_{i}'] = train_pca[:, i]
                test_new[f'pca_{i}'] = test_pca[:, i]
                
                train_new[f'pca_{i}'] = train_new[f'pca_{i}'].replace([np.inf, -np.inf], 0)
                test_new[f'pca_{i}'] = test_new[f'pca_{i}'].replace([np.inf, -np.inf], 0)
            
            return train_new, test_new
            
        except Exception as e:
            print(f"PCA 피처 오류: {e}")
            return train_df, test_df
    
    def create_ratio_features(self, df):
        """비율 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        ratio_pairs = [
            ('age', 'tenure'),
            ('frequent', 'tenure'),
            ('age', 'frequent'),
            ('payment_interval', 'contract_length'),
            ('frequent', 'contract_length')
        ]
        
        for num_col, den_col in ratio_pairs:
            if num_col in df.columns and den_col in df.columns:
                try:
                    numerator = df_new[num_col].fillna(0)
                    denominator = df_new[den_col].fillna(1) + 1
                    
                    ratio_val = numerator / denominator
                    df_new[f'{num_col}_{den_col}_ratio'] = ratio_val.replace([np.inf, -np.inf], 0)
                    
                except Exception as e:
                    print(f"비율 피처 오류 {num_col}/{den_col}: {e}")
        
        return df_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        interaction_pairs = [
            ('age', 'tenure'),
            ('frequent', 'tenure'), 
            ('age', 'frequent'),
            ('payment_interval', 'contract_length'),
            ('frequent', 'payment_interval'),
            ('age', 'contract_length'),
            ('tenure', 'contract_length')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                try:
                    val1 = df_new[feat1].fillna(0)
                    val2 = df_new[feat2].fillna(0)
                    
                    df_new[f'{feat1}_{feat2}_mult'] = (val1 * val2).replace([np.inf, -np.inf], 0)
                    df_new[f'{feat1}_{feat2}_add'] = (val1 + val2).replace([np.inf, -np.inf], 0)
                    df_new[f'{feat1}_{feat2}_diff'] = np.abs(val1 - val2).replace([np.inf, -np.inf], 0)
                    
                except Exception as e:
                    print(f"상호작용 피처 오류 {feat1}×{feat2}: {e}")
        
        return df_new
    
    def create_statistical_features(self, df):
        """통계 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            try:
                numeric_data = df_new[available_cols].fillna(0)
                numeric_data = numeric_data.replace([np.inf, -np.inf], 0)
                
                df_new['numeric_mean'] = numeric_data.mean(axis=1)
                df_new['numeric_std'] = numeric_data.std(axis=1)
                df_new['numeric_median'] = numeric_data.median(axis=1)
                df_new['numeric_range'] = numeric_data.max(axis=1) - numeric_data.min(axis=1)
                
                skew_values = []
                for idx in range(len(numeric_data)):
                    try:
                        row_skew = skew(numeric_data.iloc[idx])
                        if np.isnan(row_skew) or np.isinf(row_skew):
                            row_skew = 0
                        skew_values.append(row_skew)
                    except:
                        skew_values.append(0)
                
                df_new['numeric_skew'] = skew_values
                
                for col_name in ['numeric_mean', 'numeric_std', 'numeric_median', 'numeric_range', 'numeric_skew']:
                    df_new[col_name] = df_new[col_name].replace([np.inf, -np.inf], 0)
                    
            except Exception as e:
                print(f"통계 피처 오류: {e}")
        
        for col in available_cols:
            if col in df.columns:
                try:
                    values = df_new[col].fillna(0)
                    
                    q25 = values.quantile(0.25)
                    q75 = values.quantile(0.75)
                    q05 = values.quantile(0.05)
                    q95 = values.quantile(0.95)
                    
                    df_new[f'{col}_low_flag'] = (values <= q25).astype(int)
                    df_new[f'{col}_high_flag'] = (values >= q75).astype(int)
                    df_new[f'{col}_extreme_flag'] = ((values <= q05) | (values >= q95)).astype(int)
                    
                except Exception as e:
                    print(f"플래그 피처 오류 {col}: {e}")
        
        return df_new
    
    def create_class1_specific_features(self, df):
        """클래스 1 특화 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        if all(col in df.columns for col in ['payment_interval', 'frequent', 'age']):
            try:
                payment_val = df_new['payment_interval'].fillna(30)
                frequent_val = df_new['frequent'].fillna(10)
                age_val = df_new['age'].fillna(35)
                
                df_new['payment_frequency_mismatch'] = np.abs(
                    payment_val - (30 / (frequent_val + 1))
                )
                
                df_new['age_payment_pattern'] = (
                    age_val * payment_val / (frequent_val + 1)
                )
                
                df_new['payment_frequency_mismatch'] = df_new['payment_frequency_mismatch'].replace([np.inf, -np.inf], 0)
                df_new['age_payment_pattern'] = df_new['age_payment_pattern'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"클래스 1 페이먼트 피처 오류: {e}")
        
        if all(col in df.columns for col in ['tenure', 'contract_length', 'after_interaction']):
            try:
                tenure_val = df_new['tenure'].fillna(100)
                contract_val = df_new['contract_length'].fillna(90)
                after_val = df_new['after_interaction'].fillna(0)
                
                df_new['support_dependency'] = (
                    after_val / (tenure_val * contract_val + 1)
                )
                
                df_new['support_dependency'] = df_new['support_dependency'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"서포트 의존성 피처 오류: {e}")
        
        if all(col in df.columns for col in ['frequent', 'tenure', 'age']):
            try:
                frequent_val = df_new['frequent'].fillna(10)
                tenure_val = df_new['tenure'].fillna(100)
                age_val = df_new['age'].fillna(35)
                
                df_new['usage_decline_signal'] = np.exp(
                    -frequent_val / (tenure_val + 1)
                ) * (age_val / 100)
                
                df_new['usage_decline_signal'] = df_new['usage_decline_signal'].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"사용량 감소 신호 피처 오류: {e}")
        
        return df_new
    
    def encode_categorical(self, train_df, test_df):
        """범주형 변수 인코딩"""
        categorical_cols = ['gender', 'subscription_type']
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                try:
                    combined_data = pd.concat([train_df[col], test_df[col]])
                    
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        self.label_encoders[col].fit(combined_data.fillna('Unknown'))
                    
                    train_new[col] = self.label_encoders[col].transform(train_new[col].fillna('Unknown'))
                    test_new[col] = self.label_encoders[col].transform(test_new[col].fillna('Unknown'))
                    
                except Exception as e:
                    print(f"범주형 인코딩 오류 {col}: {e}")
        
        return train_new, test_new
    
    def create_temporal_splits(self, df):
        """시간적 분할 기반 피처"""
        df_new = df.copy()
        
        if 'temporal_id' in df.columns:
            try:
                temporal_ids = df_new['temporal_id']
                
                if len(temporal_ids) > 0 and temporal_ids.nunique() > 1:
                    cutoff_80 = np.percentile(temporal_ids, 80)
                    cutoff_60 = np.percentile(temporal_ids, 60)
                    cutoff_40 = np.percentile(temporal_ids, 40)
                    cutoff_20 = np.percentile(temporal_ids, 20)
                    
                    df_new['temporal_period'] = 0
                    df_new.loc[temporal_ids > cutoff_80, 'temporal_period'] = 4
                    df_new.loc[(temporal_ids > cutoff_60) & (temporal_ids <= cutoff_80), 'temporal_period'] = 3
                    df_new.loc[(temporal_ids > cutoff_40) & (temporal_ids <= cutoff_60), 'temporal_period'] = 2
                    df_new.loc[(temporal_ids > cutoff_20) & (temporal_ids <= cutoff_40), 'temporal_period'] = 1
                else:
                    df_new['temporal_period'] = 0
                    
            except Exception as e:
                print(f"시간 분할 피처 오류: {e}")
                df_new['temporal_period'] = 0
        
        return df_new
    
    def ensure_feature_consistency(self, train_df, test_df):
        """피처 일관성 보장"""
        try:
            if self.feature_names_order is None and not self.is_fitted:
                feature_cols = [col for col in train_df.columns 
                               if col not in ['ID', 'support_needs']]
                self.feature_names_order = sorted(feature_cols)
                self.is_fitted = True
            
            if self.feature_names_order is not None:
                train_consistent = train_df.copy()
                test_consistent = test_df.copy()
                
                for col in self.feature_names_order:
                    if col not in train_consistent.columns:
                        train_consistent[col] = 0
                    if col not in test_consistent.columns:
                        test_consistent[col] = 0
                
                train_ordered_cols = ['ID'] + self.feature_names_order
                if 'support_needs' in train_consistent.columns:
                    train_ordered_cols.append('support_needs')
                
                test_ordered_cols = ['ID'] + self.feature_names_order
                
                train_ordered_cols = [col for col in train_ordered_cols if col in train_consistent.columns]
                test_ordered_cols = [col for col in test_ordered_cols if col in test_consistent.columns]
                
                return train_consistent[train_ordered_cols], test_consistent[test_ordered_cols]
            
        except Exception as e:
            print(f"피처 일관성 보장 오류: {e}")
        
        return train_df, test_df
    
    def create_features(self, train_df, test_df):
        """전체 피처 생성 파이프라인"""
        print("피처 생성 시작")
        print("=" * 30)
        
        if train_df is None or test_df is None:
            print("입력 데이터가 None입니다")
            return None, None
        
        if train_df.empty or test_df.empty:
            print("입력 데이터가 비어있습니다")
            return None, None
        
        try:
            original_features = train_df.shape[1]
            
            train_df = self.create_temporal_features(train_df)
            test_df = self.create_temporal_features(test_df)
            
            train_df = self.create_customer_segments(train_df)
            test_df = self.create_customer_segments(test_df)
            
            train_df = self.create_mathematical_features(train_df)
            test_df = self.create_mathematical_features(test_df)
            
            train_df = self.create_business_logic_features(train_df)
            test_df = self.create_business_logic_features(test_df)
            
            train_df, test_df = self.create_polynomial_features(train_df, test_df)
            
            train_df = self.create_ratio_features(train_df)
            test_df = self.create_ratio_features(test_df)
            
            train_df = self.create_interaction_features(train_df)
            test_df = self.create_interaction_features(test_df)
            
            train_df = self.create_statistical_features(train_df)
            test_df = self.create_statistical_features(test_df)
            
            train_df = self.create_class1_specific_features(train_df)
            test_df = self.create_class1_specific_features(test_df)
            
            train_df, test_df = self.create_target_encoding(train_df, test_df)
            
            train_df, test_df = self.create_clustering_features(train_df, test_df)
            
            train_df, test_df = self.create_pca_features(train_df, test_df)
            
            train_df = self.create_temporal_splits(train_df)
            test_df = self.create_temporal_splits(test_df)
            
            train_df, test_df = self.encode_categorical(train_df, test_df)
            
            train_df, test_df = self.ensure_feature_consistency(train_df, test_df)
            
            train_df = self.safe_data_conversion(train_df)
            test_df = self.safe_data_conversion(test_df)
            
            final_features = train_df.shape[1]
            created_features = final_features - original_features
            
            print(f"피처 생성 완료: {original_features} → {final_features} (+{created_features})")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"피처 생성 파이프라인 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return None, None

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        engineer = FeatureEngineer()
        train_processed, test_processed = engineer.create_features(train_df, test_df)
        
        return engineer, train_processed, test_processed
        
    except Exception as e:
        print(f"메인 함수 오류: {e}")
        return None, None, None

if __name__ == "__main__":
    main()