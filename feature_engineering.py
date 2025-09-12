# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.feature_stats = {}
        self.selected_features = None
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
        
    def handle_after_interaction(self, train_df, test_df, temporal_threshold=None):
        """after_interaction 피처 처리"""
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        if 'after_interaction' in train_df.columns:
            # 시간적 안전 지표 생성
            if temporal_threshold is not None:
                train_id_nums = []
                for id_val in train_df['ID']:
                    try:
                        if '_' in str(id_val):
                            num = int(str(id_val).split('_')[1])
                            train_id_nums.append(num)
                        else:
                            train_id_nums.append(0)
                    except:
                        train_id_nums.append(0)
                
                safe_mask = np.array(train_id_nums) <= temporal_threshold
                safe_data = train_processed[safe_mask]
                
                # 안전 구간에서의 통계 계산
                if len(safe_data) > 1000:
                    safe_mean = safe_data['after_interaction'].mean()
                    safe_std = safe_data['after_interaction'].std()
                    
                    # 안전한 변형 피처만 생성
                    train_processed['after_safe_normalized'] = (train_processed['after_interaction'] - safe_mean) / (safe_std + 1e-8)
                    test_processed['after_safe_normalized'] = (test_processed['after_interaction'] - safe_mean) / (safe_std + 1e-8)
                    
                    # 분위수 변환
                    safe_quantiles = safe_data['after_interaction'].quantile([0.25, 0.5, 0.75]).values
                    train_processed['after_quartile'] = pd.cut(
                        train_processed['after_interaction'], 
                        bins=[-np.inf] + safe_quantiles.tolist() + [np.inf], 
                        labels=[0, 1, 2, 3]
                    ).astype(float)
                    test_processed['after_quartile'] = pd.cut(
                        test_processed['after_interaction'], 
                        bins=[-np.inf] + safe_quantiles.tolist() + [np.inf], 
                        labels=[0, 1, 2, 3]
                    ).astype(float)
                    
                    print("after_interaction 안전 변형 피처 생성")
                else:
                    print("after_interaction 피처 제거 (안전 데이터 부족)")
                    if 'after_interaction' in train_processed.columns:
                        train_processed = train_processed.drop('after_interaction', axis=1)
                    if 'after_interaction' in test_processed.columns:
                        test_processed = test_processed.drop('after_interaction', axis=1)
            else:
                print("after_interaction 피처 제거 (시간 정보 없음)")
                if 'after_interaction' in train_processed.columns:
                    train_processed = train_processed.drop('after_interaction', axis=1)
                if 'after_interaction' in test_processed.columns:
                    test_processed = test_processed.drop('after_interaction', axis=1)
            
            # 원본 제거
            if 'after_interaction' in train_processed.columns:
                train_processed = train_processed.drop('after_interaction', axis=1)
            if 'after_interaction' in test_processed.columns:
                test_processed = test_processed.drop('after_interaction', axis=1)
                
        return train_processed, test_processed
    
    def create_temporal_features(self, df):
        """시간 피처 생성"""
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
            
            if len(id_numbers) > 0 and max(id_numbers) > min(id_numbers):
                id_min = min(id_numbers)
                id_max = max(id_numbers)
                df_new['temporal_position'] = [(x - id_min) / (id_max - id_min) for x in id_numbers]
                df_new['temporal_quartile'] = pd.qcut(id_numbers, q=4, labels=False, duplicates='drop')
                df_new['temporal_rank'] = pd.Series(id_numbers).rank(pct=True)
                
                # 시간 윈도우 기반 피처
                df_new['temporal_window'] = [int(x / 1000) for x in id_numbers]
                df_new['temporal_cycle'] = [x % 100 for x in id_numbers]
            else:
                df_new['temporal_position'] = [0.5] * len(id_numbers)
                df_new['temporal_quartile'] = [1] * len(id_numbers)
                df_new['temporal_rank'] = [0.5] * len(id_numbers)
                df_new['temporal_window'] = [0] * len(id_numbers)
                df_new['temporal_cycle'] = [0] * len(id_numbers)
        
        return df_new
    
    def create_business_features(self, df):
        """비즈니스 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 고객 가치 지표
        if all(col in df.columns for col in ['tenure', 'frequent']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            
            df_new['customer_value'] = np.log1p(tenure_safe) * np.sqrt(frequent_safe)
            df_new['customer_value'] = np.clip(df_new['customer_value'], 0, 100)
            
            # 활동 집중도
            df_new['activity_concentration'] = frequent_safe / (np.log1p(tenure_safe) + 1)
            df_new['activity_concentration'] = np.clip(df_new['activity_concentration'], 0, 50)
        
        # 결제 패턴
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['payment_stability'] = contract_safe / payment_safe
            df_new['payment_stability'] = np.clip(df_new['payment_stability'], 0, 50)
            
            # 약정 강도
            df_new['commitment_level'] = contract_safe / 90
            df_new['commitment_level'] = np.clip(df_new['commitment_level'], 0, 20)
        
        # 사용 패턴
        if all(col in df.columns for col in ['frequent', 'tenure']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 0.1, 200)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            
            df_new['usage_intensity'] = frequent_safe / (tenure_safe / 30 + 1)
            df_new['usage_intensity'] = np.clip(df_new['usage_intensity'], 0, 50)
            
            # 사용 추세
            df_new['usage_trend'] = frequent_safe * np.log1p(tenure_safe)
            df_new['usage_trend'] = np.clip(df_new['usage_trend'], 0, 500)
        
        # 연령 기반 세그먼트
        if 'age' in df.columns:
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            df_new['age_group'] = pd.cut(age_safe, bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4])
            df_new['age_group'] = df_new['age_group'].fillna(2)
            
            # 생애주기 단계
            if 'tenure' in df.columns:
                tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
                df_new['lifecycle_stage'] = (age_safe / 10) * (np.log1p(tenure_safe) / 5)
                df_new['lifecycle_stage'] = np.clip(df_new['lifecycle_stage'], 0, 100)
        
        return df_new
    
    def create_statistical_features(self, df):
        """통계 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        key_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for col in key_features:
            if col in df.columns:
                values = np.clip(df_new[col].fillna(0), 0, 5000)
                
                # 로그 변환
                df_new[f'{col}_log'] = np.log1p(values)
                df_new[f'{col}_sqrt'] = np.sqrt(values)
                
                # 분위수 변환
                df_new[f'{col}_rank'] = values.rank(pct=True)
                
                # Z-score 표준화
                col_mean = values.mean()
                col_std = values.std()
                if col_std > 0:
                    df_new[f'{col}_zscore'] = (values - col_mean) / col_std
                else:
                    df_new[f'{col}_zscore'] = 0
                
                # 극값 지표
                q95 = values.quantile(0.95)
                q05 = values.quantile(0.05)
                df_new[f'{col}_is_extreme'] = ((values > q95) | (values < q05)).astype(int)
        
        return df_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 핵심 상호작용
        key_interactions = [
            ('age', 'tenure'),
            ('frequent', 'payment_interval'),
            ('tenure', 'contract_length'),
            ('age', 'frequent'),
            ('payment_interval', 'contract_length')
        ]
        
        for feat1, feat2 in key_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                val1 = np.clip(df_new[feat1].fillna(0), 0, 2000)
                val2 = np.clip(df_new[feat2].fillna(0), 0, 2000)
                
                val2_safe = np.where(val2 == 0, 1, val2)
                
                # 비율
                df_new[f'{feat1}_{feat2}_ratio'] = val1 / val2_safe
                df_new[f'{feat1}_{feat2}_ratio'] = np.clip(df_new[f'{feat1}_{feat2}_ratio'], 0, 100)
                
                # 곱
                df_new[f'{feat1}_{feat2}_product'] = np.sqrt(val1 * val2)
                df_new[f'{feat1}_{feat2}_product'] = np.clip(df_new[f'{feat1}_{feat2}_product'], 0, 1000)
                
                # 차이
                df_new[f'{feat1}_{feat2}_diff'] = abs(val1 - val2)
                df_new[f'{feat1}_{feat2}_diff'] = np.clip(df_new[f'{feat1}_{feat2}_diff'], 0, 2000)
        
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
                train_encoded = np.zeros(len(train_df))
                
                for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    # 베이지안 스무딩
                    target_mean = fold_train.groupby(col)['support_needs'].mean()
                    global_mean = fold_train['support_needs'].mean()
                    category_counts = fold_train.groupby(col).size()
                    
                    alpha = 30
                    smoothed_means = (target_mean * category_counts + global_mean * alpha) / (category_counts + alpha)
                    
                    encoded_vals = fold_val[col].map(smoothed_means).fillna(global_mean)
                    train_encoded[val_idx] = encoded_vals
                
                train_new[f'{col}_target_mean'] = train_encoded
                
                # 테스트 데이터 인코딩
                target_mean_all = train_df.groupby(col)['support_needs'].mean()
                global_mean_all = train_df['support_needs'].mean()
                category_counts_all = train_df.groupby(col).size()
                
                alpha_all = 30
                smoothed_means_all = (target_mean_all * category_counts_all + global_mean_all * alpha_all) / (category_counts_all + alpha_all)
                
                test_encoded = test_df[col].map(smoothed_means_all).fillna(global_mean_all)
                test_new[f'{col}_target_mean'] = test_encoded
                
                # 클래스별 확률 인코딩
                for cls in [0, 1, 2]:
                    cls_prob = fold_train.groupby(col)['support_needs'].apply(lambda x: (x == cls).mean())
                    global_cls_prob = (fold_train['support_needs'] == cls).mean()
                    
                    cls_smoothed = (cls_prob * category_counts_all + global_cls_prob * alpha_all) / (category_counts_all + alpha_all)
                    
                    train_new[f'{col}_target_cls{cls}'] = train_df[col].map(cls_smoothed).fillna(global_cls_prob)
                    test_new[f'{col}_target_cls{cls}'] = test_df[col].map(cls_smoothed).fillna(global_cls_prob)
        
        return train_new, test_new
    
    def create_clustering_features(self, train_df, test_df):
        """클러스터링 피처"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
        
        if len(available_cols) < 3:
            return train_df, test_df
        
        train_numeric = train_df[available_cols].fillna(0)
        test_numeric = test_df[available_cols].fillna(0)
        
        # 이상치 처리
        for col in available_cols:
            q01 = train_numeric[col].quantile(0.01)
            q99 = train_numeric[col].quantile(0.99)
            
            train_numeric[col] = np.clip(train_numeric[col], q01, q99)
            test_numeric[col] = np.clip(test_numeric[col], q01, q99)
        
        # 정규화
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_numeric)
        test_scaled = scaler.transform(test_numeric)
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        # K-Means 클러스터링
        for k in [3, 5, 8]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            train_clusters = kmeans.fit_predict(train_scaled)
            test_clusters = kmeans.predict(test_scaled)
            
            train_new[f'cluster_{k}'] = train_clusters
            test_new[f'cluster_{k}'] = test_clusters
            
            # 클러스터 거리
            train_distances = kmeans.transform(train_scaled)
            test_distances = kmeans.transform(test_scaled)
            
            train_new[f'cluster_{k}_min_dist'] = np.min(train_distances, axis=1)
            test_new[f'cluster_{k}_min_dist'] = np.min(test_distances, axis=1)
        
        return train_new, test_new
    
    def encode_categorical(self, train_df, test_df):
        """범주형 인코딩"""
        categorical_cols = ['gender', 'subscription_type']
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                combined_data = pd.concat([train_df[col], test_df[col]])
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    unique_vals = combined_data.fillna('Unknown').unique()
                    self.label_encoders[col].fit(unique_vals)
                
                train_new[col] = self.label_encoders[col].transform(train_new[col].fillna('Unknown'))
                test_new[col] = self.label_encoders[col].transform(test_new[col].fillna('Unknown'))
        
        return train_new, test_new
    
    def select_features(self, train_df, target_col='support_needs', max_features=75):
        """피처 선택"""
        if target_col not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', target_col]]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df[target_col], 0, 2)
        
        # 분산 필터링
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        # 상호정보량 기반 선택
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        mi_scores = mutual_info_classif(X_var_df, y, random_state=42)
        
        feature_scores = list(zip(variance_features, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [f[0] for f in feature_scores[:max_features]]
        
        self.selected_features = selected_features
        return selected_features
    
    def create_features(self, train_df, test_df, temporal_threshold=None):
        """피처 생성 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        print("피처 생성 시작:")
        
        # after_interaction 처리
        train_df, test_df = self.handle_after_interaction(train_df, test_df, temporal_threshold)
        print("✓ after_interaction 처리 완료")
        
        # 시간 피처
        train_df = self.create_temporal_features(train_df)
        test_df = self.create_temporal_features(test_df)
        
        # 비즈니스 피처
        train_df = self.create_business_features(train_df)
        test_df = self.create_business_features(test_df)
        
        # 통계 피처
        train_df = self.create_statistical_features(train_df)
        test_df = self.create_statistical_features(test_df)
        
        # 상호작용 피처
        train_df = self.create_interaction_features(train_df)
        test_df = self.create_interaction_features(test_df)
        
        # 타겟 인코딩
        train_df, test_df = self.create_target_encoding(train_df, test_df)
        
        # 클러스터링 피처
        train_df, test_df = self.create_clustering_features(train_df, test_df)
        
        # 범주형 인코딩
        train_df, test_df = self.encode_categorical(train_df, test_df)
        
        # 안전한 데이터 변환
        train_df = self.safe_data_conversion(train_df)
        test_df = self.safe_data_conversion(test_df)
        
        print("✓ 모든 피처 생성 완료")
        
        return train_df, test_df

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