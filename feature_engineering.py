# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.kmeans_model = None
        self.dbscan_model = None
        self.feature_stats = {}
        self.selected_features = None
        self.scaler = None
        self.power_transformer = None
        self.pca_model = None
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
        """after_interaction 피처 처리 - 개선된 누수 방지"""
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        if 'after_interaction' in train_df.columns:
            # 데이터 누수 위험성 재평가
            leakage_detected = False
            
            if 'support_needs' in train_df.columns:
                # 전체 데이터에서의 상관관계 확인
                overall_correlation = train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                
                # 상호정보량 계산
                after_clean = train_df['after_interaction'].fillna(0)
                target_clean = train_df['support_needs']
                mi_score = mutual_info_classif(after_clean.values.reshape(-1, 1), target_clean, random_state=42)[0]
                
                # 클래스별 분리도 확인
                class_means = []
                for cls in [0, 1, 2]:
                    class_data = train_df[train_df['support_needs'] == cls]['after_interaction'].dropna()
                    if len(class_data) > 0:
                        class_means.append(class_data.mean())
                
                if len(class_means) >= 2:
                    class_separation = max(class_means) - min(class_means)
                    mean_class_mean = np.mean(class_means)
                    separation_ratio = class_separation / (mean_class_mean + 1e-8) if mean_class_mean != 0 else 0
                else:
                    separation_ratio = 0
                
                # 엄격한 누수 기준 적용
                high_correlation = abs(overall_correlation) > 0.12
                high_mutual_info = mi_score > 0.25
                high_separation = separation_ratio > 0.5
                
                # 시간적 안전성 확인
                temporal_leakage = False
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
                    
                    # 시간적으로 안전한 구간에서의 상관관계 확인
                    safe_mask = np.array(train_id_nums) <= temporal_threshold
                    
                    if np.sum(safe_mask) > 500:  # 충분한 안전 데이터가 있는 경우만
                        safe_data = train_df[safe_mask]
                        safe_correlation = safe_data[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                        temporal_leakage = abs(safe_correlation) > 0.15
                    else:
                        temporal_leakage = True  # 안전 데이터가 부족하면 위험으로 간주
                
                # 누수 판정
                leakage_detected = high_correlation or high_mutual_info or high_separation or temporal_leakage
                
                print(f"after_interaction 누수 검사:")
                print(f"  상관관계: {overall_correlation:.4f} (위험: {high_correlation})")
                print(f"  상호정보량: {mi_score:.4f} (위험: {high_mutual_info})")
                print(f"  클래스 분리도: {separation_ratio:.4f} (위험: {high_separation})")
                print(f"  시간적 누수: {temporal_leakage}")
                print(f"  최종 누수 판정: {leakage_detected}")
            
            if leakage_detected:
                # 누수 위험이 높으면 해당 피처 완전 제거
                print("after_interaction 피처 누수 위험으로 인해 제거")
                if 'after_interaction' in train_processed.columns:
                    train_processed = train_processed.drop('after_interaction', axis=1)
                if 'after_interaction' in test_processed.columns:
                    test_processed = test_processed.drop('after_interaction', axis=1)
                    
                # 대체 피처 생성 (시간적으로 안전한 방식)
                self.create_safe_interaction_features(train_processed, test_processed, temporal_threshold)
                    
            else:
                # 누수 위험이 낮으면 안전한 변형만 사용
                print("after_interaction 피처 안전한 변형 적용")
                
                # 시간적으로 안전한 파생 피처만 생성
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
                    
                    if np.sum(safe_mask) > 1000:
                        # 안전 구간 데이터로 통계 계산
                        safe_data = train_processed[safe_mask]['after_interaction']
                        safe_mean = safe_data.mean()
                        safe_std = safe_data.std()
                        
                        # 안전한 정규화 및 변형
                        train_processed['after_interaction_normalized'] = (train_processed['after_interaction'] - safe_mean) / (safe_std + 1e-8)
                        test_processed['after_interaction_normalized'] = (test_processed['after_interaction'] - safe_mean) / (safe_std + 1e-8)
                        
                        # 원본 피처는 제거
                        train_processed = train_processed.drop('after_interaction', axis=1)
                        test_processed = test_processed.drop('after_interaction', axis=1)
                    else:
                        # 안전 데이터가 부족하면 제거
                        train_processed = train_processed.drop('after_interaction', axis=1)
                        if 'after_interaction' in test_processed.columns:
                            test_processed = test_processed.drop('after_interaction', axis=1)
                else:
                    # temporal_threshold가 없으면 제거
                    train_processed = train_processed.drop('after_interaction', axis=1)
                    if 'after_interaction' in test_processed.columns:
                        test_processed = test_processed.drop('after_interaction', axis=1)
        
        return train_processed, test_processed
    
    def create_safe_interaction_features(self, train_df, test_df, temporal_threshold=None):
        """안전한 상호작용 피처 생성"""
        # 기존 피처들 간의 안전한 상호작용만 생성
        safe_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for i, feat1 in enumerate(safe_features):
            for feat2 in safe_features[i+1:]:
                if feat1 in train_df.columns and feat2 in test_df.columns:
                    # 기본 수학적 연산 (누수 위험 없음)
                    train_df[f'safe_{feat1}_{feat2}_ratio'] = train_df[feat1] / (train_df[feat2] + 1e-8)
                    test_df[f'safe_{feat1}_{feat2}_ratio'] = test_df[feat1] / (test_df[feat2] + 1e-8)
                    
                    train_df[f'safe_{feat1}_{feat2}_sum'] = train_df[feat1] + train_df[feat2]
                    test_df[f'safe_{feat1}_{feat2}_sum'] = test_df[feat1] + test_df[feat2]
    
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
                
                # 시간 기반 추가 피처
                df_new['temporal_sin'] = np.sin(2 * np.pi * df_new['temporal_position'])
                df_new['temporal_cos'] = np.cos(2 * np.pi * df_new['temporal_position'])
                
                # 시간적 밀도
                temporal_counts = pd.Series(id_numbers).value_counts()
                df_new['temporal_density'] = [temporal_counts.get(x, 1) for x in id_numbers]
                
            else:
                df_new['temporal_position'] = [0.5] * len(id_numbers)
                df_new['temporal_quartile'] = [1] * len(id_numbers)
                df_new['temporal_rank'] = [0.5] * len(id_numbers)
                df_new['temporal_sin'] = [0.0] * len(id_numbers)
                df_new['temporal_cos'] = [1.0] * len(id_numbers)
                df_new['temporal_density'] = [1] * len(id_numbers)
        
        return df_new
    
    def create_business_features(self, df):
        """비즈니스 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 고객 생애 가치
        if all(col in df.columns for col in ['tenure', 'frequent', 'contract_length']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['customer_value'] = (
                np.log1p(tenure_safe) * 
                np.sqrt(frequent_safe) * 
                np.cbrt(contract_safe)
            )
            df_new['customer_value'] = np.clip(df_new['customer_value'], 0, 200)
            
            # 비선형 변환
            df_new['customer_value_squared'] = df_new['customer_value'] ** 2
            df_new['customer_value_log'] = np.log1p(df_new['customer_value'])
            
            df_new['customer_value_tier'] = pd.qcut(df_new['customer_value'], q=5, labels=False, duplicates='drop')
        
        # 결제 안정성
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['payment_stability'] = contract_safe / payment_safe
            df_new['payment_stability'] = np.clip(df_new['payment_stability'], 0, 50)
            
            # 비선형 변환
            df_new['payment_stability_sqrt'] = np.sqrt(df_new['payment_stability'])
            df_new['payment_stability_log'] = np.log1p(df_new['payment_stability'])
            
            df_new['payment_stability_tier'] = pd.qcut(df_new['payment_stability'], q=3, labels=False, duplicates='drop')
        
        # 사용 패턴
        if all(col in df.columns for col in ['frequent', 'tenure']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 0.1, 200)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            
            df_new['usage_intensity'] = frequent_safe / (tenure_safe / 30 + 1)
            df_new['usage_intensity'] = np.clip(df_new['usage_intensity'], 0, 100)
            
            # 비선형 변환
            df_new['usage_intensity_squared'] = df_new['usage_intensity'] ** 2
            df_new['usage_intensity_log'] = np.log1p(df_new['usage_intensity'])
            
            df_new['usage_pattern'] = pd.cut(df_new['usage_intensity'], 
                                          bins=[0, 2, 8, 25, 100], 
                                          labels=[0, 1, 2, 3])
            df_new['usage_pattern'] = df_new['usage_pattern'].fillna(1)
        
        # 고객 세그먼트
        if 'age' in df.columns:
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            df_new['age_group'] = pd.cut(age_safe, 
                                       bins=[0, 25, 35, 50, 65, 100], 
                                       labels=[0, 1, 2, 3, 4])
            df_new['age_group'] = df_new['age_group'].fillna(2)
            
            df_new['age_mature'] = (age_safe >= 40).astype(int)
            df_new['age_senior'] = (age_safe >= 60).astype(int)
            df_new['age_young'] = (age_safe <= 30).astype(int)
            
            # 연령 비선형 변환
            df_new['age_squared'] = age_safe ** 2
            df_new['age_log'] = np.log1p(age_safe)
        
        return df_new
    
    def create_statistical_features(self, df):
        """통계 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        key_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        for col in key_features:
            if col in df.columns:
                values = np.clip(df_new[col].fillna(0), 0, 10000)
                
                # 기본 변환
                df_new[f'{col}_log'] = np.log1p(values)
                df_new[f'{col}_sqrt'] = np.sqrt(values)
                df_new[f'{col}_cbrt'] = np.cbrt(values)
                df_new[f'{col}_squared'] = values ** 2
                
                # 분위수 변환
                df_new[f'{col}_rank'] = values.rank(pct=True)
                
                # 표준화
                col_mean = values.mean()
                col_std = values.std()
                if col_std > 0:
                    df_new[f'{col}_std'] = (values - col_mean) / col_std
                    
                    # Z-score 기반 이상치 지시자
                    df_new[f'{col}_outlier'] = (np.abs(df_new[f'{col}_std']) > 2.5).astype(int)
                else:
                    df_new[f'{col}_std'] = 0
                    df_new[f'{col}_outlier'] = 0
                
                # 분위수 구간
                df_new[f'{col}_decile'] = pd.qcut(values, q=10, labels=False, duplicates='drop')
                df_new[f'{col}_quintile'] = pd.qcut(values, q=5, labels=False, duplicates='drop')
        
        return df_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        key_interactions = [
            ('age', 'tenure'),
            ('frequent', 'payment_interval'),
            ('tenure', 'contract_length'),
            ('age', 'frequent'),
            ('age', 'contract_length'),
            ('frequent', 'contract_length')
        ]
        
        for feat1, feat2 in key_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                val1 = np.clip(df_new[feat1].fillna(0), 0, 2000)
                val2 = np.clip(df_new[feat2].fillna(0), 0, 2000)
                
                val2_safe = np.where(val2 == 0, 1, val2)
                
                # 기본 상호작용
                df_new[f'{feat1}_{feat2}_ratio'] = val1 / val2_safe
                df_new[f'{feat1}_{feat2}_ratio'] = np.clip(df_new[f'{feat1}_{feat2}_ratio'], 0, 100)
                
                df_new[f'{feat1}_{feat2}_product'] = val1 * val2
                df_new[f'{feat1}_{feat2}_product'] = np.clip(df_new[f'{feat1}_{feat2}_product'], 0, 1000000)
                
                df_new[f'{feat1}_{feat2}_diff'] = np.abs(val1 - val2)
                df_new[f'{feat1}_{feat2}_sum'] = val1 + val2
                
                # 비선형 상호작용
                df_new[f'{feat1}_{feat2}_harmonic'] = 2 * val1 * val2 / (val1 + val2 + 1e-8)
                df_new[f'{feat1}_{feat2}_geometric'] = np.sqrt(val1 * val2)
                
                # 조건부 상호작용
                df_new[f'{feat1}_gt_{feat2}'] = (val1 > val2).astype(int)
                df_new[f'{feat1}_eq_{feat2}'] = (np.abs(val1 - val2) < 1e-6).astype(int)
        
        # 3차 상호작용
        if all(col in df.columns for col in ['age', 'tenure', 'frequent']):
            age_val = np.clip(df_new['age'].fillna(35), 18, 100)
            tenure_val = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            frequent_val = np.clip(df_new['frequent'].fillna(10), 0.1, 200)
            
            df_new['age_tenure_frequent_product'] = age_val * tenure_val * frequent_val
            df_new['age_tenure_frequent_mean'] = (age_val + tenure_val + frequent_val) / 3
            df_new['age_tenure_frequent_harmonic'] = 3 / (1/age_val + 1/tenure_val + 1/frequent_val + 1e-8)
        
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
                
                # 교차 검증 기반 인코딩
                for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    # 베이지안 스무딩
                    target_mean = fold_train.groupby(col)['support_needs'].mean()
                    global_mean = fold_train['support_needs'].mean()
                    category_counts = fold_train.groupby(col).size()
                    
                    alpha = min(28, max(5, len(fold_train) // 140))
                    smoothed_means = (target_mean * category_counts + global_mean * alpha) / (category_counts + alpha)
                    
                    encoded_vals = fold_val[col].map(smoothed_means).fillna(global_mean)
                    train_encoded[val_idx] = encoded_vals
                
                train_new[f'{col}_target_encoded'] = train_encoded
                
                # 테스트 데이터 인코딩
                target_mean_all = train_df.groupby(col)['support_needs'].mean()
                global_mean_all = train_df['support_needs'].mean()
                category_counts_all = train_df.groupby(col).size()
                
                alpha_all = min(28, max(5, len(train_df) // 140))
                smoothed_means_all = (target_mean_all * category_counts_all + global_mean_all * alpha_all) / (category_counts_all + alpha_all)
                
                test_encoded = test_df[col].map(smoothed_means_all).fillna(global_mean_all)
                test_new[f'{col}_target_encoded'] = test_encoded
                
                # 추가 타겟 기반 피처
                for target_class in [0, 1, 2]:
                    class_mean = fold_train[fold_train['support_needs'] == target_class].groupby(col).size() / fold_train.groupby(col).size()
                    train_new[f'{col}_class_{target_class}_ratio'] = train_df[col].map(class_mean).fillna(0)
                    test_new[f'{col}_class_{target_class}_ratio'] = test_df[col].map(class_mean).fillna(0)
        
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
        for k in [4, 6, 8]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            train_clusters = kmeans.fit_predict(train_scaled)
            test_clusters = kmeans.predict(test_scaled)
            
            train_new[f'kmeans_{k}'] = train_clusters
            test_new[f'kmeans_{k}'] = test_clusters
            
            # 클러스터 중심까지의 거리
            train_distances = np.min(kmeans.transform(train_scaled), axis=1)
            test_distances = np.min(kmeans.transform(test_scaled), axis=1)
            
            train_new[f'kmeans_{k}_distance'] = train_distances
            test_new[f'kmeans_{k}_distance'] = test_distances
            
            # 클러스터별 밀도
            train_new[f'kmeans_{k}_density'] = 1 / (1 + train_distances)
            test_new[f'kmeans_{k}_density'] = 1 / (1 + test_distances)
        
        # DBSCAN 클러스터링
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            train_dbscan = dbscan.fit_predict(train_scaled)
            
            # 모든 포인트에 대해 가장 가까운 클러스터 할당
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=1).fit(train_scaled[train_dbscan != -1])
            
            test_distances, test_indices = nbrs.kneighbors(test_scaled)
            test_dbscan = train_dbscan[train_dbscan != -1][test_indices.flatten()]
            
            train_new['dbscan_cluster'] = train_dbscan
            test_new['dbscan_cluster'] = test_dbscan
            
            train_new['dbscan_is_outlier'] = (train_dbscan == -1).astype(int)
            test_new['dbscan_is_outlier'] = 0  # 테스트는 이상치로 분류하지 않음
            
        except:
            train_new['dbscan_cluster'] = 0
            test_new['dbscan_cluster'] = 0
            train_new['dbscan_is_outlier'] = 0
            test_new['dbscan_is_outlier'] = 0
        
        return train_new, test_new
    
    def create_polynomial_features(self, df):
        """다항식 피처"""
        df_new = self.safe_data_conversion(df)
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval']
        
        for col in numeric_cols:
            if col in df.columns:
                values = np.clip(df_new[col].fillna(0), 0, 1000)
                
                # 2차 다항식
                df_new[f'{col}_poly2'] = values ** 2
                
                # 3차 다항식 (일부만)
                if col in ['age', 'tenure']:
                    df_new[f'{col}_poly3'] = values ** 3
                
                # 역수
                df_new[f'{col}_inv'] = 1 / (values + 1e-8)
                
                # 제곱근의 역수
                df_new[f'{col}_inv_sqrt'] = 1 / (np.sqrt(values) + 1e-8)
        
        return df_new
    
    def create_time_series_features(self, df):
        """시계열 기반 피처"""
        df_new = self.safe_data_conversion(df)
        
        if 'temporal_id' in df_new.columns:
            # 시간 기반 통계
            df_new['temporal_id_lag'] = df_new['temporal_id'].shift(1).fillna(0)
            df_new['temporal_id_diff'] = df_new['temporal_id'] - df_new['temporal_id_lag']
            
            # 시간 기반 롤링 통계
            if 'age' in df_new.columns:
                df_new['age_rolling_mean'] = df_new['age'].rolling(5, min_periods=1).mean()
                df_new['age_rolling_std'] = df_new['age'].rolling(5, min_periods=1).std().fillna(0)
            
            if 'frequent' in df_new.columns:
                df_new['frequent_rolling_mean'] = df_new['frequent'].rolling(3, min_periods=1).mean()
                df_new['frequent_trend'] = df_new['frequent'] - df_new['frequent_rolling_mean']
        
        return df_new
    
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
        variance_selector = VarianceThreshold(threshold=0.0001)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        # 상호정보량 기반 선택
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        mi_scores = mutual_info_classif(X_var_df, y, random_state=42)
        
        # 상위 피처 선택
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
        
        # 다항식 피처
        train_df = self.create_polynomial_features(train_df)
        test_df = self.create_polynomial_features(test_df)
        
        # 시계열 피처
        train_df = self.create_time_series_features(train_df)
        test_df = self.create_time_series_features(test_df)
        
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