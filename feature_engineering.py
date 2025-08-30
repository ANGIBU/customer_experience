# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.kmeans_model = None
        self.pca_model = None
        
    def create_basic_features(self, df):
        """기본 피처 생성"""
        df_new = df.copy()
        
        # 연령 구간
        df_new['age_group'] = pd.cut(df_new['age'], 
                                   bins=[0, 25, 35, 45, 55, 100], 
                                   labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 계약 기간 구간
        df_new['contract_group'] = pd.cut(df_new['contract_length'],
                                        bins=[0, 30, 90, 180, 365, 1000],
                                        labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 사용 빈도 구간
        df_new['frequent_group'] = pd.cut(df_new['frequent'],
                                        bins=[0, 5, 15, 30, 50, 1000],
                                        labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 비율 피처
        df_new['freq_per_tenure'] = df_new['frequent'] / (df_new['tenure'] + 1)
        df_new['age_tenure_ratio'] = df_new['age'] / (df_new['tenure'] + 1)
        df_new['payment_contract_ratio'] = df_new['payment_interval'] / (df_new['contract_length'] + 1)
        
        return df_new
    
    def create_customer_behavior_features(self, df):
        """고객 행동 패턴 피처 생성"""
        df_new = df.copy()
        
        # 고객 가치 점수
        if all(col in df.columns for col in ['frequent', 'tenure', 'contract_length']):
            df_new['customer_value_score'] = (
                df_new['frequent'] * 0.4 + 
                df_new['tenure'] * 0.3 + 
                df_new['contract_length'] * 0.3
            ) / 3
        
        # 서비스 사용 패턴
        if 'frequent' in df.columns and 'tenure' in df.columns:
            df_new['usage_intensity'] = df_new['frequent'] / (df_new['tenure'] + 1)
            df_new['usage_stability'] = np.where(df_new['tenure'] > 0, 
                                               df_new['frequent'] / df_new['tenure'], 0)
        
        # 계약 충성도
        if 'contract_length' in df.columns and 'payment_interval' in df.columns:
            df_new['contract_loyalty'] = df_new['contract_length'] / (df_new['payment_interval'] + 1)
        
        # 연령대별 행동 패턴
        if 'age' in df.columns and 'frequent' in df.columns:
            age_normalized = (df_new['age'] - df_new['age'].min()) / (df_new['age'].max() - df_new['age'].min())
            df_new['age_usage_pattern'] = age_normalized * df_new['frequent']
        
        # 상호작용 품질 지표
        if all(col in df.columns for col in ['after_interaction', 'frequent', 'tenure']):
            df_new['interaction_quality'] = df_new['after_interaction'] / (df_new['frequent'] + df_new['tenure'] + 1)
            df_new['interaction_per_month'] = df_new['after_interaction'] / ((df_new['tenure'] / 30) + 1)
        
        return df_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성"""
        df_new = df.copy()
        
        # 핵심 수치 피처
        base_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_features = [feat for feat in base_features if feat in df.columns]
        
        # 2차 상호작용
        interaction_pairs = [
            ('age', 'tenure'),
            ('frequent', 'tenure'),
            ('age', 'frequent'),
            ('payment_interval', 'contract_length'),
            ('frequent', 'payment_interval')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df_new[f'{feat1}_{feat2}_mult'] = df_new[feat1] * df_new[feat2]
                df_new[f'{feat1}_{feat2}_add'] = df_new[feat1] + df_new[feat2]
                df_new[f'{feat1}_{feat2}_diff'] = abs(df_new[feat1] - df_new[feat2])
        
        # 제곱 피처
        for feat in available_features:
            df_new[f'{feat}_squared'] = df_new[feat] ** 2
            df_new[f'{feat}_sqrt'] = np.sqrt(abs(df_new[feat]))
        
        return df_new
    
    def create_target_encoding(self, train_df, test_df):
        """타겟 인코딩 생성"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        categorical_cols = ['gender', 'subscription_type', 'age_group', 
                           'contract_group', 'frequent_group']
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        # 5-Fold 교차 검증으로 타겟 인코딩
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col in train_df.columns:
                # 훈련 데이터 타겟 인코딩
                train_encoded = np.zeros(len(train_df))
                
                for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    # 평균 계산
                    target_mean = fold_train.groupby(col)['support_needs'].mean()
                    global_mean = fold_train['support_needs'].mean()
                    
                    # 검증 폴드에 적용
                    encoded_vals = fold_val[col].map(target_mean).fillna(global_mean)
                    train_encoded[val_idx] = encoded_vals
                
                train_new[f'{col}_target_mean'] = train_encoded
                
                # 테스트 데이터 인코딩
                target_mean_all = train_df.groupby(col)['support_needs'].mean()
                global_mean_all = train_df['support_needs'].mean()
                
                test_encoded = test_df[col].map(target_mean_all).fillna(global_mean_all)
                test_new[f'{col}_target_mean'] = test_encoded
        
        return train_new, test_new
    
    def create_clustering_features(self, train_df, test_df):
        """클러스터링 피처 생성"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        train_numeric = train_df[numeric_cols].fillna(0)
        test_numeric = test_df[numeric_cols].fillna(0)
        
        # K-means 클러스터링
        self.kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10)
        train_clusters = self.kmeans_model.fit_predict(train_numeric)
        test_clusters = self.kmeans_model.predict(test_numeric)
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        train_new['cluster'] = train_clusters
        test_new['cluster'] = test_clusters
        
        # 클러스터 중심까지 거리
        train_distances = self.kmeans_model.transform(train_numeric)
        test_distances = self.kmeans_model.transform(test_numeric)
        
        train_new['min_cluster_dist'] = train_distances.min(axis=1)
        test_new['min_cluster_dist'] = test_distances.min(axis=1)
        
        return train_new, test_new
    
    def create_pca_features(self, train_df, test_df):
        """PCA 피처 생성"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        train_numeric = train_df[numeric_cols].fillna(0)
        test_numeric = test_df[numeric_cols].fillna(0)
        
        # PCA 변환
        self.pca_model = PCA(n_components=3, random_state=42)
        train_pca = self.pca_model.fit_transform(train_numeric)
        test_pca = self.pca_model.transform(test_numeric)
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for i in range(3):
            train_new[f'pca_{i}'] = train_pca[:, i]
            test_new[f'pca_{i}'] = test_pca[:, i]
        
        return train_new, test_new
    
    def create_statistical_features(self, df):
        """통계 피처 생성"""
        df_new = df.copy()
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 행별 통계
        df_new['numeric_mean'] = df_new[numeric_cols].mean(axis=1)
        df_new['numeric_std'] = df_new[numeric_cols].std(axis=1)
        df_new['numeric_median'] = df_new[numeric_cols].median(axis=1)
        df_new['numeric_range'] = df_new[numeric_cols].max(axis=1) - df_new[numeric_cols].min(axis=1)
        
        # 분위수 기반 플래그
        for col in numeric_cols:
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            
            df_new[f'{col}_low'] = (df_new[col] <= q25).astype(int)
            df_new[f'{col}_high'] = (df_new[col] >= q75).astype(int)
        
        return df_new
    
    def encode_categorical(self, train_df, test_df):
        """범주형 변수 인코딩"""
        categorical_cols = ['gender', 'subscription_type']
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for col in categorical_cols:
            # 전체 데이터로 인코더 학습
            combined_data = pd.concat([train_df[col], test_df[col]])
            
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(combined_data)
            
            train_new[col] = self.label_encoders[col].transform(train_new[col])
            test_new[col] = self.label_encoders[col].transform(test_new[col])
        
        return train_new, test_new
    
    def remove_leakage_features(self, train_df, test_df):
        """누수 위험 피처 변환"""
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        if 'after_interaction' in train_df.columns:
            # 고객별 과거 상호작용 패턴으로 변환
            train_new['interaction_normalized'] = train_new['after_interaction'] / (train_new['frequent'] + 1)
            test_new['interaction_normalized'] = test_new['after_interaction'] / (test_new['frequent'] + 1)
            
            # 상호작용 강도 구간화
            train_new['interaction_level'] = pd.cut(train_new['after_interaction'],
                                                   bins=[0, 10, 25, 50, 1000],
                                                   labels=[0, 1, 2, 3]).astype(int)
            test_new['interaction_level'] = pd.cut(test_new['after_interaction'],
                                                  bins=[0, 10, 25, 50, 1000],
                                                  labels=[0, 1, 2, 3]).astype(int)
            
            print("after_interaction 피처 변환 완료")
        
        return train_new, test_new
    
    def create_features(self, train_df, test_df):
        """전체 피처 생성 파이프라인"""
        print("피처 생성 시작")
        print("=" * 30)
        
        original_features = train_df.shape[1]
        
        # 1. 누수 위험 피처 변환
        train_df, test_df = self.remove_leakage_features(train_df, test_df)
        
        # 2. 기본 피처 생성
        train_df = self.create_basic_features(train_df)
        test_df = self.create_basic_features(test_df)
        
        # 3. 고객 행동 패턴 피처 생성
        train_df = self.create_customer_behavior_features(train_df)
        test_df = self.create_customer_behavior_features(test_df)
        
        # 4. 상호작용 피처 생성
        train_df = self.create_interaction_features(train_df)
        test_df = self.create_interaction_features(test_df)
        
        # 5. 타겟 인코딩
        train_df, test_df = self.create_target_encoding(train_df, test_df)
        
        # 6. 클러스터링 피처
        train_df, test_df = self.create_clustering_features(train_df, test_df)
        
        # 7. PCA 피처
        train_df, test_df = self.create_pca_features(train_df, test_df)
        
        # 8. 통계 피처
        train_df = self.create_statistical_features(train_df)
        test_df = self.create_statistical_features(test_df)
        
        # 9. 범주형 인코딩
        train_df, test_df = self.encode_categorical(train_df, test_df)
        
        final_features = train_df.shape[1]
        created_features = final_features - original_features
        
        print(f"피처 생성 완료: {original_features} → {final_features} (+{created_features})")
        
        return train_df, test_df

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    engineer = FeatureEngineer()
    train_processed, test_processed = engineer.create_features(train_df, test_df)
    
    return engineer, train_processed, test_processed

if __name__ == "__main__":
    main()