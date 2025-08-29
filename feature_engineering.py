# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """피처 엔지니어링 클래스"""
    
    def __init__(self):
        self.target_encoders = {}
        self.label_encoders = {}
        self.kmeans_model = None
        self.pca_model = None
        self.feature_selector = None
        
    def create_basic_features(self, df, is_train=True):
        """기본 피처 생성"""
        print("=== 기본 피처 생성 ===")
        
        df_new = df.copy()
        
        # 연령 그룹
        df_new['age_group'] = pd.cut(df_new['age'], bins=[0, 25, 35, 45, 55, 100], 
                                   labels=[0, 1, 2, 3, 4]).astype(int)
        
        # 계약 기간 그룹
        df_new['contract_group'] = pd.cut(df_new['contract_length'], 
                                        bins=[0, 30, 90, 180, 360, 1000],
                                        labels=[0, 1, 2, 3, 4]).astype(int)
        
    def create_basic_features(self, df, is_train=True):
        """기본 피처 생성"""
        print("=== 기본 피처 생성 ===")
        
        df_new = df.copy()
        
        # 연령 그룹
        def age_group_func(age):
            if age < 25: return 0
            elif age < 35: return 1
            elif age < 45: return 2
            elif age < 55: return 3
            else: return 4
        
        df_new['age_group'] = df_new['age'].apply(age_group_func)
        
        # 계약 기간 그룹
        def contract_group_func(contract):
            if contract <= 30: return 0
            elif contract <= 90: return 1
            elif contract <= 180: return 2
            elif contract <= 360: return 3
            else: return 4
        
        df_new['contract_group'] = df_new['contract_length'].apply(contract_group_func)
        
        # 비율 피처
        df_new['freq_per_tenure'] = df_new['frequent'] / (df_new['tenure'] + 1)
        df_new['interaction_per_freq'] = df_new['after_interaction'] / (df_new['frequent'] + 1)
        df_new['age_tenure_ratio'] = df_new['age'] / (df_new['tenure'] + 1)
        df_new['payment_contract_ratio'] = df_new['payment_interval'] / (df_new['contract_length'] + 1)
        
        # 통계적 피처
        df_new['age_zscore'] = (df_new['age'] - df_new['age'].mean()) / df_new['age'].std()
        df_new['tenure_zscore'] = (df_new['tenure'] - df_new['tenure'].mean()) / df_new['tenure'].std()
        df_new['frequent_zscore'] = (df_new['frequent'] - df_new['frequent'].mean()) / df_new['frequent'].std()
        
        return df_new
    
    def create_polynomial_features(self, df, degree=2):
        """다항식 피처 생성"""
        print("=== 다항식 피처 생성 ===")
        
        df_new = df.copy()
        
        # 핵심 수치형 피처 선택
        numeric_features = ['age', 'tenure', 'frequent', 'payment_interval', 'after_interaction']
        
        # 2차 교호작용 피처 수동 생성
        interaction_pairs = [
            ('age', 'tenure'),
            ('frequent', 'after_interaction'),
            ('age', 'frequent'),
            ('tenure', 'payment_interval'),
            ('contract_length', 'payment_interval')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df_new.columns and feat2 in df_new.columns:
                df_new[f'{feat1}_{feat2}_mult'] = df_new[feat1] * df_new[feat2]
                df_new[f'{feat1}_{feat2}_add'] = df_new[feat1] + df_new[feat2]
                df_new[f'{feat1}_{feat2}_diff'] = abs(df_new[feat1] - df_new[feat2])
        
        # 제곱 피처
        for feat in numeric_features:
            if feat in df_new.columns:
                df_new[f'{feat}_squared'] = df_new[feat] ** 2
                df_new[f'{feat}_sqrt'] = np.sqrt(abs(df_new[feat]))
        
        print(f"생성된 다항식 피처 수: {len([col for col in df_new.columns if col not in df.columns])}")
        
        return df_new
    
    def create_target_encoding(self, train_df, test_df, target_col='support_needs'):
        """타겟 인코딩 생성"""
        print("=== 타겟 인코딩 생성 ===")
        
        categorical_cols = ['gender', 'subscription_type', 'age_group', 'contract_group']
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for col in categorical_cols:
            if col in train_df.columns:
                # 평균 타겟 인코딩
                target_mean = train_df.groupby(col)[target_col].mean()
                self.target_encoders[f'{col}_target_mean'] = target_mean
                
                train_new[f'{col}_target_mean'] = train_new[col].map(target_mean).fillna(train_df[target_col].mean())
                test_new[f'{col}_target_mean'] = test_new[col].map(target_mean).fillna(train_df[target_col].mean())
                
                # 표준편차 인코딩
                target_std = train_df.groupby(col)[target_col].std()
                self.target_encoders[f'{col}_target_std'] = target_std
                
                train_new[f'{col}_target_std'] = train_new[col].map(target_std).fillna(train_df[target_col].std())
                test_new[f'{col}_target_std'] = test_new[col].map(target_std).fillna(train_df[target_col].std())
        
        print(f"생성된 타겟 인코딩 피처 수: {len(categorical_cols) * 2}")
        
        return train_new, test_new
    
    def create_clustering_features(self, train_df, test_df, n_clusters=5):
        """클러스터링 기반 피처 생성"""
        print("=== 클러스터링 피처 생성 ===")
        
        # 수치형 피처만 사용
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        train_numeric = train_df[numeric_cols].fillna(0)
        test_numeric = test_df[numeric_cols].fillna(0)
        
        # K-means 클러스터링
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        train_clusters = self.kmeans_model.fit_predict(train_numeric)
        test_clusters = self.kmeans_model.predict(test_numeric)
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        train_new['cluster'] = train_clusters
        test_new['cluster'] = test_clusters
        
        # 클러스터 중심까지의 거리
        train_distances = self.kmeans_model.transform(train_numeric)
        test_distances = self.kmeans_model.transform(test_numeric)
        
        for i in range(n_clusters):
            train_new[f'dist_cluster_{i}'] = train_distances[:, i]
            test_new[f'dist_cluster_{i}'] = test_distances[:, i]
        
        # 가장 가까운 클러스터까지의 거리
        train_new['min_cluster_dist'] = train_distances.min(axis=1)
        test_new['min_cluster_dist'] = test_distances.min(axis=1)
        
        print(f"생성된 클러스터링 피처 수: {n_clusters + 2}")
        
        return train_new, test_new
    
    def create_pca_features(self, train_df, test_df, n_components=5):
        """PCA 기반 피처 생성"""
        print("=== PCA 피처 생성 ===")
        
        # 수치형 피처만 사용
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        train_numeric = train_df[numeric_cols].fillna(0)
        test_numeric = test_df[numeric_cols].fillna(0)
        
        # PCA 학습
        self.pca_model = PCA(n_components=n_components, random_state=42)
        
        train_pca = self.pca_model.fit_transform(train_numeric)
        test_pca = self.pca_model.transform(test_numeric)
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for i in range(n_components):
            train_new[f'pca_{i}'] = train_pca[:, i]
            test_new[f'pca_{i}'] = test_pca[:, i]
        
        # 설명된 분산 비율
        explained_ratio = self.pca_model.explained_variance_ratio_
        print(f"PCA 설명 분산 비율: {explained_ratio}")
        print(f"총 설명 분산: {explained_ratio.sum():.3f}")
        
        return train_new, test_new
    
    def create_statistical_features(self, df):
        """통계적 피처 생성"""
        print("=== 통계적 피처 생성 ===")
        
        df_new = df.copy()
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 
                       'contract_length', 'after_interaction']
        
        # 행별 통계
        df_new['numeric_mean'] = df_new[numeric_cols].mean(axis=1)
        df_new['numeric_std'] = df_new[numeric_cols].std(axis=1)
        df_new['numeric_median'] = df_new[numeric_cols].median(axis=1)
        df_new['numeric_max'] = df_new[numeric_cols].max(axis=1)
        df_new['numeric_min'] = df_new[numeric_cols].min(axis=1)
        df_new['numeric_range'] = df_new['numeric_max'] - df_new['numeric_min']
        
        # 분위수 기반 피처
        for col in numeric_cols:
            q25 = df[col].quantile(0.25)
            q75 = df[col].quantile(0.75)
            
            df_new[f'{col}_q25_flag'] = (df_new[col] <= q25).astype(int)
            df_new[f'{col}_q75_flag'] = (df_new[col] >= q75).astype(int)
        
        print(f"생성된 통계적 피처 수: {6 + len(numeric_cols) * 2}")
        
        return df_new
    
    def encode_categorical_features(self, train_df, test_df):
        """범주형 피처 인코딩"""
        print("=== 범주형 피처 인코딩 ===")
        
        categorical_cols = ['gender', 'subscription_type']
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        for col in categorical_cols:
            if col in train_df.columns:
                # 전체 데이터로 라벨 인코더 학습
                combined_data = pd.concat([train_df[col], test_df[col]], axis=0)
                
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(combined_data)
                
                train_new[col] = self.label_encoders[col].transform(train_new[col])
                test_new[col] = self.label_encoders[col].transform(test_new[col])
        
        return train_new, test_new
    
    def process_all_features(self, train_df, test_df):
        """모든 피처 엔지니어링 실행"""
        print("피처 엔지니어링 시작")
        print("="*40)
        
        print(f"원본 피처 수: {train_df.shape[1]}")
        
        # 1. 기본 피처 생성
        train_df = self.create_basic_features(train_df, is_train=True)
        test_df = self.create_basic_features(test_df, is_train=False)
        
        # 2. 다항식 피처 생성
        train_df = self.create_polynomial_features(train_df)
        test_df = self.create_polynomial_features(test_df)
        
        # 3. 타겟 인코딩 (train에 support_needs가 있을 때만)
        if 'support_needs' in train_df.columns:
            train_df, test_df = self.create_target_encoding(train_df, test_df)
        
        # 4. 클러스터링 피처
        train_df, test_df = self.create_clustering_features(train_df, test_df)
        
        # 5. PCA 피처
        train_df, test_df = self.create_pca_features(train_df, test_df)
        
        # 6. 통계적 피처
        train_df = self.create_statistical_features(train_df)
        test_df = self.create_statistical_features(test_df)
        
        # 7. 범주형 인코딩
        train_df, test_df = self.encode_categorical_features(train_df, test_df)
        
        print(f"최종 피처 수: {train_df.shape[1]}")
        print(f"생성된 피처 수: {train_df.shape[1] - len(['ID', 'age', 'gender', 'tenure', 'frequent', 'payment_interval', 'subscription_type', 'contract_length', 'after_interaction'])}")
        
        return train_df, test_df

def main():
    """메인 실행 함수"""
    # 데이터 로드
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # 피처 엔지니어링 실행
    engineer = FeatureEngineer()
    train_processed, test_processed = engineer.process_all_features(train_df, test_df)
    
    print("\n피처 엔지니어링 완료!")
    return engineer, train_processed, test_processed

if __name__ == "__main__":
    main()