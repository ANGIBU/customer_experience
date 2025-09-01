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
        
    def safe_data_conversion(self, df):
        """안전한 데이터 변환"""
        df_clean = df.copy()
        
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
        
        # after_interaction 완전 제거
        if 'after_interaction' in train_clean.columns:
            train_clean = train_clean.drop('after_interaction', axis=1)
        if 'after_interaction' in test_clean.columns:
            test_clean = test_clean.drop('after_interaction', axis=1)
        
        return train_clean, test_clean
    
    def create_basic_features(self, df):
        """기본 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 나이 그룹
        if 'age' in df.columns:
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            df_new['age_group'] = pd.cut(age_safe, 
                                       bins=[0, 30, 45, 60, 100], 
                                       labels=[0, 1, 2, 3])
            df_new['age_group'] = df_new['age_group'].fillna(1).astype(int)
        
        # 활동 점수
        if all(col in df.columns for col in ['tenure', 'frequent']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            
            df_new['activity_score'] = frequent_safe / np.sqrt(tenure_safe + 1)
            df_new['activity_score'] = np.clip(df_new['activity_score'], 0, 10)
        
        # 결제 안정성
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['payment_stability'] = contract_safe / payment_safe
            df_new['payment_stability'] = np.clip(df_new['payment_stability'], 0, 20)
        
        return df_new
    
    def create_ratio_features(self, df):
        """비율 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 안전한 비율만 생성
        if all(col in df.columns for col in ['age', 'tenure']):
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            df_new['age_tenure_ratio'] = age_safe / np.sqrt(tenure_safe + 1)
            df_new['age_tenure_ratio'] = np.clip(df_new['age_tenure_ratio'], 0, 20)
        
        if all(col in df.columns for col in ['frequent', 'payment_interval']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            df_new['usage_payment_ratio'] = frequent_safe / payment_safe
            df_new['usage_payment_ratio'] = np.clip(df_new['usage_payment_ratio'], 0, 5)
        
        return df_new
    
    def create_target_encoding(self, train_df, test_df):
        """타겟 인코딩"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        categorical_cols = ['gender', 'subscription_type']
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                train_encoded = np.full(len(train_df), np.nan)
                
                global_mean = train_df['support_needs'].mean()
                
                # 교차검증으로 타겟 인코딩
                for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    category_stats = fold_train.groupby(col)['support_needs'].agg(['mean', 'count'])
                    
                    smoothing_factor = 20.0  # 더 강한 스무딩
                    
                    for category in category_stats.index:
                        mean_val = category_stats.loc[category, 'mean']
                        count_val = category_stats.loc[category, 'count']
                        
                        # 베이지안 평균 적용
                        smoothed_mean = (mean_val * count_val + global_mean * smoothing_factor) / (count_val + smoothing_factor)
                        
                        mask = fold_val[col] == category
                        train_encoded[fold_val.index[mask]] = smoothed_mean
                
                # 누락값 처리
                train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
                
                # 더 보수적인 인코딩 적용
                train_encoded = 0.7 * train_encoded + 0.3 * global_mean
                train_new[f'{col}_encoded'] = train_encoded
                
                # 테스트 데이터 인코딩
                category_stats_all = train_df.groupby(col)['support_needs'].agg(['mean', 'count'])
                
                test_encoded = np.full(len(test_df), global_mean)
                for category in category_stats_all.index:
                    if category in test_df[col].values:
                        mean_val = category_stats_all.loc[category, 'mean']
                        count_val = category_stats_all.loc[category, 'count']
                        
                        smoothed_mean = (mean_val * count_val + global_mean * smoothing_factor) / (count_val + smoothing_factor)
                        smoothed_mean = 0.7 * smoothed_mean + 0.3 * global_mean
                        
                        mask = test_df[col] == category
                        test_encoded[mask] = smoothed_mean
                
                test_new[f'{col}_encoded'] = test_encoded
                
                # 원본 범주형 피처 유지
                train_new[col] = train_df[col]
                test_new[col] = test_df[col]
        
        return train_new, test_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 단순한 곱셈 피처만 생성
        if all(col in df.columns for col in ['age', 'tenure']):
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            df_new['age_tenure_product'] = np.log1p(age_safe * tenure_safe / 1000)
        
        if all(col in df.columns for col in ['frequent', 'contract_length']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            df_new['usage_contract_product'] = np.log1p(frequent_safe * contract_safe / 100)
        
        return df_new
    
    def select_features_conservative(self, train_df, target_col='support_needs', max_features=15):
        """보수적 피처 선택"""
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
        
        # 상호 정보량 기반 선택
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        mi_selector.fit(X_var_df, y)
        
        selected_features = [variance_features[i] for i, selected in enumerate(mi_selector.get_support()) if selected]
        
        self.selected_features = selected_features
        return selected_features
    
    def create_features(self, train_df, test_df):
        """피처 생성 파이프라인"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        # 누수 피처 제거
        train_df, test_df = self.remove_leakage_features(train_df, test_df)
        
        # 기본 피처 생성
        train_df = self.create_basic_features(train_df)
        test_df = self.create_basic_features(test_df)
        
        # 비율 피처 생성
        train_df = self.create_ratio_features(train_df)
        test_df = self.create_ratio_features(test_df)
        
        # 상호작용 피처 생성
        train_df = self.create_interaction_features(train_df)
        test_df = self.create_interaction_features(test_df)
        
        # 타겟 인코딩
        train_df, test_df = self.create_target_encoding(train_df, test_df)
        
        # 안전한 데이터 변환
        train_df = self.safe_data_conversion(train_df)
        test_df = self.safe_data_conversion(test_df)
        
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