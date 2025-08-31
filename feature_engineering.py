# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
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
                df_clean[col] = df_clean[col].fillna(0)
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
            else:
                df_new['temporal_position'] = [0.5] * len(id_numbers)
        
        return df_new
    
    def create_business_features(self, df):
        """비즈니스 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 고객 활동성
        if all(col in df.columns for col in ['tenure', 'frequent']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            
            df_new['activity_score'] = frequent_safe / (tenure_safe / 30 + 1)
            df_new['activity_score'] = np.clip(df_new['activity_score'], 0, 50)
        
        # 결제 안정성
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['payment_stability'] = contract_safe / payment_safe
            df_new['payment_stability'] = np.clip(df_new['payment_stability'], 0, 30)
        
        # 고객 세그먼트
        if 'age' in df.columns:
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            df_new['age_group'] = pd.cut(age_safe, 
                                       bins=[0, 30, 45, 60, 100], 
                                       labels=[0, 1, 2, 3])
            df_new['age_group'] = df_new['age_group'].fillna(1)
        
        return df_new
    
    def create_ratio_features(self, df):
        """비율 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 핵심 비율들만 생성
        ratios = [
            ('age', 'tenure'),
            ('frequent', 'payment_interval'),
            ('tenure', 'contract_length')
        ]
        
        for feat1, feat2 in ratios:
            if feat1 in df.columns and feat2 in df.columns:
                val1 = np.clip(df_new[feat1].fillna(0), 0, 1000)
                val2 = np.clip(df_new[feat2].fillna(1), 1, 1000)
                
                df_new[f'{feat1}_{feat2}_ratio'] = val1 / val2
                df_new[f'{feat1}_{feat2}_ratio'] = np.clip(df_new[f'{feat1}_{feat2}_ratio'], 0, 50)
        
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
                    
                    target_mean = fold_train.groupby(col)['support_needs'].mean()
                    global_mean = fold_train['support_needs'].mean()
                    
                    encoded_vals = fold_val[col].map(target_mean).fillna(global_mean)
                    train_encoded[val_idx] = encoded_vals
                
                train_new[f'{col}_target_encoded'] = train_encoded
                
                # 테스트 데이터 인코딩
                target_mean_all = train_df.groupby(col)['support_needs'].mean()
                global_mean_all = train_df['support_needs'].mean()
                
                test_encoded = test_df[col].map(target_mean_all).fillna(global_mean_all)
                test_new[f'{col}_target_encoded'] = test_encoded
        
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
    
    def select_features_conservative(self, train_df, target_col='support_needs', max_features=22):
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
        variance_selector = VarianceThreshold(threshold=0.001)
        X_variance = variance_selector.fit_transform(X)
        variance_features = [feature_cols[i] for i, selected in enumerate(variance_selector.get_support()) if selected]
        
        if len(variance_features) <= max_features:
            self.selected_features = variance_features
            return variance_features
        
        # 상호정보량 기반 선택
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
        
        # 시간 피처
        train_df = self.create_temporal_features(train_df)
        test_df = self.create_temporal_features(test_df)
        
        # 비즈니스 피처
        train_df = self.create_business_features(train_df)
        test_df = self.create_business_features(test_df)
        
        # 비율 피처
        train_df = self.create_ratio_features(train_df)
        test_df = self.create_ratio_features(test_df)
        
        # 타겟 인코딩
        train_df, test_df = self.create_target_encoding(train_df, test_df)
        
        # 범주형 인코딩
        train_df, test_df = self.encode_categorical(train_df, test_df)
        
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