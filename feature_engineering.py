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
    
    def create_business_features(self, df):
        """비즈니스 로직 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 고객 가치 점수
        if all(col in df.columns for col in ['tenure', 'frequent', 'contract_length']):
            tenure_val = df_new['tenure'].fillna(100)
            frequent_val = df_new['frequent'].fillna(10)
            contract_val = df_new['contract_length'].fillna(90)
            
            df_new['customer_value'] = (
                np.log1p(tenure_val) * np.sqrt(frequent_val) * 
                np.log1p(contract_val)
            ) / 100
            
            df_new['customer_value'] = df_new['customer_value'].replace([np.inf, -np.inf], 0)
        
        # 결제 안정성 점수
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_val = df_new['payment_interval'].fillna(30)
            contract_val = df_new['contract_length'].fillna(90)
            
            df_new['payment_stability'] = contract_val / (payment_val + 1)
            df_new['payment_stability'] = df_new['payment_stability'].replace([np.inf, -np.inf], 0)
        
        # 사용 패턴 점수
        if all(col in df.columns for col in ['frequent', 'tenure']):
            frequent_val = df_new['frequent'].fillna(10)
            tenure_val = df_new['tenure'].fillna(100)
            
            df_new['usage_intensity'] = frequent_val / (tenure_val / 30 + 1)
            df_new['usage_intensity'] = df_new['usage_intensity'].replace([np.inf, -np.inf], 0)
        
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
                
                # 기본 변환
                df_new[f'{col}_log'] = np.log1p(np.abs(values))
                df_new[f'{col}_sqrt'] = np.sqrt(np.abs(values))
                
                # Z-score
                if values.std() > 0:
                    df_new[f'{col}_zscore'] = (values - values.mean()) / values.std()
                else:
                    df_new[f'{col}_zscore'] = 0
                    
                # 안전성 검사
                for suffix in ['_log', '_sqrt', '_zscore']:
                    new_col = f'{col}{suffix}'
                    df_new[new_col] = df_new[new_col].replace([np.inf, -np.inf], 0)
                
            except Exception as e:
                print(f"수학적 변환 오류 {col}: {e}")
                continue
        
        return df_new
    
    def create_ratio_features(self, df):
        """비율 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        ratio_pairs = [
            ('age', 'tenure'),
            ('frequent', 'tenure'),
            ('payment_interval', 'contract_length')
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
        
        # 핵심 상호작용만 선별
        key_interactions = [
            ('age', 'tenure'),
            ('frequent', 'tenure'), 
            ('payment_interval', 'contract_length')
        ]
        
        for feat1, feat2 in key_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                try:
                    val1 = df_new[feat1].fillna(0)
                    val2 = df_new[feat2].fillna(0)
                    
                    df_new[f'{feat1}_{feat2}_mult'] = (val1 * val2).replace([np.inf, -np.inf], 0)
                    
                except Exception as e:
                    print(f"상호작용 피처 오류 {feat1}×{feat2}: {e}")
        
        return df_new
    
    def create_safe_target_encoding(self, train_df, test_df):
        """안전한 타겟 인코딩"""
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
                    
                    # 교차 검증 기반 인코딩
                    for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                        fold_train = train_df.iloc[train_idx]
                        fold_val = train_df.iloc[val_idx]
                        
                        target_mean = fold_train.groupby(col)['support_needs'].mean()
                        global_mean = fold_train['support_needs'].mean()
                        
                        # 스무딩 적용
                        alpha = 10
                        category_counts = fold_train.groupby(col).size()
                        smoothed_means = (target_mean * category_counts + global_mean * alpha) / (category_counts + alpha)
                        
                        encoded_vals = fold_val[col].map(smoothed_means).fillna(global_mean)
                        train_encoded[val_idx] = encoded_vals
                    
                    train_new[f'{col}_target_encoded'] = train_encoded
                    
                    # 테스트 데이터 인코딩
                    target_mean_all = train_df.groupby(col)['support_needs'].mean()
                    global_mean_all = train_df['support_needs'].mean()
                    category_counts_all = train_df.groupby(col).size()
                    
                    smoothed_means_all = (target_mean_all * category_counts_all + global_mean_all * 10) / (category_counts_all + 10)
                    
                    test_encoded = test_df[col].map(smoothed_means_all).fillna(global_mean_all)
                    test_new[f'{col}_target_encoded'] = test_encoded
                    
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
            
            # 이상치 처리
            for col in available_cols:
                q25 = train_numeric[col].quantile(0.25)
                q75 = train_numeric[col].quantile(0.75)
                iqr = q75 - q25
                
                if iqr > 0:
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    
                    train_numeric[col] = np.clip(train_numeric[col], lower, upper)
                    test_numeric[col] = np.clip(test_numeric[col], lower, upper)
            
            # 정규화
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_numeric)
            test_scaled = scaler.transform(test_numeric)
            
            if self.kmeans_model is None:
                self.kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10)
                train_clusters = self.kmeans_model.fit_predict(train_scaled)
            else:
                train_clusters = self.kmeans_model.predict(train_scaled)
                
            test_clusters = self.kmeans_model.predict(test_scaled)
            
            train_new = train_df.copy()
            test_new = test_df.copy()
            
            train_new['cluster'] = train_clusters
            test_new['cluster'] = test_clusters
            
            return train_new, test_new
            
        except Exception as e:
            print(f"클러스터링 피처 오류: {e}")
            return train_df, test_df
    
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
    
    def exclude_leakage_features(self, df):
        """누수 피처 제외"""
        df_clean = df.copy()
        
        # after_interaction 피처 제외
        if 'after_interaction' in df_clean.columns:
            df_clean = df_clean.drop('after_interaction', axis=1)
            print("after_interaction 피처 제외됨")
        
        return df_clean
    
    def create_polynomial_features_selective(self, train_df, test_df):
        """선별적 다항식 피처 생성"""
        # 핵심 피처만 선택
        core_features = ['age', 'tenure', 'frequent']
        available_features = [f for f in core_features if f in train_df.columns and f in test_df.columns]
        
        if len(available_features) < 2:
            return train_df, test_df
        
        try:
            train_core = train_df[available_features].fillna(0)
            test_core = test_df[available_features].fillna(0)
            
            # 2차 항만 생성 (interaction_only=True)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            
            train_poly = poly.fit_transform(train_core)
            test_poly = poly.transform(test_core)
            
            poly_feature_names = poly.get_feature_names_out(available_features)
            
            train_new = train_df.copy()
            test_new = test_df.copy()
            
            # 원본 피처가 아닌 상호작용 항만 추가
            for i, name in enumerate(poly_feature_names):
                if name not in available_features:
                    train_new[f'poly_{name}'] = train_poly[:, i]
                    test_new[f'poly_{name}'] = test_poly[:, i]
                    
                    train_new[f'poly_{name}'] = train_new[f'poly_{name}'].replace([np.inf, -np.inf], 0)
                    test_new[f'poly_{name}'] = test_new[f'poly_{name}'].replace([np.inf, -np.inf], 0)
            
            return train_new, test_new
            
        except Exception as e:
            print(f"다항식 피처 생성 오류: {e}")
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
            
            # 누수 피처 제외
            train_df = self.exclude_leakage_features(train_df)
            test_df = self.exclude_leakage_features(test_df)
            
            # 시간적 피처 생성
            train_df = self.create_temporal_features(train_df)
            test_df = self.create_temporal_features(test_df)
            
            # 비즈니스 로직 피처
            train_df = self.create_business_features(train_df)
            test_df = self.create_business_features(test_df)
            
            # 수학적 변환
            train_df = self.create_mathematical_features(train_df)
            test_df = self.create_mathematical_features(test_df)
            
            # 비율 피처
            train_df = self.create_ratio_features(train_df)
            test_df = self.create_ratio_features(test_df)
            
            # 선별적 상호작용 피처
            train_df = self.create_interaction_features(train_df)
            test_df = self.create_interaction_features(test_df)
            
            # 선별적 다항식 피처
            train_df, test_df = self.create_polynomial_features_selective(train_df, test_df)
            
            # 타겟 인코딩
            train_df, test_df = self.create_safe_target_encoding(train_df, test_df)
            
            # 클러스터링 피처
            train_df, test_df = self.create_clustering_features(train_df, test_df)
            
            # 범주형 인코딩
            train_df, test_df = self.encode_categorical(train_df, test_df)
            
            # 최종 안전성 검사
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