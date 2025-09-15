# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.kmeans_model = None
        self.feature_stats = {}
        self.selected_features = None
        self.scaler = None
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
        
    def analyze_after_interaction_safety(self, train_df, temporal_threshold=None):
        """after_interaction 피처 안전성 분석"""
        if 'after_interaction' not in train_df.columns or 'support_needs' not in train_df.columns:
            return False, "피처 없음"
        
        # 시간적 안전성 검증 (더 엄격한 기준)
        if temporal_threshold is not None and 'ID' in train_df.columns:
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
            
            # 더 엄격한 기준 (85% 안전 비율)
            safe_threshold = np.percentile(train_id_nums, 85)
            safe_mask = np.array(train_id_nums) <= safe_threshold
            safe_data = train_df[safe_mask]
            
            if len(safe_data) < 2000:
                return False, "안전 데이터 부족"
            
            # 안전 구간에서의 상관관계 분석
            correlation = safe_data[['after_interaction', 'support_needs']].corr().iloc[0, 1]
            
            # 매우 보수적인 기준 (0.08)
            if abs(correlation) > 0.08:
                return False, f"상관관계 위험: {correlation:.3f}"
            
            # 클래스별 분리도 검증
            class_means = []
            for cls in [0, 1, 2]:
                class_data = safe_data[safe_data['support_needs'] == cls]['after_interaction'].dropna()
                if len(class_data) > 0:
                    class_means.append(class_data.mean())
            
            if len(class_means) >= 2:
                separation = max(class_means) - min(class_means)
                if separation > 1.8:  # 매우 보수적 기준
                    return False, f"클래스 분리도 위험: {separation:.3f}"
            
            return True, f"안전: 상관관계 {correlation:.3f}, 분리도 {separation:.3f}"
        
        # 전체 데이터에서의 기본 검증
        correlation = train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
        if abs(correlation) > 0.06:  # 매우 엄격한 기준
            return False, f"전체 상관관계 위험: {correlation:.3f}"
        
        return True, f"기본 안전: {correlation:.3f}"
    
    def handle_after_interaction(self, train_df, test_df, temporal_threshold=None):
        """after_interaction 피처 처리 (완전 제거)"""
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        if 'after_interaction' in train_df.columns:
            # 안전성 분석
            is_safe, safety_msg = self.analyze_after_interaction_safety(train_df, temporal_threshold)
            
            # 누수 위험 최소화를 위해 완전 제거
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
                
                # 시간적 그룹핑 (더 보수적)
                df_new['temporal_quartile'] = pd.qcut(id_numbers, q=4, labels=False, duplicates='drop')
                df_new['temporal_decile'] = pd.qcut(id_numbers, q=8, labels=False, duplicates='drop')
            else:
                df_new['temporal_position'] = [0.5] * len(id_numbers)
                df_new['temporal_quartile'] = [1] * len(id_numbers)
                df_new['temporal_decile'] = [4] * len(id_numbers)
        
        return df_new
    
    def create_business_features(self, df):
        """비즈니스 피처 생성"""
        df_new = self.safe_data_conversion(df)
        
        # 고객 가치 지표 (보수적)
        if all(col in df.columns for col in ['tenure', 'frequent', 'contract_length']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 1500)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 150)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 800)
            
            df_new['customer_value'] = (
                np.log1p(tenure_safe) * 
                np.sqrt(frequent_safe) * 
                np.cbrt(contract_safe)
            )
            df_new['customer_value'] = np.clip(df_new['customer_value'], 0, 150)
        
        # 결제 패턴 (보수적)
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 300)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 800)
            
            df_new['payment_stability'] = contract_safe / payment_safe
            df_new['payment_stability'] = np.clip(df_new['payment_stability'], 0, 40)
        
        # 사용 강도 (보수적)
        if all(col in df.columns for col in ['frequent', 'tenure']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 0.1, 150)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 1500)
            
            df_new['usage_intensity'] = frequent_safe / (tenure_safe / 30 + 1)
            df_new['usage_intensity'] = np.clip(df_new['usage_intensity'], 0, 80)
        
        # 고객 세그먼트 (간소화)
        if 'age' in df.columns:
            age_safe = np.clip(df_new['age'].fillna(35), 18, 80)
            df_new['age_group'] = pd.cut(age_safe, 
                                       bins=[0, 30, 45, 65, 100], 
                                       labels=[0, 1, 2, 3])
            df_new['age_group'] = df_new['age_group'].fillna(1)
        
        return df_new
    
    def create_statistical_features(self, df):
        """통계 피처 생성 (축소)"""
        df_new = self.safe_data_conversion(df)
        
        # 핵심 피처만 변환
        key_features = ['age', 'tenure', 'frequent']
        
        for col in key_features:
            if col in df.columns:
                values = np.clip(df_new[col].fillna(0), 0, 5000)
                
                # 로그 변환만 (과적합 방지)
                df_new[f'{col}_log'] = np.log1p(values)
                
                # 분위수 변환
                df_new[f'{col}_rank'] = values.rank(pct=True)
        
        return df_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성 (축소)"""
        df_new = self.safe_data_conversion(df)
        
        # 핵심 상호작용만
        key_interactions = [
            ('age', 'tenure'),
            ('frequent', 'payment_interval')
        ]
        
        for feat1, feat2 in key_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                val1 = np.clip(df_new[feat1].fillna(0), 0, 1500)
                val2 = np.clip(df_new[feat2].fillna(0), 0, 1500)
                
                # 비율만 (과적합 방지)
                val2_safe = np.where(val2 == 0, 1, val2)
                df_new[f'{feat1}_{feat2}_ratio'] = val1 / val2_safe
                df_new[f'{feat1}_{feat2}_ratio'] = np.clip(df_new[f'{feat1}_{feat2}_ratio'], 0, 80)
        
        return df_new
    
    def create_target_encoding(self, train_df, test_df):
        """타겟 인코딩 (더 보수적)"""
        if 'support_needs' not in train_df.columns:
            return train_df, test_df
        
        categorical_cols = ['gender', 'subscription_type']
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                train_encoded = np.zeros(len(train_df))
                
                # 교차 검증 기반 인코딩
                for train_idx, val_idx in skf.split(train_df, train_df['support_needs']):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    # 베이지안 스무딩 (더 보수적)
                    target_mean = fold_train.groupby(col)['support_needs'].mean()
                    global_mean = fold_train['support_needs'].mean()
                    category_counts = fold_train.groupby(col).size()
                    
                    # 스무딩 파라미터 (더 보수적)
                    alpha = min(50, max(10, len(fold_train) // 80))
                    smoothed_means = (target_mean * category_counts + global_mean * alpha) / (category_counts + alpha)
                    
                    encoded_vals = fold_val[col].map(smoothed_means).fillna(global_mean)
                    train_encoded[val_idx] = encoded_vals
                
                train_new[f'{col}_target_encoded'] = train_encoded
                
                # 테스트 데이터 인코딩
                target_mean_all = train_df.groupby(col)['support_needs'].mean()
                global_mean_all = train_df['support_needs'].mean()
                category_counts_all = train_df.groupby(col).size()
                
                alpha_all = min(50, max(10, len(train_df) // 80))
                smoothed_means_all = (target_mean_all * category_counts_all + global_mean_all * alpha_all) / (category_counts_all + alpha_all)
                
                test_encoded = test_df[col].map(smoothed_means_all).fillna(global_mean_all)
                test_new[f'{col}_target_encoded'] = test_encoded
        
        return train_new, test_new
    
    def create_clustering_features(self, train_df, test_df):
        """클러스터링 피처 (간소화)"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval']
        available_cols = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
        
        if len(available_cols) < 3:
            return train_df, test_df
        
        train_numeric = train_df[available_cols].fillna(0)
        test_numeric = test_df[available_cols].fillna(0)
        
        # 이상치 처리 (더 보수적)
        for col in available_cols:
            q01 = train_numeric[col].quantile(0.01)
            q99 = train_numeric[col].quantile(0.99)
            
            train_numeric[col] = np.clip(train_numeric[col], q01, q99)
            test_numeric[col] = np.clip(test_numeric[col], q01, q99)
        
        # 정규화
        scaler = RobustScaler(quantile_range=(15.0, 85.0))
        train_scaled = scaler.fit_transform(train_numeric)
        test_scaled = scaler.transform(test_numeric)
        
        # 클러스터 수 축소 (과적합 방지)
        best_k = 3
        
        # 클러스터링
        if self.kmeans_model is None:
            self.kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=8)
            train_clusters = self.kmeans_model.fit_predict(train_scaled)
        else:
            train_clusters = self.kmeans_model.predict(train_scaled)
            
        test_clusters = self.kmeans_model.predict(test_scaled)
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        train_new['cluster'] = train_clusters
        test_new['cluster'] = test_clusters
        
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
    
    def select_features(self, train_df, target_col='support_needs', max_features=45):
        """피처 선택 (더 보수적)"""
        if target_col not in train_df.columns:
            feature_cols = [col for col in train_df.columns if col not in ['ID']]
            return feature_cols[:min(max_features, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', target_col]]
        
        if len(feature_cols) <= max_features:
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = np.clip(train_df[target_col], 0, 2)
        
        # 분산 필터링 (더 보수적)
        variance_selector = VarianceThreshold(threshold=0.0005)
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
        """피처 생성 파이프라인 (보수적)"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        # after_interaction 완전 제거
        train_df, test_df = self.handle_after_interaction(train_df, test_df, temporal_threshold)
        
        # 시간 피처
        train_df = self.create_temporal_features(train_df)
        test_df = self.create_temporal_features(test_df)
        
        # 비즈니스 피처 (보수적)
        train_df = self.create_business_features(train_df)
        test_df = self.create_business_features(test_df)
        
        # 통계 피처 (축소)
        train_df = self.create_statistical_features(train_df)
        test_df = self.create_statistical_features(test_df)
        
        # 상호작용 피처 (축소)
        train_df = self.create_interaction_features(train_df)
        test_df = self.create_interaction_features(test_df)
        
        # 타겟 인코딩 (보수적)
        train_df, test_df = self.create_target_encoding(train_df, test_df)
        
        # 클러스터링 피처 (간소화)
        train_df, test_df = self.create_clustering_features(train_df, test_df)
        
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