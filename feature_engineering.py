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
        
    def handle_after_interaction(self, train_df, test_df, temporal_threshold=None):
        """after_interaction 피처 처리 - 개선"""
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        if 'after_interaction' in train_df.columns:
            if temporal_threshold is not None:
                # 시간적 안전성 검증
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
                
                if 'support_needs' in train_df.columns:
                    # 안전 구간에서의 상관관계 분석
                    safe_data = train_df[safe_mask]
                    if len(safe_data) > 1500:  # 1000 -> 1500
                        correlation = safe_data[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                        
                        # 더 엄격한 기준 적용 (0.15)
                        if abs(correlation) < 0.15:  # 0.2 -> 0.15
                            # 안전한 변환 적용
                            train_processed['after_interaction_safe'] = train_processed['after_interaction'] * 0.8
                            train_processed['after_interaction_log'] = np.log1p(train_processed['after_interaction'].fillna(0))
                            train_processed['after_interaction_sqrt'] = np.sqrt(train_processed['after_interaction'].fillna(0))
                            
                            # 이동 평균 및 표준편차
                            train_processed['after_interaction_ma3'] = train_processed.groupby('ID')['after_interaction'].transform(
                                lambda x: x.rolling(3, min_periods=1).mean()
                            )
                            train_processed['after_interaction_std'] = train_processed.groupby('ID')['after_interaction'].transform(
                                lambda x: x.rolling(3, min_periods=1).std()
                            ).fillna(0)
                            
                            if 'after_interaction' in test_df.columns:
                                test_processed['after_interaction_safe'] = test_processed['after_interaction'] * 0.8
                                test_processed['after_interaction_log'] = np.log1p(test_processed['after_interaction'].fillna(0))
                                test_processed['after_interaction_sqrt'] = np.sqrt(test_processed['after_interaction'].fillna(0))
                                test_processed['after_interaction_ma3'] = test_processed.groupby('ID')['after_interaction'].transform(
                                    lambda x: x.rolling(3, min_periods=1).mean()
                                )
                                test_processed['after_interaction_std'] = test_processed.groupby('ID')['after_interaction'].transform(
                                    lambda x: x.rolling(3, min_periods=1).std()
                                ).fillna(0)
                            
                            # 원본 제거
                            train_processed = train_processed.drop('after_interaction', axis=1)
                            if 'after_interaction' in test_processed.columns:
                                test_processed = test_processed.drop('after_interaction', axis=1)
                        else:
                            # 누수 위험으로 제거
                            train_processed = train_processed.drop('after_interaction', axis=1)
                            if 'after_interaction' in test_processed.columns:
                                test_processed = test_processed.drop('after_interaction', axis=1)
                    else:
                        # 데이터 부족으로 제거
                        train_processed = train_processed.drop('after_interaction', axis=1)
                        if 'after_interaction' in test_processed.columns:
                            test_processed = test_processed.drop('after_interaction', axis=1)
            else:
                # 기본 제거
                train_processed = train_processed.drop('after_interaction', axis=1)
                if 'after_interaction' in test_processed.columns:
                    test_processed = test_processed.drop('after_interaction', axis=1)
        
        return train_processed, test_processed
    
    def create_temporal_features(self, df):
        """시간 피처 생성 - 개선"""
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
                
                # 정규화된 시간 위치
                df_new['temporal_position'] = [(x - id_min) / (id_max - id_min) for x in id_numbers]
                
                # 시간적 그룹핑 (더 세분화)
                df_new['temporal_quartile'] = pd.qcut(id_numbers, q=4, labels=False, duplicates='drop')
                df_new['temporal_decile'] = pd.qcut(id_numbers, q=10, labels=False, duplicates='drop')  # 추가
                
                # 시간적 변화율
                df_new['temporal_velocity'] = df_new['temporal_position'].diff().fillna(0)
                df_new['temporal_acceleration'] = df_new['temporal_velocity'].diff().fillna(0)
            else:
                df_new['temporal_position'] = [0.5] * len(id_numbers)
                df_new['temporal_quartile'] = [1] * len(id_numbers)
                df_new['temporal_decile'] = [5] * len(id_numbers)
                df_new['temporal_velocity'] = [0] * len(id_numbers)
                df_new['temporal_acceleration'] = [0] * len(id_numbers)
        
        return df_new
    
    def create_business_features(self, df):
        """비즈니스 피처 생성 - 개선"""
        df_new = self.safe_data_conversion(df)
        
        # 고객 생애 가치 (개선된 공식)
        if all(col in df.columns for col in ['tenure', 'frequent', 'contract_length']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 1, 200)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            # 더 정교한 고객 가치 계산
            df_new['customer_value'] = (
                np.log1p(tenure_safe) * 0.4 +
                np.sqrt(frequent_safe) * 0.35 +
                np.cbrt(contract_safe) * 0.25
            ) * 10  # 스케일 조정
            df_new['customer_value'] = np.clip(df_new['customer_value'], 0, 100)
            
            # 고객 가치 등급
            df_new['customer_value_tier'] = pd.qcut(df_new['customer_value'], q=5, labels=False, duplicates='drop')
        
        # 결제 안정성 (개선)
        if all(col in df.columns for col in ['payment_interval', 'contract_length']):
            payment_safe = np.clip(df_new['payment_interval'].fillna(30), 1, 365)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['payment_stability'] = contract_safe / payment_safe
            df_new['payment_stability'] = np.clip(df_new['payment_stability'], 0, 50)
            
            # 결제 규칙성
            df_new['payment_regularity'] = 1 / (1 + np.std([payment_safe, contract_safe], axis=0))
        
        # 사용 패턴 (개선)
        if all(col in df.columns for col in ['frequent', 'tenure']):
            frequent_safe = np.clip(df_new['frequent'].fillna(10), 0.1, 200)
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            
            df_new['usage_intensity'] = frequent_safe / (tenure_safe / 30 + 1)
            df_new['usage_intensity'] = np.clip(df_new['usage_intensity'], 0, 100)
            
            # 사용 일관성
            df_new['usage_consistency'] = np.log1p(frequent_safe) / np.log1p(tenure_safe)
        
        # 고객 세그먼트 (개선)
        if 'age' in df.columns:
            age_safe = np.clip(df_new['age'].fillna(35), 18, 100)
            
            # 연령 그룹 (더 세분화)
            df_new['age_group'] = pd.cut(age_safe, 
                                       bins=[0, 25, 35, 45, 55, 65, 100], 
                                       labels=[0, 1, 2, 3, 4, 5])
            df_new['age_group'] = df_new['age_group'].fillna(2)
            
            # 연령 제곱 피처 (비선형 관계 포착)
            df_new['age_squared'] = age_safe ** 2 / 100
            df_new['age_log'] = np.log1p(age_safe)
        
        # 계약 효율성
        if all(col in df.columns for col in ['tenure', 'contract_length']):
            tenure_safe = np.clip(df_new['tenure'].fillna(100), 1, 2000)
            contract_safe = np.clip(df_new['contract_length'].fillna(90), 1, 1000)
            
            df_new['contract_efficiency'] = tenure_safe / (contract_safe + 1)
            df_new['contract_efficiency'] = np.clip(df_new['contract_efficiency'], 0, 50)
        
        return df_new
    
    def create_statistical_features(self, df):
        """통계 피처 생성 - 개선"""
        df_new = self.safe_data_conversion(df)
        
        # 핵심 피처 및 변환
        key_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']  # 추가 피처
        important_transforms = ['log', 'sqrt', 'square', 'reciprocal']  # 추가 변환
        
        for col in key_features:
            if col in df.columns:
                values = np.clip(df_new[col].fillna(0), 0, 10000)
                
                # 로그 변환
                if 'log' in important_transforms:
                    df_new[f'{col}_log'] = np.log1p(values)
                
                # 제곱근 변환
                if 'sqrt' in important_transforms:
                    df_new[f'{col}_sqrt'] = np.sqrt(values)
                
                # 제곱 변환
                if 'square' in important_transforms and col in ['age', 'tenure']:  # 선택적 적용
                    df_new[f'{col}_square'] = (values / 100) ** 2  # 스케일 조정
                
                # 역수 변환
                if 'reciprocal' in important_transforms and col in ['payment_interval', 'contract_length']:
                    df_new[f'{col}_reciprocal'] = 1 / (values + 1)
                
                # 분위수 변환
                df_new[f'{col}_rank'] = values.rank(pct=True)
                
                # 표준화된 값
                mean_val = values.mean()
                std_val = values.std()
                if std_val > 0:
                    df_new[f'{col}_zscore'] = (values - mean_val) / std_val
                else:
                    df_new[f'{col}_zscore'] = 0
        
        return df_new
    
    def create_interaction_features(self, df):
        """상호작용 피처 생성 - 개선"""
        df_new = self.safe_data_conversion(df)
        
        # 확장된 상호작용 리스트
        key_interactions = [
            ('age', 'tenure'),
            ('frequent', 'payment_interval'),
            ('tenure', 'contract_length'),
            ('age', 'frequent'),  # 추가
            ('tenure', 'payment_interval'),  # 추가
            ('contract_length', 'payment_interval')  # 추가
        ]
        
        for feat1, feat2 in key_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                val1 = np.clip(df_new[feat1].fillna(0), 0, 2000)
                val2 = np.clip(df_new[feat2].fillna(0), 0, 2000)
                
                # 비율
                val2_safe = np.where(val2 == 0, 1, val2)
                df_new[f'{feat1}_{feat2}_ratio'] = val1 / val2_safe
                df_new[f'{feat1}_{feat2}_ratio'] = np.clip(df_new[f'{feat1}_{feat2}_ratio'], 0, 100)
                
                # 곱셈 상호작용
                df_new[f'{feat1}_{feat2}_product'] = (val1 * val2) / 1000  # 스케일 조정
                df_new[f'{feat1}_{feat2}_product'] = np.clip(df_new[f'{feat1}_{feat2}_product'], 0, 1000)
                
                # 합계 상호작용
                df_new[f'{feat1}_{feat2}_sum'] = val1 + val2
                
                # 차이 상호작용
                df_new[f'{feat1}_{feat2}_diff'] = np.abs(val1 - val2)
        
        return df_new
    
    def create_target_encoding(self, train_df, test_df):
        """타겟 인코딩 - 개선"""
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
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['support_needs'])):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    # 베이지안 스무딩 (개선된 파라미터)
                    target_mean = fold_train.groupby(col)['support_needs'].mean()
                    global_mean = fold_train['support_needs'].mean()
                    category_counts = fold_train.groupby(col).size()
                    
                    # 적응적 스무딩 파라미터
                    alpha = min(30, max(5, len(fold_train) // 100))  # 조정된 범위
                    smoothed_means = (target_mean * category_counts + global_mean * alpha) / (category_counts + alpha)
                    
                    encoded_vals = fold_val[col].map(smoothed_means).fillna(global_mean)
                    train_encoded[val_idx] = encoded_vals
                
                train_new[f'{col}_target_encoded'] = train_encoded
                
                # 추가: 빈도 인코딩
                freq_encoding = train_df[col].value_counts(normalize=True)
                train_new[f'{col}_frequency'] = train_df[col].map(freq_encoding).fillna(0)
                test_new[f'{col}_frequency'] = test_df[col].map(freq_encoding).fillna(0)
                
                # 테스트 데이터 인코딩
                target_mean_all = train_df.groupby(col)['support_needs'].mean()
                global_mean_all = train_df['support_needs'].mean()
                category_counts_all = train_df.groupby(col).size()
                
                alpha_all = min(30, max(5, len(train_df) // 100))
                smoothed_means_all = (target_mean_all * category_counts_all + global_mean_all * alpha_all) / (category_counts_all + alpha_all)
                
                test_encoded = test_df[col].map(smoothed_means_all).fillna(global_mean_all)
                test_new[f'{col}_target_encoded'] = test_encoded
        
        return train_new, test_new
    
    def create_clustering_features(self, train_df, test_df):
        """클러스터링 피처 - 개선"""
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in train_df.columns and col in test_df.columns]
        
        if len(available_cols) < 3:
            return train_df, test_df
        
        train_numeric = train_df[available_cols].fillna(0)
        test_numeric = test_df[available_cols].fillna(0)
        
        # 이상치 처리 (개선)
        for col in available_cols:
            q01 = train_numeric[col].quantile(0.01)
            q99 = train_numeric[col].quantile(0.99)
            
            train_numeric[col] = np.clip(train_numeric[col], q01, q99)
            test_numeric[col] = np.clip(test_numeric[col], q01, q99)
        
        # 정규화
        scaler = RobustScaler(quantile_range=(5, 95))  # 더 넓은 범위
        train_scaled = scaler.fit_transform(train_numeric)
        test_scaled = scaler.transform(test_numeric)
        
        # 최적 클러스터 수 (6개로 증가)
        best_k = 6  # 5 -> 6
        
        # 클러스터링
        if self.kmeans_model is None:
            self.kmeans_model = KMeans(
                n_clusters=best_k, 
                random_state=42, 
                n_init=15,  # 10 -> 15
                max_iter=500  # 추가
            )
            train_clusters = self.kmeans_model.fit_predict(train_scaled)
        else:
            train_clusters = self.kmeans_model.predict(train_scaled)
            
        test_clusters = self.kmeans_model.predict(test_scaled)
        
        train_new = train_df.copy()
        test_new = test_df.copy()
        
        train_new['cluster'] = train_clusters
        test_new['cluster'] = test_clusters
        
        # 클러스터 중심까지의 거리
        train_distances = self.kmeans_model.transform(train_scaled)
        test_distances = self.kmeans_model.transform(test_scaled)
        
        # 최소 거리
        train_new['cluster_distance'] = np.min(train_distances, axis=1)
        test_new['cluster_distance'] = np.min(test_distances, axis=1)
        
        # 각 클러스터까지의 거리 (상위 3개)
        for i in range(min(3, best_k)):
            train_new[f'distance_to_cluster_{i}'] = train_distances[:, i]
            test_new[f'distance_to_cluster_{i}'] = test_distances[:, i]
        
        # 클러스터 확률 (softmax 기반)
        train_probs = np.exp(-train_distances) / np.sum(np.exp(-train_distances), axis=1, keepdims=True)
        test_probs = np.exp(-test_distances) / np.sum(np.exp(-test_distances), axis=1, keepdims=True)
        
        train_new['cluster_confidence'] = np.max(train_probs, axis=1)
        test_new['cluster_confidence'] = np.max(test_probs, axis=1)
        
        return train_new, test_new
    
    def encode_categorical(self, train_df, test_df):
        """범주형 인코딩 - 개선"""
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
                
                # One-hot encoding 추가 (선택적)
                if col == 'subscription_type':  # 중요한 범주형 변수
                    unique_categories = self.label_encoders[col].classes_
                    for category in unique_categories[:5]:  # 상위 5개 카테고리만
                        train_new[f'{col}_{category}'] = (train_df[col] == category).astype(int)
                        test_new[f'{col}_{category}'] = (test_df[col] == category).astype(int)
        
        return train_new, test_new
    
    def select_features(self, train_df, target_col='support_needs', max_features=80):  # 70 -> 80
        """피처 선택 - 개선"""
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
        
        # 상호정보량 기반 선택 (개선)
        X_var_df = pd.DataFrame(X_variance, columns=variance_features)
        
        # 클래스별 가중 상호정보량
        mi_scores = mutual_info_classif(X_var_df, y, random_state=42)
        
        # 클래스 1 특화 피처 추가 고려
        class_1_mask = y == 1
        if class_1_mask.sum() > 100:
            class_1_mi = mutual_info_classif(X_var_df, class_1_mask.astype(int), random_state=42)
            combined_mi = 0.65 * mi_scores + 0.35 * class_1_mi  # 클래스 1 중요도 증가
        else:
            combined_mi = mi_scores
        
        # 상위 피처 선택
        feature_scores = list(zip(variance_features, combined_mi))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_features = [f[0] for f in feature_scores[:max_features]]
        
        self.selected_features = selected_features
        return selected_features
    
    def create_features(self, train_df, test_df, temporal_threshold=None):
        """피처 생성 파이프라인 - 개선"""
        if train_df is None or test_df is None or train_df.empty or test_df.empty:
            return None, None
        
        print("피처 생성 시작...")
        
        # after_interaction 처리
        train_df, test_df = self.handle_after_interaction(train_df, test_df, temporal_threshold)
        
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
        
        print(f"피처 생성 완료: {train_df.shape[1]}개 피처")
        
        return train_df, test_df

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        engineer = FeatureEngineer()
        
        # temporal_threshold 가져오기
        try:
            from data_analysis import DataAnalyzer
            analyzer = DataAnalyzer()
            analysis_results = analyzer.run_analysis()
            temporal_threshold = analysis_results.get('temporal', {}).get('temporal_threshold')
        except:
            temporal_threshold = None
        
        train_processed, test_processed = engineer.create_features(train_df, test_df, temporal_threshold)
        
        if train_processed is not None and test_processed is not None:
            print(f"피처 엔지니어링 완료:")
            print(f"훈련 세트: {train_processed.shape}")
            print(f"테스트 세트: {test_processed.shape}")
        
        return engineer, train_processed, test_processed
        
    except Exception as e:
        print(f"피처 엔지니어링 오류: {e}")
        return None, None, None

if __name__ == "__main__":
    main()