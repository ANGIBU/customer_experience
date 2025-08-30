# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        self.scalers = {}
        self.outlier_detector = None
        self.feature_selector = None
        self.selected_features = None
        
    def detect_outliers(self, train_df, contamination=0.1):
        """이상치 탐지"""
        print("=== 이상치 탐지 ===")
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ID', 'support_needs']]
        
        self.outlier_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        outlier_scores = self.outlier_detector.fit_predict(train_df[numeric_cols])
        outlier_count = (outlier_scores == -1).sum()
        
        print(f"탐지된 이상치: {outlier_count}개 ({outlier_count/len(train_df)*100:.2f}%)")
        
        train_clean = train_df.copy()
        outlier_mask = outlier_scores == -1
        
        for col in numeric_cols:
            if col in train_df.columns:
                Q1 = train_df[col].quantile(0.25)
                Q3 = train_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                train_clean[col] = np.clip(train_clean[col], lower_bound, upper_bound)
        
        return train_clean, outlier_mask
    
    def apply_multiple_scaling(self, train_df, test_df):
        """다중 스케일링 적용"""
        print("=== 다중 스케일링 적용 ===")
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ID', 'support_needs']]
        
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        self.scalers['standard'] = StandardScaler()
        train_scaled[numeric_cols] = self.scalers['standard'].fit_transform(train_df[numeric_cols])
        test_scaled[numeric_cols] = self.scalers['standard'].transform(test_df[numeric_cols])
        
        robust_scaler = RobustScaler()
        train_robust = robust_scaler.fit_transform(train_df[numeric_cols])
        test_robust = robust_scaler.transform(test_df[numeric_cols])
        
        for i, col in enumerate(numeric_cols):
            train_scaled[f'{col}_robust'] = train_robust[:, i]
            test_scaled[f'{col}_robust'] = test_robust[:, i]
        
        quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        train_quantile = quantile_transformer.fit_transform(train_df[numeric_cols])
        test_quantile = quantile_transformer.transform(test_df[numeric_cols])
        
        for i, col in enumerate(numeric_cols):
            train_scaled[f'{col}_quantile'] = train_quantile[:, i]
            test_scaled[f'{col}_quantile'] = test_quantile[:, i]
        
        print(f"스케일링 완료: 원본 {len(numeric_cols)}개 → 총 {len(numeric_cols)*3}개 피처")
        
        return train_scaled, test_scaled
    
    def select_features(self, train_df, target_col='support_needs', k=50):
        """피처 선택"""
        print("=== 피처 선택 ===")
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', target_col]]
        
        numeric_feature_cols = []
        for col in feature_cols:
            if train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_feature_cols.append(col)
        
        X = train_df[numeric_feature_cols]
        y = train_df[target_col]
        
        print(f"수치형 피처 수: {len(numeric_feature_cols)}")
        
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(numeric_feature_cols)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [numeric_feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"선택된 피처 수: {len(self.selected_features)}")
        print("상위 10개 피처:")
        
        feature_scores = list(zip(numeric_feature_cols, self.feature_selector.scores_))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(feature_scores[:10]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        return self.selected_features
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        print("=== 결측치 처리 ===")
        
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if train_clean[col].isnull().sum() > 0:
                median_val = train_clean[col].median()
                train_clean[col].fillna(median_val, inplace=True)
                test_clean[col].fillna(median_val, inplace=True)
                print(f"{col}: 중앙값 {median_val:.2f}로 대체")
        
        categorical_cols = train_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'ID' and train_clean[col].isnull().sum() > 0:
                mode_val = train_clean[col].mode().iloc[0]
                train_clean[col].fillna(mode_val, inplace=True)
                test_clean[col].fillna(mode_val, inplace=True)
                print(f"{col}: 최빈값 {mode_val}로 대체")
        
        return train_clean, test_clean
    
    def create_feature_interactions(self, df):
        """피처 상호작용 생성"""
        print("=== 피처 상호작용 생성 ===")
        
        df_new = df.copy()
        
        core_features = ['age', 'tenure', 'frequent', 'payment_interval', 'after_interaction']
        
        if all(feat in df.columns for feat in ['frequent', 'tenure']):
            df_new['freq_tenure_interaction'] = df_new['frequent'] * df_new['tenure']
        
        if all(feat in df.columns for feat in ['age', 'contract_length']):
            df_new['age_contract_interaction'] = df_new['age'] * df_new['contract_length']
        
        if all(feat in df.columns for feat in ['frequent', 'after_interaction']):
            df_new['freq_interaction_diff'] = abs(df_new['frequent'] - df_new['after_interaction'])
        
        if 'age' in df.columns:
            df_new['age_senior_flag'] = (df_new['age'] >= 60).astype(int)
            df_new['age_young_flag'] = (df_new['age'] <= 25).astype(int)
        
        if 'contract_length' in df.columns:
            df_new['long_contract_flag'] = (df_new['contract_length'] >= 360).astype(int)
            df_new['short_contract_flag'] = (df_new['contract_length'] <= 30).astype(int)
        
        created_features = [col for col in df_new.columns if col not in df.columns]
        print(f"생성된 상호작용 피처: {len(created_features)}개")
        
        return df_new
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        print("=== 데이터 품질 검증 ===")
        
        issues = []
        
        train_inf = np.isinf(train_df.select_dtypes(include=[np.number])).sum().sum()
        test_inf = np.isinf(test_df.select_dtypes(include=[np.number])).sum().sum()
        
        if train_inf > 0 or test_inf > 0:
            issues.append(f"무한대 값 - 훈련: {train_inf}, 테스트: {test_inf}")
        
        train_nan = train_df.isnull().sum().sum()
        test_nan = test_df.isnull().sum().sum()
        
        if train_nan > 0 or test_nan > 0:
            issues.append(f"결측치 - 훈련: {train_nan}, 테스트: {test_nan}")
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ID', 'support_needs']]
        
        correlation_matrix = train_df[numeric_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            issues.append(f"높은 상관관계 피처 쌍: {len(high_corr_pairs)}개")
        
        low_variance_features = []
        for col in numeric_cols:
            if train_df[col].var() < 1e-6:
                low_variance_features.append(col)
        
        if low_variance_features:
            issues.append(f"저분산 피처: {len(low_variance_features)}개")
        
        if issues:
            print("발견된 문제:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("데이터 품질 검증 통과")
        
        return len(issues) == 0
    
    def process_complete_pipeline(self, train_df, test_df):
        """전체 전처리 파이프라인"""
        print("데이터 전처리 파이프라인 시작")
        print("="*45)
        
        train_df, test_df = self.handle_missing_values(train_df, test_df)
        
        train_df, outlier_mask = self.detect_outliers(train_df)
        
        train_df = self.create_feature_interactions(train_df)
        test_df = self.create_feature_interactions(test_df)
        
        train_df, test_df = self.apply_multiple_scaling(train_df, test_df)
        
        if 'support_needs' in train_df.columns:
            selected_features = self.select_features(train_df)
            
            keep_cols = ['ID'] + selected_features
            if 'support_needs' in train_df.columns:
                keep_cols.append('support_needs')
            
            train_df = train_df[keep_cols]
            test_df = test_df[['ID'] + selected_features]
        
        quality_ok = self.validate_data_quality(train_df, test_df)
        
        print(f"\n전처리 완료:")
        print(f"  훈련 데이터: {train_df.shape}")
        print(f"  테스트 데이터: {test_df.shape}")
        print(f"  데이터 품질: {'통과' if quality_ok else '문제 있음'}")
        
        return train_df, test_df
    
    def prepare_model_data(self, train_df, test_df, test_size=0.2):
        """모델링용 데이터 준비"""
        print("=== 모델링용 데이터 준비 ===")
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        X = train_df[feature_cols]
        y = train_df['support_needs']
        X_test = test_df[feature_cols]
        test_ids = test_df['ID']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"훈련 데이터: {X_train.shape}")
        print(f"검증 데이터: {X_val.shape}")
        print(f"테스트 데이터: {X_test.shape}")
        print(f"피처 수: {len(feature_cols)}")
        
        return X_train, X_val, y_train, y_val, X_test, test_ids

def main():
    """메인 실행 함수"""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    preprocessor = DataPreprocessor()
    train_processed, test_processed = preprocessor.process_complete_pipeline(train_df, test_df)
    
    X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_model_data(
        train_processed, test_processed
    )
    
    print("\n전처리 완료!")
    return preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids

if __name__ == "__main__":
    main()