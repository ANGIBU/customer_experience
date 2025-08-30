# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.outlier_detector = None
        self.feature_selector = None
        self.imputers = {}
        self.selected_features = None
        
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        print("결측치 처리")
        
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        # 공통 수치형 컬럼 찾기
        train_numeric = train_df.select_dtypes(include=[np.number]).columns
        test_numeric = test_df.select_dtypes(include=[np.number]).columns
        
        common_numeric = [col for col in train_numeric if col in test_numeric and col not in ['ID']]
        
        for col in common_numeric:
            if train_clean[col].isnull().sum() > 0 or test_clean[col].isnull().sum() > 0:
                self.imputers[col] = SimpleImputer(strategy='median')
                train_clean[col] = self.imputers[col].fit_transform(train_clean[[col]]).flatten()
                test_clean[col] = self.imputers[col].transform(test_clean[[col]]).flatten()
        
        # 공통 범주형 컬럼 찾기
        train_categorical = train_df.select_dtypes(include=['object']).columns
        test_categorical = test_df.select_dtypes(include=['object']).columns
        
        common_categorical = [col for col in train_categorical if col in test_categorical and col != 'ID']
        
        for col in common_categorical:
            if train_clean[col].isnull().sum() > 0 or test_clean[col].isnull().sum() > 0:
                mode_val = train_clean[col].mode().iloc[0]
                train_clean[col].fillna(mode_val, inplace=True)
                test_clean[col].fillna(mode_val, inplace=True)
        
        return train_clean, test_clean
    
    def detect_and_handle_outliers(self, train_df):
        """이상치 탐지 및 처리"""
        print("이상치 처리")
        
        train_clean = train_df.copy()
        
        # 수치형 컬럼 선택 (ID와 support_needs 제외)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ID', 'support_needs']]
        
        # IQR 방법으로 이상치 클리핑
        for col in numeric_cols:
            Q1 = train_df[col].quantile(0.25)
            Q3 = train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((train_df[col] < lower_bound) | (train_df[col] > upper_bound)).sum()
            
            train_clean[col] = np.clip(train_clean[col], lower_bound, upper_bound)
            
            if outliers_before > 0:
                print(f"  {col}: {outliers_before}개 이상치 클리핑")
        
        return train_clean
    
    def apply_scaling(self, train_df, test_df):
        """다중 스케일링 적용"""
        print("스케일링 적용")
        
        # 공통 수치형 컬럼 찾기
        train_numeric = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        test_numeric = [col for col in test_df.select_dtypes(include=[np.number]).columns 
                       if col not in ['ID', 'support_needs']]
        
        common_numeric_cols = [col for col in train_numeric if col in test_numeric]
        
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        if not common_numeric_cols:
            print("공통 수치형 컬럼 없음")
            return train_scaled, test_scaled
        
        # Standard Scaler
        self.scalers['standard'] = StandardScaler()
        train_scaled[common_numeric_cols] = self.scalers['standard'].fit_transform(train_df[common_numeric_cols])
        test_scaled[common_numeric_cols] = self.scalers['standard'].transform(test_df[common_numeric_cols])
        
        # Robust Scaler 추가
        self.scalers['robust'] = RobustScaler()
        train_robust = self.scalers['robust'].fit_transform(train_df[common_numeric_cols])
        test_robust = self.scalers['robust'].transform(test_df[common_numeric_cols])
        
        for i, col in enumerate(common_numeric_cols):
            train_scaled[f'{col}_robust'] = train_robust[:, i]
            test_scaled[f'{col}_robust'] = test_robust[:, i]
        
        # Quantile Transformer 추가
        self.scalers['quantile'] = QuantileTransformer(output_distribution='normal', random_state=42)
        train_quantile = self.scalers['quantile'].fit_transform(train_df[common_numeric_cols])
        test_quantile = self.scalers['quantile'].transform(test_df[common_numeric_cols])
        
        for i, col in enumerate(common_numeric_cols):
            train_scaled[f'{col}_quantile'] = train_quantile[:, i]
            test_scaled[f'{col}_quantile'] = test_quantile[:, i]
        
        print(f"스케일링 완료: {len(common_numeric_cols)} → {len(common_numeric_cols) * 3} 피처")
        
        return train_scaled, test_scaled
    
    def select_best_features(self, train_df, k=60):
        """최적 피처 선택"""
        print("피처 선택")
        
        if 'support_needs' not in train_df.columns:
            print("타겟 변수 없음 - 피처 선택 건너뛰기")
            feature_cols = [col for col in train_df.columns if col != 'ID']
            return feature_cols[:min(k, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= k:
            print(f"피처 수가 k보다 적음: {len(feature_cols)} <= {k}")
            return feature_cols
        
        X = train_df[feature_cols]
        y = train_df['support_needs']
        
        # 상호정보량 기반 피처 선택
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        print(f"선택된 피처: {len(self.selected_features)}개")
        
        # 상위 피처 출력
        if hasattr(self.feature_selector, 'scores_'):
            feature_scores = list(zip(feature_cols, self.feature_selector.scores_))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            print("상위 5개 피처:")
            for i, (feature, score) in enumerate(feature_scores[:5]):
                print(f"  {i+1}. {feature}: {score:.3f}")
        
        return self.selected_features
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        print("데이터 품질 검증")
        
        issues = []
        
        # 각 데이터프레임의 수치형 컬럼 개별 선택
        train_numeric = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        test_numeric = [col for col in test_df.select_dtypes(include=[np.number]).columns 
                       if col not in ['ID', 'support_needs']]
        
        # 무한대값 확인
        if train_numeric:
            train_inf = np.isinf(train_df[train_numeric]).sum().sum()
        else:
            train_inf = 0
            
        if test_numeric:
            test_inf = np.isinf(test_df[test_numeric]).sum().sum()
        else:
            test_inf = 0
        
        if train_inf > 0 or test_inf > 0:
            issues.append(f"무한대값 - 훈련: {train_inf}, 테스트: {test_inf}")
        
        # 결측치 확인
        train_null = train_df.isnull().sum().sum()
        test_null = test_df.isnull().sum().sum()
        
        if train_null > 0 or test_null > 0:
            issues.append(f"결측치 - 훈련: {train_null}, 테스트: {test_null}")
        
        # 상수 피처 확인 (훈련 데이터만)
        constant_features = []
        for col in train_numeric:
            if train_df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"상수 피처: {len(constant_features)}개")
        
        if issues:
            print("품질 문제:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("품질 검증 통과")
        
        return len(issues) == 0, issues
    
    def create_feature_interactions(self, df):
        """피처 상호작용 생성"""
        print("피처 상호작용 생성")
        
        df_new = df.copy()
        
        # 기본 수치형 피처만 사용
        base_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_features = [feat for feat in base_features if feat in df.columns]
        
        # 상호작용 생성
        interaction_pairs = [
            ('frequent', 'tenure'),
            ('age', 'contract_length')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df_new[f'{feat1}_{feat2}_interaction'] = df_new[feat1] * df_new[feat2]
        
        # 플래그 피처
        if 'age' in df.columns:
            df_new['age_senior'] = (df_new['age'] >= 60).astype(int)
            df_new['age_young'] = (df_new['age'] <= 25).astype(int)
        
        if 'contract_length' in df.columns:
            df_new['long_contract'] = (df_new['contract_length'] >= 365).astype(int)
            df_new['short_contract'] = (df_new['contract_length'] <= 30).astype(int)
        
        created_count = len([col for col in df_new.columns if col not in df.columns])
        print(f"생성된 상호작용 피처: {created_count}개")
        
        return df_new
    
    def process_data(self, train_df, test_df):
        """전체 전처리 파이프라인"""
        print("데이터 전처리 시작")
        print("=" * 40)
        
        # 1. 결측치 처리
        train_df, test_df = self.handle_missing_values(train_df, test_df)
        
        # 2. 이상치 처리 (훈련 데이터만)
        train_df = self.detect_and_handle_outliers(train_df)
        
        # 3. 피처 상호작용 생성
        train_df = self.create_feature_interactions(train_df)
        test_df = self.create_feature_interactions(test_df)
        
        # 4. 스케일링
        train_df, test_df = self.apply_scaling(train_df, test_df)
        
        # 5. 피처 선택 (support_needs가 있는 경우만)
        if 'support_needs' in train_df.columns:
            selected_features = self.select_best_features(train_df, k=60)
            
            # 테스트 데이터에 존재하는 피처만 필터링
            available_features = [col for col in selected_features if col in test_df.columns]
            
            # 훈련 데이터 컬럼 준비
            train_keep_cols = ['ID'] + available_features + ['support_needs']
            train_df = train_df[train_keep_cols]
            
            # 테스트 데이터 컬럼 준비
            test_keep_cols = ['ID'] + available_features
            test_df = test_df[test_keep_cols]
            
            print(f"최종 사용 피처: {len(available_features)}개")
        
        # 6. 품질 검증
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        
        print(f"\n전처리 완료:")
        print(f"  훈련: {train_df.shape}")
        print(f"  테스트: {test_df.shape}")
        print(f"  품질: {'통과' if quality_ok else '문제있음'}")
        
        return train_df, test_df
    
    def prepare_data(self, train_df, test_df, test_size=0.25):
        """모델링용 데이터 준비"""
        print("모델링 데이터 준비")
        
        # 안전하게 피처 컬럼 선택
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        # 공통 피처 컬럼 (ID와 support_needs 제외)
        common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
        
        if not common_features:
            raise ValueError("공통 피처가 없습니다")
        
        # support_needs 컬럼 확인
        if 'support_needs' not in train_df.columns:
            raise ValueError("훈련 데이터에 support_needs 컬럼이 없습니다")
        
        X = train_df[common_features]
        y = train_df['support_needs']
        X_test = test_df[common_features]
        test_ids = test_df['ID']
        
        # 시간 기반 분할 (ID 순서 고려)
        train_size = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_val = y.iloc[train_size:]
        
        print(f"훈련: {X_train.shape}")
        print(f"검증: {X_val.shape}")
        print(f"테스트: {X_test.shape}")
        print(f"공통 피처: {len(common_features)}개")
        
        return X_train, X_val, y_train, y_val, X_test, test_ids

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    preprocessor = DataPreprocessor()
    train_processed, test_processed = preprocessor.process_data(train_df, test_df)
    
    X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data(
        train_processed, test_processed
    )
    
    return preprocessor, X_train, X_val, y_train, y_val, X_test, test_ids

if __name__ == "__main__":
    main()