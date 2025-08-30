# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.feature_selector = None
        self.imputers = {}
        self.selected_features = None
        self.variance_selector = None
        self.feature_order = None
        self.temporal_cutoff = None
        self.is_fitted = False
        
    def temporal_data_split(self, train_df):
        """시간 기반 데이터 분할"""
        print("시간 기반 분할")
        
        if 'temporal_id' not in train_df.columns:
            return train_df
        
        temporal_ids = train_df['temporal_id'].values
        
        if self.temporal_cutoff is None:
            self.temporal_cutoff = np.percentile(temporal_ids, 85)
        
        clean_mask = temporal_ids <= self.temporal_cutoff
        clean_data = train_df[clean_mask].copy()
        
        removed_count = len(train_df) - len(clean_data)
        if removed_count > 0:
            print(f"시간적 누수 데이터 제거: {removed_count}개")
        
        return clean_data
    
    def handle_missing_values(self, train_df, test_df):
        """결측치 처리"""
        print("결측치 처리")
        
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        train_numeric = train_df.select_dtypes(include=[np.number]).columns
        test_numeric = test_df.select_dtypes(include=[np.number]).columns
        
        common_numeric = [col for col in train_numeric if col in test_numeric and col not in ['ID']]
        
        for col in common_numeric:
            if train_clean[col].isnull().sum() > 0 or test_clean[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    train_clean[col] = self.imputers[col].fit_transform(train_clean[[col]]).flatten()
                else:
                    train_clean[col] = self.imputers[col].transform(train_clean[[col]]).flatten()
                
                test_clean[col] = self.imputers[col].transform(test_clean[[col]]).flatten()
        
        train_categorical = train_df.select_dtypes(include=['object']).columns
        test_categorical = test_df.select_dtypes(include=['object']).columns
        
        common_categorical = [col for col in train_categorical if col in test_categorical and col != 'ID']
        
        for col in common_categorical:
            if train_clean[col].isnull().sum() > 0 or test_clean[col].isnull().sum() > 0:
                if col not in self.imputers:
                    mode_val = train_clean[col].mode()
                    if len(mode_val) > 0:
                        mode_val = mode_val.iloc[0]
                    else:
                        mode_val = 'Unknown'
                    self.imputers[col] = mode_val
                
                train_clean[col].fillna(self.imputers[col], inplace=True)
                test_clean[col].fillna(self.imputers[col], inplace=True)
        
        return train_clean, test_clean
    
    def robust_outlier_handling(self, train_df):
        """정밀 이상치 처리"""
        print("이상치 처리")
        
        train_clean = train_df.copy()
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ID', 'support_needs']]
        
        for col in numeric_cols:
            if col in train_df.columns:
                values = train_df[col].dropna()
                
                if len(values) > 100:
                    z_scores = np.abs(zscore(values))
                    outlier_threshold = 3.5
                    
                    outlier_mask = z_scores > outlier_threshold
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        q01 = values.quantile(0.01)
                        q99 = values.quantile(0.99)
                        
                        train_clean[col] = np.clip(train_clean[col], q01, q99)
                        
                        if outlier_count > 50:
                            print(f"  {col}: {outlier_count}개 이상치 처리")
        
        return train_clean
    
    def remove_low_variance_features(self, train_df, test_df):
        """저분산 피처 제거"""
        print("저분산 피처 제거")
        
        numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                       if col not in ['ID', 'support_needs']]
        
        if not numeric_cols:
            return train_df, test_df
        
        if self.variance_selector is None:
            self.variance_selector = VarianceThreshold(threshold=0.001)
            
            train_numeric = train_df[numeric_cols].fillna(0)
            self.variance_selector.fit(train_numeric)
        
        selected_mask = self.variance_selector.get_support()
        selected_numeric_cols = [col for i, col in enumerate(numeric_cols) if selected_mask[i]]
        removed_cols = [col for i, col in enumerate(numeric_cols) if not selected_mask[i]]
        
        if removed_cols:
            print(f"제거된 저분산 피처: {len(removed_cols)}개")
        
        keep_cols_train = ['ID'] + selected_numeric_cols
        keep_cols_test = ['ID'] + selected_numeric_cols
        
        categorical_cols = train_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'ID':
                keep_cols_train.append(col)
                if col in test_df.columns:
                    keep_cols_test.append(col)
        
        if 'support_needs' in train_df.columns:
            keep_cols_train.append('support_needs')
        
        keep_cols_train = [col for col in keep_cols_train if col in train_df.columns]
        keep_cols_test = [col for col in keep_cols_test if col in test_df.columns]
        
        train_clean = train_df[keep_cols_train]
        test_clean = test_df[keep_cols_test]
        
        return train_clean, test_clean
    
    def apply_multi_scaling(self, train_df, test_df):
        """다중 스케일링 적용"""
        print("스케일링 적용")
        
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
        
        if 'standard' not in self.scalers:
            self.scalers['standard'] = StandardScaler()
            train_scaled[common_numeric_cols] = self.scalers['standard'].fit_transform(train_df[common_numeric_cols])
        else:
            train_scaled[common_numeric_cols] = self.scalers['standard'].transform(train_df[common_numeric_cols])
        
        test_scaled[common_numeric_cols] = self.scalers['standard'].transform(test_df[common_numeric_cols])
        
        if 'robust' not in self.scalers:
            self.scalers['robust'] = RobustScaler()
            train_robust = self.scalers['robust'].fit_transform(train_df[common_numeric_cols])
        else:
            train_robust = self.scalers['robust'].transform(train_df[common_numeric_cols])
        
        test_robust = self.scalers['robust'].transform(test_df[common_numeric_cols])
        
        for i, col in enumerate(common_numeric_cols):
            train_scaled[f'{col}_robust'] = train_robust[:, i]
            test_scaled[f'{col}_robust'] = test_robust[:, i]
        
        print(f"스케일링 완료: {len(common_numeric_cols)} → {len(common_numeric_cols) * 2} 피처")
        
        return train_scaled, test_scaled
    
    def multi_criteria_feature_selection(self, train_df, k=80):
        """다기준 피처 선택"""
        print("피처 선택")
        
        if 'support_needs' not in train_df.columns:
            print("타겟 변수 없음 - 피처 선택 건너뛰기")
            feature_cols = [col for col in train_df.columns if col != 'ID']
            return feature_cols[:min(k, len(feature_cols))]
        
        feature_cols = [col for col in train_df.columns if col not in ['ID', 'support_needs']]
        
        if len(feature_cols) <= k:
            print(f"피처 수가 k보다 적음: {len(feature_cols)} <= {k}")
            self.selected_features = feature_cols
            return feature_cols
        
        X = train_df[feature_cols]
        y = train_df['support_needs']
        
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(k+20, len(feature_cols)))
        X_mi = mi_selector.fit_transform(X, y)
        mi_selected = mi_selector.get_support()
        mi_features = [feature_cols[i] for i, selected in enumerate(mi_selected) if selected]
        
        rf_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=50, random_state=42),
            threshold='median'
        )
        rf_selector.fit(X, y)
        rf_selected = rf_selector.get_support()
        rf_features = [feature_cols[i] for i, selected in enumerate(rf_selected) if selected]
        
        correlation_matrix = X.corr().abs()
        
        def remove_correlated_features(features, corr_matrix, threshold=0.9):
            features_to_remove = set()
            
            for i, feat1 in enumerate(features):
                if feat1 in features_to_remove:
                    continue
                    
                for feat2 in features[i+1:]:
                    if feat2 in features_to_remove:
                        continue
                        
                    if feat1 in corr_matrix.index and feat2 in corr_matrix.columns:
                        if corr_matrix.loc[feat1, feat2] > threshold:
                            mi_score1 = mi_selector.scores_[feature_cols.index(feat1)]
                            mi_score2 = mi_selector.scores_[feature_cols.index(feat2)]
                            
                            if mi_score1 < mi_score2:
                                features_to_remove.add(feat1)
                            else:
                                features_to_remove.add(feat2)
            
            return [f for f in features if f not in features_to_remove]
        
        combined_features = list(set(mi_features + rf_features))
        final_features = remove_correlated_features(combined_features, correlation_matrix)
        
        if len(final_features) > k:
            mi_scores_dict = dict(zip(feature_cols, mi_selector.scores_))
            final_features.sort(key=lambda x: mi_scores_dict.get(x, 0), reverse=True)
            final_features = final_features[:k]
        
        self.selected_features = final_features
        
        print(f"선택된 피처: {len(self.selected_features)}개")
        
        if hasattr(mi_selector, 'scores_'):
            feature_scores = [(feat, mi_selector.scores_[i]) for i, feat in enumerate(feature_cols) 
                             if feat in final_features]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            print("상위 5개 피처:")
            for i, (feature, score) in enumerate(feature_scores[:5]):
                print(f"  {i+1}. {feature}: {score:.3f}")
        
        return self.selected_features
    
    def validate_data_quality(self, train_df, test_df):
        """데이터 품질 검증"""
        print("데이터 품질 검증")
        
        issues = []
        
        train_numeric = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                        if col not in ['ID', 'support_needs']]
        test_numeric = [col for col in test_df.select_dtypes(include=[np.number]).columns 
                       if col not in ['ID', 'support_needs']]
        
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
        
        train_null = train_df.isnull().sum().sum()
        test_null = test_df.isnull().sum().sum()
        
        if train_null > 0 or test_null > 0:
            issues.append(f"결측치 - 훈련: {train_null}, 테스트: {test_null}")
        
        if issues:
            print("품질 문제:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("품질 검증 통과")
        
        return len(issues) == 0, issues
    
    def create_class_specific_features(self, df):
        """클래스별 특화 피처 생성"""
        df_new = df.copy()
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            class_0_pattern = (
                (df_new['age'].fillna(40) > 30) & 
                (df_new['tenure'].fillna(100) > 60) & 
                (df_new['frequent'].fillna(10) < 20)
            ).astype(int)
            df_new['class_0_pattern'] = class_0_pattern
            
            class_1_pattern = (
                (df_new['payment_interval'].fillna(30) > 60) &
                (df_new['frequent'].fillna(10) < 10) &
                (df_new['age'].fillna(40) < 40)
            ).astype(int)
            df_new['class_1_pattern'] = class_1_pattern
            
            class_2_pattern = (
                (df_new['contract_length'].fillna(90) > 180) &
                (df_new['frequent'].fillna(10) > 15) &
                (df_new['age'].fillna(40) > 35)
            ).astype(int)
            df_new['class_2_pattern'] = class_2_pattern
        
        return df_new
    
    def ensure_consistent_features(self, train_df, test_df):
        """일관된 피처 구조 보장"""
        if self.selected_features is None:
            return train_df, test_df
        
        if self.feature_order is None and not self.is_fitted:
            self.feature_order = sorted(self.selected_features)
            self.is_fitted = True
        
        if self.feature_order is not None:
            for col in self.feature_order:
                if col not in train_df.columns:
                    train_df[col] = 0
                if col not in test_df.columns:
                    test_df[col] = 0
            
            train_cols = ['ID'] + self.feature_order
            if 'support_needs' in train_df.columns:
                train_cols.append('support_needs')
            
            test_cols = ['ID'] + self.feature_order
            
            train_cols = [col for col in train_cols if col in train_df.columns]
            test_cols = [col for col in test_cols if col in test_df.columns]
            
            return train_df[train_cols], test_df[test_cols]
        
        return train_df, test_df
    
    def process_data(self, train_df, test_df):
        """전체 전처리 파이프라인"""
        print("데이터 전처리 시작")
        print("=" * 40)
        
        train_df = self.temporal_data_split(train_df)
        
        train_df, test_df = self.handle_missing_values(train_df, test_df)
        
        train_df = self.robust_outlier_handling(train_df)
        
        train_df, test_df = self.remove_low_variance_features(train_df, test_df)
        
        train_df = self.create_class_specific_features(train_df)
        test_df = self.create_class_specific_features(test_df)
        
        train_df, test_df = self.apply_multi_scaling(train_df, test_df)
        
        if 'support_needs' in train_df.columns:
            selected_features = self.multi_criteria_feature_selection(train_df, k=80)
            
            train_df, test_df = self.ensure_consistent_features(train_df, test_df)
            
            print(f"최종 사용 피처: {len(self.selected_features)}개")
        
        quality_ok, issues = self.validate_data_quality(train_df, test_df)
        
        print(f"\n전처리 완료:")
        print(f"  훈련: {train_df.shape}")
        print(f"  테스트: {test_df.shape}")
        print(f"  품질: {'통과' if quality_ok else '문제있음'}")
        
        return train_df, test_df
    
    def prepare_data(self, train_df, test_df, test_size=0.2):
        """모델링용 데이터 준비"""
        print("모델링 데이터 준비")
        
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
        
        if not common_features:
            raise ValueError("공통 피처가 없습니다")
        
        if 'support_needs' not in train_df.columns:
            raise ValueError("훈련 데이터에 support_needs 컬럼이 없습니다")
        
        common_features = sorted(common_features)
        
        X = train_df[common_features]
        y = train_df['support_needs']
        X_test = test_df[common_features]
        test_ids = test_df['ID']
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
        
        if 'temporal_id' in train_df.columns:
            temporal_ids = train_df['temporal_id'].values
            cutoff_idx = int(len(temporal_ids) * (1 - test_size))
            sorted_indices = np.argsort(temporal_ids)
            
            train_indices = sorted_indices[:cutoff_idx]
            val_indices = sorted_indices[cutoff_idx:]
            
            X_train = X.iloc[train_indices]
            X_val = X.iloc[val_indices]
            y_train = y.iloc[train_indices]
            y_val = y.iloc[val_indices]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        self.feature_order = common_features
        
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