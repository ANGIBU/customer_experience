# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp, chi2_contingency
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ValidationSystem:
    def __init__(self):
        self.validation_results = {}
        self.gap_ratio = 0.01  # 갭 비율 축소
        
    def safe_data_conversion(self, X, y=None):
        """안전한 데이터 변환"""
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if y is not None:
            if hasattr(y, 'values'):
                y_array = y.values
            else:
                y_array = np.array(y)
            y_clean = np.clip(y_array, 0, 2)
            return X_clean, y_clean
        
        return X_clean
        
    def check_temporal_leakage(self, train_df, test_df):
        """시간적 누수 확인 - 더 엄격한 기준"""
        issues = []
        
        # ID 겹침 확인
        train_ids = set(train_df['ID']) if 'ID' in train_df.columns else set()
        test_ids = set(test_df['ID']) if 'ID' in test_df.columns else set()
        
        common_ids = train_ids & test_ids
        if common_ids:
            issues.append(f"common_ids: {len(common_ids)}")
        
        # 시간 순서 확인
        if train_ids and test_ids:
            def extract_numbers(id_set):
                numbers = []
                for id_val in id_set:
                    if '_' in str(id_val):
                        try:
                            num = int(str(id_val).split('_')[1])
                            numbers.append(num)
                        except:
                            continue
                return numbers
            
            train_nums = extract_numbers(train_ids)
            test_nums = extract_numbers(test_ids)
            
            if train_nums and test_nums:
                train_max = max(train_nums)
                test_min = min(test_nums)
                
                if train_max >= test_min:
                    overlap_count = len([x for x in train_nums if x >= test_min])
                    overlap_ratio = overlap_count / len(train_nums)
                    
                    # 더 엄격한 기준 (1% 이상 겹침도 위험)
                    if overlap_ratio > 0.01:
                        issues.append(f"temporal_overlap: {overlap_ratio:.3f}")
        
        return len(issues) == 0, issues
    
    def create_simple_validation_model(self, X_train, y_train):
        """단순한 검증용 모델"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        # 클래스 가중치 계산
        class_counts = np.bincount(y_train_clean.astype(int))
        total_samples = len(y_train_clean)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
        
        # 클래스 불균형 보정
        class_weights[1] *= 1.15
        class_weights[2] *= 1.09
        
        # Random Forest (단순화)
        model = RandomForestClassifier(
            n_estimators=100,  # 200 → 100으로 축소
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.8,
            class_weight=class_weights,
            random_state=42,
            n_jobs=1
        )
        
        model.fit(X_train_clean, y_train_clean)
        return model
    
    def simple_cross_validation(self, X, y, n_splits=3):  # 5 → 3으로 축소
        """단순화된 교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 제거
        if 'temporal_id' in X.columns:
            temporal_col_idx = list(X.columns).index('temporal_id')
            X_for_cv = np.delete(X_clean, temporal_col_idx, axis=1)
        else:
            X_for_cv = X_clean
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_for_cv, y_clean)):
            try:
                # 단순한 모델 학습
                model = self.create_simple_validation_model(X_for_cv[train_idx], y_clean[train_idx])
                y_pred = model.predict(X_for_cv[val_idx])
                accuracy = accuracy_score(y_clean[val_idx], y_pred)
                fold_scores.append(accuracy)
            except Exception:
                continue
        
        if fold_scores:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
        else:
            mean_score = 0.0
            std_score = 0.0
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_type': 'stratified_kfold_simple'
        }
    
    def holdout_validation_simple(self, X_train, y_train, X_val, y_val):
        """단순화된 홀드아웃 검증"""
        if any(data is None for data in [X_train, y_train, X_val, y_val]):
            return self.get_default_holdout_results()
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        try:
            # 단순한 모델 학습
            model = self.create_simple_validation_model(X_train_clean, y_train_clean)
            y_pred = model.predict(X_val_clean)
            
            accuracy = accuracy_score(y_val_clean, y_pred)
            f1_macro = f1_score(y_val_clean, y_pred, average='macro')
            f1_weighted = f1_score(y_val_clean, y_pred, average='weighted')
            
            # 클래스별 성능
            class_scores = {}
            for cls in range(3):
                mask = y_val_clean == cls
                if mask.sum() > 0:
                    class_acc = accuracy_score(y_val_clean[mask], y_pred[mask])
                    class_f1 = f1_score(y_val_clean == cls, y_pred == cls)
                    class_scores[cls] = {'accuracy': class_acc, 'f1': class_f1}
                else:
                    class_scores[cls] = {'accuracy': 0.0, 'f1': 0.0}
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'class_scores': class_scores,
                'total_samples': len(y_val_clean)
            }
            
        except Exception:
            return self.get_default_holdout_results()
    
    def stability_test_simple(self, X, y, n_runs=5):  # 15 → 5로 축소
        """단순화된 안정성 테스트"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 제거
        if 'temporal_id' in X.columns:
            temporal_col_idx = list(X.columns).index('temporal_id')
            X_temp = np.delete(X_clean, temporal_col_idx, axis=1)
        else:
            X_temp = X_clean
        
        from sklearn.model_selection import train_test_split
        
        accuracy_scores = []
        f1_scores = []
        
        for run in range(n_runs):
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_clean, 
                    test_size=0.25, 
                    random_state=run * 13, 
                    stratify=y_clean
                )
                
                # 단순한 모델 학습
                model = self.create_simple_validation_model(X_train, y_train)
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='macro')
                
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                
            except Exception:
                continue
        
        if len(accuracy_scores) < 3:
            return self.get_default_stability_results()
        
        # 통계 계산
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        # 안정성 점수
        accuracy_stability = max(0, 1 - (std_accuracy / mean_accuracy)) if mean_accuracy > 0 else 0
        f1_stability = max(0, 1 - (std_f1 / mean_f1)) if mean_f1 > 0 else 0
        
        overall_stability = (accuracy_stability * 0.6 + f1_stability * 0.4)
        
        return {
            'accuracy_scores': accuracy_scores,
            'f1_scores': f1_scores,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'accuracy_stability': accuracy_stability,
            'f1_stability': f1_stability,
            'overall_stability': overall_stability,
            'n_runs': len(accuracy_scores)
        }
    
    def feature_importance_validation_simple(self, X_train, y_train):
        """단순화된 피처 중요도 검증"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        try:
            model = self.create_simple_validation_model(X_train_clean, y_train_clean)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # 중요도 통계
                mean_importance = np.mean(importances)
                std_importance = np.std(importances)
                max_importance = np.max(importances)
                
                # 상위 피처 비율
                sorted_imp = np.sort(importances)
                top_5_ratio = np.sum(sorted_imp[-5:]) / np.sum(importances) if len(sorted_imp) >= 5 else 1.0
                
                return {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'max_importance': max_importance,
                    'top_5_ratio': top_5_ratio,
                    'n_features': len(importances)
                }
        except Exception:
            pass
        
        return {}
    
    def get_default_holdout_results(self):
        """기본 홀드아웃 결과"""
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'class_scores': {0: {'accuracy': 0.0, 'f1': 0.0}, 
                           1: {'accuracy': 0.0, 'f1': 0.0}, 
                           2: {'accuracy': 0.0, 'f1': 0.0}},
            'total_samples': 0
        }
    
    def get_default_stability_results(self):
        """기본 안정성 결과"""
        return {
            'accuracy_scores': [0.0],
            'f1_scores': [0.0],
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'mean_f1': 0.0,
            'std_f1': 0.0,
            'accuracy_stability': 0.0,
            'f1_stability': 0.0,
            'overall_stability': 0.0,
            'n_runs': 0
        }
    
    def validate_system(self, X_train, y_train, X_val=None, y_val=None):
        """단순화된 검증 시스템"""
        if X_train is None or y_train is None or len(X_train) == 0:
            return self.get_comprehensive_default_results()
        
        X_clean, y_clean = self.safe_data_conversion(X_train, y_train)
        
        # 홀드아웃 검증
        if X_val is not None and y_val is not None:
            holdout_results = self.holdout_validation_simple(X_train, y_train, X_val, y_val)
        else:
            # 데이터 분할해서 홀드아웃
            from sklearn.model_selection import train_test_split
            try:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
                )
            except:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42
                )
            holdout_results = self.holdout_validation_simple(X_train_split, y_train_split, X_val_split, y_val_split)
        
        # 교차검증 (단순화)
        cv_results = self.simple_cross_validation(X_train, y_train)
        
        # 안정성 테스트 (단순화)
        stability_results = self.stability_test_simple(X_train, y_train)
        
        # 피처 중요도 검증 (단순화)
        feature_results = self.feature_importance_validation_simple(X_train, y_train)
        
        # 종합 점수 계산
        holdout_score = holdout_results.get('accuracy', 0.0)
        cv_score = cv_results.get('mean_score', 0.0)
        stability_score = stability_results.get('overall_stability', 0.0)
        
        # 가중 평균으로 종합 점수
        overall_score = (
            holdout_score * 0.50 +      # 홀드아웃 가중치 증가
            cv_score * 0.30 +           # 교차검증 가중치 감소  
            stability_score * 0.20      # 안정성 가중치 감소
        )
        
        self.validation_results = {
            'holdout': holdout_results,
            'cross_validation': cv_results,
            'stability': stability_results,
            'feature_importance': feature_results,
            'overall_score': overall_score,
            'component_scores': {
                'holdout_score': holdout_score,
                'cv_ensemble_score': cv_score,
                'stability_score': stability_score
            }
        }
        
        return self.validation_results
    
    def get_comprehensive_default_results(self):
        """포괄적 기본 결과"""
        return {
            'holdout': self.get_default_holdout_results(),
            'cross_validation': {'mean_score': 0.0, 'std_score': 0.0, 'fold_scores': [], 'cv_type': 'none'},
            'stability': self.get_default_stability_results(),
            'feature_importance': {},
            'overall_score': 0.0,
            'component_scores': {
                'holdout_score': 0.0,
                'cv_ensemble_score': 0.0,
                'stability_score': 0.0
            }
        }

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        validator = ValidationSystem()
        leakage_free, issues = validator.check_temporal_leakage(train_df, test_df)
        
        if not leakage_free:
            print(f"시간적 누수 감지: {issues}")
        else:
            print("시간적 누수 없음")
        
        return validator
        
    except Exception as e:
        print(f"검증 시스템 오류: {e}")
        return None

if __name__ == "__main__":
    main()