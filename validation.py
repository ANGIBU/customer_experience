# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

class ValidationSystem:
    def __init__(self):
        self.validation_results = {}
        self.gap_ratio = 0.015  # 0.01 -> 0.015 (갭 약간 증가)
        
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
        """시간적 누수 확인 - 개선"""
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
                    
                    if overlap_ratio > 0.03:  # 0.05 -> 0.03 (더 엄격한 기준)
                        issues.append(f"temporal_overlap: {overlap_ratio:.3f}")
        
        return len(issues) == 0, issues
    
    def create_validation_model(self, X_train, y_train):
        """검증용 모델 생성 - 개선"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        # 클래스 가중치 계산 (정밀 조정)
        class_counts = np.bincount(y_train_clean.astype(int))
        total_samples = len(y_train_clean)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
        
        # 클래스 불균형 보정 (개선)
        class_weights[0] *= 0.98  # 추가
        class_weights[1] *= 1.18  # 1.15 -> 1.18
        class_weights[2] *= 1.04  # 1.05 -> 1.04
        
        model = RandomForestClassifier(
            n_estimators=300,  # 250 -> 300
            max_depth=11,  # 10 -> 11
            min_samples_split=12,  # 10 -> 12
            min_samples_leaf=6,  # 5 -> 6
            max_features=0.7,  # 0.75 -> 0.7
            bootstrap=True,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            criterion='entropy'  # 추가
        )
        
        model.fit(X_train_clean, y_train_clean)
        return model
    
    def gap_walk_forward_cv(self, X, y, n_splits=5):
        """갭이 있는 워크포워드 교차검증 - 개선"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 기반 분할
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            total_samples = len(sorted_indices)
            gap_size = int(total_samples * self.gap_ratio)  # 갭 크기
            
            # 최소 폴드 크기 확보
            min_fold_size = total_samples // (n_splits * 2.2)  # 2.5 -> 2.2
            if min_fold_size < 600:  # 500 -> 600
                return self.standard_cv(X, y, n_splits)
            
            fold_scores = []
            temporal_col_idx = list(X.columns).index('temporal_id')
            
            for fold in range(n_splits):
                # 동적 윈도우 크기 (개선)
                window_size = int(min_fold_size * (1.2 + fold * 0.1))  # 더 점진적 증가
                
                train_start = int(fold * min_fold_size * 0.9)  # 오버랩 감소
                train_end = int(train_start + window_size)
                
                val_start = int(train_end + gap_size)
                val_end = int(val_start + min_fold_size)
                
                if val_end > total_samples:
                    break
                
                train_idx = sorted_indices[train_start:train_end]
                val_idx = sorted_indices[val_start:val_end]
                
                if len(train_idx) < 300 or len(val_idx) < 150:  # 200, 100 -> 300, 150
                    continue
                
                X_train_fold = np.delete(X_clean[train_idx], temporal_col_idx, axis=1)
                y_train_fold = y_clean[train_idx]
                X_val_fold = np.delete(X_clean[val_idx], temporal_col_idx, axis=1)
                y_val_fold = y_clean[val_idx]
                
                model = self.create_validation_model(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                accuracy = accuracy_score(y_val_fold, y_pred)
                fold_scores.append(accuracy)
            
            if len(fold_scores) >= 3:
                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                
                return {
                    'fold_scores': fold_scores,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'cv_type': 'gap_walk_forward'
                }
            else:
                return self.standard_cv(X, y, n_splits)
        else:
            return self.standard_cv(X, y, n_splits)
    
    def standard_cv(self, X, y, n_splits=5):
        """표준 교차검증 - 개선"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 컬럼 제거
        if 'temporal_id' in X.columns:
            temporal_col_idx = list(X.columns).index('temporal_id')
            X_for_cv = np.delete(X_clean, temporal_col_idx, axis=1)
        else:
            X_for_cv = X_clean
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        fold_f1_scores = []  # F1 점수도 추가
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_for_cv, y_clean)):
            model = self.create_validation_model(X_for_cv[train_idx], y_clean[train_idx])
            y_pred = model.predict(X_for_cv[val_idx])
            
            accuracy = accuracy_score(y_clean[val_idx], y_pred)
            f1 = f1_score(y_clean[val_idx], y_pred, average='macro')
            
            fold_scores.append(accuracy)
            fold_f1_scores.append(f1)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        mean_f1 = np.mean(fold_f1_scores)
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_f1': mean_f1,
            'cv_type': 'stratified_kfold'
        }
    
    def repeated_cv(self, X, y, n_repeats=3, n_splits=5):
        """반복 교차검증 - 개선"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        if 'temporal_id' in X.columns:
            temporal_col_idx = list(X.columns).index('temporal_id')
            X_for_cv = np.delete(X_clean, temporal_col_idx, axis=1)
        else:
            X_for_cv = X_clean
        
        all_scores = []
        all_f1_scores = []
        
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat * 7)
            
            for train_idx, val_idx in skf.split(X_for_cv, y_clean):
                model = self.create_validation_model(X_for_cv[train_idx], y_clean[train_idx])
                y_pred = model.predict(X_for_cv[val_idx])
                
                accuracy = accuracy_score(y_clean[val_idx], y_pred)
                f1 = f1_score(y_clean[val_idx], y_pred, average='macro')
                
                all_scores.append(accuracy)
                all_f1_scores.append(f1)
        
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        mean_f1 = np.mean(all_f1_scores)
        
        return {
            'scores': all_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'mean_f1': mean_f1,
            'cv_type': 'repeated_stratified_kfold'
        }
    
    def holdout_validation(self, X_train, y_train, X_val, y_val):
        """홀드아웃 검증 - 개선"""
        if any(data is None for data in [X_train, y_train, X_val, y_val]):
            return self.get_default_holdout_results()
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = self.create_validation_model(X_train_clean, y_train_clean)
        y_pred = model.predict(X_val_clean)
        y_pred_proba = model.predict_proba(X_val_clean)
        
        # 전체 성능
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
        
        # 예측 신뢰도 분석
        confidence_scores = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)
        high_confidence_ratio = np.mean(confidence_scores > 0.7)  # 높은 신뢰도 비율
        
        # 혼동 행렬
        cm = confusion_matrix(y_val_clean, y_pred)
        
        # 클래스 1 성능 특별 체크 (중요)
        class_1_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        class_1_precision = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'class_scores': class_scores,
            'avg_confidence': avg_confidence,
            'high_confidence_ratio': high_confidence_ratio,
            'confusion_matrix': cm.tolist(),
            'class_1_recall': class_1_recall,
            'class_1_precision': class_1_precision,
            'total_samples': len(y_val_clean)
        }
    
    def stability_test(self, X, y, n_runs=15):  # 10 -> 15
        """안정성 테스트 - 개선"""
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
        class_balance_scores = []
        class_1_performance = []  # 클래스 1 성능 추적
        
        for run in range(n_runs):
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_clean, 
                    test_size=0.22,  # 0.25 -> 0.22
                    random_state=run * 13,  # 다양성 증가
                    stratify=y_clean
                )
                
                model = self.create_validation_model(X_train, y_train)
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='macro')
                
                # 클래스별 F1 점수
                class_f1_scores = f1_score(y_val, y_pred, average=None)
                class_balance = 1 - np.std(class_f1_scores) if len(class_f1_scores) > 1 else 0
                
                # 클래스 1 성능
                if len(class_f1_scores) > 1:
                    class_1_performance.append(class_f1_scores[1])
                
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                class_balance_scores.append(class_balance)
                
            except Exception:
                continue
        
        if len(accuracy_scores) < 8:  # 5 -> 8
            return self.get_default_stability_results()
        
        # 통계 계산
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_balance = np.mean(class_balance_scores)
        mean_class_1_perf = np.mean(class_1_performance) if class_1_performance else 0
        
        # 안정성 점수 (개선된 계산)
        accuracy_stability = max(0, 1 - (std_accuracy / (mean_accuracy + 0.01)))
        f1_stability = max(0, 1 - (std_f1 / (mean_f1 + 0.01)))
        
        # 가중 안정성 점수 (클래스 1 성능 포함)
        overall_stability = (
            accuracy_stability * 0.3 +
            f1_stability * 0.3 +
            mean_balance * 0.25 +
            mean_class_1_perf * 0.15
        )
        
        return {
            'accuracy_scores': accuracy_scores,
            'f1_scores': f1_scores,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'accuracy_stability': accuracy_stability,
            'f1_stability': f1_stability,
            'class_balance': mean_balance,
            'class_1_performance': mean_class_1_perf,
            'overall_stability': overall_stability,
            'n_runs': len(accuracy_scores)
        }
    
    def feature_importance_validation(self, X_train, y_train):
        """피처 중요도 검증 - 개선"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        model = self.create_validation_model(X_train_clean, y_train_clean)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 중요도 통계
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            max_importance = np.max(importances)
            
            # 상위 피처 비율
            top_10_ratio = np.sum(np.sort(importances)[-10:]) / np.sum(importances)
            top_5_ratio = np.sum(np.sort(importances)[-5:]) / np.sum(importances)
            
            # 중요도 집중도
            concentration = 1 - (np.sum(importances > mean_importance) / len(importances))
            
            # Gini 계수 계산 (피처 중요도 불평등 측정)
            sorted_importances = np.sort(importances)
            n = len(importances)
            cumsum = np.cumsum(sorted_importances)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            
            return {
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'max_importance': max_importance,
                'top_10_ratio': top_10_ratio,
                'top_5_ratio': top_5_ratio,
                'concentration': concentration,
                'gini_coefficient': gini,
                'n_features': len(importances)
            }
        
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
            'avg_confidence': 0.0,
            'high_confidence_ratio': 0.0,
            'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'class_1_recall': 0.0,
            'class_1_precision': 0.0,
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
            'class_balance': 0.0,
            'class_1_performance': 0.0,
            'overall_stability': 0.0,
            'n_runs': 0
        }
    
    def validate_system(self, X_train, y_train, X_val=None, y_val=None):
        """검증 시스템 실행 - 개선"""
        if X_train is None or y_train is None or len(X_train) == 0:
            return self.get_comprehensive_default_results()
        
        X_clean, y_clean = self.safe_data_conversion(X_train, y_train)
        
        # 홀드아웃 검증
        if X_val is not None and y_val is not None:
            holdout_results = self.holdout_validation(X_train, y_train, X_val, y_val)
        else:
            # 데이터 분할해서 홀드아웃
            from sklearn.model_selection import train_test_split
            try:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_clean, y_clean, test_size=0.22, random_state=42, stratify=y_clean
                )
            except:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_clean, y_clean, test_size=0.22, random_state=42
                )
            holdout_results = self.holdout_validation(X_train_split, y_train_split, X_val_split, y_val_split)
        
        # 교차검증
        cv_results = self.gap_walk_forward_cv(X_train, y_train)
        
        # 반복 교차검증
        repeated_cv_results = self.repeated_cv(X_train, y_train, n_repeats=3, n_splits=5)
        
        # 안정성 테스트
        stability_results = self.stability_test(X_train, y_train, n_runs=15)
        
        # 피처 중요도 검증
        feature_results = self.feature_importance_validation(X_train, y_train)
        
        # 종합 점수 계산 (개선된 가중치)
        holdout_score = holdout_results.get('accuracy', 0.0)
        cv_score = cv_results.get('mean_score', 0.0)
        repeated_cv_score = repeated_cv_results.get('mean_score', 0.0)
        stability_score = stability_results.get('overall_stability', 0.0)
        
        # 클래스 1 성능 보너스
        class_1_bonus = 0
        if 'class_1_recall' in holdout_results and 'class_1_precision' in holdout_results:
            class_1_f1 = 2 * (holdout_results['class_1_recall'] * holdout_results['class_1_precision']) / \
                        (holdout_results['class_1_recall'] + holdout_results['class_1_precision'] + 0.001)
            class_1_bonus = class_1_f1 * 0.05  # 5% 보너스
        
        # 가중 평균으로 종합 점수 (개선된 가중치)
        overall_score = (
            holdout_score * 0.25 +
            cv_score * 0.25 +
            repeated_cv_score * 0.15 +
            stability_score * 0.35 +
            class_1_bonus
        )
        
        self.validation_results = {
            'holdout': holdout_results,
            'cross_validation': cv_results,
            'repeated_cv': repeated_cv_results,
            'stability': stability_results,
            'feature_importance': feature_results,
            'overall_score': overall_score,
            'component_scores': {
                'holdout_score': holdout_score,
                'cv_score': cv_score,
                'repeated_cv_score': repeated_cv_score,
                'stability_score': stability_score,
                'class_1_bonus': class_1_bonus
            }
        }
        
        return self.validation_results
    
    def get_comprehensive_default_results(self):
        """포괄적 기본 결과"""
        return {
            'holdout': self.get_default_holdout_results(),
            'cross_validation': {'mean_score': 0.0, 'std_score': 0.0, 'fold_scores': [], 'cv_type': 'none'},
            'repeated_cv': {'mean_score': 0.0, 'std_score': 0.0, 'scores': [], 'cv_type': 'none'},
            'stability': self.get_default_stability_results(),
            'feature_importance': {},
            'overall_score': 0.0,
            'component_scores': {
                'holdout_score': 0.0,
                'cv_score': 0.0,
                'repeated_cv_score': 0.0,
                'stability_score': 0.0,
                'class_1_bonus': 0.0
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