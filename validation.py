# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class DirichletCalibrator:
    """3클래스 확률 보정"""
    
    def __init__(self, regularization=1.0):
        self.regularization = regularization
        self.calibration_model = None
        
    def fit(self, uncalibrated_probs, true_labels):
        """보정 모델 학습"""
        log_probs = np.log(uncalibrated_probs + 1e-10)
        
        self.calibration_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=self.regularization,
            max_iter=1000,
            random_state=42
        )
        
        self.calibration_model.fit(log_probs, true_labels)
        
    def predict_proba(self, uncalibrated_probs):
        """보정된 확률 예측"""
        if self.calibration_model is None:
            raise ValueError("보정 모델이 학습되지 않았습니다.")
            
        log_probs = np.log(uncalibrated_probs + 1e-10)
        calibrated_probs = self.calibration_model.predict_proba(log_probs)
        
        return calibrated_probs
    
    def evaluate_calibration(self, y_probs, y_true, n_bins=10):
        """보정 품질 평가"""
        ece = 0
        for class_idx in range(3):
            class_probs = y_probs[:, class_idx]
            class_true = (y_true == class_idx).astype(int)
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            for i in range(n_bins):
                bin_mask = (class_probs > bin_boundaries[i]) & (class_probs <= bin_boundaries[i+1])
                if np.sum(bin_mask) > 0:
                    bin_confidence = np.mean(class_probs[bin_mask])
                    bin_accuracy = np.mean(class_true[bin_mask])
                    bin_weight = np.sum(bin_mask) / len(class_probs)
                    ece += bin_weight * np.abs(bin_confidence - bin_accuracy)
        
        return ece / 3

class OptimalThresholdFinder:
    """3클래스 임계값 최적화"""
    
    def __init__(self, metric='f1_macro'):
        self.metric = metric
        self.optimal_thresholds = None
    
    def optimize_thresholds(self, y_probs, y_true):
        """최적 임계값 탐색"""
        def objective(thresholds):
            adjusted_probs = y_probs - thresholds.reshape(1, -1)
            y_pred = np.argmax(adjusted_probs, axis=1)
            
            if self.metric == 'f1_macro':
                score = f1_score(y_true, y_pred, average='macro')
            elif self.metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                score = accuracy_score(y_true, y_pred)
            
            return -score
        
        initial_thresholds = np.array([0.33, 0.33, 0.34])
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.1, 0.9)] * 3
        
        result = minimize(
            objective,
            initial_thresholds,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.optimal_thresholds = result.x
        return self.optimal_thresholds

class UltraPrecisionValidator:
    """초정밀 교차검증"""
    
    def __init__(self, n_outer_splits=8, n_inner_splits=5, n_repeats=25):
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.n_repeats = n_repeats
        self.gap_ratio = 0.04
        
    def validate_with_calibration(self, X, y, timestamps, model_pipeline):
        """보정 포함 중첩 교차검증"""
        tscv = self._create_temporal_splits(X, timestamps)
        
        all_scores = []
        
        for repeat in range(self.n_repeats):
            repeat_scores = []
            
            for train_outer, test_outer in tscv:
                if len(train_outer) < 500 or len(test_outer) < 100:
                    continue
                    
                X_train_outer = X.iloc[train_outer]
                y_train_outer = y.iloc[train_outer]
                X_test_outer = X.iloc[test_outer]
                y_test_outer = y.iloc[test_outer]
                
                # 내부 교차검증
                inner_cv = StratifiedKFold(
                    n_splits=self.n_inner_splits, 
                    shuffle=True, 
                    random_state=repeat
                )
                
                # 보정용 예측 생성
                try:
                    cal_predictions = cross_val_predict(
                        model_pipeline,
                        X_train_outer, y_train_outer,
                        cv=inner_cv,
                        method='predict_proba'
                    )
                except:
                    # predict_proba 실패시 일반 예측 사용
                    cal_predictions_class = cross_val_predict(
                        model_pipeline,
                        X_train_outer, y_train_outer,
                        cv=inner_cv
                    )
                    # 원핫 인코딩
                    cal_predictions = np.zeros((len(cal_predictions_class), 3))
                    for i, cls in enumerate(cal_predictions_class):
                        cal_predictions[i, int(cls)] = 1.0
                
                # 보정기 훈련
                calibrator = DirichletCalibrator()
                calibrator.fit(cal_predictions, y_train_outer.values)
                
                # 최종 평가
                try:
                    model_pipeline.fit(X_train_outer, y_train_outer)
                    if hasattr(model_pipeline, 'predict_proba'):
                        test_probs = model_pipeline.predict_proba(X_test_outer)
                    else:
                        test_pred_class = model_pipeline.predict(X_test_outer)
                        test_probs = np.zeros((len(test_pred_class), 3))
                        for i, cls in enumerate(test_pred_class):
                            test_probs[i, int(cls)] = 1.0
                            
                    calibrated_probs = calibrator.predict_proba(test_probs)
                except:
                    continue
                
                # 임계값 최적화
                threshold_finder = OptimalThresholdFinder()
                try:
                    optimal_thresholds = threshold_finder.optimize_thresholds(
                        cal_predictions, y_train_outer.values
                    )
                except:
                    optimal_thresholds = np.array([0.33, 0.33, 0.34])
                
                # 최종 예측 및 점수 계산
                final_predictions = np.argmax(
                    calibrated_probs - optimal_thresholds.reshape(1, -1), 
                    axis=1
                )
                
                score = accuracy_score(y_test_outer, final_predictions)
                repeat_scores.append(score)
            
            if repeat_scores:
                all_scores.extend(repeat_scores)
        
        if not all_scores:
            return self._get_default_results()
        
        scores_array = np.array(all_scores)
        
        return {
            'mean_accuracy': np.mean(scores_array),
            'std_accuracy': np.std(scores_array),
            'confidence_interval': self._bootstrap_ci(scores_array),
            'all_scores': scores_array
        }
    
    def _create_temporal_splits(self, X, timestamps):
        """시간적 누수 방지 분할"""
        if timestamps is None or len(timestamps) == 0:
            # 시간 정보가 없으면 순서 기반 분할
            indices = np.arange(len(X))
        else:
            indices = np.argsort(timestamps)
        
        n_samples = len(indices)
        gap_size = int(self.gap_ratio * n_samples)
        
        splits = []
        for i in range(self.n_outer_splits):
            train_size = int(0.6 * n_samples) + i * gap_size // 2
            
            if train_size >= n_samples - gap_size - 100:
                break
                
            train_end = min(train_size, n_samples - gap_size - 100)
            val_start = train_end + gap_size
            val_end = min(val_start + gap_size, n_samples)
            
            if val_end - val_start < 100:
                break
                
            train_indices = indices[:train_end]
            val_indices = indices[val_start:val_end]
            
            splits.append((train_indices, val_indices))
        
        return splits
    
    def _bootstrap_ci(self, scores, confidence=0.95, n_bootstrap=5000):
        """부트스트랩 신뢰구간"""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(
                scores, size=len(scores), replace=True
            )
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def _get_default_results(self):
        """기본 결과"""
        return {
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'confidence_interval': (0.0, 0.0),
            'all_scores': np.array([0.0])
        }

class LeakageDetector:
    """시간적 누수 탐지"""
    
    def __init__(self):
        self.leakage_indicators = {}
    
    def detect_temporal_leakage(self, X, y, model, timestamps):
        """시간적 누수 통계적 검증"""
        if timestamps is None or len(timestamps) == 0:
            return {'has_leakage': False, 'leakage_severity': 0.0}
            
        sorted_indices = np.argsort(timestamps)
        n_samples = len(sorted_indices)
        zero_point = int(0.7 * n_samples)
        
        # 정상 훈련
        try:
            model.fit(X, y)
            normal_predictions = model.predict(X)
        except:
            return {'has_leakage': False, 'leakage_severity': 0.0}
        
        # 제로 수정된 타겟으로 훈련
        y_modified = y.copy()
        y_modified[sorted_indices[zero_point:]] = 0
        
        try:
            model.fit(X, y_modified)
            zero_predictions = model.predict(X)
        except:
            return {'has_leakage': False, 'leakage_severity': 0.0}
        
        # 조기 분기점 탐지
        divergence_scores = []
        for i in range(zero_point, n_samples):
            idx = sorted_indices[i]
            if normal_predictions[idx] != zero_predictions[idx]:
                divergence_scores.append(i - zero_point)
        
        # 누수 판정
        early_divergence = len(divergence_scores) > 0 and min(divergence_scores) < 100
        
        return {
            'has_leakage': early_divergence,
            'divergence_point': min(divergence_scores) if divergence_scores else None,
            'leakage_severity': len(divergence_scores) / (n_samples - zero_point) if n_samples > zero_point else 0.0
        }

class ValidationSystem:
    def __init__(self):
        self.validation_results = {}
        self.gap_ratio = 0.04
        
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
        """시간적 누수 확인"""
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
                    
                    if overlap_ratio > 0.02:
                        issues.append(f"temporal_overlap: {overlap_ratio:.3f}")
        
        return len(issues) == 0, issues
    
    def create_validation_model(self, X_train, y_train):
        """검증용 모델 생성"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        # 클래스 가중치 계산
        class_counts = np.bincount(y_train_clean.astype(int), minlength=3)
        total_samples = len(y_train_clean)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (3 * count)
            else:
                class_weights[i] = 1.0
        
        # 클래스 불균형 보정
        class_weights[1] *= 1.2
        class_weights[2] *= 1.05
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features=0.8,
            bootstrap=True,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_clean, y_train_clean)
        return model
    
    def gap_walk_forward_cv(self, X, y, n_splits=6):
        """갭이 있는 워크포워드 교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 기반 분할
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            total_samples = len(sorted_indices)
            gap_size = int(total_samples * self.gap_ratio)
            
            min_fold_size = total_samples // (n_splits * 2.5)
            if min_fold_size < 300:
                return self.standard_cv(X, y, n_splits)
            
            fold_scores = []
            temporal_col_idx = list(X.columns).index('temporal_id')
            
            for fold in range(n_splits):
                window_size = int(min_fold_size + (fold * min_fold_size // 6))
                
                train_start = int(fold * min_fold_size)
                train_end = int(train_start + window_size)
                
                val_start = int(train_end + gap_size)
                val_end = int(val_start + min_fold_size)
                
                if val_end > total_samples:
                    break
                
                train_idx = sorted_indices[train_start:train_end]
                val_idx = sorted_indices[val_start:val_end]
                
                if len(train_idx) < 100 or len(val_idx) < 50:
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
        """표준 교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 컬럼 제거
        if 'temporal_id' in X.columns:
            temporal_col_idx = list(X.columns).index('temporal_id')
            X_for_cv = np.delete(X_clean, temporal_col_idx, axis=1)
        else:
            X_for_cv = X_clean
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_for_cv, y_clean)):
            model = self.create_validation_model(X_for_cv[train_idx], y_clean[train_idx])
            y_pred = model.predict(X_for_cv[val_idx])
            
            accuracy = accuracy_score(y_clean[val_idx], y_pred)
            fold_scores.append(accuracy)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_type': 'stratified_kfold'
        }
    
    def repeated_cv(self, X, y, n_repeats=3, n_splits=4):
        """반복 교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        if 'temporal_id' in X.columns:
            temporal_col_idx = list(X.columns).index('temporal_id')
            X_for_cv = np.delete(X_clean, temporal_col_idx, axis=1)
        else:
            X_for_cv = X_clean
        
        all_scores = []
        
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
            
            for train_idx, val_idx in skf.split(X_for_cv, y_clean):
                model = self.create_validation_model(X_for_cv[train_idx], y_clean[train_idx])
                y_pred = model.predict(X_for_cv[val_idx])
                
                accuracy = accuracy_score(y_clean[val_idx], y_pred)
                all_scores.append(accuracy)
        
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        return {
            'scores': all_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_type': 'repeated_stratified_kfold'
        }
    
    def holdout_validation(self, X_train, y_train, X_val, y_val):
        """홀드아웃 검증"""
        if any(data is None for data in [X_train, y_train, X_val, y_val]):
            return self.get_default_holdout_results()
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = self.create_validation_model(X_train_clean, y_train_clean)
        y_pred = model.predict(X_val_clean)
        
        try:
            y_pred_proba = model.predict_proba(X_val_clean)
        except:
            y_pred_proba = np.zeros((len(y_pred), 3))
            for i, cls in enumerate(y_pred):
                y_pred_proba[i, int(cls)] = 1.0
        
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
        low_confidence_ratio = np.mean(confidence_scores < 0.5)
        
        # 혼동 행렬
        cm = confusion_matrix(y_val_clean, y_pred, labels=[0, 1, 2])
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'class_scores': class_scores,
            'avg_confidence': avg_confidence,
            'low_confidence_ratio': low_confidence_ratio,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(y_val_clean)
        }
    
    def stability_test(self, X, y, n_runs=6):
        """안정성 테스트"""
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
        
        for run in range(n_runs):
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_clean, 
                    test_size=0.25, 
                    random_state=run * 7, 
                    stratify=y_clean
                )
                
                model = self.create_validation_model(X_train, y_train)
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='macro')
                
                # 클래스 균형 점수
                class_f1_scores = f1_score(y_val, y_pred, average=None)
                class_balance = 1 - np.std(class_f1_scores) if len(class_f1_scores) > 1 else 0
                
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                class_balance_scores.append(class_balance)
                
            except Exception:
                continue
        
        if len(accuracy_scores) < 3:
            return self.get_default_stability_results()
        
        # 통계 계산
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_balance = np.mean(class_balance_scores)
        
        # 안정성 점수
        accuracy_stability = max(0, 1 - (std_accuracy / mean_accuracy)) if mean_accuracy > 0 else 0
        f1_stability = max(0, 1 - (std_f1 / mean_f1)) if mean_f1 > 0 else 0
        overall_stability = (accuracy_stability + f1_stability + mean_balance) / 3
        
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
            'overall_stability': overall_stability,
            'n_runs': len(accuracy_scores)
        }
    
    def feature_importance_validation(self, X_train, y_train):
        """피처 중요도 검증"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        model = self.create_validation_model(X_train_clean, y_train_clean)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 중요도 통계
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            max_importance = np.max(importances)
            
            # 상위 피처 비율
            top_10_ratio = np.sum(np.sort(importances)[-10:]) / np.sum(importances) if len(importances) >= 10 else 1.0
            top_5_ratio = np.sum(np.sort(importances)[-5:]) / np.sum(importances) if len(importances) >= 5 else 1.0
            
            # 중요도 집중도
            concentration = 1 - (np.sum(importances > mean_importance) / len(importances))
            
            return {
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'max_importance': max_importance,
                'top_10_ratio': top_10_ratio,
                'top_5_ratio': top_5_ratio,
                'concentration': concentration,
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
            'low_confidence_ratio': 1.0,
            'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
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
            'overall_stability': 0.0,
            'n_runs': 0
        }
    
    def validate_system(self, X_train, y_train, X_val=None, y_val=None):
        """검증 시스템 실행"""
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
        repeated_cv_results = self.repeated_cv(X_train, y_train, n_repeats=2, n_splits=4)
        
        # 안정성 테스트
        stability_results = self.stability_test(X_train, y_train)
        
        # 피처 중요도 검증
        feature_results = self.feature_importance_validation(X_train, y_train)
        
        # 종합 점수 계산
        holdout_score = holdout_results.get('accuracy', 0.0)
        cv_score = cv_results.get('mean_score', 0.0)
        repeated_cv_score = repeated_cv_results.get('mean_score', 0.0)
        stability_score = stability_results.get('overall_stability', 0.0)
        
        # 가중 평균으로 종합 점수
        overall_score = (
            holdout_score * 0.25 +
            cv_score * 0.30 +
            repeated_cv_score * 0.20 +
            stability_score * 0.25
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
                'stability_score': stability_score
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
                'stability_score': 0.0
            }
        }

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        validator = ValidationSystem()
        leakage_free, issues = validator.check_temporal_leakage(train_df, test_df)
        
        return validator
        
    except Exception as e:
        return None

if __name__ == "__main__":
    main()