# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp, chi2_contingency
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ValidationSystem:
    def __init__(self):
        self.validation_results = {}
        self.gap_ratio = 0.006  # 축소된 갭 (0.6%)
        self.model_cache = {}
        
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
                    
                    if overlap_ratio > 0.03:  # 관대한 기준 (3%)
                        issues.append(f"temporal_overlap: {overlap_ratio:.3f}")
        
        return len(issues) == 0, issues
    
    def create_validation_model_ensemble(self, X_train, y_train):
        """검증용 모델 앙상블 생성"""
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
        class_weights[1] *= 1.20
        class_weights[2] *= 1.12
        
        # 다양한 모델 구성
        models = []
        
        # Random Forest - 기본
        rf_basic = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.75,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        models.append(('rf_basic', rf_basic))
        
        # Random Forest - 깊은 모델
        rf_deep = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight=class_weights,
            random_state=43,
            n_jobs=-1
        )
        models.append(('rf_deep', rf_deep))
        
        # Random Forest - 넓은 모델
        rf_wide = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features=0.7,
            class_weight=class_weights,
            random_state=44,
            n_jobs=-1
        )
        models.append(('rf_wide', rf_wide))
        
        # 모델 학습
        trained_models = []
        for name, model in models:
            try:
                model.fit(X_train_clean, y_train_clean)
                trained_models.append((name, model))
            except Exception:
                continue
        
        return trained_models
    
    def advanced_cross_validation(self, X, y, n_splits=5):
        """교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 기반 분할
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            
            # 시간 기반 그룹 생성
            n_groups = min(n_splits * 2, len(np.unique(temporal_ids)))
            if n_groups >= n_splits:
                return self.temporal_group_cv(X, y, n_groups, n_splits)
            else:
                return self.standard_cv(X, y, n_splits)
        else:
            return self.standard_cv(X, y, n_splits)
    
    def temporal_group_cv(self, X, y, n_groups, n_splits):
        """시간 기반 그룹 교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        temporal_ids = X['temporal_id'].values
        temporal_col_idx = list(X.columns).index('temporal_id')
        
        # 시간 기반 그룹 할당
        sorted_indices = np.argsort(temporal_ids)
        group_size = len(sorted_indices) // n_groups
        groups = np.zeros(len(sorted_indices), dtype=int)
        
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < n_groups - 1 else len(sorted_indices)
            groups[sorted_indices[start_idx:end_idx]] = i
        
        # GroupKFold 사용
        group_kfold = GroupKFold(n_splits=n_splits)
        fold_scores = []
        
        X_for_cv = np.delete(X_clean, temporal_col_idx, axis=1)
        
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_for_cv, y_clean, groups)):
            if len(train_idx) < 100 or len(val_idx) < 50:
                continue
            
            # 앙상블 모델 학습
            models = self.create_validation_model_ensemble(X_for_cv[train_idx], y_clean[train_idx])
            
            if not models:
                continue
            
            # 앙상블 예측
            ensemble_predictions = []
            for name, model in models:
                try:
                    y_pred = model.predict(X_for_cv[val_idx])
                    ensemble_predictions.append(y_pred)
                except Exception:
                    continue
            
            if ensemble_predictions:
                # 다수결 투표
                ensemble_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
                accuracy = accuracy_score(y_clean[val_idx], ensemble_pred)
                fold_scores.append(accuracy)
        
        if len(fold_scores) >= 3:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            return {
                'fold_scores': fold_scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_type': 'temporal_group'
            }
        else:
            return self.standard_cv(X, y, n_splits)
    
    def gap_walk_forward_cv(self, X, y, n_splits=5):
        """갭이 있는 워크포워드 교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        # temporal_id 기반 분할
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            total_samples = len(sorted_indices)
            gap_size = int(total_samples * self.gap_ratio)  # 0.6%
            
            # 최소 폴드 크기 확보
            min_fold_size = total_samples // (n_splits * 2.5)
            if min_fold_size < 500:
                return self.standard_cv(X, y, n_splits)
            
            fold_scores = []
            temporal_col_idx = list(X.columns).index('temporal_id')
            
            for fold in range(n_splits):
                # 동적 윈도우 크기
                window_size = int(min_fold_size + (fold * min_fold_size // 4))
                
                train_start = int(fold * min_fold_size)
                train_end = int(train_start + window_size)
                
                val_start = int(train_end + gap_size)
                val_end = int(val_start + min_fold_size)
                
                if val_end > total_samples:
                    break
                
                train_idx = sorted_indices[train_start:train_end]
                val_idx = sorted_indices[val_start:val_end]
                
                if len(train_idx) < 200 or len(val_idx) < 100:
                    continue
                
                X_train_fold = np.delete(X_clean[train_idx], temporal_col_idx, axis=1)
                y_train_fold = y_clean[train_idx]
                X_val_fold = np.delete(X_clean[val_idx], temporal_col_idx, axis=1)
                y_val_fold = y_clean[val_idx]
                
                # 앙상블 모델 학습
                models = self.create_validation_model_ensemble(X_train_fold, y_train_fold)
                
                if not models:
                    continue
                
                # 앙상블 예측
                ensemble_predictions = []
                for name, model in models:
                    try:
                        y_pred = model.predict(X_val_fold)
                        ensemble_predictions.append(y_pred)
                    except Exception:
                        continue
                
                if ensemble_predictions:
                    # 가중 투표 (깊은 모델에 더 높은 가중치)
                    weights = [0.4, 0.35, 0.25]  # rf_basic, rf_deep, rf_wide
                    if len(ensemble_predictions) >= 3:
                        weighted_pred = np.zeros(len(ensemble_predictions[0]))
                        for i, pred in enumerate(ensemble_predictions[:3]):
                            weighted_pred += weights[i] * pred
                        final_pred = np.round(weighted_pred).astype(int)
                    else:
                        final_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
                    
                    accuracy = accuracy_score(y_val_fold, final_pred)
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
            # 앙상블 모델 학습
            models = self.create_validation_model_ensemble(X_for_cv[train_idx], y_clean[train_idx])
            
            if not models:
                continue
            
            # 앙상블 예측
            ensemble_predictions = []
            for name, model in models:
                try:
                    y_pred = model.predict(X_for_cv[val_idx])
                    ensemble_predictions.append(y_pred)
                except Exception:
                    continue
            
            if ensemble_predictions:
                # 다수결 투표
                ensemble_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
                accuracy = accuracy_score(y_clean[val_idx], ensemble_pred)
                fold_scores.append(accuracy)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_type': 'stratified_kfold'
        }
    
    def repeated_cv(self, X, y, n_repeats=3, n_splits=5):
        """반복 교차검증"""
        X_clean, y_clean = self.safe_data_conversion(X, y)
        
        if 'temporal_id' in X.columns:
            temporal_col_idx = list(X.columns).index('temporal_id')
            X_for_cv = np.delete(X_clean, temporal_col_idx, axis=1)
        else:
            X_for_cv = X_clean
        
        all_scores = []
        
        for repeat in range(n_repeats):
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=42 + repeat)
            
            for train_idx, val_idx in rskf.split(X_for_cv, y_clean):
                # 앙상블 모델 학습
                models = self.create_validation_model_ensemble(X_for_cv[train_idx], y_clean[train_idx])
                
                if not models:
                    continue
                
                # 앙상블 예측
                ensemble_predictions = []
                for name, model in models:
                    try:
                        y_pred = model.predict(X_for_cv[val_idx])
                        ensemble_predictions.append(y_pred)
                    except Exception:
                        continue
                
                if ensemble_predictions:
                    # 다수결 투표
                    ensemble_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
                    accuracy = accuracy_score(y_clean[val_idx], ensemble_pred)
                    all_scores.append(accuracy)
        
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        return {
            'scores': all_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_type': 'repeated_stratified_kfold'
        }
    
    def holdout_validation_comprehensive(self, X_train, y_train, X_val, y_val):
        """홀드아웃 검증"""
        if any(data is None for data in [X_train, y_train, X_val, y_val]):
            return self.get_default_holdout_results()
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 앙상블 모델 학습
        models = self.create_validation_model_ensemble(X_train_clean, y_train_clean)
        
        if not models:
            return self.get_default_holdout_results()
        
        # 개별 모델 성능 평가
        model_scores = {}
        ensemble_predictions = []
        ensemble_probabilities = []
        
        for name, model in models:
            try:
                y_pred = model.predict(X_val_clean)
                y_pred_proba = model.predict_proba(X_val_clean) if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_val_clean, y_pred)
                f1_macro = f1_score(y_val_clean, y_pred, average='macro')
                f1_weighted = f1_score(y_val_clean, y_pred, average='weighted')
                
                model_scores[name] = {
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted
                }
                
                ensemble_predictions.append(y_pred)
                if y_pred_proba is not None:
                    ensemble_probabilities.append(y_pred_proba)
                    
            except Exception:
                continue
        
        if not ensemble_predictions:
            return self.get_default_holdout_results()
        
        # 앙상블 예측
        if len(ensemble_predictions) >= 3:
            # 가중 투표
            weights = [0.4, 0.35, 0.25]
            weighted_pred = np.zeros(len(ensemble_predictions[0]))
            for i, pred in enumerate(ensemble_predictions[:3]):
                weighted_pred += weights[i] * pred
            final_pred = np.round(weighted_pred).astype(int)
        else:
            # 다수결 투표
            final_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
        
        # 앙상블 성능
        ensemble_accuracy = accuracy_score(y_val_clean, final_pred)
        ensemble_f1_macro = f1_score(y_val_clean, final_pred, average='macro')
        ensemble_f1_weighted = f1_score(y_val_clean, final_pred, average='weighted')
        
        # 클래스별 성능
        class_scores = {}
        for cls in range(3):
            mask = y_val_clean == cls
            if mask.sum() > 0:
                class_acc = accuracy_score(y_val_clean[mask], final_pred[mask])
                class_f1 = f1_score(y_val_clean == cls, final_pred == cls)
                class_scores[cls] = {'accuracy': class_acc, 'f1': class_f1}
            else:
                class_scores[cls] = {'accuracy': 0.0, 'f1': 0.0}
        
        # 예측 신뢰도 분석
        if ensemble_probabilities:
            avg_proba = np.mean(ensemble_probabilities, axis=0)
            confidence_scores = np.max(avg_proba, axis=1)
            avg_confidence = np.mean(confidence_scores)
            low_confidence_ratio = np.mean(confidence_scores < 0.5)
            
            # 보정 곡선 분석
            try:
                calibration_scores = {}
                for cls in range(3):
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_val_clean == cls, avg_proba[:, cls], n_bins=10
                    )
                    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                    calibration_scores[cls] = calibration_error
            except:
                calibration_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        else:
            avg_confidence = 0.5
            low_confidence_ratio = 1.0
            calibration_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        
        # 혼동 행렬
        cm = confusion_matrix(y_val_clean, final_pred)
        
        # 다양성 점수 (모델간 불일치율)
        if len(ensemble_predictions) >= 2:
            diversity_scores = []
            for i in range(len(ensemble_predictions)):
                for j in range(i + 1, len(ensemble_predictions)):
                    disagreement = np.mean(ensemble_predictions[i] != ensemble_predictions[j])
                    diversity_scores.append(disagreement)
            avg_diversity = np.mean(diversity_scores)
        else:
            avg_diversity = 0.0
        
        return {
            'accuracy': ensemble_accuracy,
            'f1_macro': ensemble_f1_macro,
            'f1_weighted': ensemble_f1_weighted,
            'class_scores': class_scores,
            'model_scores': model_scores,
            'avg_confidence': avg_confidence,
            'low_confidence_ratio': low_confidence_ratio,
            'calibration_scores': calibration_scores,
            'avg_calibration_error': np.mean(list(calibration_scores.values())),
            'confusion_matrix': cm.tolist(),
            'diversity_score': avg_diversity,
            'total_samples': len(y_val_clean)
        }
    
    def stability_test_comprehensive(self, X, y, n_runs=15):
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
        prediction_variances = []
        
        for run in range(n_runs):
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_clean, 
                    test_size=0.25, 
                    random_state=run * 7, 
                    stratify=y_clean
                )
                
                # 앙상블 모델 학습
                models = self.create_validation_model_ensemble(X_train, y_train)
                
                if not models:
                    continue
                
                # 앙상블 예측
                ensemble_predictions = []
                for name, model in models:
                    try:
                        y_pred = model.predict(X_val)
                        ensemble_predictions.append(y_pred)
                    except Exception:
                        continue
                
                if not ensemble_predictions:
                    continue
                
                # 다수결 투표
                final_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
                
                accuracy = accuracy_score(y_val, final_pred)
                f1 = f1_score(y_val, final_pred, average='macro')
                
                # 클래스 균형 점수
                class_f1_scores = f1_score(y_val, final_pred, average=None)
                class_balance = 1 - np.std(class_f1_scores) if len(class_f1_scores) > 1 else 0
                
                # 예측 분산
                pred_variance = np.var(ensemble_predictions, axis=0).mean()
                
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                class_balance_scores.append(class_balance)
                prediction_variances.append(pred_variance)
                
            except Exception:
                continue
        
        if len(accuracy_scores) < 8:
            return self.get_default_stability_results()
        
        # 통계 계산
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_balance = np.mean(class_balance_scores)
        mean_pred_variance = np.mean(prediction_variances)
        
        # 안정성 점수
        accuracy_stability = max(0, 1 - (std_accuracy / mean_accuracy)) if mean_accuracy > 0 else 0
        f1_stability = max(0, 1 - (std_f1 / mean_f1)) if mean_f1 > 0 else 0
        prediction_stability = max(0, 1 - mean_pred_variance) if mean_pred_variance <= 1 else 0
        
        overall_stability = (accuracy_stability * 0.4 + 
                           f1_stability * 0.3 + 
                           mean_balance * 0.2 + 
                           prediction_stability * 0.1)
        
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
            'prediction_stability': prediction_stability,
            'prediction_variance': mean_pred_variance,
            'overall_stability': overall_stability,
            'n_runs': len(accuracy_scores)
        }
    
    def feature_importance_validation(self, X_train, y_train):
        """피처 중요도 검증"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        # 앙상블 모델로 피처 중요도 계산
        models = self.create_validation_model_ensemble(X_train_clean, y_train_clean)
        
        if not models:
            return {}
        
        importance_arrays = []
        
        for name, model in models:
            if hasattr(model, 'feature_importances_'):
                importance_arrays.append(model.feature_importances_)
        
        if not importance_arrays:
            return {}
        
        # 평균 중요도
        avg_importances = np.mean(importance_arrays, axis=0)
        std_importances = np.std(importance_arrays, axis=0)
        
        # 중요도 통계
        mean_importance = np.mean(avg_importances)
        std_importance = np.std(avg_importances)
        max_importance = np.max(avg_importances)
        
        # 상위 피처 비율
        top_10_ratio = np.sum(np.sort(avg_importances)[-10:]) / np.sum(avg_importances)
        top_5_ratio = np.sum(np.sort(avg_importances)[-5:]) / np.sum(avg_importances)
        
        # 중요도 집중도
        concentration = 1 - (np.sum(avg_importances > mean_importance) / len(avg_importances))
        
        # 중요도 안정성
        stability = np.mean(1 - (std_importances / (avg_importances + 1e-8)))
        
        return {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'max_importance': max_importance,
            'top_10_ratio': top_10_ratio,
            'top_5_ratio': top_5_ratio,
            'concentration': concentration,
            'stability': stability,
            'n_features': len(avg_importances),
            'n_models': len(importance_arrays)
        }
    
    def cross_validation_ensemble(self, X, y):
        """교차검증 앙상블"""
        # 시간 기반 교차검증
        temporal_cv = self.gap_walk_forward_cv(X, y, n_splits=5)
        
        # 표준 교차검증
        standard_cv = self.standard_cv(X, y, n_splits=5)
        
        # 반복 교차검증
        repeated_cv = self.repeated_cv(X, y, n_repeats=2, n_splits=5)
        
        # 그룹 기반 교차검증
        group_cv = self.advanced_cross_validation(X, y, n_splits=5)
        
        # 앙상블 점수 계산
        scores = []
        weights = []
        
        if temporal_cv['mean_score'] > 0:
            scores.append(temporal_cv['mean_score'])
            weights.append(0.35)  # 시간 기반 CV에 높은 가중치
        
        if standard_cv['mean_score'] > 0:
            scores.append(standard_cv['mean_score'])
            weights.append(0.25)
        
        if repeated_cv['mean_score'] > 0:
            scores.append(repeated_cv['mean_score'])
            weights.append(0.2)
        
        if group_cv['mean_score'] > 0:
            scores.append(group_cv['mean_score'])
            weights.append(0.2)
        
        # 가중 평균
        if scores:
            ensemble_score = np.average(scores, weights=weights[:len(scores)])
        else:
            ensemble_score = 0.0
        
        return {
            'temporal_cv': temporal_cv,
            'standard_cv': standard_cv,
            'repeated_cv': repeated_cv,
            'group_cv': group_cv,
            'ensemble_score': ensemble_score
        }
    
    def get_default_holdout_results(self):
        """기본 홀드아웃 결과"""
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'class_scores': {0: {'accuracy': 0.0, 'f1': 0.0}, 
                           1: {'accuracy': 0.0, 'f1': 0.0}, 
                           2: {'accuracy': 0.0, 'f1': 0.0}},
            'model_scores': {},
            'avg_confidence': 0.0,
            'low_confidence_ratio': 1.0,
            'calibration_scores': {0: 0.0, 1: 0.0, 2: 0.0},
            'avg_calibration_error': 0.0,
            'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'diversity_score': 0.0,
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
            'prediction_stability': 0.0,
            'prediction_variance': 0.0,
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
            holdout_results = self.holdout_validation_comprehensive(X_train, y_train, X_val, y_val)
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
            holdout_results = self.holdout_validation_comprehensive(X_train_split, y_train_split, X_val_split, y_val_split)
        
        # 교차검증 앙상블
        cv_ensemble_results = self.cross_validation_ensemble(X_train, y_train)
        
        # 안정성 테스트
        stability_results = self.stability_test_comprehensive(X_train, y_train)
        
        # 피처 중요도 검증
        feature_results = self.feature_importance_validation(X_train, y_train)
        
        # 종합 점수 계산
        holdout_score = holdout_results.get('accuracy', 0.0)
        cv_ensemble_score = cv_ensemble_results.get('ensemble_score', 0.0)
        stability_score = stability_results.get('overall_stability', 0.0)
        
        # 가중 평균으로 종합 점수 (홀드아웃과 CV 앙상블에 높은 가중치)
        overall_score = (
            holdout_score * 0.40 +
            cv_ensemble_score * 0.35 +
            stability_score * 0.25
        )
        
        self.validation_results = {
            'holdout': holdout_results,
            'cross_validation_ensemble': cv_ensemble_results,
            'stability': stability_results,
            'feature_importance': feature_results,
            'overall_score': overall_score,
            'component_scores': {
                'holdout_score': holdout_score,
                'cv_ensemble_score': cv_ensemble_score,
                'stability_score': stability_score
            }
        }
        
        return self.validation_results
    
    def get_comprehensive_default_results(self):
        """포괄적 기본 결과"""
        return {
            'holdout': self.get_default_holdout_results(),
            'cross_validation_ensemble': {
                'temporal_cv': {'mean_score': 0.0, 'std_score': 0.0, 'fold_scores': [], 'cv_type': 'none'},
                'standard_cv': {'mean_score': 0.0, 'std_score': 0.0, 'fold_scores': [], 'cv_type': 'none'},
                'repeated_cv': {'mean_score': 0.0, 'std_score': 0.0, 'scores': [], 'cv_type': 'none'},
                'group_cv': {'mean_score': 0.0, 'std_score': 0.0, 'fold_scores': [], 'cv_type': 'none'},
                'ensemble_score': 0.0
            },
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
        
        return validator
        
    except Exception as e:
        return None

if __name__ == "__main__":
    main()