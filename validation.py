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
        self.gap_ratio = 0.005
        
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
                    
                    if overlap_ratio > 0.008:  # 더 엄격한 기준
                        issues.append(f"temporal_overlap: {overlap_ratio:.3f}")
        
        return len(issues) == 0, issues
    
    def create_validation_model(self, X_train, y_train):
        """검증용 모델"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        # 실제 분포 기반 클래스 가중치
        class_counts = np.bincount(y_train_clean.astype(int))
        total_samples = len(y_train_clean)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
        
        # 실제 분포에 맞춘 조정
        class_weights[0] *= 0.90  # 클래스 0이 가장 많으므로 가중치 감소
        class_weights[1] *= 1.12  # 클래스 1 가중치 증가
        class_weights[2] *= 1.15  # 클래스 2 가중치 약간 더 증가
        
        # Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features=0.7,
            class_weight=class_weights,
            random_state=42,
            n_jobs=1,
            bootstrap=True
        )
        
        model.fit(X_train_clean, y_train_clean)
        return model
    
    def cross_validation_stratified(self, X, y, n_splits=5):
        """교차검증"""
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
                model = self.create_validation_model(X_for_cv[train_idx], y_clean[train_idx])
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
            'cv_type': 'stratified_kfold',
            'n_folds': len(fold_scores)
        }
    
    def holdout_validation(self, X_train, y_train, X_val, y_val):
        """홀드아웃 검증"""
        if any(data is None for data in [X_train, y_train, X_val, y_val]):
            return self.get_default_holdout_results()
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        try:
            model = self.create_validation_model(X_train_clean, y_train_clean)
            y_pred = model.predict(X_val_clean)
            
            accuracy = accuracy_score(y_val_clean, y_pred)
            f1_macro = f1_score(y_val_clean, y_pred, average='macro')
            f1_weighted = f1_score(y_val_clean, y_pred, average='weighted')
            
            # 클래스별 성능
            class_scores = {}
            for cls in range(3):
                mask = y_val_clean == cls
                if mask.sum() > 0:
                    class_pred = y_pred[mask]
                    class_actual = y_val_clean[mask]
                    
                    class_acc = accuracy_score(class_actual, class_pred)
                    class_f1 = f1_score(y_val_clean == cls, y_pred == cls)
                    
                    # 클래스별 재현율과 정밀도
                    true_positive = np.sum((y_pred == cls) & (y_val_clean == cls))
                    predicted_positive = np.sum(y_pred == cls)
                    actual_positive = np.sum(y_val_clean == cls)
                    
                    precision = true_positive / predicted_positive if predicted_positive > 0 else 0
                    recall = true_positive / actual_positive if actual_positive > 0 else 0
                    
                    class_scores[cls] = {
                        'accuracy': class_acc, 
                        'f1': class_f1,
                        'precision': precision,
                        'recall': recall
                    }
                else:
                    class_scores[cls] = {
                        'accuracy': 0.0, 
                        'f1': 0.0,
                        'precision': 0.0,
                        'recall': 0.0
                    }
            
            # 예측 분포 확인
            pred_dist = np.bincount(y_pred, minlength=3) / len(y_pred)
            actual_dist = np.bincount(y_val_clean, minlength=3) / len(y_val_clean)
            
            # 분포 일치도
            dist_similarity = 1 - np.sum(np.abs(pred_dist - actual_dist)) / 2
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'class_scores': class_scores,
                'total_samples': len(y_val_clean),
                'pred_distribution': pred_dist.tolist(),
                'actual_distribution': actual_dist.tolist(),
                'distribution_similarity': dist_similarity
            }
            
        except Exception:
            return self.get_default_holdout_results()
    
    def stability_test(self, X, y, n_runs=7):
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
        class_0_recalls = []
        
        for run in range(n_runs):
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_clean, 
                    test_size=0.30,
                    random_state=run * 17, 
                    stratify=y_clean
                )
                
                model = self.create_validation_model(X_train, y_train)
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='macro')
                
                # 클래스 0 재현율 (가장 중요한 클래스)
                class_0_recall = np.sum((y_pred == 0) & (y_val == 0)) / np.sum(y_val == 0) if np.sum(y_val == 0) > 0 else 0
                
                accuracy_scores.append(accuracy)
                f1_scores.append(f1)
                class_0_recalls.append(class_0_recall)
                
            except Exception:
                continue
        
        if len(accuracy_scores) < 3:
            return self.get_default_stability_results()
        
        # 통계 계산
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_class_0_recall = np.mean(class_0_recalls)
        std_class_0_recall = np.std(class_0_recalls)
        
        # 안정성 점수 (변동계수 기반)
        accuracy_cv = std_accuracy / mean_accuracy if mean_accuracy > 0 else 1
        f1_cv = std_f1 / mean_f1 if mean_f1 > 0 else 1
        class_0_cv = std_class_0_recall / mean_class_0_recall if mean_class_0_recall > 0 else 1
        
        accuracy_stability = max(0, 1 - accuracy_cv * 2)
        f1_stability = max(0, 1 - f1_cv * 2)
        class_0_stability = max(0, 1 - class_0_cv * 2)
        
        # 종합 안정성 (클래스 0 성능에 더 높은 가중치)
        overall_stability = (0.4 * accuracy_stability + 
                           0.3 * f1_stability + 
                           0.3 * class_0_stability)
        
        return {
            'accuracy_scores': accuracy_scores,
            'f1_scores': f1_scores,
            'class_0_recalls': class_0_recalls,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'mean_class_0_recall': mean_class_0_recall,
            'std_class_0_recall': std_class_0_recall,
            'accuracy_stability': accuracy_stability,
            'f1_stability': f1_stability,
            'class_0_stability': class_0_stability,
            'overall_stability': overall_stability,
            'n_runs': len(accuracy_scores)
        }
    
    def feature_importance_validation(self, X_train, y_train):
        """피처 중요도 검증"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        
        try:
            model = self.create_validation_model(X_train_clean, y_train_clean)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # 중요도 통계
                mean_importance = np.mean(importances)
                std_importance = np.std(importances)
                max_importance = np.max(importances)
                
                # 상위 피처 비율
                sorted_imp = np.sort(importances)[::-1]
                top_5_ratio = np.sum(sorted_imp[:5]) / np.sum(importances) if len(sorted_imp) >= 5 and np.sum(importances) > 0 else 1.0
                top_10_ratio = np.sum(sorted_imp[:10]) / np.sum(importances) if len(sorted_imp) >= 10 and np.sum(importances) > 0 else 1.0
                
                # 피처 집중도 (너무 집중되면 과적합 위험)
                concentration_score = 1 - top_5_ratio if top_5_ratio < 0.8 else 0.5
                
                return {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'max_importance': max_importance,
                    'top_5_ratio': top_5_ratio,
                    'top_10_ratio': top_10_ratio,
                    'concentration_score': concentration_score,
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
            'class_scores': {0: {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}, 
                           1: {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}, 
                           2: {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}},
            'total_samples': 0,
            'pred_distribution': [0.33, 0.33, 0.34],
            'actual_distribution': [0.33, 0.33, 0.34],
            'distribution_similarity': 1.0
        }
    
    def get_default_stability_results(self):
        """기본 안정성 결과"""
        return {
            'accuracy_scores': [0.0],
            'f1_scores': [0.0],
            'class_0_recalls': [0.0],
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'mean_f1': 0.0,
            'std_f1': 0.0,
            'mean_class_0_recall': 0.0,
            'std_class_0_recall': 0.0,
            'accuracy_stability': 0.0,
            'f1_stability': 0.0,
            'class_0_stability': 0.0,
            'overall_stability': 0.0,
            'n_runs': 0
        }
    
    def validate_system(self, X_train, y_train, X_val=None, y_val=None):
        """검증 시스템"""
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
                    X_clean, y_clean, test_size=0.25, random_state=42, stratify=y_clean
                )
            except:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_clean, y_clean, test_size=0.25, random_state=42
                )
            holdout_results = self.holdout_validation(X_train_split, y_train_split, X_val_split, y_val_split)
        
        # 교차검증
        cv_results = self.cross_validation_stratified(X_train, y_train)
        
        # 안정성 테스트
        stability_results = self.stability_test(X_train, y_train)
        
        # 피처 중요도 검증
        feature_results = self.feature_importance_validation(X_train, y_train)
        
        # 종합 점수 계산
        holdout_score = holdout_results.get('accuracy', 0.0)
        cv_score = cv_results.get('mean_score', 0.0)
        stability_score = stability_results.get('overall_stability', 0.0)
        
        # 클래스 0 성능 보너스
        class_0_recall = holdout_results.get('class_scores', {}).get(0, {}).get('recall', 0.0)
        class_0_bonus = min(0.05, class_0_recall * 0.1)
        
        # 분포 일치도 보너스
        dist_similarity = holdout_results.get('distribution_similarity', 0.0)
        dist_bonus = min(0.03, (dist_similarity - 0.8) * 0.15) if dist_similarity > 0.8 else 0
        
        # 가중 평균으로 종합 점수
        base_score = (
            holdout_score * 0.40 +
            cv_score * 0.35 +
            stability_score * 0.25
        )
        
        overall_score = base_score + class_0_bonus + dist_bonus
        
        # 보수적 조정 (실제 테스트에서는 약간 낮을 수 있음)
        conservative_penalty = 0.92
        overall_score = overall_score * conservative_penalty
        
        self.validation_results = {
            'holdout': holdout_results,
            'cross_validation': cv_results,
            'stability': stability_results,
            'feature_importance': feature_results,
            'overall_score': overall_score,
            'component_scores': {
                'holdout_score': holdout_score,
                'cv_ensemble_score': cv_score,
                'stability_score': stability_score,
                'class_0_bonus': class_0_bonus,
                'distribution_bonus': dist_bonus
            },
            'conservative_adjustment': True,
            'penalty_applied': conservative_penalty
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
                'stability_score': 0.0,
                'class_0_bonus': 0.0,
                'distribution_bonus': 0.0
            },
            'conservative_adjustment': True,
            'penalty_applied': 0.92
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