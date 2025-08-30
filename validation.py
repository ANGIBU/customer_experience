# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

class ValidationSystem:
    def __init__(self):
        self.validation_results = {}
        self.leakage_issues = []
        
    def check_temporal_leakage(self, train_df, test_df):
        """시간적 누수 확인"""
        print("시간적 누수 확인")
        
        issues = []
        
        train_ids = set(train_df['ID']) if 'ID' in train_df.columns else set()
        test_ids = set(test_df['ID']) if 'ID' in test_df.columns else set()
        
        common_ids = train_ids & test_ids
        if common_ids:
            issues.append(f"공통 ID {len(common_ids)}개 발견")
        
        if train_ids and test_ids:
            try:
                def extract_temporal_numbers(id_set):
                    numbers = []
                    for id_val in id_set:
                        if '_' in str(id_val):
                            try:
                                num = int(str(id_val).split('_')[1])
                                numbers.append(num)
                            except:
                                continue
                    return numbers
                
                train_id_nums = extract_temporal_numbers(train_ids)
                test_id_nums = extract_temporal_numbers(test_ids)
                
                if train_id_nums and test_id_nums:
                    train_max = max(train_id_nums)
                    test_min = min(test_id_nums)
                    
                    if train_max >= test_min:
                        overlap_count = len([x for x in train_id_nums if x >= test_min])
                        overlap_ratio = overlap_count / len(train_id_nums)
                        
                        if overlap_ratio > 0.05:
                            issues.append(f"시간적 순서 위반: 겹침 비율 {overlap_ratio:.1%}")
                        else:
                            print("시간적 순서 정상 (경미한 겹침)")
                    else:
                        print("시간적 순서 정상")
                        
            except Exception as e:
                print(f"ID 시간 순서 분석 오류: {e}")
        
        if 'support_needs' in train_df.columns and 'after_interaction' in train_df.columns:
            try:
                correlation = train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                if abs(correlation) > 0.08:
                    issues.append(f"after_interaction 상관관계: {correlation:.3f}")
            except:
                pass
        
        self.leakage_issues = issues
        
        if issues:
            print("누수 위험:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("누수 위험 없음")
            return True
    
    def create_validation_model(self, X_train, y_train):
        """검증용 모델 생성"""
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
        
        class_weights[1] *= 1.3
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            max_features=0.8
        )
        
        X_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
        
        model.fit(X_clean, y_train)
        return model
    
    def temporal_cross_validation(self, X, y, n_splits=5):
        """시간 기반 교차 검증"""
        print(f"시간 기반 교차 검증 (K={n_splits})")
        
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            fold_size = len(sorted_indices) // (n_splits + 1)
            fold_scores = []
            all_predictions = []
            all_actuals = []
            
            for fold in range(n_splits):
                train_end = (fold + 1) * fold_size
                val_start = train_end
                val_end = val_start + fold_size
                
                if val_end > len(sorted_indices):
                    break
                
                train_idx = sorted_indices[:train_end]
                val_idx = sorted_indices[val_start:val_end]
                
                try:
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    feature_cols = [col for col in X_train_fold.columns if col != 'temporal_id']
                    
                    model = self.create_validation_model(X_train_fold[feature_cols], y_train_fold)
                    
                    X_val_clean = X_val_fold[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                    y_pred = model.predict(X_val_clean)
                    
                    accuracy = accuracy_score(y_val_fold, y_pred)
                    fold_scores.append(accuracy)
                    
                    all_predictions.extend(y_pred)
                    all_actuals.extend(y_val_fold)
                    
                    print(f"  시점 {fold + 1}: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  시점 {fold + 1} 오류: {e}")
                    fold_scores.append(0.0)
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_scores = []
            all_predictions = []
            all_actuals = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    model = self.create_validation_model(X_train_fold, y_train_fold)
                    
                    X_val_clean = X_val_fold.fillna(0).replace([np.inf, -np.inf], 0)
                    y_pred = model.predict(X_val_clean)
                    
                    accuracy = accuracy_score(y_val_fold, y_pred)
                    fold_scores.append(accuracy)
                    
                    all_predictions.extend(y_pred)
                    all_actuals.extend(y_val_fold)
                    
                    print(f"  시점 {fold + 1}: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  시점 {fold + 1} 오류: {e}")
                    fold_scores.append(0.0)
        
        mean_score = np.mean(fold_scores) if fold_scores else 0.0
        std_score = np.std(fold_scores) if fold_scores else 0.0
        
        print(f"시간 기반 평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
        
        class_performance = {}
        if all_predictions and all_actuals:
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    all_actuals, all_predictions, average=None, zero_division=0
                )
                
                print("클래스별 성능:")
                for i in range(len(precision)):
                    print(f"  클래스 {i}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")
                    class_performance[i] = {
                        'precision': precision[i],
                        'recall': recall[i],
                        'f1': f1[i]
                    }
                    
            except Exception as e:
                print(f"클래스별 성능 계산 오류: {e}")
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'class_performance': class_performance
        }
    
    def stratified_cross_validation(self, X, y, n_splits=5):
        """계층화 교차 검증"""
        print(f"계층화 교차 검증 (K={n_splits})")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_scores = []
        all_predictions = []
        all_actuals = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            try:
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                model = self.create_validation_model(X_train_fold, y_train_fold)
                
                X_val_clean = X_val_fold.fillna(0).replace([np.inf, -np.inf], 0)
                y_pred = model.predict(X_val_clean)
                
                accuracy = accuracy_score(y_val_fold, y_pred)
                fold_scores.append(accuracy)
                
                all_predictions.extend(y_pred)
                all_actuals.extend(y_val_fold)
                
                print(f"  Fold {fold + 1}: {accuracy:.4f}")
                
            except Exception as e:
                print(f"  Fold {fold + 1} 오류: {e}")
                fold_scores.append(0.0)
        
        mean_score = np.mean(fold_scores) if fold_scores else 0.0
        std_score = np.std(fold_scores) if fold_scores else 0.0
        
        print(f"평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
        
        class_performance = {}
        if all_predictions and all_actuals:
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    all_actuals, all_predictions, average=None, zero_division=0
                )
                
                print("클래스별 성능:")
                for i in range(len(precision)):
                    print(f"  클래스 {i}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")
                    class_performance[i] = {
                        'precision': precision[i],
                        'recall': recall[i],
                        'f1': f1[i]
                    }
                    
            except Exception as e:
                print(f"클래스별 성능 계산 오류: {e}")
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'class_performance': class_performance
        }
    
    def holdout_validation(self, X_train, y_train, X_val, y_val):
        """홀드아웃 검증"""
        print("홀드아웃 검증")
        
        try:
            model = self.create_validation_model(X_train, y_train)
            
            X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)
            y_pred = model.predict(X_val_clean)
            
            accuracy = accuracy_score(y_val, y_pred)
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
            
            print(f"홀드아웃 정확도: {accuracy:.4f}")
            
            class_accuracies = []
            for i in range(3):
                if cm.sum(axis=1)[i] > 0:
                    class_acc = cm[i, i] / cm.sum(axis=1)[i]
                else:
                    class_acc = 0.0
                class_accuracies.append(class_acc)
                print(f"  클래스 {i}: {class_acc:.3f}")
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'class_accuracies': class_accuracies
            }
            
        except Exception as e:
            print(f"홀드아웃 검증 오류: {e}")
            return {
                'accuracy': 0.0,
                'confusion_matrix': np.zeros((3, 3)),
                'class_accuracies': [0.0, 0.0, 0.0]
            }
    
    def stability_test(self, X, y, n_runs=10):
        """모델 안정성 테스트"""
        print(f"안정성 테스트 (실행 {n_runs}회)")
        
        scores = []
        
        for run in range(n_runs):
            try:
                if 'temporal_id' in X.columns:
                    X_temp = X.drop(['temporal_id'], axis=1)
                else:
                    X_temp = X
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y, test_size=0.2, random_state=run, stratify=y
                )
                
                model = self.create_validation_model(X_train, y_train)
                
                X_val_clean = X_val.fillna(0).replace([np.inf, -np.inf], 0)
                y_pred = model.predict(X_val_clean)
                
                accuracy = accuracy_score(y_val, y_pred)
                scores.append(accuracy)
                
            except Exception as e:
                print(f"  실행 {run + 1} 오류: {e}")
                scores.append(0.0)
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
        else:
            mean_score = std_score = min_score = max_score = 0.0
        
        print(f"평균: {mean_score:.4f}")
        print(f"표준편차: {std_score:.4f}")
        print(f"범위: {min_score:.4f} - {max_score:.4f}")
        
        stability_score = 1 - (std_score / mean_score) if mean_score > 0 else 0
        print(f"안정성 점수: {stability_score:.3f}")
        
        return {
            'scores': scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'stability_score': stability_score
        }
    
    def distribution_analysis(self, y_true, y_pred):
        """분포 분석"""
        print("예측 분포 분석")
        
        try:
            true_dist = pd.Series(y_true).value_counts(normalize=True).sort_index()
            pred_dist = pd.Series(y_pred).value_counts(normalize=True).sort_index()
            
            print("분포 비교:")
            total_diff = 0
            for cls in [0, 1, 2]:
                true_pct = true_dist.get(cls, 0) * 100
                pred_pct = pred_dist.get(cls, 0) * 100
                diff = abs(true_pct - pred_pct)
                total_diff += diff
                
                print(f"  클래스 {cls}: 실제 {true_pct:.1f}% vs 예측 {pred_pct:.1f}%")
            
            similarity_score = max(0, 100 - total_diff) / 100
            print(f"분포 유사성: {similarity_score:.3f}")
            
            return {
                'true_distribution': true_dist.to_dict(),
                'pred_distribution': pred_dist.to_dict(),
                'similarity_score': similarity_score
            }
            
        except Exception as e:
            print(f"분포 분석 오류: {e}")
            return {
                'true_distribution': {},
                'pred_distribution': {},
                'similarity_score': 0.0
            }
    
    def feature_stability_validation(self, train_df, test_df):
        """피처 안정성 검증"""
        print("피처 안정성 검증")
        
        numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns
                       if col not in ['ID', 'support_needs'] and col in test_df.columns]
        
        stability_results = {}
        unstable_features = []
        
        for col in numeric_cols:
            try:
                train_vals = train_df[col].dropna()
                test_vals = test_df[col].dropna()
                
                if len(train_vals) > 100 and len(test_vals) > 100:
                    statistic, p_value = ks_2samp(train_vals, test_vals)
                    
                    stability_score = 1 - statistic
                    stability_results[col] = {
                        'ks_statistic': statistic,
                        'p_value': p_value,
                        'stability_score': stability_score
                    }
                    
                    if stability_score < 0.9:
                        unstable_features.append(col)
                        print(f"  {col}: 불안정 (점수: {stability_score:.3f})")
                        
            except Exception as e:
                print(f"  {col} 안정성 분석 오류: {e}")
                continue
        
        print(f"불안정 피처: {len(unstable_features)}개")
        
        return stability_results, unstable_features
    
    def class_balance_validation(self, y_true, y_pred):
        """클래스 균형 검증"""
        print("클래스 균형 검증")
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        class_metrics = {}
        for i in range(3):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"  클래스 {i}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        return class_metrics
    
    def validate_system(self, X_train, y_train):
        """전체 검증 시스템"""
        print("검증 시스템 실행")
        print("=" * 40)
        
        try:
            X_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)
            
            if 'temporal_id' in X_clean.columns:
                feature_cols = [col for col in X_clean.columns if col != 'temporal_id']
                X_for_split = X_clean[feature_cols]
            else:
                X_for_split = X_clean
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_for_split, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            holdout_results = self.holdout_validation(
                X_train_split, y_train_split, X_val_split, y_val_split
            )
            
            stratified_results = self.stratified_cross_validation(X_clean, y_train)
            
            temporal_results = self.temporal_cross_validation(X_clean, y_train)
            
            stability_results = self.stability_test(X_clean, y_train)
            
            model = self.create_validation_model(X_train_split, y_train_split)
            X_val_clean = X_val_split.fillna(0).replace([np.inf, -np.inf], 0)
            y_pred = model.predict(X_val_clean)
            distribution_results = self.distribution_analysis(y_val_split, y_pred)
            
            class_balance_results = self.class_balance_validation(y_val_split, y_pred)
            
            validation_score = (
                holdout_results['accuracy'] * 0.25 +
                stratified_results['mean_score'] * 0.25 +
                temporal_results['mean_score'] * 0.30 +
                stability_results['stability_score'] * 0.20
            )
            
            print(f"\n검증 결과 요약:")
            print(f"홀드아웃: {holdout_results['accuracy']:.4f}")
            print(f"계층화 CV: {stratified_results['mean_score']:.4f}")
            print(f"시간 기반 CV: {temporal_results['mean_score']:.4f}")
            print(f"안정성: {stability_results['stability_score']:.3f}")
            print(f"종합 점수: {validation_score:.4f}")
            
            if validation_score >= 0.55:
                print("검증 성능: 목표 달성")
            elif validation_score >= 0.50:
                print("검증 성능: 양호")
            else:
                print("검증 성능: 미흡")
            
            self.validation_results = {
                'holdout': holdout_results,
                'stratified_cv': stratified_results,
                'temporal_cv': temporal_results,
                'stability': stability_results,
                'distribution': distribution_results,
                'class_balance': class_balance_results,
                'overall_score': validation_score
            }
            
            return self.validation_results
            
        except Exception as e:
            print(f"검증 시스템 오류: {e}")
            return {
                'holdout': {'accuracy': 0.0},
                'stratified_cv': {'mean_score': 0.0},
                'temporal_cv': {'mean_score': 0.0},
                'stability': {'stability_score': 0.0},
                'distribution': {'similarity_score': 0.0},
                'class_balance': {},
                'overall_score': 0.0
            }
    
    def validate_final_model(self, model, X_test, y_test):
        """최종 모델 검증"""
        print("최종 모델 검증")
        
        try:
            X_test_clean = X_test.fillna(0).replace([np.inf, -np.inf], 0)
            
            y_pred = model.predict(X_test_clean)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"최종 테스트 정확도: {accuracy:.4f}")
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            
            print("분류 보고서:")
            for cls in ['0', '1', '2']:
                if cls in report:
                    precision = report[cls]['precision']
                    recall = report[cls]['recall']
                    f1 = report[cls]['f1-score']
                    print(f"  클래스 {cls}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"최종 모델 검증 오류: {e}")
            return {
                'accuracy': 0.0,
                'classification_report': {},
                'confusion_matrix': np.zeros((3, 3))
            }

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    validator = ValidationSystem()
    
    leakage_free = validator.check_temporal_leakage(train_df, test_df)
    
    if not leakage_free:
        print("경고: 데이터 누수 위험 탐지")
    
    return validator

if __name__ == "__main__":
    main()