# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ValidationSystem:
    """검증 시스템 클래스"""
    
    def __init__(self):
        self.validation_results = {}
        self.leakage_checks = {}
        
    def detect_data_leakage(self, train_df, test_df):
        """데이터 누수 탐지"""
        print("=== 데이터 누수 탐지 ===")
        
        leakage_issues = []
        
        # 1. ID 중복 확인
        train_ids = set(train_df['ID'].tolist()) if 'ID' in train_df.columns else set()
        test_ids = set(test_df['ID'].tolist()) if 'ID' in test_df.columns else set()
        
        common_ids = train_ids & test_ids
        if common_ids:
            leakage_issues.append(f"공통 ID {len(common_ids)}개 발견")
        
        # 2. 동일한 행 패턴 확인
        feature_cols = [col for col in train_df.columns 
                       if col in test_df.columns and col not in ['ID', 'support_needs']]
        
        if feature_cols:
            train_patterns = train_df[feature_cols].apply(lambda x: hash(tuple(x)), axis=1)
            test_patterns = test_df[feature_cols].apply(lambda x: hash(tuple(x)), axis=1)
            
            common_patterns = len(set(train_patterns) & set(test_patterns))
            if common_patterns > 0:
                leakage_issues.append(f"동일한 패턴 {common_patterns}개 발견")
        
        # 3. 타겟 관련 피처 확인
        target_related_features = []
        if 'support_needs' in train_df.columns:
            for col in feature_cols:
                if 'support' in col.lower() or 'target' in col.lower():
                    target_related_features.append(col)
        
        if target_related_features:
            leakage_issues.append(f"타겟 관련 피처 {len(target_related_features)}개: {target_related_features}")
        
        # 4. 시간적 순서 확인
        if 'ID' in train_df.columns and 'ID' in test_df.columns:
            train_id_numbers = [int(id.split('_')[1]) for id in train_df['ID'] if '_' in id]
            test_id_numbers = [int(id.split('_')[1]) for id in test_df['ID'] if '_' in id]
            
            if train_id_numbers and test_id_numbers:
                train_max = max(train_id_numbers)
                test_min = min(test_id_numbers)
                
                if train_max >= test_min:
                    leakage_issues.append(f"시간적 순서 문제: 훈련 최대 ID({train_max}) >= 테스트 최소 ID({test_min})")
        
        # 결과 저장
        self.leakage_checks = {
            'common_ids': len(common_ids),
            'common_patterns': common_patterns if 'common_patterns' in locals() else 0,
            'target_related_features': target_related_features,
            'issues': leakage_issues
        }
        
        if leakage_issues:
            print("발견된 누수 위험:")
            for issue in leakage_issues:
                print(f"  - {issue}")
        else:
            print("데이터 누수 위험 없음")
        
        return len(leakage_issues) == 0
    
    def stratified_validation(self, X, y, model_func, n_splits=5):
        """계층화 교차 검증"""
        print(f"=== 계층화 교차 검증 (K={n_splits}) ===")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_scores = []
        fold_predictions = []
        fold_actuals = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            # 모델 학습 및 예측
            model = model_func(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # 점수 계산
            accuracy = accuracy_score(y_val_fold, y_pred)
            fold_scores.append(accuracy)
            
            fold_predictions.extend(y_pred)
            fold_actuals.extend(y_val_fold)
            
            print(f"Fold {fold + 1}: {accuracy:.4f}")
        
        # 전체 결과
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # 클래스별 성능
        precision, recall, f1, support = precision_recall_fscore_support(
            fold_actuals, fold_predictions, average=None
        )
        
        print("클래스별 성능:")
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
            print(f"  클래스 {i}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'predictions': fold_predictions,
            'actuals': fold_actuals
        }
    
    def temporal_validation(self, X, y, model_func, n_splits=5):
        """시간 기반 검증"""
        print(f"=== 시간 기반 검증 (K={n_splits}) ===")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            # 모델 학습 및 예측
            model = model_func(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # 점수 계산
            accuracy = accuracy_score(y_val_fold, y_pred)
            fold_scores.append(accuracy)
            
            print(f"시점 {fold + 1}: {accuracy:.4f} (훈련: {len(train_idx)}, 검증: {len(val_idx)})")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"시간 기반 평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score
        }
    
    def holdout_validation(self, X_train, y_train, X_val, y_val, model_func):
        """홀드아웃 검증"""
        print("=== 홀드아웃 검증 ===")
        
        # 모델 학습
        model = model_func(X_train, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        # 혼동 행렬
        cm = confusion_matrix(y_val, y_pred)
        
        print(f"홀드아웃 정확도: {accuracy:.4f}")
        print("혼동 행렬:")
        print(cm)
        
        # 클래스별 정확도
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            print(f"클래스 {i} 정확도: {acc:.4f}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'class_accuracies': class_accuracies,
            'predictions': y_pred
        }
    
    def distribution_validation(self, train_target, val_predictions):
        """분포 기반 검증"""
        print("=== 분포 기반 검증 ===")
        
        # 훈련 데이터 타겟 분포
        train_dist = pd.Series(train_target).value_counts(normalize=True).sort_index()
        
        # 예측 분포
        pred_dist = pd.Series(val_predictions).value_counts(normalize=True).sort_index()
        
        print("분포 비교:")
        print(f"{'클래스':<8} {'훈련':<8} {'예측':<8} {'차이':<8}")
        print("-" * 32)
        
        total_diff = 0
        for cls in train_dist.index:
            train_pct = train_dist.get(cls, 0) * 100
            pred_pct = pred_dist.get(cls, 0) * 100
            diff = abs(train_pct - pred_pct)
            total_diff += diff
            
            print(f"{cls:<8} {train_pct:<7.1f}% {pred_pct:<7.1f}% {diff:<7.1f}%")
        
        print(f"총 분포 차이: {total_diff:.1f}%")
        
        # 분포 유사성 점수
        similarity_score = max(0, 100 - total_diff) / 100
        print(f"분포 유사성 점수: {similarity_score:.3f}")
        
        return {
            'train_distribution': train_dist.to_dict(),
            'prediction_distribution': pred_dist.to_dict(),
            'total_difference': total_diff,
            'similarity_score': similarity_score
        }
    
    def stability_validation(self, X, y, model_func, n_runs=10):
        """안정성 검증"""
        print(f"=== 안정성 검증 (실행 횟수: {n_runs}) ===")
        
        scores = []
        predictions_list = []
        
        for run in range(n_runs):
            # 다른 시드로 검증 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=run, stratify=y
            )
            
            # 모델 학습 및 예측
            model = model_func(X_train, y_train)
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            scores.append(accuracy)
            predictions_list.append(y_pred)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        print(f"평균 정확도: {mean_score:.4f}")
        print(f"표준편차: {std_score:.4f}")
        print(f"최소-최대: {min_score:.4f} - {max_score:.4f}")
        print(f"변동 계수: {std_score/mean_score:.3f}")
        
        # 안정성 점수 (낮은 변동성이 좋음)
        stability_score = 1 - (std_score / mean_score)
        print(f"안정성 점수: {stability_score:.3f}")
        
        return {
            'scores': scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'stability_score': stability_score
        }
    
    def comprehensive_validation(self, X_train, y_train, X_val, y_val, model_func):
        """종합 검증"""
        print("종합 검증 시작")
        print("="*40)
        
        # 1. 홀드아웃 검증
        holdout_results = self.holdout_validation(X_train, y_train, X_val, y_val, model_func)
        
        # 2. 전체 데이터에 대한 교차 검증
        X_full = pd.concat([X_train, X_val]) if hasattr(X_train, 'columns') else np.vstack([X_train, X_val])
        y_full = pd.concat([y_train, y_val]) if hasattr(y_train, 'index') else np.concatenate([y_train, y_val])
        
        stratified_results = self.stratified_validation(X_full, y_full, model_func)
        
        # 3. 분포 검증
        distribution_results = self.distribution_validation(y_train, holdout_results['predictions'])
        
        # 4. 안정성 검증
        stability_results = self.stability_validation(X_full, y_full, model_func)
        
        # 종합 점수 계산
        validation_score = (
            holdout_results['accuracy'] * 0.3 +
            stratified_results['mean_score'] * 0.3 +
            distribution_results['similarity_score'] * 0.2 +
            stability_results['stability_score'] * 0.2
        )
        
        print(f"\n=== 종합 검증 결과 ===")
        print(f"홀드아웃 정확도: {holdout_results['accuracy']:.4f}")
        print(f"교차 검증 정확도: {stratified_results['mean_score']:.4f} (+/- {stratified_results['std_score']:.4f})")
        print(f"분포 유사성: {distribution_results['similarity_score']:.3f}")
        print(f"안정성 점수: {stability_results['stability_score']:.3f}")
        print(f"종합 검증 점수: {validation_score:.4f}")
        
        self.validation_results = {
            'holdout': holdout_results,
            'cross_validation': stratified_results,
            'distribution': distribution_results,
            'stability': stability_results,
            'overall_score': validation_score
        }
        
        return self.validation_results

def main():
    """메인 실행 함수"""
    # 데이터 로드 (예시)
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # 검증 시스템 생성
    validator = ValidationSystem()
    
    # 데이터 누수 탐지
    validator.detect_data_leakage(train_df, test_df)
    
    print("\n검증 시스템 준비 완료!")
    return validator

if __name__ == "__main__":
    main()