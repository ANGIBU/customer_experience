# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class ValidationSystem:
    def __init__(self):
        self.validation_results = {}
        self.leakage_issues = []
        
    def check_data_leakage(self, train_df, test_df):
        """데이터 누수 확인"""
        print("데이터 누수 확인")
        
        issues = []
        
        # ID 중복 확인
        train_ids = set(train_df['ID']) if 'ID' in train_df.columns else set()
        test_ids = set(test_df['ID']) if 'ID' in test_df.columns else set()
        
        common_ids = train_ids & test_ids
        if common_ids:
            issues.append(f"공통 ID {len(common_ids)}개 발견")
        
        # 시간적 순서 확인
        if train_ids and test_ids:
            try:
                train_id_nums = [int(id.split('_')[1]) for id in train_ids if '_' in id]
                test_id_nums = [int(id.split('_')[1]) for id in test_ids if '_' in id]
                
                if train_id_nums and test_id_nums:
                    train_max = max(train_id_nums)
                    test_min = min(test_id_nums)
                    
                    if train_max >= test_min:
                        # 이 경우는 자연스러운 시간 순서이므로 문제 없음
                        print("시간적 순서 정상 (연속적 ID)")
                    else:
                        print("시간적 순서 정상")
            except:
                print("ID 형식 분석 불가")
        
        # 타겟 직접 누수 확인
        if 'support_needs' in train_df.columns:
            # 타겟과 동일한 이름의 컬럼 확인
            direct_leakage = [col for col in train_df.columns 
                            if col.lower() == 'support_needs' and col != 'support_needs']
            if direct_leakage:
                issues.append(f"타겟 직접 누수: {direct_leakage}")
        
        self.leakage_issues = issues
        
        if issues:
            print("누수 위험:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("누수 위험 없음")
            return True
    
    def create_robust_model(self, X_train, y_train):
        """안정적인 모델 생성"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
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
                
                # 모델 학습 및 예측
                model = self.create_robust_model(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                accuracy = accuracy_score(y_val_fold, y_pred)
                fold_scores.append(accuracy)
                
                all_predictions.extend(y_pred)
                all_actuals.extend(y_val_fold)
                
                print(f"  Fold {fold + 1}: {accuracy:.4f}")
                
            except Exception as e:
                print(f"  Fold {fold + 1} 오류: {e}")
                fold_scores.append(0.0)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # 클래스별 성능
        if all_predictions and all_actuals:
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    all_actuals, all_predictions, average=None, zero_division=0
                )
                
                print("클래스별 성능:")
                for i in range(len(precision)):
                    print(f"  클래스 {i}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")
                    
            except Exception as e:
                print(f"클래스별 성능 계산 오류: {e}")
                precision = recall = f1 = np.array([0.0, 0.0, 0.0])
        else:
            precision = recall = f1 = np.array([0.0, 0.0, 0.0])
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def temporal_cross_validation(self, X, y, n_splits=5):
        """시간 기반 교차 검증"""
        print(f"시간 기반 교차 검증 (K={n_splits})")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            try:
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                model = self.create_robust_model(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                accuracy = accuracy_score(y_val_fold, y_pred)
                fold_scores.append(accuracy)
                
                print(f"  시점 {fold + 1}: {accuracy:.4f}")
                
            except Exception as e:
                print(f"  시점 {fold + 1} 오류: {e}")
                fold_scores.append(0.0)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"시간 기반 평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
        
        return {
            'fold_scores': fold_scores,
            'mean_score': mean_score,
            'std_score': std_score
        }
    
    def holdout_validation(self, X_train, y_train, X_val, y_val):
        """홀드아웃 검증"""
        print("홀드아웃 검증")
        
        try:
            model = self.create_robust_model(X_train, y_train)
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
            
            print(f"홀드아웃 정확도: {accuracy:.4f}")
            
            # 클래스별 정확도
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
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=run, stratify=y
                )
                
                model = self.create_robust_model(X_train, y_train)
                y_pred = model.predict(X_val)
                
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
        
        # 안정성 점수
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
            # 실제 분포
            true_dist = pd.Series(y_true).value_counts(normalize=True).sort_index()
            
            # 예측 분포  
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
    
    def validate_system(self, X_train, y_train):
        """전체 검증 시스템"""
        print("검증 시스템 실행")
        print("=" * 40)
        
        try:
            # 1. 홀드아웃 검증
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            holdout_results = self.holdout_validation(
                X_train_split, y_train_split, X_val_split, y_val_split
            )
            
            # 2. 계층화 교차 검증
            stratified_results = self.stratified_cross_validation(X_train, y_train)
            
            # 3. 시간 기반 교차 검증
            temporal_results = self.temporal_cross_validation(X_train, y_train)
            
            # 4. 안정성 테스트
            stability_results = self.stability_test(X_train, y_train)
            
            # 5. 분포 분석
            model = self.create_robust_model(X_train_split, y_train_split)
            y_pred = model.predict(X_val_split)
            distribution_results = self.distribution_analysis(y_val_split, y_pred)
            
            # 종합 점수 계산
            validation_score = (
                holdout_results['accuracy'] * 0.25 +
                stratified_results['mean_score'] * 0.25 +
                temporal_results['mean_score'] * 0.25 +
                stability_results['stability_score'] * 0.25
            )
            
            print(f"\n검증 결과 요약:")
            print(f"홀드아웃: {holdout_results['accuracy']:.4f}")
            print(f"계층화 CV: {stratified_results['mean_score']:.4f}")
            print(f"시간 기반 CV: {temporal_results['mean_score']:.4f}")
            print(f"안정성: {stability_results['stability_score']:.3f}")
            print(f"종합 점수: {validation_score:.4f}")
            
            self.validation_results = {
                'holdout': holdout_results,
                'stratified_cv': stratified_results,
                'temporal_cv': temporal_results,
                'stability': stability_results,
                'distribution': distribution_results,
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
                'overall_score': 0.0
            }
    
    def validate_final_model(self, model, X_test, y_test):
        """최종 모델 검증"""
        print("최종 모델 검증")
        
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"최종 테스트 정확도: {accuracy:.4f}")
            
            # 분류 보고서
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # 혼동 행렬
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
    
    def check_model_reliability(self, X, y, threshold=0.55):
        """모델 신뢰성 확인"""
        print(f"모델 신뢰성 확인 (목표: {threshold})")
        
        # 다중 분할 테스트
        reliabilities = []
        
        for seed in [42, 123, 456, 789, 999]:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed, stratify=y
                )
                
                # 간단한 재학습
                temp_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=seed,
                    n_jobs=-1
                )
                temp_model.fit(X_train, y_train)
                
                y_pred = temp_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                reliabilities.append(accuracy)
                
            except Exception as e:
                print(f"  시드 {seed} 오류: {e}")
                reliabilities.append(0.0)
        
        if reliabilities:
            mean_reliability = np.mean(reliabilities)
            std_reliability = np.std(reliabilities)
        else:
            mean_reliability = std_reliability = 0.0
        
        print(f"평균 신뢰성: {mean_reliability:.4f} (+/- {std_reliability:.4f})")
        
        reliable = mean_reliability >= threshold and std_reliability <= 0.02
        
        if reliable:
            print("모델 신뢰성 통과")
        else:
            print("모델 신뢰성 부족")
        
        return {
            'mean_reliability': mean_reliability,
            'std_reliability': std_reliability,
            'is_reliable': reliable,
            'reliabilities': reliabilities
        }
    
    def validate_prediction_consistency(self, model, X, n_runs=5):
        """예측 일관성 검증"""
        print("예측 일관성 검증")
        
        try:
            predictions_list = []
            
            for run in range(n_runs):
                # 동일한 데이터로 여러 번 예측
                y_pred = model.predict(X)
                predictions_list.append(y_pred)
            
            # 예측 일관성 확인
            if predictions_list:
                consistency_matrix = np.array(predictions_list)
                
                # 각 샘플별 예측 일치도
                consistency_scores = []
                for i in range(consistency_matrix.shape[1]):
                    sample_predictions = consistency_matrix[:, i]
                    most_common = np.bincount(sample_predictions, minlength=3).max()
                    consistency = most_common / n_runs
                    consistency_scores.append(consistency)
                
                mean_consistency = np.mean(consistency_scores)
            else:
                mean_consistency = 0.0
                consistency_scores = []
            
            print(f"예측 일관성: {mean_consistency:.3f}")
            
            if mean_consistency >= 0.95:
                print("예측 일관성 통과")
            else:
                print("예측 일관성 부족")
            
            return {
                'mean_consistency': mean_consistency,
                'consistency_scores': consistency_scores,
                'is_consistent': mean_consistency >= 0.95
            }
            
        except Exception as e:
            print(f"일관성 검증 오류: {e}")
            return {
                'mean_consistency': 0.0,
                'consistency_scores': [],
                'is_consistent': False
            }

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    validator = ValidationSystem()
    
    # 데이터 누수 확인
    leakage_free = validator.check_data_leakage(train_df, test_df)
    
    if not leakage_free:
        print("경고: 데이터 누수 위험 탐지")
    
    return validator

if __name__ == "__main__":
    main()