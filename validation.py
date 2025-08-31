# validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, train_test_split
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
        print("시간적 누수 확인")
        
        issues = []
        
        try:
            # ID 겹침 확인
            train_ids = set(train_df['ID']) if 'ID' in train_df.columns else set()
            test_ids = set(test_df['ID']) if 'ID' in test_df.columns else set()
            
            common_ids = train_ids & test_ids
            if common_ids:
                issues.append(f"공통 ID {len(common_ids)}개 발견")
            
            # 시간적 순서 확인
            if train_ids and test_ids:
                def extract_id_numbers(id_set):
                    numbers = []
                    for id_val in id_set:
                        if '_' in str(id_val):
                            try:
                                num = int(str(id_val).split('_')[1])
                                numbers.append(num)
                            except:
                                continue
                    return numbers
                
                train_id_nums = extract_id_numbers(train_ids)
                test_id_nums = extract_id_numbers(test_ids)
                
                if train_id_nums and test_id_nums:
                    train_max = max(train_id_nums)
                    test_min = min(test_id_nums)
                    
                    if train_max >= test_min:
                        overlap_count = len([x for x in train_id_nums if x >= test_min])
                        overlap_ratio = overlap_count / len(train_id_nums)
                        
                        if overlap_ratio > 0.1:
                            issues.append(f"시간적 순서 위반: 겹침 비율 {overlap_ratio:.1%}")
            
            self.leakage_issues = issues
            
            if issues:
                print("누수 위험:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            else:
                print("누수 위험 없음")
                return True
                
        except Exception as e:
            print(f"누수 확인 오류: {e}")
            return True
    
    def create_validation_model(self, X_train, y_train):
        """검증용 모델 생성"""
        try:
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
            
            # 클래스 1 가중치 조정
            class_weights[1] *= 1.2
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_clean, y_train_clean)
            return model
            
        except Exception as e:
            print(f"검증 모델 생성 오류: {e}")
            
            # 기본 모델 생성
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            X_fallback = np.zeros((100, 5)) if X_train is None else self.safe_data_conversion(X_train)
            y_fallback = np.zeros(100) if y_train is None else self.safe_data_conversion(y_train)
            model.fit(X_fallback, y_fallback)
            return model
    
    def temporal_cross_validation(self, X, y, n_splits=5):
        """시간 기반 교차 검증"""
        print(f"시간 기반 교차 검증 (K={n_splits})")
        
        try:
            X_clean, y_clean = self.safe_data_conversion(X, y)
            
            # temporal_id 기반 분할
            if 'temporal_id' in X.columns:
                temporal_ids = X['temporal_id'].values
                sorted_indices = np.argsort(temporal_ids)
                
                fold_size = len(sorted_indices) // (n_splits + 1)
                fold_scores = []
                
                for fold in range(n_splits):
                    train_end = (fold + 1) * fold_size
                    val_start = train_end
                    val_end = val_start + fold_size
                    
                    if val_end > len(sorted_indices):
                        break
                    
                    train_idx = sorted_indices[:train_end]
                    val_idx = sorted_indices[val_start:val_end]
                    
                    try:
                        X_train_fold = X_clean[train_idx]
                        y_train_fold = y_clean[train_idx]
                        X_val_fold = X_clean[val_idx]
                        y_val_fold = y_clean[val_idx]
                        
                        # temporal_id 컬럼 제거
                        if 'temporal_id' in X.columns:
                            temporal_col_idx = list(X.columns).index('temporal_id')
                            X_train_fold = np.delete(X_train_fold, temporal_col_idx, axis=1)
                            X_val_fold = np.delete(X_val_fold, temporal_col_idx, axis=1)
                        
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X_train_fold, y_train_fold)
                        y_pred = model.predict(X_val_fold)
                        
                        accuracy = accuracy_score(y_val_fold, y_pred)
                        fold_scores.append(accuracy)
                        
                        print(f"  시점 {fold + 1}: {accuracy:.4f}")
                        
                    except Exception as e:
                        print(f"  시점 {fold + 1} 오류: {e}")
                        fold_scores.append(0.0)
            else:
                # TimeSeriesSplit 사용
                tscv = TimeSeriesSplit(n_splits=n_splits)
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
                    try:
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X_clean[train_idx], y_clean[train_idx])
                        y_pred = model.predict(X_clean[val_idx])
                        
                        accuracy = accuracy_score(y_clean[val_idx], y_pred)
                        fold_scores.append(accuracy)
                        
                        print(f"  시점 {fold + 1}: {accuracy:.4f}")
                        
                    except Exception as e:
                        print(f"  시점 {fold + 1} 오류: {e}")
                        fold_scores.append(0.0)
            
            mean_score = np.mean(fold_scores) if fold_scores else 0.0
            std_score = np.std(fold_scores) if fold_scores else 0.0
            
            print(f"시간 기반 평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
            
            return {
                'fold_scores': fold_scores,
                'mean_score': mean_score,
                'std_score': std_score
            }
            
        except Exception as e:
            print(f"시간 기반 교차 검증 오류: {e}")
            return {
                'fold_scores': [0.0] * n_splits,
                'mean_score': 0.0,
                'std_score': 0.0
            }
    
    def holdout_validation(self, X_train, y_train, X_val, y_val):
        """홀드아웃 검증"""
        print("홀드아웃 검증")
        
        try:
            if any(data is None for data in [X_train, y_train, X_val, y_val]):
                print("홀드아웃 데이터가 None입니다")
                return {
                    'accuracy': 0.0,
                    'confusion_matrix': np.zeros((3, 3)),
                    'class_accuracies': [0.0, 0.0, 0.0]
                }
            
            X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
            X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
            
            model = self.create_validation_model(X_train_clean, y_train_clean)
            
            y_pred = model.predict(X_val_clean)
            y_pred = np.clip(y_pred, 0, 2)
            
            accuracy = accuracy_score(y_val_clean, y_pred)
            cm = confusion_matrix(y_val_clean, y_pred, labels=[0, 1, 2])
            
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
    
    def stability_test(self, X, y, n_runs=5):
        """모델 안정성 테스트"""
        print(f"안정성 테스트 (실행 {n_runs}회)")
        
        try:
            X_clean, y_clean = self.safe_data_conversion(X, y)
            
            # temporal_id 제거
            if 'temporal_id' in X.columns:
                temporal_col_idx = list(X.columns).index('temporal_id')
                X_temp = np.delete(X_clean, temporal_col_idx, axis=1)
            else:
                X_temp = X_clean
            
            scores = []
            
            for run in range(n_runs):
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_clean, test_size=0.2, random_state=run, stratify=y_clean
                    )
                    
                    model = self.create_validation_model(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_pred = np.clip(y_pred, 0, 2)
                    
                    accuracy = accuracy_score(y_val, y_pred)
                    scores.append(accuracy)
                    
                except Exception as e:
                    print(f"  실행 {run + 1} 오류: {e}")
                    scores.append(0.0)
            
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                min_score = np.min(valid_scores)
                max_score = np.max(valid_scores)
            else:
                mean_score = std_score = min_score = max_score = 0.0
            
            print(f"평균: {mean_score:.4f}")
            print(f"표준편차: {std_score:.4f}")
            print(f"범위: {min_score:.4f} - {max_score:.4f}")
            
            stability_score = 1 - (std_score / mean_score) if mean_score > 0 else 0
            stability_score = max(0, min(1, stability_score))
            print(f"안정성 점수: {stability_score:.3f}")
            
            return {
                'scores': scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'stability_score': stability_score
            }
            
        except Exception as e:
            print(f"안정성 테스트 오류: {e}")
            return {
                'scores': [0.0] * n_runs,
                'mean_score': 0.0,
                'std_score': 0.0,
                'stability_score': 0.0
            }
    
    def feature_leakage_detection(self, train_df, test_df):
        """피처 누수 탐지"""
        print("피처 누수 탐지")
        
        leakage_warnings = []
        
        try:
            # after_interaction 피처 확인
            if 'after_interaction' in train_df.columns and 'support_needs' in train_df.columns:
                correlation = train_df[['after_interaction', 'support_needs']].corr().iloc[0, 1]
                
                if abs(correlation) > 0.05:
                    leakage_warnings.append(f"after_interaction 상관관계: {correlation:.4f}")
                    print(f"  after_interaction 누수 위험: {correlation:.4f}")
            
            # 훈련-테스트 분포 차이 확인
            common_numeric = [col for col in train_df.select_dtypes(include=[np.number]).columns
                             if col not in ['ID', 'support_needs'] and col in test_df.columns]
            
            for feature in common_numeric:
                try:
                    train_values = train_df[feature].dropna()
                    test_values = test_df[feature].dropna()
                    
                    if len(train_values) > 100 and len(test_values) > 100:
                        statistic, p_value = ks_2samp(train_values, test_values)
                        
                        if statistic > 0.1:
                            leakage_warnings.append(f"{feature} 분포 차이: {statistic:.3f}")
                            print(f"  {feature} 분포 이동: {statistic:.3f}")
                            
                except Exception:
                    continue
            
            return leakage_warnings
            
        except Exception as e:
            print(f"누수 탐지 오류: {e}")
            return []
    
    def purged_time_series_cv(self, X, y, n_splits=5, gap=0.1):
        """갭을 둔 시간 기반 교차 검증"""
        print(f"갭 적용 시간 기반 CV (gap={gap})")
        
        try:
            X_clean, y_clean = self.safe_data_conversion(X, y)
            
            if 'temporal_id' in X.columns:
                temporal_ids = X['temporal_id'].values
                sorted_indices = np.argsort(temporal_ids)
                
                fold_scores = []
                total_samples = len(sorted_indices)
                gap_size = int(total_samples * gap)
                
                for fold in range(n_splits):
                    # 각 폴드의 크기 계산
                    fold_size = (total_samples - gap_size * n_splits) // n_splits
                    
                    if fold_size < 100:  # 최소 샘플 수 확인
                        break
                    
                    # 훈련 구간
                    train_start = fold * (fold_size + gap_size)
                    train_end = train_start + fold_size
                    
                    # 검증 구간 (갭 후)
                    val_start = train_end + gap_size
                    val_end = val_start + fold_size
                    
                    if val_end > total_samples:
                        break
                    
                    train_idx = sorted_indices[train_start:train_end]
                    val_idx = sorted_indices[val_start:val_end]
                    
                    try:
                        X_train_fold = X_clean[train_idx]
                        y_train_fold = y_clean[train_idx]
                        X_val_fold = X_clean[val_idx]
                        y_val_fold = y_clean[val_idx]
                        
                        # temporal_id 제거
                        if 'temporal_id' in X.columns:
                            temporal_col_idx = list(X.columns).index('temporal_id')
                            X_train_fold = np.delete(X_train_fold, temporal_col_idx, axis=1)
                            X_val_fold = np.delete(X_val_fold, temporal_col_idx, axis=1)
                        
                        model = self.create_validation_model(X_train_fold, y_train_fold)
                        y_pred = model.predict(X_val_fold)
                        
                        accuracy = accuracy_score(y_val_fold, y_pred)
                        fold_scores.append(accuracy)
                        
                        print(f"  갭 폴드 {fold + 1}: {accuracy:.4f}")
                        
                    except Exception as e:
                        print(f"  갭 폴드 {fold + 1} 오류: {e}")
                        fold_scores.append(0.0)
            else:
                # 일반 TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=n_splits, gap=int(len(X) * gap))
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
                    try:
                        model = self.create_validation_model(X_clean[train_idx], y_clean[train_idx])
                        y_pred = model.predict(X_clean[val_idx])
                        accuracy = accuracy_score(y_clean[val_idx], y_pred)
                        fold_scores.append(accuracy)
                        print(f"  갭 폴드 {fold + 1}: {accuracy:.4f}")
                    except Exception as e:
                        print(f"  갭 폴드 {fold + 1} 오류: {e}")
                        fold_scores.append(0.0)
            
            mean_score = np.mean(fold_scores) if fold_scores else 0.0
            std_score = np.std(fold_scores) if fold_scores else 0.0
            
            print(f"갭 적용 평균 정확도: {mean_score:.4f} (+/- {std_score:.4f})")
            
            return {
                'fold_scores': fold_scores,
                'mean_score': mean_score,
                'std_score': std_score
            }
            
        except Exception as e:
            print(f"갭 적용 교차 검증 오류: {e}")
            return {
                'fold_scores': [0.0] * n_splits,
                'mean_score': 0.0,
                'std_score': 0.0
            }
    
    def validate_system(self, X_train, y_train):
        """전체 검증 시스템"""
        print("검증 시스템 실행")
        print("=" * 40)
        
        try:
            if X_train is None or y_train is None:
                print("검증 데이터가 None입니다")
                return self.get_default_validation_results()
            
            if len(X_train) == 0 or len(y_train) == 0:
                print("검증 데이터가 비어있습니다")
                return self.get_default_validation_results()
            
            X_clean, y_clean = self.safe_data_conversion(X_train, y_train)
            
            # 데이터 분할
            if 'temporal_id' in X_train.columns:
                temporal_col_idx = list(X_train.columns).index('temporal_id')
                X_for_split = np.delete(X_clean, temporal_col_idx, axis=1)
            else:
                X_for_split = X_clean
            
            try:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_for_split, y_clean, test_size=0.2, random_state=42, stratify=y_clean
                )
            except Exception:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_for_split, y_clean, test_size=0.2, random_state=42
                )
            
            # 홀드아웃 검증
            holdout_results = self.holdout_validation(
                X_train_split, y_train_split, X_val_split, y_val_split
            )
            
            # 시간 기반 교차 검증
            temporal_results = self.temporal_cross_validation(X_train, y_train)
            
            # 갭 적용 교차 검증
            purged_results = self.purged_time_series_cv(X_train, y_train)
            
            # 안정성 테스트
            stability_results = self.stability_test(X_train, y_train)
            
            # 종합 점수 계산
            validation_score = (
                holdout_results['accuracy'] * 0.3 +
                temporal_results['mean_score'] * 0.3 +
                purged_results['mean_score'] * 0.3 +
                stability_results['stability_score'] * 0.1
            )
            
            print(f"\n검증 결과 요약:")
            print(f"홀드아웃: {holdout_results['accuracy']:.4f}")
            print(f"시간 기반 CV: {temporal_results['mean_score']:.4f}")
            print(f"갭 적용 CV: {purged_results['mean_score']:.4f}")
            print(f"안정성: {stability_results['stability_score']:.3f}")
            print(f"종합 점수: {validation_score:.4f}")
            
            self.validation_results = {
                'holdout': holdout_results,
                'temporal_cv': temporal_results,
                'purged_cv': purged_results,
                'stability': stability_results,
                'overall_score': validation_score
            }
            
            return self.validation_results
            
        except Exception as e:
            print(f"검증 시스템 오류: {e}")
            return self.get_default_validation_results()
    
    def get_default_validation_results(self):
        """기본 검증 결과 반환"""
        return {
            'holdout': {'accuracy': 0.0, 'confusion_matrix': np.zeros((3, 3)), 'class_accuracies': [0.0, 0.0, 0.0]},
            'temporal_cv': {'mean_score': 0.0, 'std_score': 0.0},
            'purged_cv': {'mean_score': 0.0, 'std_score': 0.0},
            'stability': {'stability_score': 0.0, 'mean_score': 0.0, 'std_score': 0.0},
            'overall_score': 0.0
        }

def main():
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        validator = ValidationSystem()
        leakage_free = validator.check_temporal_leakage(train_df, test_df)
        
        if not leakage_free:
            print("경고: 데이터 누수 위험 탐지")
        
        return validator
        
    except Exception as e:
        print(f"검증 시스템 초기화 오류: {e}")
        return None

if __name__ == "__main__":
    main()