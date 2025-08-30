# prediction.py

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictionSystem:
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.preprocessor = None
        
    def load_trained_models(self):
        """학습된 모델 로드"""
        print("학습된 모델 로드")
        
        try:
            # 전처리기 및 피처 엔지니어 로드
            self.preprocessor = joblib.load('models/preprocessor.pkl')
            self.feature_engineer = joblib.load('models/feature_engineer.pkl')
            print("전처리기 로드 완료")
            
            # 모델 파일 목록
            model_files = [
                ('lightgbm', 'models/lightgbm_model.txt', 'lgb'),
                ('xgboost', 'models/xgboost_model.json', 'xgb'),
                ('catboost', 'models/catboost_model.pkl', 'pkl'),
                ('random_forest', 'models/random_forest_model.pkl', 'pkl'),
                ('extra_trees', 'models/extra_trees_model.pkl', 'pkl'),
                ('neural_network', 'models/neural_network_model.pkl', 'pkl'),
                ('svm', 'models/svm_model.pkl', 'pkl'),
                ('stacking', 'models/stacking_model.pkl', 'pkl'),
                ('voting', 'models/voting_model.pkl', 'pkl')
            ]
            
            loaded_count = 0
            for name, filepath, model_type in model_files:
                if os.path.exists(filepath):
                    try:
                        if model_type == 'lgb':
                            self.models[name] = lgb.Booster(model_file=filepath)
                        elif model_type == 'xgb':
                            model = xgb.Booster()
                            model.load_model(filepath)
                            self.models[name] = model
                        else:
                            self.models[name] = joblib.load(filepath)
                        
                        loaded_count += 1
                        print(f"  {name} 로드 완료")
                    except Exception as e:
                        print(f"  {name} 로드 실패: {e}")
            
            print(f"총 {loaded_count}개 모델 로드 완료")
            return loaded_count > 0
            
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return False
    
    def prepare_test_data(self):
        """테스트 데이터 준비"""
        print("테스트 데이터 준비")
        
        # 원본 데이터 로드
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # 피처 엔지니어링 적용
        train_processed, test_processed = self.feature_engineer.create_features(train_df, test_df)
        
        # 전처리 적용
        train_final, test_final = self.preprocessor.process_data(train_processed, test_processed)
        
        # 모델링용 데이터 준비
        feature_cols = [col for col in test_final.columns if col not in ['ID', 'support_needs']]
        
        X_test = test_final[feature_cols]
        test_ids = test_final['ID']
        
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"피처 수: {len(feature_cols)}")
        
        return X_test, test_ids
    
    def predict_individual_models(self, X_test):
        """개별 모델 예측"""
        print("개별 모델 예측")
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    pred_proba = model.predict(X_test)
                    if pred_proba.ndim == 1:
                        # 이진 분류 결과를 다중 분류로 변환
                        pred_proba_multi = np.zeros((len(pred_proba), 3))
                        pred_proba_multi[:, 1] = pred_proba  # 클래스 1 확률
                        pred_proba_multi[:, 0] = 1 - pred_proba  # 클래스 0 확률
                        pred_proba_multi[:, 2] = 0  # 클래스 2 확률
                        pred_proba = pred_proba_multi
                    predictions[name] = pred_proba
                    
                elif name == 'xgboost':
                    xgb_test = xgb.DMatrix(X_test)
                    pred_proba = model.predict(xgb_test)
                    if pred_proba.ndim == 1:
                        pred_proba_multi = np.zeros((len(pred_proba), 3))
                        pred_proba_multi[:, 1] = pred_proba
                        pred_proba_multi[:, 0] = 1 - pred_proba
                        pred_proba_multi[:, 2] = 0
                        pred_proba = pred_proba_multi
                    predictions[name] = pred_proba
                    
                else:
                    # sklearn 모델들
                    pred_proba = model.predict_proba(X_test)
                    predictions[name] = pred_proba
                
                print(f"  {name}: {predictions[name].shape}")
                
            except Exception as e:
                print(f"  {name} 예측 오류: {e}")
                continue
        
        return predictions
    
    def ensemble_predictions(self, predictions):
        """앙상블 예측 생성"""
        print("앙상블 예측 생성")
        
        # 성능 기반 재조정된 가중치
        weights = {
            'lightgbm': 0.35,      # 부스팅 계열 강화
            'xgboost': 0.30,
            'catboost': 0.25,
            'random_forest': 0.10,  # 트리 계열 축소
            'extra_trees': 0.00,    # 제외
            'neural_network': 0.00, # 제외
            'stacking': 0.00        # 제외
        }
        
        # 사용 가능한 모델의 가중치 정규화
        available_weights = {}
        total_weight = 0
        
        for name in predictions.keys():
            if name in weights and weights[name] > 0:
                available_weights[name] = weights[name]
                total_weight += weights[name]
        
        # 가중치 정규화
        if total_weight > 0:
            for name in available_weights:
                available_weights[name] /= total_weight
        
        print("모델별 가중치:")
        for name, weight in available_weights.items():
            print(f"  {name}: {weight:.2f}")
        
        # 가중 평균 계산
        ensemble_proba = np.zeros_like(list(predictions.values())[0])
        
        for name, weight in available_weights.items():
            if name in predictions:
                ensemble_proba += weight * predictions[name]
        
        return ensemble_proba
    
    def optimize_thresholds(self, pred_proba):
        """임계값 최적화"""
        print("임계값 최적화")
        
        optimized_proba = pred_proba.copy()
        
        # 각 클래스별 강화된 임계값 조정
        for cls in range(3):
            class_proba = optimized_proba[:, cls]
            
            # 분위수 기반 강화 조정
            q90 = np.percentile(class_proba, 90)
            q75 = np.percentile(class_proba, 75)
            q25 = np.percentile(class_proba, 25)
            q10 = np.percentile(class_proba, 10)
            
            # 매우 높은 확률은 더 강화
            very_high_mask = class_proba > q90
            high_mask = (class_proba > q75) & (class_proba <= q90)
            low_mask = (class_proba >= q10) & (class_proba < q25)
            very_low_mask = class_proba < q10
            
            optimized_proba[very_high_mask, cls] *= 1.15
            optimized_proba[high_mask, cls] *= 1.08
            optimized_proba[low_mask, cls] *= 0.92
            optimized_proba[very_low_mask, cls] *= 0.85
        
        # 확률 정규화
        row_sums = optimized_proba.sum(axis=1, keepdims=True)
        optimized_proba = optimized_proba / row_sums
        
        return optimized_proba
    
    def calibrate_probabilities(self, pred_proba):
        """확률 보정"""
        print("확률 보정")
        
        calibrated_proba = pred_proba.copy()
        
        # 확률 범위 제한
        calibrated_proba = np.clip(calibrated_proba, 0.001, 0.999)
        
        # 온도 스케일링 유사 기법
        temperature = 1.2
        calibrated_proba = np.exp(np.log(calibrated_proba) / temperature)
        
        # 재정규화
        row_sums = calibrated_proba.sum(axis=1, keepdims=True)
        calibrated_proba = calibrated_proba / row_sums
        
        return calibrated_proba
    
    def apply_class_balancing(self, pred_proba):
        """클래스 균형 조정"""
        print("클래스 균형 조정")
        
        pred_classes = np.argmax(pred_proba, axis=1)
        current_dist = np.bincount(pred_classes, minlength=3)
        total_samples = len(pred_classes)
        
        # 훈련 데이터 실제 분포 기반 목표 설정
        target_distribution = np.array([0.463, 0.269, 0.268])
        target_counts = (target_distribution * total_samples).astype(int)
        
        print("분포 조정:")
        for cls in range(3):
            current_pct = current_dist[cls] / total_samples * 100
            target_pct = target_distribution[cls] * 100
            print(f"  클래스 {cls}: {current_pct:.1f}% → {target_pct:.1f}%")
        
        # 확률 강화 조정
        adjusted_proba = pred_proba.copy()
        
        for cls in range(3):
            current_count = current_dist[cls]
            target_count = target_counts[cls]
            
            if current_count > 0:
                ratio = target_count / current_count
                
                if ratio < 0.8:  # 과다 예측
                    cls_mask = pred_classes == cls
                    # 확률 크게 감소
                    adjusted_proba[cls_mask, cls] *= 0.7
                    
                elif ratio > 1.2:  # 과소 예측
                    cls_mask = pred_classes == cls
                    # 확률 크게 증가
                    adjusted_proba[cls_mask, cls] *= 1.4
                    
                    # 다른 클래스에서 확률 이동
                    for other_cls in range(3):
                        if other_cls != cls:
                            # 상위 확률 샘플에서 클래스 변경
                            top_candidates = pred_proba[:, cls] > np.percentile(pred_proba[:, cls], 85)
                            change_mask = top_candidates & (pred_classes != cls)
                            
                            if change_mask.sum() > 0:
                                adjusted_proba[change_mask, cls] *= 1.3
                                adjusted_proba[change_mask, other_cls] *= 0.8
        
        # 재정규화
        row_sums = adjusted_proba.sum(axis=1, keepdims=True)
        adjusted_proba = adjusted_proba / row_sums
        
        return adjusted_proba
    
    def post_process_predictions(self, pred_proba):
        """예측 후처리"""
        print("예측 후처리")
        
        # 1. 확률 보정
        calibrated_proba = self.calibrate_probabilities(pred_proba)
        
        # 2. 임계값 최적화
        optimized_proba = self.optimize_thresholds(calibrated_proba)
        
        # 3. 클래스 균형 조정
        balanced_proba = self.apply_class_balancing(optimized_proba)
        
        # 최종 예측 클래스
        final_predictions = np.argmax(balanced_proba, axis=1)
        
        return final_predictions, balanced_proba
    
    def create_submission(self, test_ids, predictions):
        """제출 파일 생성"""
        print("제출 파일 생성")
        
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'support_needs': predictions
        })
        
        # 제출 파일 저장
        submission_df.to_csv('submission.csv', index=False)
        
        # 예측 분포 확인
        pred_dist = submission_df['support_needs'].value_counts().sort_index()
        
        print("최종 예측 분포:")
        for cls, count in pred_dist.items():
            pct = count / len(submission_df) * 100
            print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
        
        print(f"제출 파일 저장: submission.csv ({submission_df.shape})")
        
        return submission_df
    
    def validate_submission(self, submission_df):
        """제출 파일 검증"""
        print("제출 파일 검증")
        
        # 형식 확인
        required_cols = ['ID', 'support_needs']
        if not all(col in submission_df.columns for col in required_cols):
            print("오류: 필수 컬럼 누락")
            return False
        
        # 데이터 타입 확인
        if not submission_df['support_needs'].dtype in ['int64', 'int32']:
            print("오류: support_needs가 정수형이 아님")
            return False
        
        # 클래스 범위 확인
        valid_classes = {0, 1, 2}
        pred_classes = set(submission_df['support_needs'].unique())
        
        if not pred_classes.issubset(valid_classes):
            print(f"오류: 잘못된 클래스 값 {pred_classes - valid_classes}")
            return False
        
        # 결측치 확인
        if submission_df.isnull().sum().sum() > 0:
            print("오류: 결측치 존재")
            return False
        
        print("제출 파일 검증 통과")
        return True
    
    def generate_predictions(self):
        """전체 예측 파이프라인"""
        print("예측 시스템 시작")
        print("=" * 40)
        
        # 1. 모델 로드
        if not self.load_trained_models():
            print("모델 로드 실패")
            return None
        
        # 2. 테스트 데이터 준비
        X_test, test_ids = self.prepare_test_data()
        
        # 3. 개별 모델 예측
        individual_predictions = self.predict_individual_models(X_test)
        
        if not individual_predictions:
            print("예측 실패")
            return None
        
        # 4. 앙상블 예측
        ensemble_proba = self.ensemble_predictions(individual_predictions)
        
        # 5. 후처리
        final_predictions, final_proba = self.post_process_predictions(ensemble_proba)
        
        # 6. 제출 파일 생성
        submission_df = self.create_submission(test_ids, final_predictions)
        
        # 7. 검증
        if self.validate_submission(submission_df):
            print("예측 시스템 완료")
            return submission_df
        else:
            print("제출 파일 검증 실패")
            return None
    
    def analyze_prediction_confidence(self, pred_proba):
        """예측 신뢰도 분석"""
        print("예측 신뢰도 분석")
        
        # 최대 확률 기반 신뢰도
        max_proba = np.max(pred_proba, axis=1)
        
        # 신뢰도 구간별 분포
        high_conf = (max_proba > 0.8).sum()
        medium_conf = ((max_proba > 0.5) & (max_proba <= 0.8)).sum()
        low_conf = (max_proba <= 0.5).sum()
        
        total = len(pred_proba)
        
        print("신뢰도 분포:")
        print(f"  높음 (>0.8): {high_conf}개 ({high_conf/total*100:.1f}%)")
        print(f"  보통 (0.5-0.8): {medium_conf}개 ({medium_conf/total*100:.1f}%)")
        print(f"  낮음 (<0.5): {low_conf}개 ({low_conf/total*100:.1f}%)")
        
        # 평균 신뢰도
        mean_confidence = np.mean(max_proba)
        print(f"평균 신뢰도: {mean_confidence:.3f}")
        
        return {
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf,
            'mean_confidence': mean_confidence
        }
    
    def generate_prediction_report(self, submission_df, confidence_analysis):
        """예측 보고서 생성"""
        print("\n예측 성과 보고서")
        print("=" * 40)
        
        # 기본 정보
        print(f"총 예측 샘플: {len(submission_df):,}개")
        print(f"평균 신뢰도: {confidence_analysis['mean_confidence']:.3f}")
        
        # 클래스별 분포
        class_dist = submission_df['support_needs'].value_counts().sort_index()
        print("\n클래스별 예측 분포:")
        for cls, count in class_dist.items():
            pct = count / len(submission_df) * 100
            print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
        
        # 신뢰도별 분포
        print(f"\n신뢰도별 분포:")
        print(f"  높은 신뢰도: {confidence_analysis['high_confidence']:,}개")
        print(f"  보통 신뢰도: {confidence_analysis['medium_confidence']:,}개")
        print(f"  낮은 신뢰도: {confidence_analysis['low_confidence']:,}개")
        
        # 모델 정보
        print(f"\n사용된 모델: {len(self.models)}개")
        for name in self.models.keys():
            print(f"  - {name}")

def main():
    predictor = PredictionSystem()
    submission_df = predictor.generate_predictions()
    
    if submission_df is not None:
        print("예측 완료")
        return predictor, submission_df
    else:
        print("예측 실패")
        return predictor, None

if __name__ == "__main__":
    main()