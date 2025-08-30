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
        # 추가 초기화
        self.feature_info = None
        
    def load_trained_models(self):
        """학습된 모델 로드"""
        print("학습된 모델 로드")
        
        try:
            # 전처리기 및 피처 엔지니어 로드
            if os.path.exists('models/preprocessor.pkl'):
                self.preprocessor = joblib.load('models/preprocessor.pkl')
                print("전처리기 로드 완료")
            else:
                print("전처리기 파일 없음 - 새로 생성")
                from preprocessing import DataPreprocessor
                self.preprocessor = DataPreprocessor()
            
            if os.path.exists('models/feature_engineer.pkl'):
                self.feature_engineer = joblib.load('models/feature_engineer.pkl')
                print("피처 엔지니어 로드 완료")
            else:
                print("피처 엔지니어 파일 없음 - 새로 생성")
                from feature_engineering import FeatureEngineer
                self.feature_engineer = FeatureEngineer()
            
            # 피처 정보 로드
            self.feature_info = None
            if os.path.exists('models/feature_info.pkl'):
                self.feature_info = joblib.load('models/feature_info.pkl')
                print(f"피처 정보 로드: {self.feature_info['feature_count']}개")
            
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
                else:
                    print(f"  {name} 파일 없음: {filepath}")
            
            if loaded_count == 0:
                print("모델 파일이 없어 새로 학습합니다")
                return self.train_models_if_missing()
            
            print(f"총 {loaded_count}개 모델 로드 완료")
            return True
            
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return self.train_models_if_missing()
    
    def train_models_if_missing(self):
        """모델이 없을 경우 새로 학습"""
        print("모델 없음 - 새로 학습 시작")
        
        try:
            from model_training import ModelTrainer
            
            trainer = ModelTrainer()
            X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor = trainer.prepare_training_data()
            trainer.train_models(X_train, X_val, y_train, y_val)
            
            # 학습된 모델들 복사
            self.models = trainer.models.copy()
            self.feature_engineer = engineer
            self.preprocessor = preprocessor
            
            print("새 모델 학습 완료")
            return len(self.models) > 0
            
        except Exception as e:
            print(f"새 모델 학습 실패: {e}")
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
        
        # 저장된 피처 정보 사용
        if self.feature_info and 'feature_names' in self.feature_info:
            expected_features = self.feature_info['feature_names']
            print(f"저장된 피처 정보 사용: {len(expected_features)}개")
            
            # 누락된 피처를 0으로 채우기
            for feature in expected_features:
                if feature not in test_final.columns:
                    test_final[feature] = 0
                    print(f"누락 피처 추가: {feature}")
            
            # 예상 피처만 선택하고 순서 맞추기
            available_features = [f for f in expected_features if f in test_final.columns]
            X_test = test_final[available_features]
            
        else:
            # 기존 방식 (공통 피처 추출)
            train_cols = set(train_final.columns)
            test_cols = set(test_final.columns)
            
            common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
            common_features = sorted(common_features)
            
            X_test = test_final[common_features]
        
        test_ids = test_final['ID']
        
        print(f"테스트 데이터 형태: {X_test.shape}")
        print(f"피처 수: {X_test.shape[1]}")
        
        return X_test, test_ids
    
    def predict_individual_models(self, X_test):
        """개별 모델 예측"""
        print("개별 모델 예측")
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    pred_proba = model.predict(X_test.values)
                    if pred_proba.ndim == 1:
                        pred_proba_multi = np.zeros((len(pred_proba), 3))
                        pred_proba_multi[:, 1] = pred_proba
                        pred_proba_multi[:, 0] = 1 - pred_proba
                        pred_proba_multi[:, 2] = 0
                        pred_proba = pred_proba_multi
                    predictions[name] = pred_proba
                    
                elif name == 'xgboost':
                    # XGBoost는 피처 이름에 매우 민감하므로 값만 사용
                    xgb_test = xgb.DMatrix(X_test.values)
                    pred_proba = model.predict(xgb_test)
                    if pred_proba.ndim == 1:
                        pred_proba_multi = np.zeros((len(pred_proba), 3))
                        pred_proba_multi[:, 1] = pred_proba
                        pred_proba_multi[:, 0] = 1 - pred_proba
                        pred_proba_multi[:, 2] = 0
                        pred_proba = pred_proba_multi
                    predictions[name] = pred_proba
                    
                elif name == 'catboost':
                    # CatBoost도 값만 사용
                    pred_proba = model.predict_proba(X_test.values)
                    predictions[name] = pred_proba
                
                elif name == 'stacking':
                    # Stacking 모델은 기본 모델들의 예측을 입력으로 사용
                    if len(predictions) >= 2:
                        # 기존 예측들을 메타 피처로 사용
                        base_predictions = []
                        for pred_name, pred_proba in predictions.items():
                            if pred_name != 'stacking':
                                base_predictions.append(pred_proba)
                        
                        if base_predictions:
                            meta_features = np.hstack(base_predictions)
                            
                            if hasattr(model, 'predict_proba'):
                                pred_proba = model.predict_proba(meta_features)
                            else:
                                pred_class = model.predict(meta_features)
                                pred_proba = np.zeros((len(pred_class), 3))
                                for i, cls in enumerate(pred_class):
                                    pred_proba[i, cls] = 1.0
                            
                            predictions[name] = pred_proba
                    else:
                        print(f"  {name}: 기본 모델 부족으로 건너뜀")
                        continue
                    
                else:
                    # sklearn 모델들 - 값만 사용
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_test.values)
                        predictions[name] = pred_proba
                    else:
                        pred_class = model.predict(X_test.values)
                        pred_proba = np.zeros((len(pred_class), 3))
                        for i, cls in enumerate(pred_class):
                            pred_proba[i, cls] = 1.0
                        predictions[name] = pred_proba
                
                print(f"  {name}: {predictions[name].shape}")
                
            except Exception as e:
                print(f"  {name} 예측 오류: {e}")
                continue
        
        return predictions
    
    def ensemble_predictions(self, predictions):
        """앙상블 예측 생성"""
        print("앙상블 예측 생성")
        
        # 성능 기반 가중치
        weights = {
            'lightgbm': 0.35,
            'xgboost': 0.25,
            'catboost': 0.20,
            'random_forest': 0.10,
            'extra_trees': 0.05,
            'neural_network': 0.05,
            'stacking': 0.00,
            'voting': 0.00,
            'svm': 0.00
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
        else:
            # 모든 모델에 동일 가중치
            for name in predictions.keys():
                available_weights[name] = 1.0 / len(predictions)
        
        print("모델별 가중치:")
        for name, weight in available_weights.items():
            print(f"  {name}: {weight:.2f}")
        
        # 가중 평균 계산
        ensemble_proba = None
        
        for name, weight in available_weights.items():
            if name in predictions:
                if ensemble_proba is None:
                    ensemble_proba = weight * predictions[name]
                else:
                    ensemble_proba += weight * predictions[name]
        
        # 예외 처리: 앙상블이 실패한 경우 가장 좋은 모델 사용
        if ensemble_proba is None:
            print("앙상블 실패 - 첫 번째 모델 사용")
            first_model = list(predictions.keys())[0]
            ensemble_proba = predictions[first_model]
        
        return ensemble_proba
    
    def optimize_thresholds(self, pred_proba):
        """임계값 최적화"""
        print("임계값 최적화")
        
        optimized_proba = pred_proba.copy()
        
        # 각 클래스별 강화된 임계값 조정
        for cls in range(3):
            class_proba = optimized_proba[:, cls]
            
            # 분위수 기반 조정
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
        optimized_proba = np.where(row_sums > 0, optimized_proba / row_sums, optimized_proba)
        
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
        calibrated_proba = np.where(row_sums > 0, calibrated_proba / row_sums, calibrated_proba)
        
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
        
        # 확률 조정
        adjusted_proba = pred_proba.copy()
        
        for cls in range(3):
            current_count = current_dist[cls]
            target_count = target_counts[cls]
            
            if current_count > 0:
                ratio = target_count / current_count
                
                if ratio < 0.8:
                    cls_mask = pred_classes == cls
                    adjusted_proba[cls_mask, cls] *= 0.7
                    
                elif ratio > 1.2:
                    cls_mask = pred_classes == cls
                    adjusted_proba[cls_mask, cls] *= 1.4
                    
                    # 다른 클래스에서 확률 이동
                    for other_cls in range(3):
                        if other_cls != cls:
                            top_candidates = pred_proba[:, cls] > np.percentile(pred_proba[:, cls], 85)
                            change_mask = top_candidates & (pred_classes != cls)
                            
                            if change_mask.sum() > 0:
                                adjusted_proba[change_mask, cls] *= 1.3
                                adjusted_proba[change_mask, other_cls] *= 0.8
        
        # 재정규화
        row_sums = adjusted_proba.sum(axis=1, keepdims=True)
        adjusted_proba = np.where(row_sums > 0, adjusted_proba / row_sums, adjusted_proba)
        
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
            try:
                submission_df['support_needs'] = submission_df['support_needs'].astype(int)
            except:
                print("오류: support_needs를 정수형으로 변환할 수 없음")
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
    
    def create_fallback_predictions(self, X_test, test_ids):
        """대체 예측 생성"""
        print("대체 예측 생성")
        
        # 간단한 모델로 예측
        from sklearn.ensemble import RandomForestClassifier
        
        # 훈련 데이터 다시 로드 및 처리
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # 간단한 전처리
        from sklearn.preprocessing import LabelEncoder
        
        # 수치형 피처
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
        
        # 범주형 피처 인코딩
        categorical_cols = ['gender', 'subscription_type']
        
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        le = LabelEncoder()
        for col in categorical_cols:
            if col in train_df.columns and col in test_df.columns:
                combined = pd.concat([train_df[col], test_df[col]])
                le.fit(combined)
                train_processed[col] = le.transform(train_df[col])
                test_processed[col] = le.transform(test_df[col])
        
        # 피처 선택
        feature_cols = numeric_cols + categorical_cols
        feature_cols = [col for col in feature_cols if col in train_processed.columns and col in test_processed.columns]
        
        X_train = train_processed[feature_cols].fillna(0)
        y_train = train_processed['support_needs']
        X_test_simple = test_processed[feature_cols].fillna(0)
        
        # RandomForest 학습
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test_simple)
        
        print("대체 예측 완료")
        
        return predictions
    
    def generate_predictions(self):
        """전체 예측 파이프라인"""
        print("예측 시스템 시작")
        print("=" * 40)
        
        # 1. 모델 로드
        if not self.load_trained_models():
            print("모델 로드 실패")
            
        # 2. 테스트 데이터 준비
        try:
            X_test, test_ids = self.prepare_test_data()
        except Exception as e:
            print(f"테스트 데이터 준비 오류: {e}")
            return None
        
        # 3. 개별 모델 예측
        final_predictions = None
        
        if self.models:
            try:
                individual_predictions = self.predict_individual_models(X_test)
                
                if individual_predictions:
                    # 4. 앙상블 예측
                    ensemble_proba = self.ensemble_predictions(individual_predictions)
                    
                    # 5. 후처리
                    final_predictions, final_proba = self.post_process_predictions(ensemble_proba)
                else:
                    print("개별 모델 예측 실패 - 대체 방법 사용")
                    final_predictions = self.create_fallback_predictions(X_test, test_ids)
            except Exception as e:
                print(f"모델 예측 오류: {e} - 대체 방법 사용")
                final_predictions = self.create_fallback_predictions(X_test, test_ids)
        else:
            print("사용 가능한 모델 없음 - 대체 방법 사용")
            final_predictions = self.create_fallback_predictions(X_test, test_ids)
        
        # 최후의 수단: 모든 것이 실패한 경우
        if final_predictions is None:
            print("모든 예측 방법 실패 - 랜덤 예측 생성")
            np.random.seed(42)
            final_predictions = np.random.choice([0, 1, 2], size=len(test_ids), p=[0.46, 0.27, 0.27])
        
        # 6. 제출 파일 생성
        submission_df = self.create_submission(test_ids, final_predictions)
        
        # 7. 검증
        if self.validate_submission(submission_df):
            print("예측 시스템 완료")
            return submission_df
        else:
            print("제출 파일 검증 실패")
            return None

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