# prediction.py

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictionSystem:
    def __init__(self):
        self.models = {}
        self.feature_engineer = None
        self.preprocessor = None
        self.feature_info = None
        self.feature_names = None
        self.class_weights = None
        
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
            y_clean = np.nan_to_num(y_array, nan=0)
            return X_clean, y_clean
        
        return X_clean
        
    def load_trained_models(self):
        """학습된 모델 로드"""
        print("학습된 모델 로드")
        
        try:
            if os.path.exists('models/preprocessor.pkl'):
                self.preprocessor = joblib.load('models/preprocessor.pkl')
                print("전처리기 로드 완료")
            else:
                print("전처리기 파일 없음")
                from preprocessing import DataPreprocessor
                self.preprocessor = DataPreprocessor()
            
            if os.path.exists('models/feature_engineer.pkl'):
                self.feature_engineer = joblib.load('models/feature_engineer.pkl')
                print("피처 엔지니어 로드 완료")
            else:
                print("피처 엔지니어 파일 없음")
                from feature_engineering import FeatureEngineer
                self.feature_engineer = FeatureEngineer()
            
            if os.path.exists('models/feature_info.pkl'):
                self.feature_info = joblib.load('models/feature_info.pkl')
                self.feature_names = self.feature_info['feature_names']
                self.class_weights = self.feature_info.get('class_weights', {0: 1.0, 1: 1.2, 2: 1.0})
                print(f"피처 정보 로드: {self.feature_info['feature_count']}개")
            
            # 모델 파일 로드
            model_files = [
                ('lightgbm', 'models/lightgbm_model.txt', 'lgb'),
                ('xgboost', 'models/xgboost_model.json', 'xgb'),
                ('catboost', 'models/catboost_model.pkl', 'pkl'),
                ('random_forest', 'models/random_forest_model.pkl', 'pkl'),
                ('neural_network', 'models/neural_network_model.pkl', 'pkl'),
                ('ensemble', 'models/ensemble_model.pkl', 'pkl')
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
            trainer.train_models(X_train, X_val, y_train, y_val, engineer, preprocessor)
            
            self.models = trainer.models.copy()
            self.feature_engineer = engineer
            self.preprocessor = preprocessor
            self.feature_names = trainer.feature_names
            self.class_weights = trainer.class_weights
            
            print("새 모델 학습 완료")
            return len(self.models) > 0
            
        except Exception as e:
            print(f"새 모델 학습 실패: {e}")
            return False
    
    def prepare_test_data(self):
        """테스트 데이터 준비"""
        print("테스트 데이터 준비")
        
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # 피처 생성
        train_processed, test_processed = self.feature_engineer.create_features(train_df, test_df)
        
        # 전처리
        train_final, test_final = self.preprocessor.process_data(train_processed, test_processed)
        
        # 피처 순서 맞춤
        if self.feature_names is not None:
            print(f"저장된 피처 순서 사용: {len(self.feature_names)}개")
            
            # 누락된 피처 추가
            for feature in self.feature_names:
                if feature not in test_final.columns:
                    test_final[feature] = 0
            
            # 순서 맞춤
            available_features = [f for f in self.feature_names if f in test_final.columns]
            X_test = test_final[available_features].copy()
            
        else:
            # 공통 피처 사용
            train_cols = set(train_final.columns)
            test_cols = set(test_final.columns)
            common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
            X_test = test_final[sorted(common_features)]
        
        test_ids = test_final['ID']
        
        print(f"테스트 데이터 형태: {X_test.shape}")
        return X_test, test_ids
    
    def predict_individual_models(self, X_test):
        """개별 모델 예측"""
        print("개별 모델 예측")
        
        predictions = {}
        X_test_clean = self.safe_data_conversion(X_test)
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    pred_proba = model.predict(X_test_clean)
                    if pred_proba.ndim == 1:
                        # 이진 분류를 다중 분류로 변환
                        pred_proba_multi = np.zeros((len(pred_proba), 3))
                        pred_proba_multi[:, 1] = pred_proba
                        pred_proba_multi[:, 0] = 1 - pred_proba
                        pred_proba = pred_proba_multi
                    predictions[name] = pred_proba
                    
                elif name == 'xgboost':
                    if self.feature_names is not None:
                        xgb_test = xgb.DMatrix(X_test_clean, feature_names=self.feature_names)
                    else:
                        xgb_test = xgb.DMatrix(X_test_clean)
                    
                    pred_proba = model.predict(xgb_test)
                    if pred_proba.ndim == 1:
                        pred_proba_multi = np.zeros((len(pred_proba), 3))
                        pred_proba_multi[:, 1] = pred_proba
                        pred_proba_multi[:, 0] = 1 - pred_proba
                        pred_proba = pred_proba_multi
                    predictions[name] = pred_proba
                    
                else:
                    # 일반 모델들
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_test_clean)
                        if pred_proba.shape[1] == 3:
                            predictions[name] = pred_proba
                    else:
                        pred_class = model.predict(X_test_clean)
                        pred_proba = np.zeros((len(pred_class), 3))
                        for i, cls in enumerate(pred_class):
                            if 0 <= cls <= 2:
                                pred_proba[i, cls] = 1.0
                            else:
                                pred_proba[i] = [0.33, 0.33, 0.34]
                        predictions[name] = pred_proba
                
                print(f"  {name}: {predictions[name].shape}")
                
            except Exception as e:
                print(f"  {name} 예측 오류: {e}")
                continue
        
        return predictions
    
    def weighted_ensemble_prediction(self, predictions):
        """가중 앙상블 예측"""
        print("가중 앙상블 예측")
        
        if not predictions:
            return None
        
        # 성능 기반 가중치
        model_weights = {
            'lightgbm': 0.35,
            'xgboost': 0.30,
            'catboost': 0.20,
            'random_forest': 0.10,
            'neural_network': 0.05
        }
        
        # 사용 가능한 모델에 대해서만 가중치 정규화
        available_models = list(predictions.keys())
        normalized_weights = {}
        total_weight = 0
        
        for model_name in available_models:
            if model_name in model_weights:
                normalized_weights[model_name] = model_weights[model_name]
                total_weight += model_weights[model_name]
            else:
                normalized_weights[model_name] = 0.05
                total_weight += 0.05
        
        # 가중치 정규화
        if total_weight > 0:
            for model_name in normalized_weights:
                normalized_weights[model_name] /= total_weight
        
        print("모델별 가중치:")
        for name, weight in normalized_weights.items():
            print(f"  {name}: {weight:.3f}")
        
        # 가중 평균 계산
        first_pred = list(predictions.values())[0]
        ensemble_proba = np.zeros((first_pred.shape[0], 3))
        
        for model_name, pred in predictions.items():
            if model_name in normalized_weights:
                weight = normalized_weights[model_name]
                
                if isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[1] == 3:
                    ensemble_proba += weight * pred
                else:
                    print(f"  {model_name}: 잘못된 예측 형태")
        
        # 확률 정규화
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        ensemble_proba = np.where(row_sums > 0, ensemble_proba / row_sums, 
                                 np.array([0.33, 0.33, 0.34])[np.newaxis, :])
        
        return ensemble_proba
    
    def apply_calibration(self, pred_proba):
        """확률 보정"""
        print("확률 보정")
        
        # 온도 스케일링
        temperature = 1.1
        calibrated_proba = np.exp(np.log(np.clip(pred_proba, 1e-7, 1-1e-7)) / temperature)
        
        # 정규화
        row_sums = calibrated_proba.sum(axis=1, keepdims=True)
        calibrated_proba = np.where(row_sums > 0, calibrated_proba / row_sums, calibrated_proba)
        
        return calibrated_proba
    
    def natural_prediction(self, pred_proba):
        """자연스러운 예측 생성"""
        print("자연스러운 예측 생성")
        
        # 확률 보정
        calibrated_proba = self.apply_calibration(pred_proba)
        
        # 가장 높은 확률의 클래스 선택 (자연스러운 방법)
        predictions = np.argmax(calibrated_proba, axis=1)
        
        # 예측 분포 확인
        unique_predictions = np.unique(predictions)
        pred_counts = np.bincount(predictions, minlength=3)
        
        print("예측 분포:")
        for cls in range(3):
            count = pred_counts[cls]
            pct = count / len(predictions) * 100
            print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
        
        print(f"예측된 클래스 수: {len(unique_predictions)}")
        
        return predictions, calibrated_proba
    
    def create_fallback_predictions(self, X_test, test_ids):
        """대체 예측 생성"""
        print("대체 예측 생성")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # 원본 데이터 로드
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            # 기본 피처만 사용 (after_interaction 제외)
            numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
            categorical_cols = ['gender', 'subscription_type']
            
            train_processed = train_df.copy()
            test_processed = test_df.copy()
            
            # 범주형 인코딩
            le = LabelEncoder()
            for col in categorical_cols:
                if col in train_df.columns and col in test_df.columns:
                    combined = pd.concat([train_df[col], test_df[col]])
                    le.fit(combined.fillna('Unknown'))
                    train_processed[col] = le.transform(train_df[col].fillna('Unknown'))
                    test_processed[col] = le.transform(test_df[col].fillna('Unknown'))
            
            # 피처 선택
            feature_cols = numeric_cols + categorical_cols
            feature_cols = [col for col in feature_cols if col in train_processed.columns and col in test_processed.columns]
            
            X_train = train_processed[feature_cols].fillna(0)
            y_train = train_processed['support_needs']
            X_test_simple = test_processed[feature_cols].fillna(0)
            
            # 클래스 가중치 계산
            class_counts = np.bincount(y_train)
            total_samples = len(y_train)
            class_weights = {}
            
            for i, count in enumerate(class_counts):
                if count > 0:
                    class_weights[i] = total_samples / (len(class_counts) * count)
                else:
                    class_weights[i] = 1.0
            
            class_weights[1] *= 1.2
            
            # 모델 학습
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # 자연스러운 예측
            pred_proba = model.predict_proba(X_test_simple)
            predictions = np.argmax(pred_proba, axis=1)
            
            print("대체 예측 완료")
            return predictions
            
        except Exception as e:
            print(f"대체 예측 실패: {e}")
            
            # 최종 대체: 균등 분포
            np.random.seed(42)
            random_predictions = np.random.choice([0, 1, 2], size=len(test_ids))
            print("균등 분포 예측 생성")
            return random_predictions
    
    def create_submission(self, test_ids, predictions):
        """제출 파일 생성"""
        print("제출 파일 생성")
        
        predictions_clean = np.array(predictions)
        predictions_clean = np.clip(predictions_clean, 0, 2).astype(int)
        
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'support_needs': predictions_clean
        })
        
        submission_df.to_csv('submission.csv', index=False)
        
        # 분포 확인
        pred_dist = submission_df['support_needs'].value_counts().sort_index()
        
        print("최종 예측 분포:")
        for cls in [0, 1, 2]:
            count = pred_dist.get(cls, 0)
            pct = count / len(submission_df) * 100
            print(f"  클래스 {cls}: {count:,}개 ({pct:.1f}%)")
        
        print(f"제출 파일 저장: submission.csv ({submission_df.shape})")
        
        return submission_df
    
    def validate_submission(self, submission_df):
        """제출 파일 검증"""
        print("제출 파일 검증")
        
        # 필수 컬럼 확인
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
        
        # 예측 다양성 확인
        if len(pred_classes) < 2:
            print(f"경고: 예측된 클래스가 {len(pred_classes)}개뿐입니다")
        
        print("제출 파일 검증 통과")
        return True
    
    def generate_predictions(self):
        """전체 예측 파이프라인"""
        print("예측 시스템 시작")
        print("=" * 40)
        
        # 모델 로드
        if not self.load_trained_models():
            print("모델 로드 실패")
        
        try:
            # 테스트 데이터 준비
            X_test, test_ids = self.prepare_test_data()
        except Exception as e:
            print(f"테스트 데이터 준비 오류: {e}")
            return None
        
        final_predictions = None
        
        # 모델 예측 수행
        if self.models:
            try:
                # 개별 모델 예측
                individual_predictions = self.predict_individual_models(X_test)
                
                if individual_predictions and len(individual_predictions) > 0:
                    # 앙상블 예측
                    ensemble_proba = self.weighted_ensemble_prediction(individual_predictions)
                    
                    if ensemble_proba is not None:
                        # 자연스러운 예측 (분포 강제 조정 없음)
                        final_predictions, final_proba = self.natural_prediction(ensemble_proba)
                    else:
                        print("앙상블 실패 - 대체 방법 사용")
                        final_predictions = self.create_fallback_predictions(X_test, test_ids)
                else:
                    print("개별 모델 예측 실패 - 대체 방법 사용")
                    final_predictions = self.create_fallback_predictions(X_test, test_ids)
                    
            except Exception as e:
                print(f"모델 예측 오류: {e} - 대체 방법 사용")
                final_predictions = self.create_fallback_predictions(X_test, test_ids)
        else:
            print("사용 가능한 모델 없음 - 대체 방법 사용")
            final_predictions = self.create_fallback_predictions(X_test, test_ids)
        
        # 최종 안전장치
        if final_predictions is None:
            print("모든 예측 방법 실패 - 기본 예측 생성")
            np.random.seed(42)
            final_predictions = np.random.choice([0, 1, 2], size=len(test_ids))
        
        # 제출 파일 생성
        submission_df = self.create_submission(test_ids, final_predictions)
        
        # 검증
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