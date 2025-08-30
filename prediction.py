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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
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
        self.calibrators = {}
        
    def load_trained_models(self):
        """학습된 모델 로드"""
        print("학습된 모델 로드")
        
        try:
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
            
            if os.path.exists('models/feature_info.pkl'):
                self.feature_info = joblib.load('models/feature_info.pkl')
                self.feature_names = self.feature_info['feature_names']
                self.class_weights = self.feature_info.get('class_weights', {0: 1.0, 1: 1.5, 2: 1.0})
                print(f"피처 정보 로드: {self.feature_info['feature_count']}개")
            
            model_files = [
                ('lightgbm', 'models/lightgbm_model.txt', 'lgb'),
                ('xgboost', 'models/xgboost_model.json', 'xgb'),
                ('catboost', 'models/catboost_model.pkl', 'pkl'),
                ('random_forest', 'models/random_forest_model.pkl', 'pkl'),
                ('extra_trees', 'models/extra_trees_model.pkl', 'pkl'),
                ('neural_network', 'models/neural_network_model.pkl', 'pkl'),
                ('class1_specialist', 'models/class1_specialist_model.pkl', 'pkl'),
                ('rf_calibrated', 'models/rf_calibrated_model.pkl', 'pkl'),
                ('svm_calibrated', 'models/svm_calibrated_model.pkl', 'pkl'),
                ('stacking', 'models/stacking_model.pkl', 'pkl')
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
        
        train_processed, test_processed = self.feature_engineer.create_features(train_df, test_df)
        
        train_final, test_final = self.preprocessor.process_data(train_processed, test_processed)
        
        if self.feature_names is not None:
            print(f"저장된 피처 순서 사용: {len(self.feature_names)}개")
            
            for feature in self.feature_names:
                if feature not in test_final.columns:
                    test_final[feature] = 0
            
            available_features = [f for f in self.feature_names if f in test_final.columns]
            
            missing_features = set(self.feature_names) - set(available_features)
            if missing_features:
                print(f"경고: {len(missing_features)}개 피처 누락")
            
            X_test = test_final[available_features].copy()
            
            final_features = []
            for feat in self.feature_names:
                if feat in X_test.columns:
                    final_features.append(feat)
                else:
                    X_test[feat] = 0
                    final_features.append(feat)
            
            X_test = X_test[final_features]
            
        else:
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
                    if self.feature_names is not None:
                        xgb_test = xgb.DMatrix(X_test.values, feature_names=self.feature_names)
                    else:
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
                    pred_proba = model.predict_proba(X_test.values)
                    predictions[name] = pred_proba
                    
                elif name == 'class1_specialist':
                    pred_binary = model.predict_proba(X_test.values)
                    if pred_binary.shape[1] == 2:
                        class1_proba = pred_binary[:, 1]
                        predictions[name] = class1_proba
                    
                elif name == 'stacking':
                    base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
                    valid_base_models = [m for m in base_models if m in predictions]
                    
                    if len(valid_base_models) >= 2:
                        base_predictions = []
                        for base_name in valid_base_models:
                            base_predictions.append(predictions[base_name])
                        
                        meta_features = np.hstack(base_predictions)
                        
                        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
                        
                        if expected_features is not None and meta_features.shape[1] != expected_features:
                            print(f"  {name}: 피처 크기 불일치 - 건너뜀")
                            continue
                        
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(meta_features)
                            predictions[name] = pred_proba
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
    
    def class_specific_ensemble(self, predictions):
        """클래스별 특화 앙상블"""
        print("클래스별 앙상블 예측")
        
        class_weights = {
            0: {'lightgbm': 0.45, 'xgboost': 0.35, 'catboost': 0.20},
            1: {'class1_specialist': 0.40, 'neural_network': 0.30, 'lightgbm': 0.30},
            2: {'catboost': 0.40, 'lightgbm': 0.35, 'xgboost': 0.25}
        }
        
        ensemble_proba = np.zeros((len(list(predictions.values())[0]), 3))
        
        for cls in range(3):
            cls_weights = class_weights[cls]
            total_weight = 0
            
            for model_name, weight in cls_weights.items():
                if model_name in predictions:
                    if model_name == 'class1_specialist':
                        if isinstance(predictions[model_name], np.ndarray) and predictions[model_name].ndim == 1:
                            ensemble_proba[:, cls] += weight * predictions[model_name]
                        else:
                            ensemble_proba[:, cls] += weight * predictions[model_name][:, 1]
                    else:
                        ensemble_proba[:, cls] += weight * predictions[model_name][:, cls]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_proba[:, cls] /= total_weight
        
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        ensemble_proba = np.where(row_sums > 0, ensemble_proba / row_sums, ensemble_proba)
        
        return ensemble_proba
    
    def apply_isotonic_calibration(self, pred_proba, train_data=None):
        """등장회귀 보정"""
        print("확률 보정")
        
        calibrated_proba = pred_proba.copy()
        
        if train_data is not None:
            X_train, y_train = train_data
            
            for cls in range(3):
                if cls not in self.calibrators:
                    self.calibrators[cls] = IsotonicRegression(out_of_bounds='clip')
                    
                    binary_labels = (y_train == cls).astype(int)
                    class_proba = pred_proba[:, cls]
                    
                    self.calibrators[cls].fit(class_proba, binary_labels)
                
                calibrated_proba[:, cls] = self.calibrators[cls].transform(pred_proba[:, cls])
        
        temperature = 1.05
        calibrated_proba = np.exp(np.log(np.clip(calibrated_proba, 1e-7, 1-1e-7)) / temperature)
        
        row_sums = calibrated_proba.sum(axis=1, keepdims=True)
        calibrated_proba = np.where(row_sums > 0, calibrated_proba / row_sums, calibrated_proba)
        
        return calibrated_proba
    
    def optimize_class_thresholds(self, pred_proba):
        """클래스별 임계값 최적화"""
        print("임계값 최적화")
        
        optimized_proba = pred_proba.copy()
        
        class_thresholds = {
            0: {'low': 0.35, 'high': 0.65},
            1: {'low': 0.25, 'high': 0.55},
            2: {'low': 0.30, 'high': 0.60}
        }
        
        for cls in range(3):
            class_proba = optimized_proba[:, cls]
            thresholds = class_thresholds[cls]
            
            high_conf_mask = class_proba > thresholds['high']
            low_conf_mask = class_proba < thresholds['low']
            
            optimized_proba[high_conf_mask, cls] *= 1.12
            optimized_proba[low_conf_mask, cls] *= 0.88
        
        row_sums = optimized_proba.sum(axis=1, keepdims=True)
        optimized_proba = np.where(row_sums > 0, optimized_proba / row_sums, optimized_proba)
        
        return optimized_proba
    
    def apply_distribution_matching(self, pred_proba):
        """분포 정합"""
        print("분포 정합")
        
        pred_classes = np.argmax(pred_proba, axis=1)
        current_dist = np.bincount(pred_classes, minlength=3) / len(pred_classes)
        
        target_distribution = np.array([0.463, 0.269, 0.268])
        
        print("분포 조정:")
        for cls in range(3):
            current_pct = current_dist[cls] * 100
            target_pct = target_distribution[cls] * 100
            print(f"  클래스 {cls}: {current_pct:.1f}% → {target_pct:.1f}%")
        
        adjusted_proba = pred_proba.copy()
        
        for cls in range(3):
            current_count = current_dist[cls]
            target_count = target_distribution[cls]
            
            if current_count > 0:
                ratio = target_count / current_count
                
                if ratio < 0.85:
                    cls_mask = pred_classes == cls
                    adjusted_proba[cls_mask, cls] *= 0.75
                elif ratio > 1.15:
                    cls_mask = pred_classes == cls
                    adjusted_proba[cls_mask, cls] *= 1.25
                    
                    top_candidates = pred_proba[:, cls] > np.percentile(pred_proba[:, cls], 80)
                    change_mask = top_candidates & (pred_classes != cls)
                    
                    if change_mask.sum() > 0:
                        adjusted_proba[change_mask, cls] *= 1.2
                        for other_cls in range(3):
                            if other_cls != cls:
                                adjusted_proba[change_mask, other_cls] *= 0.85
        
        row_sums = adjusted_proba.sum(axis=1, keepdims=True)
        adjusted_proba = np.where(row_sums > 0, adjusted_proba / row_sums, adjusted_proba)
        
        return adjusted_proba
    
    def post_process_predictions(self, pred_proba):
        """예측 후처리"""
        print("예측 후처리")
        
        calibrated_proba = self.apply_isotonic_calibration(pred_proba)
        
        optimized_proba = self.optimize_class_thresholds(calibrated_proba)
        
        balanced_proba = self.apply_distribution_matching(optimized_proba)
        
        final_predictions = np.argmax(balanced_proba, axis=1)
        
        return final_predictions, balanced_proba
    
    def create_submission(self, test_ids, predictions):
        """제출 파일 생성"""
        print("제출 파일 생성")
        
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'support_needs': predictions
        })
        
        submission_df.to_csv('submission.csv', index=False)
        
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
        
        required_cols = ['ID', 'support_needs']
        if not all(col in submission_df.columns for col in required_cols):
            print("오류: 필수 컬럼 누락")
            return False
        
        if not submission_df['support_needs'].dtype in ['int64', 'int32']:
            try:
                submission_df['support_needs'] = submission_df['support_needs'].astype(int)
            except:
                print("오류: support_needs를 정수형으로 변환할 수 없음")
                return False
        
        valid_classes = {0, 1, 2}
        pred_classes = set(submission_df['support_needs'].unique())
        
        if not pred_classes.issubset(valid_classes):
            print(f"오류: 잘못된 클래스 값 {pred_classes - valid_classes}")
            return False
        
        if submission_df.isnull().sum().sum() > 0:
            print("오류: 결측치 존재")
            return False
        
        print("제출 파일 검증 통과")
        return True
    
    def create_fallback_predictions(self, X_test, test_ids):
        """대체 예측 생성"""
        print("대체 예측 생성")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length']
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
        
        feature_cols = numeric_cols + categorical_cols
        feature_cols = [col for col in feature_cols if col in train_processed.columns and col in test_processed.columns]
        
        X_train = train_processed[feature_cols].fillna(0)
        y_train = train_processed['support_needs']
        X_test_simple = test_processed[feature_cols].fillna(0)
        
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
        
        class_weights[1] *= 1.4
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight=class_weights,
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
        
        if not self.load_trained_models():
            print("모델 로드 실패")
            
        try:
            X_test, test_ids = self.prepare_test_data()
        except Exception as e:
            print(f"테스트 데이터 준비 오류: {e}")
            return None
        
        final_predictions = None
        
        if self.models:
            try:
                individual_predictions = self.predict_individual_models(X_test)
                
                if individual_predictions:
                    ensemble_proba = self.class_specific_ensemble(individual_predictions)
                    
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
        
        if final_predictions is None:
            print("모든 예측 방법 실패 - 랜덤 예측 생성")
            np.random.seed(42)
            final_predictions = np.random.choice([0, 1, 2], size=len(test_ids), p=[0.463, 0.269, 0.268])
        
        submission_df = self.create_submission(test_ids, final_predictions)
        
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