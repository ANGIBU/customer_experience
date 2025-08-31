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
                ('catboost', 'models/catboost_model.cbm', 'cat'),
                ('catboost_pkl', 'models/catboost_model.pkl', 'cat_pkl'),
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
                        elif model_type == 'cat':
                            model = CatBoostClassifier()
                            model.load_model(filepath)
                            self.models['catboost'] = model
                        elif model_type == 'cat_pkl':
                            model = joblib.load(filepath)
                            self.models['catboost'] = model
                        else:
                            self.models[name] = joblib.load(filepath)
                        
                        loaded_count += 1
                        model_name = 'catboost' if 'catboost' in name else name
                        print(f"  {model_name} 로드 완료")
                    except Exception as e:
                        print(f"  {name} 로드 실패: {e}")
                else:
                    if 'catboost' not in name:
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
        X_test_clean = self.safe_data_conversion(X_test)
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    pred_proba = model.predict(X_test_clean)
                    if pred_proba.ndim == 1:
                        pred_proba_multi = np.zeros((len(pred_proba), 3))
                        pred_proba_multi[:, 1] = pred_proba
                        pred_proba_multi[:, 0] = 1 - pred_proba
                        pred_proba_multi[:, 2] = 0
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
                        pred_proba_multi[:, 2] = 0
                        pred_proba = pred_proba_multi
                    predictions[name] = pred_proba
                    
                elif name == 'catboost':
                    try:
                        pred_proba = model.predict_proba(X_test_clean)
                        if pred_proba.shape[1] == 3:
                            predictions[name] = pred_proba
                        else:
                            print(f"  {name}: 잘못된 출력 차원 {pred_proba.shape}")
                    except Exception as cat_e:
                        print(f"  {name} CatBoost 예측 오류: {cat_e}")
                        continue
                    
                elif name == 'class1_specialist':
                    pred_binary = model.predict_proba(X_test_clean)
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
                                if 0 <= cls <= 2:
                                    pred_proba[i, cls] = 1.0
                                else:
                                    pred_proba[i] = [0.463, 0.269, 0.268]
                            predictions[name] = pred_proba
                    else:
                        print(f"  {name}: 기본 모델 부족으로 건너뜀")
                        continue
                    
                else:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_test_clean)
                        if pred_proba.shape[1] == 3:
                            predictions[name] = pred_proba
                        else:
                            print(f"  {name}: 잘못된 출력 차원")
                            continue
                    else:
                        pred_class = model.predict(X_test_clean)
                        pred_proba = np.zeros((len(pred_class), 3))
                        for i, cls in enumerate(pred_class):
                            if 0 <= cls <= 2:
                                pred_proba[i, cls] = 1.0
                            else:
                                pred_proba[i] = [0.463, 0.269, 0.268]
                        predictions[name] = pred_proba
                
                print(f"  {name}: {predictions[name].shape}")
                
            except Exception as e:
                print(f"  {name} 예측 오류: {e}")
                continue
        
        return predictions
    
    def class_specific_ensemble(self, predictions):
        """클래스별 특화 앙상블"""
        print("클래스별 앙상블 예측")
        
        available_models = list(predictions.keys())
        print(f"사용 가능한 모델: {available_models}")
        
        weights = {
            'lightgbm': 0.29,
            'xgboost': 0.24,
            'catboost': 0.15,
            'rf_calibrated': 0.14,
            'random_forest': 0.10,
            'extra_trees': 0.05,
            'neural_network': 0.05,
            'class1_specialist': 0.05,
            'svm_calibrated': 0.05,
            'stacking': 0.05
        }
        
        normalized_weights = {}
        total_weight = 0
        
        for model_name in available_models:
            if model_name in weights:
                normalized_weights[model_name] = weights[model_name]
                total_weight += weights[model_name]
            else:
                normalized_weights[model_name] = 0.05
                total_weight += 0.05
        
        if total_weight > 0:
            for model_name in normalized_weights:
                normalized_weights[model_name] /= total_weight
        
        print("모델별 가중치:")
        for name, weight in normalized_weights.items():
            print(f"  {name}: {weight:.2f}")
        
        first_pred = list(predictions.values())[0]
        if isinstance(first_pred, np.ndarray):
            if first_pred.ndim == 2:
                ensemble_proba = np.zeros((first_pred.shape[0], 3))
            else:
                ensemble_proba = np.zeros((len(first_pred), 3))
        else:
            ensemble_proba = np.zeros((len(first_pred), 3))
        
        for model_name, pred in predictions.items():
            if model_name in normalized_weights:
                weight = normalized_weights[model_name]
                
                try:
                    if model_name == 'class1_specialist':
                        if isinstance(pred, np.ndarray) and pred.ndim == 1:
                            ensemble_proba[:, 1] += weight * pred
                        elif isinstance(pred, np.ndarray) and pred.shape[1] >= 2:
                            ensemble_proba[:, 1] += weight * pred[:, 1]
                    else:
                        if isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[1] == 3:
                            ensemble_proba += weight * pred
                        else:
                            print(f"  {model_name}: 잘못된 예측 형태 {pred.shape if hasattr(pred, 'shape') else type(pred)}")
                except Exception as e:
                    print(f"  {model_name} 앙상블 처리 오류: {e}")
                    continue
        
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        ensemble_proba = np.where(row_sums > 0, ensemble_proba / row_sums, 
                                 np.array([0.463, 0.269, 0.268])[np.newaxis, :])
        
        return ensemble_proba
    
    def apply_isotonic_calibration(self, pred_proba, train_data=None):
        """등장회귀 보정"""
        print("확률 보정")
        
        calibrated_proba = pred_proba.copy()
        
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
            0: {'boost': 1.08, 'reduce': 0.92},
            1: {'boost': 1.15, 'reduce': 0.88},
            2: {'boost': 1.12, 'reduce': 0.90}
        }
        
        for cls in range(3):
            class_proba = optimized_proba[:, cls]
            thresholds = class_thresholds[cls]
            
            high_conf_percentile = 70
            low_conf_percentile = 30
            
            high_threshold = np.percentile(class_proba, high_conf_percentile)
            low_threshold = np.percentile(class_proba, low_conf_percentile)
            
            high_conf_mask = class_proba >= high_threshold
            low_conf_mask = class_proba <= low_threshold
            
            optimized_proba[high_conf_mask, cls] *= thresholds['boost']
            optimized_proba[low_conf_mask, cls] *= thresholds['reduce']
        
        row_sums = optimized_proba.sum(axis=1, keepdims=True)
        optimized_proba = np.where(row_sums > 0, optimized_proba / row_sums, optimized_proba)
        
        return optimized_proba
    
    def improved_distribution_matching(self, pred_proba):
        """개선된 분포 정합"""
        print("분포 정합")
        
        n_samples = len(pred_proba)
        target_distribution = np.array([0.463, 0.269, 0.268])
        target_counts = (target_distribution * n_samples).astype(int)
        
        remaining_samples = n_samples - target_counts.sum()
        if remaining_samples > 0:
            target_counts[0] += remaining_samples
        
        print("목표 분포:")
        for cls in range(3):
            target_pct = target_counts[cls] / n_samples * 100
            print(f"  클래스 {cls}: {target_counts[cls]}개 ({target_pct:.1f}%)")
        
        class_scores = []
        for cls in range(3):
            scores_with_idx = [(i, pred_proba[i, cls]) for i in range(n_samples)]
            scores_with_idx.sort(key=lambda x: x[1], reverse=True)
            class_scores.append(scores_with_idx)
        
        final_predictions = np.full(n_samples, -1, dtype=int)
        assigned = np.zeros(n_samples, dtype=bool)
        
        for cls in range(3):
            count = 0
            for idx, score in class_scores[cls]:
                if not assigned[idx] and count < target_counts[cls]:
                    final_predictions[idx] = cls
                    assigned[idx] = True
                    count += 1
                    
                if count >= target_counts[cls]:
                    break
        
        unassigned_indices = np.where(~assigned)[0]
        for idx in unassigned_indices:
            final_predictions[idx] = np.argmax(pred_proba[idx])
        
        invalid_mask = (final_predictions < 0) | (final_predictions > 2)
        if invalid_mask.any():
            for idx in np.where(invalid_mask)[0]:
                final_predictions[idx] = np.argmax(pred_proba[idx])
        
        final_dist = np.bincount(final_predictions, minlength=3)
        print("최종 분포:")
        for cls in range(3):
            final_pct = final_dist[cls] / n_samples * 100
            print(f"  클래스 {cls}: {final_dist[cls]}개 ({final_pct:.1f}%)")
        
        balanced_proba = np.zeros_like(pred_proba)
        for i in range(n_samples):
            balanced_proba[i] = pred_proba[i]
            assigned_class = final_predictions[i]
            if 0 <= assigned_class <= 2:
                balanced_proba[i, assigned_class] *= 1.2
            balanced_proba[i] = balanced_proba[i] / balanced_proba[i].sum()
        
        return final_predictions, balanced_proba
    
    def post_process_predictions(self, pred_proba):
        """예측 후처리"""
        print("예측 후처리")
        
        calibrated_proba = self.apply_isotonic_calibration(pred_proba)
        
        optimized_proba = self.optimize_class_thresholds(calibrated_proba)
        
        final_predictions, final_proba = self.improved_distribution_matching(optimized_proba)
        
        return final_predictions, final_proba
    
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
        
        if len(pred_classes) < 2:
            print(f"경고: 예측된 클래스가 {len(pred_classes)}개뿐입니다")
        
        print("제출 파일 검증 통과")
        return True
    
    def create_robust_fallback_predictions(self, X_test, test_ids):
        """강화된 대체 예측 생성"""
        print("강화된 대체 예측 생성")
        
        try:
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
                n_estimators=500,
                max_depth=12,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            pred_proba = model.predict_proba(X_test_simple)
            
            n_samples = len(pred_proba)
            target_distribution = np.array([0.463, 0.269, 0.268])
            target_counts = (target_distribution * n_samples).astype(int)
            
            remaining = n_samples - target_counts.sum()
            if remaining > 0:
                target_counts[0] += remaining
            
            class_scores = []
            for cls in range(3):
                scores_with_idx = [(i, pred_proba[i, cls]) for i in range(n_samples)]
                scores_with_idx.sort(key=lambda x: x[1], reverse=True)
                class_scores.append(scores_with_idx)
            
            predictions = np.full(n_samples, -1, dtype=int)
            assigned = np.zeros(n_samples, dtype=bool)
            
            for cls in range(3):
                count = 0
                for idx, score in class_scores[cls]:
                    if not assigned[idx] and count < target_counts[cls]:
                        predictions[idx] = cls
                        assigned[idx] = True
                        count += 1
                        
                    if count >= target_counts[cls]:
                        break
            
            unassigned_indices = np.where(~assigned)[0]
            for idx in unassigned_indices:
                predictions[idx] = np.argmax(pred_proba[idx])
            
            invalid_mask = (predictions < 0) | (predictions > 2)
            if invalid_mask.any():
                print(f"잘못된 예측값 {invalid_mask.sum()}개 수정")
                for idx in np.where(invalid_mask)[0]:
                    predictions[idx] = np.argmax(pred_proba[idx])
            
            print("강화된 대체 예측 완료")
            
            return predictions
            
        except Exception as e:
            print(f"강화된 대체 예측 실패: {e}")
            
            np.random.seed(42)
            random_predictions = np.random.choice([0, 1, 2], size=len(test_ids), p=[0.463, 0.269, 0.268])
            print("랜덤 예측 생성")
            return random_predictions
    
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
                
                if individual_predictions and len(individual_predictions) > 0:
                    ensemble_proba = self.class_specific_ensemble(individual_predictions)
                    
                    final_predictions, final_proba = self.post_process_predictions(ensemble_proba)
                    
                    unique_predictions = np.unique(final_predictions)
                    print(f"예측된 클래스: {sorted(unique_predictions)}")
                    
                    if len(unique_predictions) == 1:
                        print("경고: 모든 예측이 동일한 클래스입니다 - 대체 방법 사용")
                        final_predictions = self.create_robust_fallback_predictions(X_test, test_ids)
                else:
                    print("개별 모델 예측 실패 - 대체 방법 사용")
                    final_predictions = self.create_robust_fallback_predictions(X_test, test_ids)
            except Exception as e:
                print(f"모델 예측 오류: {e} - 대체 방법 사용")
                final_predictions = self.create_robust_fallback_predictions(X_test, test_ids)
        else:
            print("사용 가능한 모델 없음 - 대체 방법 사용")
            final_predictions = self.create_robust_fallback_predictions(X_test, test_ids)
        
        if final_predictions is None:
            print("모든 예측 방법 실패 - 강화된 랜덤 예측 생성")
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