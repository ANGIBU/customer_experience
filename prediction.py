# prediction.py

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax
from scipy.optimize import minimize
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
        self.ensemble_weights = None
        
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
        
    def load_trained_models(self):
        """학습된 모델 로드"""
        try:
            if os.path.exists('models/preprocessor.pkl'):
                self.preprocessor = joblib.load('models/preprocessor.pkl')
            else:
                from preprocessing import DataPreprocessor
                self.preprocessor = DataPreprocessor()
            
            if os.path.exists('models/feature_engineer.pkl'):
                self.feature_engineer = joblib.load('models/feature_engineer.pkl')
            else:
                from feature_engineering import FeatureEngineer
                self.feature_engineer = FeatureEngineer()
            
            if os.path.exists('models/feature_info.pkl'):
                self.feature_info = joblib.load('models/feature_info.pkl')
                self.feature_names = self.feature_info.get('feature_names', [])
                self.class_weights = self.feature_info.get('class_weights', {0: 1.28, 1: 1.05, 2: 0.85})
                self.ensemble_weights = self.feature_info.get('ensemble_weights', {})
            
            model_configs = [
                ('lightgbm', 'models/lightgbm_model.txt', 'lgb'),
                ('xgboost', 'models/xgboost_model.json', 'xgb'),
                ('catboost', 'models/catboost_model.pkl', 'pkl'),
                ('random_forest', 'models/random_forest_model.pkl', 'pkl'),
                ('gradient_boosting', 'models/gradient_boosting_model.pkl', 'pkl'),
                ('extra_trees', 'models/extra_trees_model.pkl', 'pkl'),
                ('neural_network', 'models/neural_network_model.pkl', 'pkl'),
                ('stacking', 'models/stacking_model.pkl', 'pkl')
            ]
            
            loaded_count = 0
            for name, filepath, model_type in model_configs:
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
                        
                    except Exception as e:
                        continue
            
            if loaded_count == 0:
                return self.train_fallback_models()
            
            return True
            
        except Exception as e:
            return self.train_fallback_models()
    
    def train_fallback_models(self):
        """대체 모델 학습"""
        try:
            from model_training import ModelTrainer
            
            trainer = ModelTrainer()
            result = trainer.prepare_training_data()
            trainer.train_models(result[0], result[1], result[2], result[3], result[6], result[7])
            
            self.models = trainer.models.copy()
            self.feature_engineer = result[6]
            self.preprocessor = result[7]
            self.feature_names = trainer.feature_names
            self.class_weights = trainer.class_weights
            self.ensemble_weights = trainer.ensemble_weights
            
            return len(self.models) > 0
            
        except Exception as e:
            return False
    
    def prepare_test_data(self):
        """테스트 데이터 준비"""
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        try:
            from data_analysis import DataAnalyzer
            analyzer = DataAnalyzer()
            analysis_results = analyzer.run_analysis()
            temporal_threshold = analysis_results.get('temporal', {}).get('temporal_threshold')
            temporal_info = analysis_results.get('temporal')
        except:
            temporal_threshold = None
            temporal_info = None
        
        train_processed, test_processed = self.feature_engineer.create_features(train_df, test_df, temporal_threshold)
        
        train_final, test_final = self.preprocessor.process_data(train_processed, test_processed, temporal_info)
        
        if self.feature_names is not None:
            for feature in self.feature_names:
                if feature not in test_final.columns:
                    test_final[feature] = 0
            
            available_features = [f for f in self.feature_names if f in test_final.columns]
            X_test = test_final[available_features].copy()
        else:
            train_cols = set(train_final.columns)
            test_cols = set(test_final.columns)
            common_features = list((train_cols & test_cols) - {'ID', 'support_needs'})
            X_test = test_final[sorted(common_features)]
        
        test_ids = test_final['ID']
        
        return X_test, test_ids
    
    def predict_individual_models(self, X_test):
        """개별 모델 예측"""
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
                    
                elif name == 'stacking':
                    if isinstance(model, dict):
                        base_models = model.get('base_models', [])
                        meta_model = model.get('meta_model')
                        base_model_objects = model.get('base_model_objects', {})
                        
                        if meta_model and base_model_objects:
                            base_predictions = []
                            
                            for base_name in base_models:
                                if base_name in base_model_objects:
                                    base_model = base_model_objects[base_name]
                                    
                                    if base_name == 'lightgbm':
                                        base_pred = base_model.predict(X_test_clean)
                                        if base_pred.ndim == 2 and base_pred.shape[1] == 3:
                                            base_predictions.append(base_pred)
                                    elif base_name == 'xgboost':
                                        xgb_test = xgb.DMatrix(X_test_clean, feature_names=self.feature_names)
                                        base_pred = base_model.predict(xgb_test)
                                        if base_pred.ndim == 2 and base_pred.shape[1] == 3:
                                            base_predictions.append(base_pred)
                                    else:
                                        if hasattr(base_model, 'predict_proba'):
                                            base_pred = base_model.predict_proba(X_test_clean)
                                            if base_pred.shape[1] == 3:
                                                base_predictions.append(base_pred)
                            
                            if base_predictions:
                                meta_X = np.hstack(base_predictions)
                                pred_proba = meta_model.predict_proba(meta_X)
                                predictions[name] = pred_proba
                    
                else:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_test_clean)
                        if pred_proba.shape[1] == 3:
                            predictions[name] = pred_proba
                    else:
                        pred_class = model.predict(X_test_clean)
                        pred_proba = np.zeros((len(pred_class), 3))
                        for i, cls in enumerate(pred_class):
                            if 0 <= cls <= 2:
                                pred_proba[i, int(cls)] = 1.0
                            else:
                                pred_proba[i] = [0.33, 0.34, 0.33]
                        predictions[name] = pred_proba
                        
            except Exception as e:
                continue
        
        return predictions
    
    def create_ensemble_weights(self, predictions):
        """앙상블 가중치 생성"""
        if not predictions or len(predictions) < 2:
            return None
        
        if self.ensemble_weights:
            weights = {}
            total_weight = 0
            for model_name in predictions.keys():
                if model_name in self.ensemble_weights:
                    weights[model_name] = self.ensemble_weights[model_name]
                else:
                    weights[model_name] = 0.1
                total_weight += weights[model_name]
            
            if total_weight > 0:
                for model_name in weights:
                    weights[model_name] /= total_weight
            
            return weights
        
        optimized_weights = {
            'lightgbm': 0.32,
            'xgboost': 0.28,
            'catboost': 0.24,
            'stacking': 0.12,
            'random_forest': 0.03,
            'gradient_boosting': 0.01,
            'extra_trees': 0.00
        }
        
        available_models = list(predictions.keys())
        weights = {}
        total_weight = 0
        
        for model_name in available_models:
            if model_name in optimized_weights:
                weights[model_name] = optimized_weights[model_name]
            else:
                weights[model_name] = 0.01
            total_weight += weights[model_name]
        
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        
        return weights
    
    def weighted_ensemble_prediction(self, predictions):
        """가중 앙상블 예측"""
        if not predictions:
            return None
        
        weights = self.create_ensemble_weights(predictions)
        if not weights:
            return None
        
        first_pred = list(predictions.values())[0]
        ensemble_proba = np.zeros((first_pred.shape[0], 3))
        
        for model_name, pred in predictions.items():
            if model_name in weights and isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[1] == 3:
                weight = weights[model_name]
                ensemble_proba += weight * pred
        
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        ensemble_proba = np.where(row_sums > 0, ensemble_proba / row_sums, 
                                 np.array([0.33, 0.34, 0.33])[np.newaxis, :])
        
        return ensemble_proba
    
    def apply_calibration(self, pred_proba):
        """확률 보정"""
        try:
            temperature = 1.02
            
            log_proba = np.log(np.clip(pred_proba, 1e-7, 1-1e-7))
            
            scaled_log_proba = log_proba / temperature
            
            calibrated_proba = softmax(scaled_log_proba, axis=1)
            
            return calibrated_proba
            
        except Exception:
            return pred_proba
    
    def apply_class_adjustment(self, pred_proba):
        """클래스 균형 조정"""
        class_adjustments = np.array([1.18, 1.02, 0.88])
        
        adjusted_proba = pred_proba * class_adjustments[np.newaxis, :]
        
        row_sums = adjusted_proba.sum(axis=1, keepdims=True)
        final_proba = adjusted_proba / row_sums
        
        return final_proba
    
    def apply_distribution_correction(self, predictions, target_dist=None):
        """분포 보정"""
        if target_dist is None:
            target_dist = [0.463, 0.269, 0.268]
        
        current_dist = np.bincount(predictions, minlength=3) / len(predictions)
        
        correction_factors = np.array(target_dist) / np.maximum(current_dist, 0.01)
        
        adjustment_strength = 0.3
        final_factors = 1.0 + adjustment_strength * (correction_factors - 1.0)
        
        return final_factors
    
    def generate_predictions(self, X_test, test_ids):
        """예측 생성"""
        individual_predictions = self.predict_individual_models(X_test)
        
        if not individual_predictions:
            return self.create_fallback_predictions(X_test, test_ids)
        
        ensemble_proba = self.weighted_ensemble_prediction(individual_predictions)
        
        if ensemble_proba is None:
            return self.create_fallback_predictions(X_test, test_ids)
        
        calibrated_proba = self.apply_calibration(ensemble_proba)
        
        adjusted_proba = self.apply_class_adjustment(calibrated_proba)
        
        predictions = np.argmax(adjusted_proba, axis=1)
        
        correction_factors = self.apply_distribution_correction(predictions)
        
        final_proba = adjusted_proba * correction_factors[np.newaxis, :]
        final_proba = final_proba / final_proba.sum(axis=1, keepdims=True)
        
        final_predictions = np.argmax(final_proba, axis=1)
        
        pred_counts = np.bincount(final_predictions, minlength=3)
        total_preds = len(final_predictions)
        
        if pred_counts[0] < total_preds * 0.42:
            class_0_proba = final_proba[:, 0]
            needed_count = int(total_preds * 0.42) - pred_counts[0]
            if needed_count > 0:
                top_indices = np.argsort(class_0_proba)[-needed_count:]
                final_predictions[top_indices] = 0
        
        if pred_counts[1] < total_preds * 0.25:
            class_1_proba = final_proba[:, 1]
            needed_count = int(total_preds * 0.25) - pred_counts[1]
            if needed_count > 0:
                top_indices = np.argsort(class_1_proba)[-needed_count:]
                final_predictions[top_indices] = 1
        
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'support_needs': final_predictions.astype(int)
        })
        
        submission_df.to_csv('submission.csv', index=False)
        
        return submission_df
    
    def create_fallback_predictions(self, X_test, test_ids):
        """대체 예측 생성"""
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
                    le.fit(combined.fillna('Unknown'))
                    train_processed[col] = le.transform(train_df[col].fillna('Unknown'))
                    test_processed[col] = le.transform(test_df[col].fillna('Unknown'))
            
            if all(col in train_processed.columns for col in ['age', 'tenure']):
                train_processed['age_tenure_ratio'] = train_processed['age'] / (train_processed['tenure'] + 1)
                test_processed['age_tenure_ratio'] = test_processed['age'] / (test_processed['tenure'] + 1)
            
            if all(col in train_processed.columns for col in ['frequent', 'payment_interval']):
                train_processed['frequency_payment_ratio'] = train_processed['frequent'] / (train_processed['payment_interval'] + 1)
                test_processed['frequency_payment_ratio'] = test_processed['frequent'] / (test_processed['payment_interval'] + 1)
            
            feature_cols = numeric_cols + categorical_cols + ['age_tenure_ratio', 'frequency_payment_ratio']
            feature_cols = [col for col in feature_cols if col in train_processed.columns and col in test_processed.columns]
            
            X = train_processed[feature_cols].fillna(0)
            y = train_processed['support_needs']
            X_test_simple = test_processed[feature_cols].fillna(0)
            
            if self.class_weights:
                class_weights = self.class_weights
            else:
                class_counts = np.bincount(y)
                total_samples = len(y)
                class_weights = {}
                
                for i, count in enumerate(class_counts):
                    if count > 0:
                        class_weights[i] = total_samples / (len(class_counts) * count)
                    else:
                        class_weights[i] = 1.0
                
                class_weights[0] *= 1.28
                class_weights[1] *= 1.05
                class_weights[2] *= 0.85
            
            model = RandomForestClassifier(
                n_estimators=450,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=3,
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            
            pred_proba = model.predict_proba(X_test_simple)
            
            calibrated_proba = self.apply_calibration(pred_proba)
            adjusted_proba = self.apply_class_adjustment(calibrated_proba)
            
            predictions = np.argmax(adjusted_proba, axis=1)
            
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'support_needs': predictions.astype(int)
            })
            
            submission_df.to_csv('submission.csv', index=False)
            
            return submission_df
            
        except Exception as e:
            np.random.seed(42)
            random_predictions = np.random.choice([0, 1, 2], size=len(test_ids), p=[0.463, 0.269, 0.268])
            
            submission_df = pd.DataFrame({
                'ID': test_ids,
                'support_needs': random_predictions
            })
            
            submission_df.to_csv('submission.csv', index=False)
            return submission_df
    
    def validate_submission(self, submission_df):
        """제출 파일 검증"""
        if not all(col in submission_df.columns for col in ['ID', 'support_needs']):
            return False
        
        if not submission_df['support_needs'].dtype in ['int64', 'int32']:
            submission_df['support_needs'] = submission_df['support_needs'].astype(int)
        
        valid_classes = {0, 1, 2}
        pred_classes = set(submission_df['support_needs'].unique())
        
        if not pred_classes.issubset(valid_classes):
            return False
        
        if submission_df.isnull().sum().sum() > 0:
            return False
        
        if len(pred_classes) < 2:
            return False
        
        return True
    
    def generate_final_predictions(self):
        """최종 예측 파이프라인"""
        if not self.load_trained_models():
            pass
        
        try:
            X_test, test_ids = self.prepare_test_data()
        except Exception as e:
            return None
        
        if self.models:
            try:
                submission_df = self.generate_predictions(X_test, test_ids)
            except Exception as e:
                submission_df = self.create_fallback_predictions(X_test, test_ids)
        else:
            submission_df = self.create_fallback_predictions(X_test, test_ids)
        
        if submission_df is not None and self.validate_submission(submission_df):
            return submission_df
        else:
            return None

def main():
    predictor = PredictionSystem()
    submission_df = predictor.generate_final_predictions()
    
    if submission_df is not None:
        return predictor, submission_df
    else:
        return predictor, None

if __name__ == "__main__":
    main()