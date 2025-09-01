# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.feature_names = None
        self.class_weights = None
        self.ensemble_weights = {}
        
    def calculate_class_weights(self, y_train):
        """클래스 가중치 계산"""
        y_array = np.array(y_train)
        class_counts = np.bincount(y_array.astype(int), minlength=3)
        total_samples = len(y_array)
        
        weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total_samples / (3 * count)
            else:
                weights[i] = 1.0
        
        weights[1] *= 1.10
        weights[2] *= 1.05
        
        self.class_weights = weights
        return weights
    
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
    
    def prepare_training_data(self):
        """훈련 데이터 준비"""
        try:
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            
            engineer = FeatureEngineer()
            train_df, test_df = engineer.create_features(train_df, test_df)
            
            if train_df is None or test_df is None:
                raise ValueError("피처 생성 실패")
            
            preprocessor = DataPreprocessor()
            train_df, test_df = preprocessor.process_data(train_df, test_df)
            
            if train_df is None or test_df is None:
                raise ValueError("전처리 실패")
            
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_split(
                train_df, test_df, val_size=0.15, gap_size=0.15
            )
            
            if X_train is None or X_val is None:
                raise ValueError("데이터 분할 실패")
            
            self.feature_names = list(X_train.columns)
            self.calculate_class_weights(y_train)
            
            return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
            
        except Exception as e:
            return None, None, None, None, None, None, None, None
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 40,
            'learning_rate': 0.08,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_child_weight': 8,
            'min_split_gain': 0.02,
            'reg_alpha': 0.02,
            'reg_lambda': 0.02,
            'max_depth': 7,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        train_data = lgb.Dataset(X_train_clean, label=y_train_clean, weight=sample_weight, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val_clean, label=y_val_clean, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1200,
            callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val_clean)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 7,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.02,
            'reg_lambda': 0.02,
            'min_child_weight': 8,
            'gamma': 0.02,
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'hist'
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        train_data = xgb.DMatrix(X_train_clean, label=y_train_clean, weight=sample_weight, feature_names=self.feature_names)
        val_data = xgb.DMatrix(X_val_clean, label=y_val_clean, feature_names=self.feature_names)
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=1200,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=120,
            verbose_eval=0
        )
        
        y_pred = model.predict(val_data)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.08,
            depth=7,
            l2_leaf_reg=2,
            bootstrap_type='Bernoulli',
            subsample=0.85,
            colsample_bylevel=0.85,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=120,
            task_type='CPU',
            thread_count=-1
        )
        
        model.fit(
            X_train_clean, y_train_clean,
            sample_weight=sample_weight,
            eval_set=(X_val_clean, y_val_clean),
            use_best_model=True,
            verbose=False
        )
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['catboost'] = model
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features=0.8,
            bootstrap=True,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Extra Trees 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            max_features=0.8,
            bootstrap=True,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['extra_trees'] = model
        return model, accuracy
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """로지스틱 회귀 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = LogisticRegression(
            solver='liblinear',
            C=0.5,
            penalty='l2',
            class_weight=self.class_weights,
            random_state=42,
            max_iter=2000
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['logistic_regression'] = model
        return model, accuracy
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """SVM 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        if len(X_train_clean) > 10000:
            sample_indices = np.random.choice(len(X_train_clean), 8000, replace=False)
            X_train_sample = X_train_clean[sample_indices]
            y_train_sample = y_train_clean[sample_indices]
        else:
            X_train_sample = X_train_clean
            y_train_sample = y_train_clean
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight=self.class_weights,
            probability=True,
            random_state=42
        )
        
        model.fit(X_train_sample, y_train_sample)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['svm'] = model
        return model, accuracy
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Gradient Boosting 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=3,
            subsample=0.85,
            max_features=0.8,
            random_state=42
        )
        
        model.fit(X_train_clean, y_train_clean, sample_weight=sample_weight)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['gradient_boosting'] = model
        return model, accuracy
    
    def create_ensemble_weights(self, X_val, y_val):
        """앙상블 가중치 계산"""
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    y_pred_proba = model.predict(X_val_clean)
                    if y_pred_proba.ndim == 2:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                elif name == 'xgboost':
                    val_data = xgb.DMatrix(X_val_clean, feature_names=self.feature_names)
                    y_pred_proba = model.predict(val_data)
                    if y_pred_proba.ndim == 2:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        
                else:
                    y_pred = model.predict(X_val_clean)
                
                accuracy = accuracy_score(y_val_clean, y_pred)
                model_scores[name] = accuracy
                
            except Exception as e:
                model_scores[name] = 0.0
        
        if not model_scores or all(score <= 0.45 for score in model_scores.values()):
            conservative_weights = {
                'lightgbm': 0.20,
                'xgboost': 0.18,
                'catboost': 0.16,
                'random_forest': 0.14,
                'extra_trees': 0.12,
                'logistic_regression': 0.10,
                'svm': 0.05,
                'gradient_boosting': 0.05
            }
            
            for name in self.models.keys():
                if name in conservative_weights:
                    self.ensemble_weights[name] = conservative_weights[name]
                else:
                    self.ensemble_weights[name] = 0.02
        else:
            min_score = min(model_scores.values())
            adjusted_scores = {name: max(score - min_score, 0.01) for name, score in model_scores.items()}
            
            total_score = sum(adjusted_scores.values())
            if total_score > 0:
                for name in adjusted_scores:
                    self.ensemble_weights[name] = adjusted_scores[name] / total_score
            else:
                num_models = len(model_scores)
                for name in model_scores:
                    self.ensemble_weights[name] = 1.0 / num_models
        
        total_weight = sum(self.ensemble_weights.values())
        for name in self.ensemble_weights:
            self.ensemble_weights[name] /= total_weight
        
        return self.ensemble_weights
    
    def save_models(self, engineer, preprocessor):
        """모델 저장"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        joblib.dump(engineer, 'models/feature_engineer.pkl')
        
        feature_info = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'class_weights': self.class_weights,
            'ensemble_weights': self.ensemble_weights
        }
        joblib.dump(feature_info, 'models/feature_info.pkl')
        
        saved_count = 0
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    model.save_model(f'models/{name}_model.txt')
                elif name == 'xgboost':
                    model.save_model(f'models/{name}_model.json')
                else:
                    joblib.dump(model, f'models/{name}_model.pkl')
                
                saved_count += 1
                
            except Exception as e:
                continue
        
        return saved_count
    
    def train_models(self, X_train, X_val, y_train, y_val, engineer=None, preprocessor=None):
        """모델 학습 파이프라인"""
        if engineer is None:
            engineer = FeatureEngineer()
        if preprocessor is None:
            preprocessor = DataPreprocessor()
        if self.feature_names is None:
            self.feature_names = list(X_train.columns)
        if self.class_weights is None:
            self.calculate_class_weights(y_train)
        
        model_results = {}
        
        try:
            lgb_model, lgb_acc = self.train_lightgbm(X_train, y_train, X_val, y_val)
            model_results['lightgbm'] = lgb_acc
        except Exception:
            model_results['lightgbm'] = 0.0
        
        try:
            xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val)
            model_results['xgboost'] = xgb_acc
        except Exception:
            model_results['xgboost'] = 0.0
        
        try:
            cat_model, cat_acc = self.train_catboost(X_train, y_train, X_val, y_val)
            model_results['catboost'] = cat_acc
        except Exception:
            model_results['catboost'] = 0.0
        
        try:
            rf_model, rf_acc = self.train_random_forest(X_train, y_train, X_val, y_val)
            model_results['random_forest'] = rf_acc
        except Exception:
            model_results['random_forest'] = 0.0
        
        try:
            et_model, et_acc = self.train_extra_trees(X_train, y_train, X_val, y_val)
            model_results['extra_trees'] = et_acc
        except Exception:
            model_results['extra_trees'] = 0.0
        
        try:
            lr_model, lr_acc = self.train_logistic_regression(X_train, y_train, X_val, y_val)
            model_results['logistic_regression'] = lr_acc
        except Exception:
            model_results['logistic_regression'] = 0.0
        
        try:
            svm_model, svm_acc = self.train_svm(X_train, y_train, X_val, y_val)
            model_results['svm'] = svm_acc
        except Exception:
            model_results['svm'] = 0.0
        
        try:
            gb_model, gb_acc = self.train_gradient_boosting(X_train, y_train, X_val, y_val)
            model_results['gradient_boosting'] = gb_acc
        except Exception:
            model_results['gradient_boosting'] = 0.0
        
        self.create_ensemble_weights(X_val, y_val)
        
        self.save_models(engineer, preprocessor)

def main():
    try:
        trainer = ModelTrainer()
        result = trainer.prepare_training_data()
        
        if result[0] is not None:
            X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor = result
            trainer.train_models(X_train, X_val, y_train, y_val, engineer, preprocessor)
            return trainer
        else:
            return None
            
    except Exception as e:
        return None

if __name__ == "__main__":
    main()