# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.calibration import CalibratedClassifierCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
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
        self.best_threshold = 0.5
        self.ensemble_weights = {}
        self.hyperopt_trials = {}
        
    def calculate_class_weights(self, y_train):
        """클래스 가중치 계산"""
        y_array = np.array(y_train)
        class_counts = np.bincount(y_array)
        total_samples = len(y_array)
        
        weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total_samples / (len(class_counts) * count)
            else:
                weights[i] = 1.0
        
        # 클래스 불균형 보정
        weights[1] *= 1.15
        weights[2] *= 1.09
        
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
            
            try:
                from data_analysis import DataAnalyzer
                analyzer = DataAnalyzer()
                analysis_results = analyzer.run_analysis()
                temporal_threshold = analysis_results.get('temporal', {}).get('temporal_threshold')
                temporal_info = analysis_results.get('temporal')
            except Exception:
                temporal_threshold = None
                temporal_info = None
            
            engineer = FeatureEngineer()
            train_df, test_df = engineer.create_features(train_df, test_df, temporal_threshold)
            
            if train_df is None or test_df is None:
                raise ValueError("피처 생성 실패")
            
            preprocessor = DataPreprocessor()
            train_df, test_df = preprocessor.process_data(train_df, test_df, temporal_info)
            
            if train_df is None or test_df is None:
                raise ValueError("전처리 실패")
            
            X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data_temporal_optimized(
                train_df, test_df, val_size=0.18, gap_size=0.005
            )
            
            if X_train is None or X_val is None:
                raise ValueError("데이터 분할 실패")
            
            self.feature_names = list(X_train.columns)
            self.calculate_class_weights(y_train)
            
            return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
            
        except Exception as e:
            return None, None, None, None, None, None, None, None
    
    def optimize_lightgbm_hyperopt(self, X_train, y_train, X_val, y_val):
        """LightGBM 베이지안 최적화"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        def objective(params):
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': int(params['num_leaves']),
                'learning_rate': params['learning_rate'],
                'feature_fraction': params['feature_fraction'],
                'bagging_fraction': params['bagging_fraction'],
                'bagging_freq': 5,
                'min_child_weight': int(params['min_child_weight']),
                'min_split_gain': params['min_split_gain'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'max_depth': int(params['max_depth']),
                'verbose': -1,
                'random_state': 42,
                'force_col_wise': True
            }
            
            try:
                train_data = lgb.Dataset(X_train_clean, label=y_train_clean, weight=sample_weight, feature_name=self.feature_names)
                val_data = lgb.Dataset(X_val_clean, label=y_val_clean, reference=train_data)
                
                model = lgb.train(
                    lgb_params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=800,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val_clean)
                y_pred_class = np.argmax(y_pred, axis=1)
                accuracy = accuracy_score(y_val_clean, y_pred_class)
                
                return {'loss': 1 - accuracy, 'status': STATUS_OK}
                
            except Exception:
                return {'loss': 1.0, 'status': STATUS_OK}
        
        space = {
            'num_leaves': hp.choice('num_leaves', [25, 31, 42, 55]),
            'learning_rate': hp.uniform('learning_rate', 0.02, 0.05),
            'feature_fraction': hp.uniform('feature_fraction', 0.75, 0.9),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.8, 0.95),
            'min_child_weight': hp.choice('min_child_weight', [3, 5, 7, 10]),
            'min_split_gain': hp.uniform('min_split_gain', 0.05, 0.15),
            'reg_alpha': hp.uniform('reg_alpha', 0.05, 0.15),
            'reg_lambda': hp.uniform('reg_lambda', 0.08, 0.18),
            'max_depth': hp.choice('max_depth', [6, 7, 8, 9])
        }
        
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=25, trials=trials)
        
        self.hyperopt_trials['lightgbm'] = trials
        return space_eval(space, best)
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        try:
            best_params = self.optimize_lightgbm_hyperopt(X_train, y_train, X_val, y_val)
        except:
            best_params = {
                'num_leaves': 42,
                'learning_rate': 0.028,
                'feature_fraction': 0.82,
                'bagging_fraction': 0.87,
                'min_child_weight': 7,
                'min_split_gain': 0.08,
                'reg_alpha': 0.08,
                'reg_lambda': 0.12,
                'max_depth': 7
            }
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True,
            **best_params
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
            num_boost_round=2000,
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
            'max_depth': 6,
            'learning_rate': 0.028,
            'subsample': 0.87,
            'colsample_bytree': 0.82,
            'reg_alpha': 0.08,
            'reg_lambda': 0.12,
            'min_child_weight': 7,
            'gamma': 0.08,
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
            num_boost_round=2000,
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
            iterations=2000,
            learning_rate=0.028,
            depth=6,
            l2_leaf_reg=2.8,
            bootstrap_type='Bernoulli',
            subsample=0.85,
            colsample_bylevel=0.82,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=120,
            task_type='CPU',
            thread_count=1
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
            n_estimators=450,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            max_features=0.78,
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
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Gradient Boosting 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = GradientBoostingClassifier(
            n_estimators=280,
            learning_rate=0.058,
            max_depth=6,
            min_samples_split=12,
            min_samples_leaf=6,
            subsample=0.87,
            random_state=42
        )
        
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        model.fit(X_train_clean, y_train_clean, sample_weight=sample_weight)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['gradient_boosting'] = model
        return model, accuracy
    
    def train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Extra Trees 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.82,
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
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """신경망 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = MLPClassifier(
            hidden_layer_sizes=(180, 95, 45),
            activation='relu',
            solver='adam',
            alpha=0.0008,
            learning_rate='adaptive',
            learning_rate_init=0.0008,
            max_iter=1800,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """SVM 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 데이터 크기가 클 경우 샘플링
        if len(X_train_clean) > 8000:
            from sklearn.utils import resample
            X_sample, y_sample = resample(X_train_clean, y_train_clean, n_samples=8000, random_state=42, stratify=y_train_clean)
        else:
            X_sample, y_sample = X_train_clean, y_train_clean
        
        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight=self.class_weights,
            probability=True,
            random_state=42
        )
        
        model.fit(X_sample, y_sample)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['svm'] = model
        return model, accuracy
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """로지스틱 회귀 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = LogisticRegression(
            C=0.8,
            penalty='l2',
            class_weight=self.class_weights,
            max_iter=2000,
            random_state=42,
            solver='liblinear'
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['logistic_regression'] = model
        return model, accuracy
    
    def create_stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """스태킹 앙상블"""
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'gradient_boosting']
        available_models = [name for name in base_models if name in self.models]
        
        if len(available_models) < 3:
            return None, 0.0
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 메타 피처 생성 (5-fold CV)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features_train = np.zeros((len(X_train_clean), len(available_models) * 3))
        meta_features_val = np.zeros((len(X_val_clean), len(available_models) * 3))
        
        for model_idx, model_name in enumerate(available_models):
            model = self.models[model_name]
            
            # 교차 검증으로 메타 피처 생성
            for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf.split(X_train_clean, y_train_clean)):
                X_fold_train = X_train_clean[fold_train_idx]
                y_fold_train = y_train_clean[fold_train_idx]
                X_fold_val = X_train_clean[fold_val_idx]
                
                # 임시 모델 학습
                if model_name == 'lightgbm':
                    temp_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=200)
                elif model_name == 'xgboost':
                    temp_model = xgb.XGBClassifier(random_state=42, verbosity=0, n_estimators=200)
                elif model_name == 'catboost':
                    temp_model = CatBoostClassifier(random_seed=42, verbose=0, iterations=200)
                elif model_name == 'random_forest':
                    temp_model = RandomForestClassifier(random_state=42, n_estimators=200)
                else:
                    temp_model = GradientBoostingClassifier(random_state=42, n_estimators=200)
                
                temp_model.fit(X_fold_train, y_fold_train)
                
                if hasattr(temp_model, 'predict_proba'):
                    fold_pred = temp_model.predict_proba(X_fold_val)
                    meta_features_train[fold_val_idx, model_idx*3:(model_idx+1)*3] = fold_pred
                
            # 검증 데이터에 대한 예측
            if model_name == 'lightgbm':
                val_pred = model.predict(X_val_clean)
                if val_pred.ndim == 2 and val_pred.shape[1] == 3:
                    meta_features_val[:, model_idx*3:(model_idx+1)*3] = val_pred
                    
            elif model_name == 'xgboost':
                val_data = xgb.DMatrix(X_val_clean, feature_names=self.feature_names)
                val_pred = model.predict(val_data)
                if val_pred.ndim == 2 and val_pred.shape[1] == 3:
                    meta_features_val[:, model_idx*3:(model_idx+1)*3] = val_pred
                    
            else:
                if hasattr(model, 'predict_proba'):
                    val_pred = model.predict_proba(X_val_clean)
                    if val_pred.shape[1] == 3:
                        meta_features_val[:, model_idx*3:(model_idx+1)*3] = val_pred
        
        # 메타 모델 학습 (신경망 사용)
        meta_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        meta_model.fit(meta_features_train, y_train_clean)
        
        y_pred = meta_model.predict(meta_features_val)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['stacking'] = {
            'base_models': available_models,
            'meta_model': meta_model,
            'base_model_objects': {name: self.models[name] for name in available_models}
        }
        
        return meta_model, accuracy
    
    def create_voting_ensemble(self, X_train, y_train, X_val, y_val):
        """보팅 앙상블"""
        base_models = ['random_forest', 'extra_trees', 'gradient_boosting']
        available_models = [(name, self.models[name]) for name in base_models if name in self.models]
        
        if len(available_models) < 2:
            return None, 0.0
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        voting_model = VotingClassifier(
            estimators=available_models,
            voting='soft'
        )
        
        voting_model.fit(X_train_clean, y_train_clean)
        
        y_pred = voting_model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['voting'] = voting_model
        return voting_model, accuracy
    
    def optimize_ensemble_weights_advanced(self, X_val, y_val):
        """앙상블 가중치 최적화"""
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 각 모델의 예측 수집
        model_predictions = {}
        model_scores = {}
        
        for name, model in self.models.items():
            if name in ['stacking', 'voting']:
                continue
                
            try:
                if name == 'lightgbm':
                    y_pred_proba = model.predict(X_val_clean)
                    if y_pred_proba.ndim == 2:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        model_predictions[name] = y_pred_proba
                        
                elif name == 'xgboost':
                    val_data = xgb.DMatrix(X_val_clean, feature_names=self.feature_names)
                    y_pred_proba = model.predict(val_data)
                    if y_pred_proba.ndim == 2:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        model_predictions[name] = y_pred_proba
                        
                else:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val_clean)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        model_predictions[name] = y_pred_proba
                    else:
                        y_pred = model.predict(X_val_clean)
                        model_predictions[name] = None
                
                accuracy = accuracy_score(y_val_clean, y_pred)
                f1 = f1_score(y_val_clean, y_pred, average='macro')
                
                # 다양성 점수 계산
                diversity_score = 1.0
                for other_name, other_model in self.models.items():
                    if other_name != name and other_name not in ['stacking', 'voting']:
                        try:
                            if other_name == 'lightgbm':
                                other_pred_proba = other_model.predict(X_val_clean)
                                if other_pred_proba.ndim == 2:
                                    other_pred = np.argmax(other_pred_proba, axis=1)
                            elif other_name == 'xgboost':
                                other_data = xgb.DMatrix(X_val_clean, feature_names=self.feature_names)
                                other_pred_proba = other_model.predict(other_data)
                                if other_pred_proba.ndim == 2:
                                    other_pred = np.argmax(other_pred_proba, axis=1)
                            else:
                                other_pred = other_model.predict(X_val_clean)
                            
                            # 다양성 계산 (불일치율)
                            disagreement = np.mean(y_pred != other_pred)
                            diversity_score *= (1 + disagreement)
                        except:
                            continue
                
                # 종합 점수 (성능 + 다양성)
                combined_score = 0.75 * accuracy + 0.15 * f1 + 0.1 * min(diversity_score, 2.0)
                model_scores[name] = combined_score
                
            except Exception:
                model_scores[name] = 0.0
        
        # 가중치 분배
        base_weights = {
            'lightgbm': 0.30,
            'xgboost': 0.27,
            'catboost': 0.23,
            'random_forest': 0.12,
            'gradient_boosting': 0.05,
            'extra_trees': 0.02,
            'neural_network': 0.008,
            'svm': 0.005,
            'logistic_regression': 0.002
        }
        
        # 성능 기반 조정
        total_score = sum(model_scores.values())
        if total_score > 0:
            for name in model_scores:
                performance_ratio = model_scores[name] / total_score
                base_weight = base_weights.get(name, 0.01)
                self.ensemble_weights[name] = 0.65 * base_weight + 0.35 * performance_ratio
        else:
            self.ensemble_weights = base_weights.copy()
        
        # 정규화
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            for name in self.ensemble_weights:
                self.ensemble_weights[name] /= total_weight
        
        return self.ensemble_weights
    
    def save_models(self, engineer, preprocessor):
        """모델 저장"""
        os.makedirs('models', exist_ok=True)
        
        # 전처리기 저장
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        joblib.dump(engineer, 'models/feature_engineer.pkl')
        
        # 피처 정보 저장
        feature_info = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'class_weights': self.class_weights,
            'ensemble_weights': self.ensemble_weights
        }
        joblib.dump(feature_info, 'models/feature_info.pkl')
        
        # 모델 저장
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
                
            except Exception:
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
        
        # LightGBM
        try:
            lgb_model, lgb_acc = self.train_lightgbm(X_train, y_train, X_val, y_val)
            model_results['lightgbm'] = lgb_acc
        except Exception:
            model_results['lightgbm'] = 0.0
        
        # XGBoost
        try:
            xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val)
            model_results['xgboost'] = xgb_acc
        except Exception:
            model_results['xgboost'] = 0.0
        
        # CatBoost
        try:
            cat_model, cat_acc = self.train_catboost(X_train, y_train, X_val, y_val)
            model_results['catboost'] = cat_acc
        except Exception:
            model_results['catboost'] = 0.0
        
        # Random Forest
        try:
            rf_model, rf_acc = self.train_random_forest(X_train, y_train, X_val, y_val)
            model_results['random_forest'] = rf_acc
        except Exception:
            model_results['random_forest'] = 0.0
        
        # Gradient Boosting
        try:
            gb_model, gb_acc = self.train_gradient_boosting(X_train, y_train, X_val, y_val)
            model_results['gradient_boosting'] = gb_acc
        except Exception:
            model_results['gradient_boosting'] = 0.0
        
        # Extra Trees
        try:
            et_model, et_acc = self.train_extra_trees(X_train, y_train, X_val, y_val)
            model_results['extra_trees'] = et_acc
        except Exception:
            model_results['extra_trees'] = 0.0
        
        # Neural Network
        try:
            nn_model, nn_acc = self.train_neural_network(X_train, y_train, X_val, y_val)
            model_results['neural_network'] = nn_acc
        except Exception:
            model_results['neural_network'] = 0.0
        
        # SVM
        try:
            svm_model, svm_acc = self.train_svm(X_train, y_train, X_val, y_val)
            model_results['svm'] = svm_acc
        except Exception:
            model_results['svm'] = 0.0
        
        # Logistic Regression
        try:
            lr_model, lr_acc = self.train_logistic_regression(X_train, y_train, X_val, y_val)
            model_results['logistic_regression'] = lr_acc
        except Exception:
            model_results['logistic_regression'] = 0.0
        
        # 앙상블 가중치 최적화
        self.optimize_ensemble_weights_advanced(X_val, y_val)
        
        # 스태킹 앙상블
        try:
            stacking_model, stacking_acc = self.create_stacking_ensemble(X_train, y_train, X_val, y_val)
            if stacking_model is not None:
                model_results['stacking'] = stacking_acc
        except Exception:
            pass
        
        # 보팅 앙상블
        try:
            voting_model, voting_acc = self.create_voting_ensemble(X_train, y_train, X_val, y_val)
            if voting_model is not None:
                model_results['voting'] = voting_acc
        except Exception:
            pass
        
        # 모델 저장
        self.save_models(engineer, preprocessor)
        
        # 결과 출력
        valid_results = {k: v for k, v in model_results.items() if v > 0}
        
        if valid_results:
            best_model = max(valid_results.items(), key=lambda x: x[1])

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