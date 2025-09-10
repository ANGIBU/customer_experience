# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.mixture import GaussianMixture
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
from feature_engineering import FeatureEngineer, TemporalSPOSampler
from preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class DynamicWeightCalculator:
    """동적 클래스 가중치 계산"""
    
    def __init__(self, base_weights={0: 1.0, 1: 12.0, 2: 18.0}):
        self.base_weights = base_weights
        self.adaptation_history = []
    
    def calculate_adaptive_weights(self, y_true, y_pred_proba=None, epoch=0):
        """동적 가중치 계산"""
        y_array = np.array(y_true)
        class_counts = np.bincount(y_array, minlength=3)
        total_samples = len(y_array)
        
        # 시간 감소 인수
        temporal_factor = 1 + (0.3 * np.exp(-epoch/15))
        
        # 예측 신뢰도 기반 조정
        if y_pred_proba is not None:
            confidence_scores = np.max(y_pred_proba, axis=1)
            avg_confidence_per_class = []
            for i in range(3):
                mask = y_array == i
                if mask.sum() > 0:
                    avg_confidence_per_class.append(np.mean(confidence_scores[mask]))
                else:
                    avg_confidence_per_class.append(0.5)
        else:
            avg_confidence_per_class = [0.5, 0.5, 0.5]
        
        # 동적 가중치 계산
        dynamic_weights = {}
        for class_idx in range(3):
            base_weight = self.base_weights.get(class_idx, 1.0)
            
            if class_counts[class_idx] > 0:
                frequency_adjustment = total_samples / (3 * class_counts[class_idx])
            else:
                frequency_adjustment = 1.0
                
            confidence_adjustment = 1.0 / (avg_confidence_per_class[class_idx] + 0.15)
            
            dynamic_weights[class_idx] = base_weight * temporal_factor * frequency_adjustment * confidence_adjustment
        
        return dynamic_weights

class HierarchicalPortfolioStacker:
    """계층적 스태킹"""
    
    def __init__(self):
        self.base_models = self._initialize_base_models()
        self.meta_models = {}
        self.final_aggregator = None
        self.model_weights = {}
        self.weight_calculator = DynamicWeightCalculator()
        
    def _initialize_base_models(self):
        """기본 모델 초기화"""
        models = {
            'xgb': XGBClassifier(
                n_estimators=250, learning_rate=0.04, max_depth=7,
                subsample=0.82, colsample_bytree=0.85,
                objective='multi:softprob', eval_metric='mlogloss',
                random_state=42, verbosity=0
            ),
            'lgb': LGBMClassifier(
                num_leaves=55, learning_rate=0.04,
                feature_fraction=0.82, bagging_fraction=0.78,
                objective='multiclass', random_state=42, verbose=-1
            ),
            'cat': CatBoostClassifier(
                iterations=250, learning_rate=0.04, depth=7,
                verbose=False, random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=280, max_depth=14,
                min_samples_split=5, min_samples_leaf=2,
                max_features=0.85, random_state=42, n_jobs=-1
            ),
            'et': ExtraTreesClassifier(
                n_estimators=260, max_depth=13,
                min_samples_split=4, min_samples_leaf=2,
                max_features=0.88, random_state=42, n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=220, learning_rate=0.06, max_depth=8,
                min_samples_split=8, min_samples_leaf=4,
                subsample=0.85, random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(160, 80, 40), activation='relu',
                solver='adam', alpha=0.001, learning_rate='adaptive',
                learning_rate_init=0.001, max_iter=1500,
                random_state=42, early_stopping=True
            )
        }
        return models
    
    def fit(self, X, y):
        """계층적 학습 수행"""
        # 1단계: 기본 모델 학습
        level1_features = self._generate_level1_features(X, y)
        
        # 2단계: 메타 모델 학습
        self._train_meta_models(level1_features, y)
        
        # 3단계: 최종 집계기 학습
        self._train_final_aggregator(level1_features, y)
        
    def _generate_level1_features(self, X, y):
        """1차 피처 생성"""
        n_models = len(self.base_models)
        level1_features = np.zeros((len(X), n_models * 3))
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 동적 가중치 계산
            dynamic_weights = self.weight_calculator.calculate_adaptive_weights(y_train, epoch=fold)
            
            for i, (model_name, model) in enumerate(self.base_models.items()):
                # SPO 샘플링 적용
                spo_sampler = TemporalSPOSampler(sampling_strategy={1: 0.7, 2: 0.85})
                X_resampled, y_resampled = spo_sampler.fit_resample(X_train.values, y_train.values)
                
                # 모델별 가중치 설정
                if hasattr(model, 'class_weight'):
                    model.set_params(class_weight=dynamic_weights)
                elif model_name in ['xgb', 'lgb', 'cat']:
                    # 샘플 가중치 적용
                    sample_weights = np.array([dynamic_weights[int(label)] for label in y_resampled])
                    
                    if model_name == 'xgb':
                        model.fit(X_resampled, y_resampled, sample_weight=sample_weights)
                    elif model_name == 'lgb':
                        model.fit(X_resampled, y_resampled, sample_weight=sample_weights)
                    elif model_name == 'cat':
                        model.fit(X_resampled, y_resampled, sample_weight=sample_weights)
                else:
                    model.fit(X_resampled, y_resampled)
                
                predictions = model.predict_proba(X_val)
                
                start_col = i * 3
                level1_features[val_idx, start_col:start_col+3] = predictions
        
        return level1_features
    
    def _train_meta_models(self, level1_features, y):
        """메타 모델 학습"""
        # 메타 피처 생성
        meta_features = self._engineer_meta_features(level1_features)
        
        # 다양한 메타 모델 학습
        self.meta_models['xgb_meta'] = XGBClassifier(
            n_estimators=120, learning_rate=0.02, max_depth=5,
            subsample=0.8, random_state=42, verbosity=0
        )
        
        self.meta_models['lgb_meta'] = LGBMClassifier(
            num_leaves=35, learning_rate=0.02,
            feature_fraction=0.8, random_state=42, verbose=-1
        )
        
        self.meta_models['lr_meta'] = LogisticRegression(
            multi_class='multinomial', C=0.1, max_iter=1000,
            random_state=42
        )
        
        # 메타 모델 학습
        for name, model in self.meta_models.items():
            dynamic_weights = self.weight_calculator.calculate_adaptive_weights(y)
            
            if hasattr(model, 'class_weight'):
                model.set_params(class_weight=dynamic_weights)
            
            model.fit(meta_features, y)
    
    def _engineer_meta_features(self, level1_features):
        """메타 피처 생성"""
        n_models = len(self.base_models)
        
        # 기본 예측
        raw_predictions = level1_features
        
        # 예측 신뢰도
        entropy_features = []
        for i in range(n_models):
            start_col = i * 3
            model_probs = level1_features[:, start_col:start_col+3]
            entropy = -np.sum(model_probs * np.log(model_probs + 1e-15), axis=1)
            entropy_features.append(entropy)
        
        # 모델 간 일치도
        agreement_features = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                prob_i = level1_features[:, i*3:(i+1)*3]
                prob_j = level1_features[:, j*3:(j+1)*3]
                agreement = np.sum(prob_i * prob_j, axis=1)
                agreement_features.append(agreement)
        
        # 예측 분산
        pred_variance = []
        for class_idx in range(3):
            class_predictions = level1_features[:, class_idx::3]
            variance = np.var(class_predictions, axis=1)
            pred_variance.append(variance)
        
        # 최대 예측값
        max_predictions = []
        for i in range(n_models):
            start_col = i * 3
            model_probs = level1_features[:, start_col:start_col+3]
            max_pred = np.max(model_probs, axis=1)
            max_predictions.append(max_pred)
        
        # 모든 메타 피처 결합
        meta_features = np.column_stack([
            raw_predictions,
            np.column_stack(entropy_features),
            np.column_stack(agreement_features),
            np.column_stack(pred_variance),
            np.column_stack(max_predictions)
        ])
        
        return meta_features
    
    def _train_final_aggregator(self, level1_features, y):
        """최종 집계기 학습"""
        meta_features = self._engineer_meta_features(level1_features)
        
        # 메타 모델 예측 생성
        meta_predictions = np.zeros((len(y), len(self.meta_models) * 3))
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(meta_features, y):
            for i, (name, model) in enumerate(self.meta_models.items()):
                pred = model.predict_proba(meta_features[val_idx])
                start_col = i * 3
                meta_predictions[val_idx, start_col:start_col+3] = pred
        
        # 최종 집계기
        dynamic_weights = self.weight_calculator.calculate_adaptive_weights(y)
        self.final_aggregator = LogisticRegression(
            multi_class='multinomial', C=0.05, max_iter=800,
            class_weight=dynamic_weights, random_state=42
        )
        
        # 최종 피처: 원본 + 메타 예측
        final_features = np.column_stack([level1_features, meta_predictions])
        
        self.final_aggregator.fit(final_features, y)
    
    def predict_proba(self, X):
        """확률 예측"""
        # 1차 예측 생성
        level1_preds = np.zeros((len(X), len(self.base_models) * 3))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            pred = model.predict_proba(X)
            start_col = i * 3
            level1_preds[:, start_col:start_col+3] = pred
        
        # 메타 피처 생성
        meta_features = self._engineer_meta_features(level1_preds)
        
        # 메타 모델 예측
        meta_preds = np.zeros((len(X), len(self.meta_models) * 3))
        for i, (name, model) in enumerate(self.meta_models.items()):
            pred = model.predict_proba(meta_features)
            start_col = i * 3
            meta_preds[:, start_col:start_col+3] = pred
        
        # 최종 예측
        final_features = np.column_stack([level1_preds, meta_preds])
        final_predictions = self.final_aggregator.predict_proba(final_features)
        
        return final_predictions

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.feature_names = None
        self.class_weights = None
        self.best_threshold = 0.5
        self.ensemble_weights = {}
        self.hierarchical_stacker = None
        
    def calculate_class_weights(self, y_train):
        """클래스 가중치 계산"""
        weight_calc = DynamicWeightCalculator()
        self.class_weights = weight_calc.calculate_adaptive_weights(y_train)
        return self.class_weights
    
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
            
            # 데이터 분석으로부터 temporal_threshold 가져오기
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
                train_df, test_df, val_size=0.16, gap_size=0.04
            )
            
            if X_train is None or X_val is None:
                raise ValueError("데이터 분할 실패")
            
            self.feature_names = list(X_train.columns)
            self.calculate_class_weights(y_train)
            
            return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
            
        except Exception as e:
            print(f"훈련 데이터 준비 오류: {e}")
            return None, None, None, None, None, None, None, None
    
    def train_hierarchical_stacker(self, X_train, y_train):
        """계층적 스태킹 훈련"""
        self.hierarchical_stacker = HierarchicalPortfolioStacker()
        self.hierarchical_stacker.fit(X_train, y_train)
        return self.hierarchical_stacker
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 58,
            'learning_rate': 0.026,
            'feature_fraction': 0.84,
            'bagging_fraction': 0.86,
            'bagging_freq': 5,
            'min_child_weight': 12,
            'min_split_gain': 0.10,
            'reg_alpha': 0.10,
            'reg_lambda': 0.10,
            'max_depth': 9,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 클래스 가중치 적용
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
            num_boost_round=2200,
            callbacks=[lgb.early_stopping(140), lgb.log_evaluation(0)]
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
            'max_depth': 8,
            'learning_rate': 0.026,
            'subsample': 0.86,
            'colsample_bytree': 0.84,
            'reg_alpha': 0.10,
            'reg_lambda': 0.10,
            'min_child_weight': 12,
            'gamma': 0.10,
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'hist'
        }
        
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 클래스 가중치 적용
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        train_data = xgb.DMatrix(X_train_clean, label=y_train_clean, weight=sample_weight, feature_names=self.feature_names)
        val_data = xgb.DMatrix(X_val_clean, label=y_val_clean, feature_names=self.feature_names)
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=2200,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=140,
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
        
        # 클래스 가중치 적용
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        model = CatBoostClassifier(
            iterations=2200,
            learning_rate=0.026,
            depth=8,
            l2_leaf_reg=5,
            bootstrap_type='Bernoulli',
            subsample=0.84,
            colsample_bylevel=0.84,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=140,
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
    
    def optimize_ensemble_weights(self, X_val, y_val):
        """앙상블 가중치 최적화"""
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 각 모델의 성능 평가
        model_scores = {}
        
        for name, model in self.models.items():
            if name == 'hierarchical_stacker':
                continue
                
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
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val_clean)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                    else:
                        y_pred = model.predict(X_val_clean)
                
                accuracy = accuracy_score(y_val_clean, y_pred)
                f1 = f1_score(y_val_clean, y_pred, average='macro')
                
                # 성능 점수
                combined_score = 0.7 * accuracy + 0.3 * f1
                model_scores[name] = combined_score
                
            except Exception as e:
                model_scores[name] = 0.0
        
        # 계층적 스태킹 성능 추가
        if self.hierarchical_stacker is not None:
            try:
                y_pred_proba = self.hierarchical_stacker.predict_proba(X_val)
                y_pred = np.argmax(y_pred_proba, axis=1)
                accuracy = accuracy_score(y_val_clean, y_pred)
                f1 = f1_score(y_val_clean, y_pred, average='macro')
                combined_score = 0.7 * accuracy + 0.3 * f1
                model_scores['hierarchical_stacker'] = combined_score
            except Exception:
                model_scores['hierarchical_stacker'] = 0.0
        
        # 가중치 정규화
        total_score = sum(model_scores.values())
        if total_score > 0:
            for name in model_scores:
                self.ensemble_weights[name] = model_scores[name] / total_score
        else:
            # 균등 가중치
            num_models = len(model_scores)
            for name in model_scores:
                self.ensemble_weights[name] = 1.0 / num_models
        
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
        
        # 계층적 스태킹 저장
        if self.hierarchical_stacker is not None:
            joblib.dump(self.hierarchical_stacker, 'models/hierarchical_stacker.pkl')
        
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
        
        # 계층적 스태킹
        try:
            hierarchical_stacker = self.train_hierarchical_stacker(X_train, y_train)
            y_pred_proba = hierarchical_stacker.predict_proba(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
            hs_acc = accuracy_score(y_val, y_pred)
            model_results['hierarchical_stacker'] = hs_acc
        except Exception:
            model_results['hierarchical_stacker'] = 0.0
        
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
        
        # 앙상블 가중치 최적화
        self.optimize_ensemble_weights(X_val, y_val)
        
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
            print("훈련 데이터 준비 실패")
            return None
            
    except Exception as e:
        print(f"모델 훈련 오류: {e}")
        return None

if __name__ == "__main__":
    main()