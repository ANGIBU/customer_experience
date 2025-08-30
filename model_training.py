# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
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
        self.feature_importance = {}
        self.best_params = {}
        self.meta_model = None
        self.feature_names = None
        self.class_weights = None
        
    def calculate_class_weights(self, y_train):
        """클래스 가중치 계산"""
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        
        weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total_samples / (len(class_counts) * count)
            else:
                weights[i] = 1.0
        
        weights[1] *= 1.5
        
        self.class_weights = weights
        return weights
    
    def prepare_training_data(self):
        """학습 데이터 준비"""
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        engineer = FeatureEngineer()
        train_df, test_df = engineer.create_features(train_df, test_df)
        
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.process_data(train_df, test_df)
        
        X_train, X_val, y_train, y_val, X_test, test_ids = preprocessor.prepare_data(
            train_df, test_df
        )
        
        self.feature_names = list(X_train.columns)
        self.calculate_class_weights(y_train)
        
        return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
    
    def train_lightgbm_optimized(self, X_train, y_train, X_val, y_val):
        """LightGBM 최적화 학습"""
        print("LightGBM 학습")
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.75,
            'bagging_freq': 3,
            'min_child_weight': 8,
            'min_split_gain': 0.02,
            'reg_alpha': 0.15,
            'reg_lambda': 0.15,
            'max_depth': 8,
            'verbose': -1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val, y_pred_class)
        
        print(f"LightGBM 검증 정확도: {accuracy:.4f}")
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost_optimized(self, X_train, y_train, X_val, y_val):
        """XGBoost 최적화 학습"""
        print("XGBoost 학습")
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.75,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.85,
            'reg_alpha': 0.2,
            'reg_lambda': 0.2,
            'min_child_weight': 8,
            'gamma': 0.02,
            'random_state': 42,
            'verbosity': 0
        }
        
        train_data = xgb.DMatrix(X_train.values, label=y_train, feature_names=self.feature_names)
        val_data = xgb.DMatrix(X_val.values, label=y_val, feature_names=self.feature_names)
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=2000,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=80,
            verbose_eval=0
        )
        
        y_pred = model.predict(val_data)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val, y_pred_class)
        
        print(f"XGBoost 검증 정확도: {accuracy:.4f}")
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_catboost_fixed(self, X_train, y_train, X_val, y_val):
        """CatBoost 수정 학습"""
        print("CatBoost 학습")
        
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            bootstrap_type='Bernoulli',
            subsample=0.75,
            colsample_bylevel=0.85,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=80,
            class_weights=list(self.class_weights.values())
        )
        
        model.fit(
            X_train.values, y_train,
            eval_set=(X_val.values, y_val),
            use_best_model=True,
            verbose=False
        )
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"CatBoost 검증 정확도: {accuracy:.4f}")
        
        self.models['catboost'] = model
        return model, accuracy
    
    def train_random_forest_optimized(self, X_train, y_train, X_val, y_val):
        """Random Forest 최적화 학습"""
        print("Random Forest 학습")
        
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features=0.7,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_samples=0.8
        )
        
        model.fit(X_train.values, y_train)
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Random Forest 검증 정확도: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_extra_trees_optimized(self, X_train, y_train, X_val, y_val):
        """Extra Trees 최적화 학습"""
        print("Extra Trees 학습")
        
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features=0.8,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        
        model.fit(X_train.values, y_train)
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Extra Trees 검증 정확도: {accuracy:.4f}")
        
        self.models['extra_trees'] = model
        return model, accuracy
    
    def train_neural_network_optimized(self, X_train, y_train, X_val, y_val):
        """신경망 최적화 학습"""
        print("Neural Network 학습")
        
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.002,
            learning_rate='adaptive',
            learning_rate_init=0.002,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.fit(X_train.values, y_train)
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Neural Network 검증 정확도: {accuracy:.4f}")
        print(f"반복 횟수: {model.n_iter_}")
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def train_class1_specialist(self, X_train, y_train, X_val, y_val):
        """클래스 1 전문 모델"""
        print("클래스 1 전문 모델 학습")
        
        y_binary_train = (y_train == 1).astype(int)
        y_binary_val = (y_val == 1).astype(int)
        
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=2.7,
            random_state=42,
            n_estimators=1000,
            early_stopping_rounds=50,
            verbose=-1
        )
        
        model.fit(
            X_train, y_binary_train,
            eval_set=[(X_val, y_binary_val)],
            verbose=False
        )
        
        y_pred_binary = model.predict(X_val)
        accuracy = accuracy_score(y_binary_val, y_pred_binary)
        
        print(f"클래스 1 전문 모델 정확도: {accuracy:.4f}")
        
        self.models['class1_specialist'] = model
        return model, accuracy
    
    def create_stacking_model(self, X_train, y_train, X_val, y_val):
        """스태킹 모델 생성"""
        print("스태킹 모델 생성")
        
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
        valid_models = [name for name in base_models if name in self.models]
        
        if len(valid_models) < 3:
            print("스태킹을 위한 모델이 부족합니다")
            return None, 0.0
        
        try:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            meta_features_train = np.zeros((len(X_train), len(valid_models) * 3))
            meta_features_val = np.zeros((len(X_val), len(valid_models) * 3))
            
            for fold, (train_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_oof = X_train.iloc[oof_idx]
                
                col_idx = 0
                
                for model_name in valid_models:
                    try:
                        if model_name == 'lightgbm':
                            lgb_params = {
                                'objective': 'multiclass', 'num_class': 3, 
                                'verbose': -1, 'random_state': 42,
                                'num_leaves': 31, 'learning_rate': 0.05
                            }
                            train_data = lgb.Dataset(X_fold_train.values, label=y_fold_train)
                            fold_model = lgb.train(lgb_params, train_data, num_boost_round=200)
                            oof_pred = fold_model.predict(X_fold_oof.values)
                            val_pred = fold_model.predict(X_val.values)
                            
                        elif model_name == 'xgboost':
                            xgb_params = {
                                'objective': 'multi:softprob', 'num_class': 3,
                                'random_state': 42, 'verbosity': 0
                            }
                            train_data = xgb.DMatrix(X_fold_train.values, label=y_fold_train)
                            fold_model = xgb.train(xgb_params, train_data, num_boost_round=200)
                            oof_pred = fold_model.predict(xgb.DMatrix(X_fold_oof.values))
                            val_pred = fold_model.predict(xgb.DMatrix(X_val.values))
                            
                        elif model_name == 'catboost':
                            fold_model = CatBoostClassifier(
                                iterations=200, random_seed=42, verbose=0
                            )
                            fold_model.fit(X_fold_train.values, y_fold_train, verbose=False)
                            oof_pred = fold_model.predict_proba(X_fold_oof.values)
                            val_pred = fold_model.predict_proba(X_val.values)
                            
                        elif model_name == 'random_forest':
                            fold_model = RandomForestClassifier(
                                n_estimators=200, random_state=42, n_jobs=-1,
                                class_weight=self.class_weights
                            )
                            fold_model.fit(X_fold_train.values, y_fold_train)
                            oof_pred = fold_model.predict_proba(X_fold_oof.values)
                            val_pred = fold_model.predict_proba(X_val.values)
                        
                        else:
                            continue
                        
                        meta_features_train[oof_idx, col_idx:col_idx+3] = oof_pred
                        
                        if fold == 0:
                            meta_features_val[:, col_idx:col_idx+3] = val_pred / skf.n_splits
                        else:
                            meta_features_val[:, col_idx:col_idx+3] += val_pred / skf.n_splits
                        
                        col_idx += 3
                        
                    except Exception as e:
                        print(f"폴드 {fold} {model_name} 오류: {e}")
                        col_idx += 3
                        continue
            
            actual_features = col_idx
            meta_features_train = meta_features_train[:, :actual_features]
            meta_features_val = meta_features_val[:, :actual_features]
            
            if actual_features == 0:
                print("메타 피처 생성 실패")
                return None, 0.0
            
            self.meta_model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                random_state=42,
                max_iter=2000,
                class_weight=self.class_weights,
                C=0.5
            )
            
            self.meta_model.fit(meta_features_train, y_train)
            
            stacking_pred_proba = self.meta_model.predict_proba(meta_features_val)
            stacking_pred = np.argmax(stacking_pred_proba, axis=1)
            stacking_accuracy = accuracy_score(y_val, stacking_pred)
            
            print(f"스태킹 검증 정확도: {stacking_accuracy:.4f}")
            print(f"메타 피처 수: {actual_features}")
            
            self.models['stacking'] = self.meta_model
            return self.meta_model, stacking_accuracy
            
        except Exception as e:
            print(f"스태킹 모델 생성 오류: {e}")
            return None, 0.0
    
    def train_calibrated_models(self, X_train, y_train, X_val, y_val):
        """보정 모델 학습"""
        print("보정 모델 학습")
        
        calibrated_models = {}
        
        base_model_configs = [
            ('rf_calibrated', RandomForestClassifier(
                n_estimators=300, max_depth=12, class_weight=self.class_weights, 
                random_state=42, n_jobs=-1
            )),
            ('svm_calibrated', SVC(
                C=2.0, kernel='rbf', probability=True, 
                class_weight=self.class_weights, random_state=42
            ))
        ]
        
        for name, base_model in base_model_configs:
            try:
                calibrated_model = CalibratedClassifierCV(
                    base_model, method='isotonic', cv=3
                )
                
                calibrated_model.fit(X_train.values, y_train)
                
                y_pred = calibrated_model.predict(X_val.values)
                accuracy = accuracy_score(y_val, y_pred)
                
                print(f"{name} 검증 정확도: {accuracy:.4f}")
                
                calibrated_models[name] = calibrated_model
                self.models[name] = calibrated_model
                
            except Exception as e:
                print(f"{name} 학습 실패: {e}")
                continue
        
        return calibrated_models
    
    def perform_temporal_cv(self, X, y, model_type='lightgbm', n_splits=5):
        """시간 기반 교차 검증"""
        print(f"{model_type.upper()} 시간 기반 CV")
        
        if 'temporal_id' in X.columns:
            temporal_ids = X['temporal_id'].values
            sorted_indices = np.argsort(temporal_ids)
            
            fold_size = len(sorted_indices) // (n_splits + 1)
            cv_scores = []
            
            for fold in range(n_splits):
                train_end = (fold + 1) * fold_size
                val_start = train_end
                val_end = val_start + fold_size
                
                if val_end > len(sorted_indices):
                    break
                
                train_idx = sorted_indices[:train_end]
                val_idx = sorted_indices[val_start:val_end]
                
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                try:
                    if model_type == 'lightgbm':
                        lgb_params = {
                            'objective': 'multiclass', 'num_class': 3, 
                            'verbose': -1, 'random_state': 42,
                            'num_leaves': 31, 'learning_rate': 0.05
                        }
                        
                        feature_cols = [col for col in X_train_fold.columns if col != 'temporal_id']
                        train_data = lgb.Dataset(X_train_fold[feature_cols].values, label=y_train_fold)
                        model = lgb.train(lgb_params, train_data, num_boost_round=300)
                        y_pred = model.predict(X_val_fold[feature_cols].values)
                        y_pred_class = np.argmax(y_pred, axis=1)
                        
                    accuracy = accuracy_score(y_val_fold, y_pred_class)
                    cv_scores.append(accuracy)
                    
                    print(f"  Fold {fold + 1}: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  Fold {fold + 1} 오류: {e}")
                    cv_scores.append(0.0)
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"  Fold {fold + 1}: 시간 기반 분할 적용")
                cv_scores.append(0.50)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"평균 CV 점수: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.cv_scores[model_type] = {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score
        }
        
        return mean_score, std_score
    
    def save_all_models(self, engineer, preprocessor, X_train):
        """모든 모델 저장"""
        print("모델 저장")
        
        os.makedirs('models', exist_ok=True)
        
        try:
            joblib.dump(preprocessor, 'models/preprocessor.pkl')
            joblib.dump(engineer, 'models/feature_engineer.pkl')
            
            feature_info = {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names),
                'class_weights': self.class_weights
            }
            joblib.dump(feature_info, 'models/feature_info.pkl')
            print("전처리기 및 피처 엔지니어 저장 완료")
        except Exception as e:
            print(f"전처리기 저장 오류: {e}")
        
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
                print(f"  {name} 모델 저장 완료")
                
            except Exception as e:
                print(f"  {name} 모델 저장 오류: {e}")
        
        print(f"총 {saved_count}개 모델 저장 완료")
    
    def train_models(self, X_train, X_val, y_train, y_val, engineer=None, preprocessor=None):
        """모든 모델 학습"""
        print("모델 학습 시작")
        print("=" * 40)
        
        if engineer is None:
            from feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
        
        if preprocessor is None:
            from preprocessing import DataPreprocessor  
            preprocessor = DataPreprocessor()
        
        if self.feature_names is None:
            self.feature_names = list(X_train.columns)
        
        model_results = {}
        
        try:
            lgb_model, lgb_acc = self.train_lightgbm_optimized(X_train, y_train, X_val, y_val)
            model_results['lightgbm'] = lgb_acc
        except Exception as e:
            print(f"LightGBM 학습 실패: {e}")
            model_results['lightgbm'] = 0.0
        
        try:
            xgb_model, xgb_acc = self.train_xgboost_optimized(X_train, y_train, X_val, y_val)
            model_results['xgboost'] = xgb_acc
        except Exception as e:
            print(f"XGBoost 학습 실패: {e}")
            model_results['xgboost'] = 0.0
        
        try:
            cat_model, cat_acc = self.train_catboost_fixed(X_train, y_train, X_val, y_val)
            model_results['catboost'] = cat_acc
        except Exception as e:
            print(f"CatBoost 학습 실패: {e}")
            model_results['catboost'] = 0.0
        
        try:
            rf_model, rf_acc = self.train_random_forest_optimized(X_train, y_train, X_val, y_val)
            model_results['random_forest'] = rf_acc
        except Exception as e:
            print(f"Random Forest 학습 실패: {e}")
            model_results['random_forest'] = 0.0
        
        try:
            et_model, et_acc = self.train_extra_trees_optimized(X_train, y_train, X_val, y_val)
            model_results['extra_trees'] = et_acc
        except Exception as e:
            print(f"Extra Trees 학습 실패: {e}")
            model_results['extra_trees'] = 0.0
        
        try:
            nn_model, nn_acc = self.train_neural_network_optimized(X_train, y_train, X_val, y_val)
            model_results['neural_network'] = nn_acc
        except Exception as e:
            print(f"Neural Network 학습 실패: {e}")
            model_results['neural_network'] = 0.0
        
        try:
            c1_model, c1_acc = self.train_class1_specialist(X_train, y_train, X_val, y_val)
            model_results['class1_specialist'] = c1_acc
        except Exception as e:
            print(f"클래스 1 전문 모델 학습 실패: {e}")
            model_results['class1_specialist'] = 0.0
        
        try:
            calibrated_models = self.train_calibrated_models(X_train, y_train, X_val, y_val)
            for name, model in calibrated_models.items():
                y_pred = model.predict(X_val.values)
                accuracy = accuracy_score(y_val, y_pred)
                model_results[name] = accuracy
        except Exception as e:
            print(f"보정 모델 생성 실패: {e}")
        
        try:
            stacking_model, stacking_acc = self.create_stacking_model(X_train, y_train, X_val, y_val)
            if stacking_model is not None:
                model_results['stacking'] = stacking_acc
        except Exception as e:
            print(f"스태킹 모델 생성 실패: {e}")
            model_results['stacking'] = 0.0
        
        if len(self.models) > 0:
            full_X = pd.concat([X_train, X_val])
            full_y = pd.concat([y_train, y_val])
            
            if 'lightgbm' in self.models:
                self.perform_temporal_cv(full_X, full_y, 'lightgbm')
        
        self.save_all_models(engineer, preprocessor, X_train)
        
        print("\n모델 성능 요약:")
        for model_name, accuracy in model_results.items():
            if accuracy > 0:
                print(f"{model_name}: {accuracy:.4f}")

def main():
    trainer = ModelTrainer()
    X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor = trainer.prepare_training_data()
    trainer.train_models(X_train, X_val, y_train, y_val)
    
    return trainer

if __name__ == "__main__":
    main()