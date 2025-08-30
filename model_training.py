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
        
        # 피처 이름 저장
        self.feature_names = list(X_train.columns)
        
        return X_train, X_val, y_train, y_val, X_test, test_ids, engineer, preprocessor
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        print("LightGBM 학습")
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val, y_pred_class)
        
        print(f"LightGBM 검증 정확도: {accuracy:.4f}")
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
        print("XGBoost 학습")
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
        
        # 피처 이름을 명시적으로 저장
        train_data = xgb.DMatrix(X_train.values, label=y_train, feature_names=self.feature_names)
        val_data = xgb.DMatrix(X_val.values, label=y_val, feature_names=self.feature_names)
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=1000,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=0
        )
        
        y_pred = model.predict(val_data)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val, y_pred_class)
        
        print(f"XGBoost 검증 정확도: {accuracy:.4f}")
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 모델 학습"""
        print("CatBoost 학습")
        
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            class_weights=list(class_weights.values()),
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50,
            feature_names=self.feature_names
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
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Random Forest 모델 학습"""
        print("Random Forest 학습")
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train.values, y_train)
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Random Forest 검증 정확도: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Extra Trees 모델 학습"""
        print("Extra Trees 학습")
        
        model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train.values, y_train)
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Extra Trees 검증 정확도: {accuracy:.4f}")
        
        self.models['extra_trees'] = model
        return model, accuracy
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """신경망 모델 학습"""
        print("Neural Network 학습")
        
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        model.fit(X_train.values, y_train)
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"Neural Network 검증 정확도: {accuracy:.4f}")
        print(f"반복 횟수: {model.n_iter_}")
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """SVM 모델 학습"""
        print("SVM 학습")
        
        model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train.values, y_train)
        
        y_pred = model.predict(X_val.values)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"SVM 검증 정확도: {accuracy:.4f}")
        
        self.models['svm'] = model
        return model, accuracy
    
    def perform_time_series_cv(self, X, y, model_type='lightgbm', n_splits=5):
        """시간 기반 교차 검증"""
        print(f"{model_type.upper()} 시간 기반 CV")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            try:
                if model_type == 'lightgbm':
                    lgb_params = {
                        'objective': 'multiclass',
                        'num_class': 3,
                        'metric': 'multi_logloss',
                        'verbose': -1,
                        'random_state': 42,
                        'num_leaves': 31,
                        'learning_rate': 0.05
                    }
                    
                    train_data = lgb.Dataset(X_train_fold.values, label=y_train_fold)
                    model = lgb.train(lgb_params, train_data, num_boost_round=200)
                    y_pred = model.predict(X_val_fold.values)
                    y_pred_class = np.argmax(y_pred, axis=1)
                    
                elif model_type == 'catboost':
                    model = CatBoostClassifier(
                        iterations=200,
                        learning_rate=0.05,
                        depth=6,
                        random_seed=42,
                        verbose=0
                    )
                    model.fit(X_train_fold.values, y_train_fold, verbose=False)
                    y_pred_class = model.predict(X_val_fold.values)
                
                accuracy = accuracy_score(y_val_fold, y_pred_class)
                cv_scores.append(accuracy)
                
                print(f"  Fold {fold + 1}: {accuracy:.4f}")
                
            except Exception as e:
                print(f"  Fold {fold + 1} 오류: {e}")
                cv_scores.append(0.0)
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"평균 CV 점수: {mean_score:.4f} (+/- {std_score:.4f})")
        
        self.cv_scores[model_type] = {
            'scores': cv_scores,
            'mean': mean_score,
            'std': std_score
        }
        
        return mean_score, std_score
    
    def create_stacking_model(self, X_train, y_train, X_val, y_val):
        """스태킹 모델 생성"""
        print("스태킹 모델 생성")
        
        base_models = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
        valid_models = [name for name in base_models if name in self.models]
        
        if len(valid_models) < 2:
            print("스태킹을 위한 모델이 부족합니다")
            return None, 0.0
        
        try:
            # 기본 모델들의 Out-of-Fold 예측 생성
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            meta_features_train = np.zeros((len(X_train), len(valid_models) * 3))
            meta_features_val = np.zeros((len(X_val), len(valid_models) * 3))
            
            # 각 폴드에서 메타 피처 생성
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
                                'verbose': -1, 'random_state': 42
                            }
                            train_data = lgb.Dataset(X_fold_train.values, label=y_fold_train)
                            fold_model = lgb.train(lgb_params, train_data, num_boost_round=100)
                            oof_pred = fold_model.predict(X_fold_oof.values)
                            val_pred = fold_model.predict(X_val.values)
                            
                        elif model_name == 'xgboost':
                            xgb_params = {
                                'objective': 'multi:softprob', 'num_class': 3,
                                'random_state': 42, 'verbosity': 0
                            }
                            train_data = xgb.DMatrix(X_fold_train.values, label=y_fold_train)
                            fold_model = xgb.train(xgb_params, train_data, num_boost_round=100)
                            oof_pred = fold_model.predict(xgb.DMatrix(X_fold_oof.values))
                            val_pred = fold_model.predict(xgb.DMatrix(X_val.values))
                            
                        elif model_name == 'catboost':
                            fold_model = CatBoostClassifier(
                                iterations=100, random_seed=42, verbose=0
                            )
                            fold_model.fit(X_fold_train.values, y_fold_train, verbose=False)
                            oof_pred = fold_model.predict_proba(X_fold_oof.values)
                            val_pred = fold_model.predict_proba(X_val.values)
                            
                        elif model_name == 'random_forest':
                            fold_model = RandomForestClassifier(
                                n_estimators=100, random_state=42, n_jobs=-1
                            )
                            fold_model.fit(X_fold_train.values, y_fold_train)
                            oof_pred = fold_model.predict_proba(X_fold_oof.values)
                            val_pred = fold_model.predict_proba(X_val.values)
                        
                        else:
                            continue
                        
                        # Out-of-Fold 예측 저장
                        meta_features_train[oof_idx, col_idx:col_idx+3] = oof_pred
                        
                        # 검증 데이터 예측은 평균으로 누적
                        if fold == 0:
                            meta_features_val[:, col_idx:col_idx+3] = val_pred / skf.n_splits
                        else:
                            meta_features_val[:, col_idx:col_idx+3] += val_pred / skf.n_splits
                        
                        col_idx += 3
                        
                    except Exception as e:
                        print(f"폴드 {fold} {model_name} 오류: {e}")
                        col_idx += 3
                        continue
            
            # 실제 사용할 피처 수 결정
            actual_features = col_idx
            meta_features_train = meta_features_train[:, :actual_features]
            meta_features_val = meta_features_val[:, :actual_features]
            
            if actual_features == 0:
                print("메타 피처 생성 실패")
                return None, 0.0
            
            # 메타 모델 학습
            self.meta_model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            self.meta_model.fit(meta_features_train, y_train)
            
            # 검증 데이터로 평가
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
    
    def create_voting_ensemble(self, X_train, y_train, X_val, y_val):
        """보팅 앙상블 생성"""
        print("보팅 앙상블 생성")
        
        voting_models = []
        
        if 'random_forest' in self.models:
            voting_models.append(('rf', self.models['random_forest']))
        if 'extra_trees' in self.models:
            voting_models.append(('et', self.models['extra_trees']))
        if 'neural_network' in self.models:
            voting_models.append(('nn', self.models['neural_network']))
        if 'svm' in self.models:
            voting_models.append(('svm', self.models['svm']))
        
        if len(voting_models) >= 2:
            voting_classifier = VotingClassifier(
                estimators=voting_models,
                voting='soft'
            )
            
            voting_classifier.fit(X_train.values, y_train)
            
            y_pred = voting_classifier.predict(X_val.values)
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"보팅 앙상블 검증 정확도: {accuracy:.4f}")
            
            self.models['voting'] = voting_classifier
            return voting_classifier, accuracy
        
        return None, 0.0
    
    def save_all_models(self, engineer, preprocessor, X_train):
        """모든 모델 저장"""
        print("모델 저장")
        
        os.makedirs('models', exist_ok=True)
        
        try:
            joblib.dump(preprocessor, 'models/preprocessor.pkl')
            joblib.dump(engineer, 'models/feature_engineer.pkl')
            
            # 피처 정보 저장 (이름과 순서 포함)
            feature_info = {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names)
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
        
        # 피처 이름 설정
        if self.feature_names is None:
            self.feature_names = list(X_train.columns)
        
        # 개별 모델 학습
        model_results = {}
        
        try:
            lgb_model, lgb_acc = self.train_lightgbm(X_train, y_train, X_val, y_val)
            model_results['lightgbm'] = lgb_acc
        except Exception as e:
            print(f"LightGBM 학습 실패: {e}")
            model_results['lightgbm'] = 0.0
        
        try:
            xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val)
            model_results['xgboost'] = xgb_acc
        except Exception as e:
            print(f"XGBoost 학습 실패: {e}")
            model_results['xgboost'] = 0.0
        
        try:
            cat_model, cat_acc = self.train_catboost(X_train, y_train, X_val, y_val)
            model_results['catboost'] = cat_acc
        except Exception as e:
            print(f"CatBoost 학습 실패: {e}")
            model_results['catboost'] = 0.0
        
        try:
            rf_model, rf_acc = self.train_random_forest(X_train, y_train, X_val, y_val)
            model_results['random_forest'] = rf_acc
        except Exception as e:
            print(f"Random Forest 학습 실패: {e}")
            model_results['random_forest'] = 0.0
        
        try:
            et_model, et_acc = self.train_extra_trees(X_train, y_train, X_val, y_val)
            model_results['extra_trees'] = et_acc
        except Exception as e:
            print(f"Extra Trees 학습 실패: {e}")
            model_results['extra_trees'] = 0.0
        
        try:
            nn_model, nn_acc = self.train_neural_network(X_train, y_train, X_val, y_val)
            model_results['neural_network'] = nn_acc
        except Exception as e:
            print(f"Neural Network 학습 실패: {e}")
            model_results['neural_network'] = 0.0
        
        try:
            svm_model, svm_acc = self.train_svm(X_train, y_train, X_val, y_val)
            model_results['svm'] = svm_acc
        except Exception as e:
            print(f"SVM 학습 실패: {e}")
            model_results['svm'] = 0.0
        
        # 앙상블 모델 생성 (기본 모델들이 충분히 학습된 후)
        try:
            stacking_model, stacking_acc = self.create_stacking_model(X_train, y_train, X_val, y_val)
            if stacking_model is not None:
                model_results['stacking'] = stacking_acc
        except Exception as e:
            print(f"스태킹 모델 생성 실패: {e}")
            model_results['stacking'] = 0.0
        
        try:
            voting_model, voting_acc = self.create_voting_ensemble(X_train, y_train, X_val, y_val)
            if voting_model is not None:
                model_results['voting'] = voting_acc
        except Exception as e:
            print(f"보팅 앙상블 생성 실패: {e}")
            model_results['voting'] = 0.0
        
        # 교차 검증
        if len(self.models) > 0:
            full_X = pd.concat([X_train, X_val])
            full_y = pd.concat([y_train, y_val])
            
            if 'lightgbm' in self.models:
                self.perform_time_series_cv(full_X, full_y, 'lightgbm')
            if 'catboost' in self.models:
                self.perform_time_series_cv(full_X, full_y, 'catboost')
        
        # 모델 저장
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