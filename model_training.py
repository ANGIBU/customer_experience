# model_training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
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
        self.feature_names = None
        self.class_weights = None
        self.ensemble_weights = {}
        self.model_results = {}
        
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
        
        # 분포 균형 조정 (클래스 0 과다 예측 방지)
        weights[0] *= 0.65  # 클래스 0 가중치 대폭 감소
        weights[1] *= 1.45  # 클래스 1 가중치 대폭 증가
        weights[2] *= 1.40  # 클래스 2 가중치 대폭 증가
        
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
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        # 클래스별 샘플 가중치
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.06,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 3,
            'min_child_weight': 3,
            'min_split_gain': 0.02,
            'reg_alpha': 0.08,
            'reg_lambda': 0.15,
            'max_depth': 10,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True,
            'num_threads': 1,
            'extra_trees': False
        }
        
        train_data = lgb.Dataset(X_train_clean, label=y_train_clean, weight=sample_weight, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val_clean, label=y_val_clean, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1200,
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val_clean)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_clean, y_pred_class)
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        sample_weight = np.ones(len(y_train_clean))
        for i, weight in self.class_weights.items():
            mask = y_train_clean == i
            sample_weight[mask] = weight
        
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'learning_rate': 0.06,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.08,
            'reg_lambda': 0.12,
            'min_child_weight': 3,
            'gamma': 0.02,
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'hist',
            'nthread': 1,
            'grow_policy': 'depthwise'
        }
        
        train_data = xgb.DMatrix(X_train_clean, label=y_train_clean, weight=sample_weight, feature_names=self.feature_names)
        val_data = xgb.DMatrix(X_val_clean, label=y_val_clean, feature_names=self.feature_names)
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=1200,
            evals=[(val_data, 'eval')],
            early_stopping_rounds=80,
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
            learning_rate=0.06,
            depth=8,
            l2_leaf_reg=5,
            bootstrap_type='Bernoulli',
            subsample=0.85,
            colsample_bylevel=0.85,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=80,
            task_type='CPU',
            thread_count=1,
            border_count=128,
            feature_border_type='GreedyLogSum'
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
            n_estimators=300,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=1,
            oob_score=True,
            max_samples=0.8
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
            n_estimators=250,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight=self.class_weights,
            random_state=42,
            n_jobs=1,
            oob_score=True
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['extra_trees'] = model
        return model, accuracy
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Gradient Boosting 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=3,
            subsample=0.85,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=30
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
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """신경망 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            batch_size=256
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['neural_network'] = model
        return model, accuracy
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """로지스틱 회귀 모델 학습"""
        X_train_clean, y_train_clean = self.safe_data_conversion(X_train, y_train)
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model = LogisticRegression(
            C=0.5,
            penalty='l2',
            class_weight=self.class_weights,
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        model.fit(X_train_clean, y_train_clean)
        
        y_pred = model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['logistic_regression'] = model
        return model, accuracy
    
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
            voting='soft',
            weights=[0.4, 0.35, 0.25] if len(available_models) == 3 else None
        )
        
        voting_model.fit(X_train_clean, y_train_clean)
        
        y_pred = voting_model.predict(X_val_clean)
        accuracy = accuracy_score(y_val_clean, y_pred)
        
        self.models['voting'] = voting_model
        return voting_model, accuracy
    
    def optimize_ensemble_weights(self, X_val, y_val):
        """앙상블 가중치 계산"""
        X_val_clean, y_val_clean = self.safe_data_conversion(X_val, y_val)
        
        model_scores = {}
        
        for name, model in self.models.items():
            if name in ['voting']:
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
                    y_pred = model.predict(X_val_clean)
                
                accuracy = accuracy_score(y_val_clean, y_pred)
                f1 = f1_score(y_val_clean, y_pred, average='macro')
                
                # 클래스 0 재현율 보너스
                class_0_recall = np.sum((y_pred == 0) & (y_val_clean == 0)) / np.sum(y_val_clean == 0)
                
                combined_score = 0.6 * accuracy + 0.3 * f1 + 0.1 * class_0_recall
                model_scores[name] = combined_score
                
            except Exception:
                model_scores[name] = 0.0
        
        # 새로운 가중치 분배 (클래스 0 예측에 강한 모델 선호)
        base_weights = {
            'lightgbm': 0.28,
            'xgboost': 0.26,
            'catboost': 0.24,
            'random_forest': 0.12,
            'extra_trees': 0.05,
            'gradient_boosting': 0.03,
            'neural_network': 0.015,
            'logistic_regression': 0.005
        }
        
        # 성능 기반 조정
        total_score = sum(model_scores.values())
        if total_score > 0:
            for name in model_scores:
                if name in base_weights:
                    performance_ratio = model_scores[name] / total_score
                    base_weight = base_weights[name]
                    self.ensemble_weights[name] = 0.75 * base_weight + 0.25 * performance_ratio
        else:
            self.ensemble_weights = {k: v for k, v in base_weights.items() if k in self.models}
        
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
            'feature_count': len(self.feature_names) if self.feature_names else 0,
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
                
            except Exception as e:
                print(f"{name} 모델 저장 실패: {e}")
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
        
        print("모델 학습 진행...")
        
        # 모델 학습 순서
        model_trainers = [
            ('lightgbm', self.train_lightgbm),
            ('xgboost', self.train_xgboost),
            ('catboost', self.train_catboost),
            ('random_forest', self.train_random_forest),
            ('extra_trees', self.train_extra_trees),
            ('gradient_boosting', self.train_gradient_boosting),
            ('neural_network', self.train_neural_network),
            ('logistic_regression', self.train_logistic_regression)
        ]
        
        for name, trainer in model_trainers:
            try:
                print(f"  학습 중: {name}")
                model, accuracy = trainer(X_train, y_train, X_val, y_val)
                self.model_results[name] = accuracy
            except Exception as e:
                print(f"  {name} 학습 실패: {str(e)[:50]}...")
                self.model_results[name] = 0.0
                continue
        
        # 앙상블 가중치 계산
        self.optimize_ensemble_weights(X_val, y_val)
        
        # 보팅 앙상블
        successful_models = sum(1 for score in self.model_results.values() if score > 0)
        if successful_models >= 3:
            try:
                voting_model, voting_acc = self.create_voting_ensemble(X_train, y_train, X_val, y_val)
                if voting_model is not None:
                    self.model_results['voting'] = voting_acc
            except Exception:
                pass
        
        # 모델 저장
        saved_count = self.save_models(engineer, preprocessor)
        
        # 결과 출력
        valid_results = {k: v for k, v in self.model_results.items() if v > 0}
        
        if valid_results:
            # 상위 5개 모델만 표시
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
            print("✓ 학습 완료된 모델:")
            for name, score in sorted_results[:5]:
                print(f"  - {name}: {score:.4f}")
            
            best_model = max(valid_results.items(), key=lambda x: x[1])
            print(f"✓ 최고 성능: {best_model[1]:.4f} ({best_model[0]})")
            
            if best_model[1] >= 0.48:
                print("✓ 내부 검증 통과")
            else:
                gap = 0.48 - best_model[1]
                print(f"→ 목표까지: {gap:.4f}")
            
            print(f"✓ 성공 모델: {len(valid_results)}개")
            print(f"✓ 저장된 모델: {saved_count}")
        
        print("✓ 모델 학습 완료")
        
        return valid_results

def main():
    try:
        trainer = ModelTrainer()
        
        # 데이터 준비
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        
        # 기본 전처리
        from preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        # 기본 피처만 사용
        basic_features = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length', 'gender', 'subscription_type']
        available_features = [f for f in basic_features if f in train_df.columns]
        
        # 범주형 인코딩
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in ['gender', 'subscription_type']:
            if col in available_features:
                combined = pd.concat([train_df[col], test_df[col]])
                le.fit(combined.fillna('Unknown'))
                train_df[col] = le.transform(train_df[col].fillna('Unknown'))
        
        X = train_df[available_features].fillna(0)
        y = train_df['support_needs']
        
        # 데이터 분할
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        trainer.feature_names = available_features
        trainer.calculate_class_weights(y_train)
        
        results = trainer.train_models(X_train, X_val, y_train, y_val, None, preprocessor)
        
        return trainer
        
    except Exception as e:
        print(f"모델 학습 오류: {e}")
        return None

if __name__ == "__main__":
    main()