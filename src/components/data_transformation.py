import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.utils import resample
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import sys




## Data Transformation config
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

## Data Transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        self.norm_map = {
            'age': 'standartization', 'workclass': 'onehot', 'fnlwgt': 'normalization', 'education': 'onehot', 
            'education-num': 'standartization', 'marital-status': 'onehot', 'occupation': 'onehot', 'relationship': 'onehot', 
            'race': 'onehot', 'sex': 'onehot', 'capital-gain': 'normalization', 'capital-loss': 'normalization', 
            'hours-per-week': 'standartization', 'country': 'onehot', 'salary': 'onehot'
        }

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define the categorical and numerical columns
            categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'country', 'sex']
            numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

            # Define the custom categories for ordinal variables
            custom_categories = {
                'workclass': [' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov', ' Local-gov',
                            ' Self-emp-inc', ' Without-pay', ' Never-worked','na'],
                
                'marital-status': [' Never-married', ' Married-civ-spouse', ' Divorced',
                                ' Married-spouse-absent', ' Separated', ' Married-AF-spouse', ' Widowed'],
                'occupation': [' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners', ' Prof-specialty',
                            ' Other-service', ' Sales', ' Craft-repair', ' Transport-moving',
                            ' Farming-fishing', ' Machine-op-inspct', ' Tech-support', ' Protective-serv',
                            ' Armed-Forces', ' Priv-house-serv','na'],
                'relationship': [' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried', ' Other-relative'],
                'race': [' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other'],
                
                'country': [' United-States', ' Cuba', ' Jamaica', ' India', ' Mexico', ' South',
                            ' Puerto-Rico', ' Honduras', ' England', ' Canada', ' Germany', ' Iran',
                            ' Philippines', ' Italy', ' Poland', ' Columbia', ' Cambodia', ' Thailand',
                            ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',
                            ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan', ' Yugoslavia',
                            ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland', ' Trinadad&Tobago',
                            ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland', ' Hungary',
                            ' Holand-Netherlands'],
                
                'sex': [' Male', ' Female']
            }

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[custom_categories[col] for col in categorical_cols])),
                    ('scaler', StandardScaler())
                ]
            )

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Creating a column transformer
            preprocessor = ColumnTransformer([
                ('cat_pipeline', cat_pipeline, categorical_cols),
                ('num_pipeline', num_pipeline, numerical_cols)
            ])

            logging.info('Data Transformation Completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initaite_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            logging.info(train_df)
            test_df = pd.read_csv(test_path)
            logging.info(test_df)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            logging.info('Handling imbalanced data using resampling')
            df_majority = train_df[train_df['salary'] == '<=50K']
            df_minority = train_df[train_df['salary'] == '>50K']
            df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=123)
            train_df = pd.concat([df_majority_downsampled, df_minority])

            logging.info('Data resampling completed')

            # Encoding using norm_map
            for feature_name, encoding_type in self.norm_map.items():
                if encoding_type == 'normalization':
                    max_value = train_df[feature_name].max()
                    min_value = train_df[feature_name].min()
                    train_df[feature_name] = (train_df[feature_name] - min_value) / (max_value - min_value)
                    test_df[feature_name] = (test_df[feature_name] - min_value) / (max_value - min_value)
                elif encoding_type == 'standartization':
                    mean_value = train_df[feature_name].mean()
                    std_value = train_df[feature_name].std()
                    train_df[feature_name] = (train_df[feature_name] - mean_value) / std_value
                    test_df[feature_name] = (test_df[feature_name] - mean_value) / std_value
                elif encoding_type == 'onehot':
                    dummies_train = pd.get_dummies(train_df[[feature_name]])
                    dummies_test = pd.get_dummies(test_df[[feature_name]])
                    train_df = pd.concat([train_df, dummies_train], axis=1)
                    test_df = pd.concat([test_df, dummies_test], axis=1)
                    train_df = train_df.drop([feature_name], axis=1)
                    test_df = test_df.drop([feature_name], axis=1)

            logging.info('Data encoding completed')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()
            test_df.to_csv('lol.csv')
            logging.info(test_df)
            target_column_name = 'salary'
            drop_columns = [target_column_name, 'education']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            logging.info(input_feature_train_df)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Transform using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)
