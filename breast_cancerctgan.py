import pandas as pd
import warnings
from sklearn.datasets import load_breast_cancer
from ctgan import CTGAN
from table_evaluator import TableEvaluator

warnings.filterwarnings('ignore')

print(f'---Step 1: Loading Real Dataset---')
disease = load_breast_cancer(as_frame=True)
real_data = disease.frame

print(f'Real Dataset shape:{real_data.shape}')
print(f'Columns:{list(real_data.columns)}\n')

print(f'---Step 2: Training CTGAN---')
discrete_columns = ['target']

model = CTGAN(epochs = 10, verbose= True)
model.fit(real_data,discrete_columns)

print(f'Generating synthetic data...')
synthetic_data = model.sample(500)
synthetic_data['target'] = synthetic_data['target'].round().astype(int)
print(f'Synthetic dataset shape:{synthetic_data.shape}\n')

print('---Step 2.5: Side-by-Side Data')

pd.set_option('display.max_columns',5)
pd.set_option('display.width',1000)

print('\n[REAL DATA SAMPLE -First 6 Rows]')
print(real_data.head(6).round(3))

print('\n'+'='*50)

print('\n[SYNTHETIC DATA SAMPLE - First 6 Rows]')
print(synthetic_data.head(6).round(3))

print('---Step 3: Evaluating Quality with Table Evaluator--')
sample_size = min(500,len(real_data))
real_sample = real_data.sample(sample_size,random_state=42)

evaluator = TableEvaluator(real_sample,synthetic_data)

print('Generating distribution plots...')
evaluator.visual_evaluation()

