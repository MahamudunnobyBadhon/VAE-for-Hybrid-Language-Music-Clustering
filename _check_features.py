import numpy as np
import pandas as pd

for name in ['mfcc', 'combined', 'mel']:
    try:
        f = np.load(f'data/features/{name}_features.npy')
        m = pd.read_csv(f'data/features/{name}_metadata.csv')
        lang_counts = m['language'].value_counts().to_dict()
        print(f'{name}: features={f.shape}, meta={m.shape}, langs={lang_counts}')
    except Exception as e:
        print(f'{name}: ERROR - {e}')
