import sys
def pf(s):
    print(s)
    sys.stdout.flush()

pf('Starting script...')
import torch
pf('Imported torch')
import numpy as np
pf('Imported numpy')
from config import CONFIG
pf('Imported config')
from models.transformer_model import TransformerPredictor
pf('Imported TransformerPredictor')

pf('Allocating mock data...')
X_train = np.random.randn(10, 60, 148).astype(np.float32)
y_train = np.random.randint(0, 2, size=(10,)).astype(np.int64)

pf('Initializing model...')
model = TransformerPredictor(CONFIG)
pf('Starting fit...')
model.fit(X_train, y_train)
pf('Done!')
