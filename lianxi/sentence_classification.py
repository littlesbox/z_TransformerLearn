from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

data = pd.read_csv('ChnSentiCorp_htl_all.csv')
data.head