import torch as tc
import pandas as pd
from showplot import plotprecrec

result=tc.load('result.pt')

# plotprecrec(result)

dt=pd.DataFrame(result)
dt.to_csv("./result.csv")

