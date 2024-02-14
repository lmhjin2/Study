import numpy as np
import pandas as pd

df = pd.DataFrame({'A':[1,2,3,4,5],
                   'B':[10,20,30,40,50],
                   'C':[5,4,3,2,1],
                   'D':[3,7,5,1,4],
                   'E':[0,1,2,1,0],
                   })

correlations = df.corr()
print(correlations)
