import pandas as pd

df = pd.DataFrame({'cate':[1,2,2,3]})

df['cate'] = df['cate'].astype('category')
df['code_%s' % 'cate'] = df['cate'].cat.codes
print(df)