## 图书评分预测

### 标签分布

- 训练集
```text
0     482512
8      60088
10     48938
7      43333
9      41194
5      30827
6      20545
4       4830
3       3212
2       1481
1        904
```

- ensemble.py class3 76.2221
```text
0.0    155521
8.0     41335
4.0      8496
3.0       417
1.5       399
5.5        67
```
- ensemble.py class4 75.5705	
```text
0.00    177316
9.00     21919
4.50      3388
6.00      2136
7.50       638
3.00       592
1.25       238
2.50         6
5.75         2
```

- stacking_cls.py 75.002
```text
0     184215
6       6484
8       5116
7       3052
5       2806
2       1378
1       1120
9        742
10       738
4        314
3        270                
```

- ensemble_v2.py 76.4648
```text
0.000000    154354
7.000000     40824
4.666667      7141
2.333333      3228
3.500000       428
1.166667       225
5.833333        35
```

以下为线上效果：
- lgb num_class=3: 76.333
- lgb num_class=4: 75.456
- lgb num_class=5: 74.9737
- lgb num_class=11:73.7819

## 特征工程
![](others/feature_imp.png)
```python
# 加入 效果差
# data['Publisher_count'] = data.groupby('Publisher')['ISBN'].transform('count')
```

- 所有特征 线上76.3504 线下lgb_numclasses3_0.7195052410657701.csv
