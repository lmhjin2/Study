import numpy as np
import pandas as pd
# print(hp.__version__) # 0.2.7
from hyperopt import hp,fmin,tpe, Trials, STATUS_OK
search_space = {'x1' : hp.quniform('x1', -10, 10, 1),
                'x2' : hp.quniform('x2', -15, 15, 1)}
                   # : hp.quniform(label, low, high, q)  
                   # quniform = 균등분포 생성. low부터 high까지 q단위로

# hp.quniform(label, low, high, q) : label로 지정된 입력 값 변수 검색공간을
#                      최소값 low에서 최대값 high까지 q의 간격을 가지고 설정
# hp.uniform(label, low, high) : 최소값 low에서 최대값 high까지 정규분포 형태의
#                                  검색 공간 설정
# hp.randint(label, upper) : 0부터 최대값 upper까지 
#                           random한 정수값으로 검색 공간 설정.
# hp.loguniform(label, low, high) : exp(uniform(low, high)) 값을 반환하며,
#               반환값의 log 변환 된 값은 정규분포 형태를 가지는 검색 공간 설정

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 - 20*x2
    
    return return_value

trial_val = Trials()
best = fmin(
    fn = objective_func,
    space=search_space,
    algo = tpe.suggest,     # 알고리즘, 기본값
    max_evals=20,   # 서치 횟수
    trials=trial_val,
    rstate=np.random.default_rng(seed=10),
    # rstate=333,   # 'int object'를 'integers'에 쓸 수 없다?
)
    
# print(best)
# {'x1': 0.0, 'x2': 15.0}

# print(trial_val.results)
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, {'loss': 129.0, 'status': 'ok'}, 
# {'loss': 200.0, 'status': 'ok'}, {'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'}, 
# {'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'}, 
# {'loss': -51.0, 'status': 'ok'}, {'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, 
# {'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, {'loss': 49.0, 'status': 'ok'}, 
# {'loss': -300.0, 'status': 'ok'}, {'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'},
# {'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]

# print(trial_val.vals)
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0], 
# 'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}

# ################ 졌잘싸 #################
# data = {'iter':[],
#         'target':[],
#         'x1':[],
#         'x2':[]}
# for i in range(len(trial_val.vals['x1'])):
#     data['iter'].append(int(i+1))
#     data['target'].append(trial_val.results['loss'][i])
#     data['x1'].append(trial_val.vals['x1'][i])
#     data['x2'].append(trial_val.vals['x2'][i])


# print(data)
# # trial = pd.DataFrame()
# ################ 졌잘싸 #################

# ################ 영현이꺼 ################

# print('|   iter   |  target  |    x1    |    x2    |')
# print('---------------------------------------------')
# x1_list = trial_val.vals['x1']
# x2_list = trial_val.vals['x2']
# for idx, data in enumerate(trial_val.results):
#     loss = data['loss']
#     print(f'|{idx+1:^10}|{loss:^10}|{x1_list[idx]:^10}|{x2_list[idx]:^10}|')

# ################ 영현이꺼 ################

# ################ 쌤꺼 ################

target = [aaa['loss'] for aaa in trial_val.results]
# print(target)
df = pd.DataFrame({'results' : target,
                  'x1' : trial_val.vals['x1'],
                  'x2' : trial_val.vals['x2'],
                  })


print(df)