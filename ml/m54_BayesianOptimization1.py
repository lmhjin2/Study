param_bounds = {'x1':(-1,5),
                'x2':(0,4)}

def y_function(x1, x2):
    return -x1 **2 - (x2-2) **2 + 10

# pip install bayesian-optimization
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f =  y_function,
    pbounds = param_bounds,  
    random_state=777)

optimizer.maximize(init_points=5, n_iter= 5)
print(optimizer.max)