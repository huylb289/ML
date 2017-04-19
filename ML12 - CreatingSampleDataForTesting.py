import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

def create_dataset(hm, variance, step=2, correlation=False):
    """
    hm - The value will be "how much"
    variance - This will dictate how much each point can vary from the previous point
    step - This will be how far to step on average per point
    correlation - This will be either False, pos, or neg to indicate that we want no correlation, positive correlation, or negative correlation.
    """

    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for _ in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

def best_fit_slope(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / \
         ((mean(xs)**2) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)
    return m, b

style.use('ggplot')

xs, ys = create_dataset(40,40,2,correlation='pos')
m, b = best_fit_slope(xs, ys)

regression_line = [(m*x)+b for x in xs]
predict_x = 7
predict_y = (m*predict_x) + b

print(m, b)
print(regression_line)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

##plt.xlim([0,7])
##plt.ylim([3.5,6.5])
plt.scatter(xs, ys, color='#003F72', label='data')
plt.scatter(predict_x, predict_y, color='#000000', label='prediction')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
