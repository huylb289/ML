from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

##r^2 = 1 - (SEy_hat/SEy_mean)

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

xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)
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
##plt.scatter(xs, ys, color='#003F72', label='data')
##plt.scatter(predict_x, predict_y, color='#000000', label='prediction')
##plt.plot(xs, regression_line, label='regression line')
##plt.legend(loc=4)
##plt.show()
