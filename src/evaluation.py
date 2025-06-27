import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

def get_metrics(y_pred, y_test, plot=True):
    results = {}

    results['rmse'] = root_mean_squared_error(y_pred, y_test)
    results['r2'] = r2_score(y_test, y_pred)
    results['pearson_correlation'] = pearsonr(y_test, y_pred)[0]
    results['pearson_p_value'] = pearsonr(y_test, y_pred)[1]

    if plot:
        plt.scatter(y_test, y_pred, color='mediumseagreen',
                    label='Model predictions')
        plt.plot([min(y_test), max(y_test)],
                 [min(y_pred), max(y_pred)], color='dimgray', label='Ideal Fit',
                 linewidth=2)
        plt.xlabel('Post routing result')
        plt.ylabel('HB or OpenLane result')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.show()


