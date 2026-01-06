import numpy as np
import matplotlib.pyplot as plt

def plot_logistic_curves(model,X):
    feature_names = X.columns
    coef = model.coef_[0]
    intercept = model.intercept_[0]

    for i,feature in enumerate(feature_names):
        x_vals = np.linspace(-3,3,200)

        z = coef[i]*x_vals+intercept

        y_vals = 1/(1+np.exp(-z))

        plt.figure()
        plt.plot(x_vals,y_vals)
        plt.xlabel(f"{feature} (std)")
        plt.ylabel("P(label=1)")
        plt.title(f"Logisctic curve : {feature}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()