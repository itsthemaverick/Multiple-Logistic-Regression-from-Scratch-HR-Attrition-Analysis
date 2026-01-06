import matplotlib.pyplot as plt
import numpy as np

def plot_coefficient(model,feature_names):
    coef = model.coef_[0]

    plt.figure()
    plt.barh(feature_names,coef)
    plt.xlabel("coeff value")
    plt.title("Feature Impact In Log Reg")
    plt.tight_layout()
    plt.show()

def plot_conf_matrix(cm):
    plt.figure()
    plt.imshow(cm)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

