from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.visuslize import plot_conf_matrix,plot_coefficient
from src.viz_sigmoid import plot_logistic_curves

df = load_data("data/HR_comma_sep.csv")
X,y = preprocess(df)

model,acc,cm,feature_names = train_model(X,y)

print("Accuracy : ", acc)

plot_coefficient(model,feature_names)
plot_conf_matrix(cm)

plot_logistic_curves(model,X)