import numpy as np
from tabulate import tabulate


cm = np.array([[246, 25, 1, 8],
               [23, 231, 9, 17],
               [1, 6, 270, 3],
               [9, 5, 0, 266] ])

TP = []
FP = []
FN = []
TN = []
DA = []
Sens = []
Spec = []
PPV = []
NPV = []

for i in range(4):
    TP.append(cm[i][i])
    FP.append(sum(cm[:, i]) - TP[i])
    FN.append(sum(cm[i, :]) - TP[i])
    TN.append(sum(sum(cm)) - (TP[i] + FP[i] + FN[i]))
    DA.append((TP[i] + TN[i]) / (TP[i] + FP[i] + FN[i] + TN[i]))
    Sens.append(TP[i] / (TP[i] + FN[i]))
    Spec.append(TN[i] / (TN[i] + FP[i]))
    PPV.append(TP[i] / (TP[i] + FP[i]))
    NPV.append(TN[i] / (TN[i] + FN[i]))

ModelDA = np.mean(DA)

classes = ["Sensitivity", "Specificity", "PPV", "NPV", "Accuracy"]

print(f"Model DA: {ModelDA:.4f}")
print("Class-wise metrics:")

for i in range(4):
    print(f"{classes[i]}: Sens={Sens[i]:.4f}, Spec={Spec[i]:.4f}, PPV={PPV[i]:.4f}, NPV={NPV[i]:.4f}")


# Optional: Tabulate results
table = []
for i in range(4):
    table.append([classes[i], Sens[i], Spec[i], PPV[i], NPV[i]])
table.append(["Accuracy", ModelDA, "", "", ""])
print("\n" + tabulate(table, headers=["Metric", "Glioma", "Meningioma", "noTumor", "Pituitary"], floatfmt=".4f"))
