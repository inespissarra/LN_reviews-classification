##################################################################################
# # Do the matrix of confusion
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

labels = ['TRUTHFULPOSITIVE', 'TRUTHFULNEGATIVE', 'DECEPTIVEPOSITIVE','DECEPTIVENEGATIVE']

# Print the results
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix')
plt.show()