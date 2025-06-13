from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [35, 45, 50, 55, 60, 65, 70, 75, 85, 95]

x = np.array(hours).reshape(-1, 1)
y = np.array(scores)

model = LinearRegression()
model.fit(x, y)

def predict(hours):
    return model.predict(np.array([[hours]]))[0]

print(f"Prediction of scores of studying 7hrs: {predict(7):.2f} marks")

predicted = predict(10)
print(f"Predicted Score: {predicted}")

def classify(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Needs Improvement"

print(f"Performance Category: {classify(predicted)}")

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x)

labels = kmeans.labels_

colors = plt.cm.viridis(labels / 2)

plt.bar(hours, scores, color=colors, edgecolor='black')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Scores Based on Hours Studied (Clustered)")
plt.show()

