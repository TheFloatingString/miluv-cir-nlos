from tabpfn import TabPFNClassifier
import numpy as np

X_train = np.random.rand(10, 10)
y_train = np.array([0, 1] * 5)  # Two classes
X_test = np.random.rand(10, 10)
y_test = np.array([0, 1] * 5)

clf = TabPFNClassifier()
clf.fit(X_train, y_train)  # Should not crash
predictions = clf.predict(X_test)

print("Predictions:", predictions)

