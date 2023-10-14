from typing import Union, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from helpMethods import createDict, draw_PR_ROC_Curve


# Ці дані ми застосуємо для створення ключів в дікті і для виведення результатів
endings = ['_dt', '_deep_dt', '_rf', '_deep_rf']
resultNames1 = ["Дрібне дерево рішень:",
               "Глибоке дерево рішень:",
               "Випадковий ліс на дрібних деревах:",
               "Випадковий ліс на глибоких деревах:"]
resultNames2 = ["дерева рішень:",
               "глибокого дерева рішень:",
               "випадкового лісу:",
               "випадкового лісу на глибоких деревах:"]

# Читаємо CSV файл
data = pd.read_csv("bioresponse.csv")

X = data.drop("Activity", axis=1)
y = data["Activity"]

""" Перша частина лабораторної роботи """

# Розділяємо дані на 4 частини
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Створюємо класифікатори
classifier_dt = DecisionTreeClassifier(random_state=50)
classifier_dt.fit(X_train, y_train)

classifier_deep_dt = DecisionTreeClassifier(max_depth=15, random_state=50)
classifier_deep_dt.fit(X_train, y_train)

classifier_rf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=50)
classifier_rf.fit(X_train, y_train)

classifier_deep_rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=50)
classifier_deep_rf.fit(X_train, y_train)

classifiers = [classifier_dt, classifier_deep_dt, classifier_rf, classifier_deep_rf]

# Передбачуємо дані
y_predicted_dt = classifier_dt.predict(X_test)
y_predicted_deep_dt = classifier_deep_dt.predict(X_test)
y_predicted_rf = classifier_rf.predict(X_test)
y_predicted_deep_rf = classifier_deep_rf.predict(X_test)
y_predictedList = [y_predicted_dt, y_predicted_deep_dt, y_predicted_rf, y_predicted_deep_rf]

# Метрики: частка правильних відповідей, точність, повнота, f1 результат, логарифмічна втрата
accuracy = createDict('accuracy', endings, float)
precision = createDict('precision', endings, Any)
recall = createDict('recall', endings, Any)
f1score = createDict('f1score', endings, Any)
logLoss = createDict('logLoss', endings, float)

accKeys = list(accuracy.keys())  # accuracy_dt, accuracy_deep_dt, accuracy_rf, accuracy_deep_rf
precisionKeys = list(precision.keys())
recallKeys = list(recall.keys())
f1scoreKeys = list(f1score.keys())
logLossKeys = list(logLoss.keys())


# Вираховуємо метрики для кожного класифікатора
for i in range(0, len(endings)):
    accuracy[accKeys[i]] = accuracy_score(y_test, y_predictedList[i])
    precision[precisionKeys[i]] = precision_score(y_test, y_predictedList[i])
    recall[recallKeys[i]] = recall_score(y_test, y_predictedList[i])
    f1score[f1scoreKeys[i]] = f1_score(y_test, y_predictedList[i])
    logLoss[logLossKeys[i]] = log_loss(y_test, classifiers[i].predict_proba(X_test)[:, 1])

# Виведення результатів 5 метрик
for i in range(0, len(resultNames1)):
    print(resultNames1[i])
    print(f"Частка: {accuracy[accKeys[i]]}")
    print(f"Точність: {precision[precisionKeys[i]]}")
    print(f"Повнота: {recall[recallKeys[i]]}")
    print(f"F1 Score: {f1score[f1scoreKeys[i]]}")
    print(f"Логарифмічна втрата: {logLoss[logLossKeys[i]]}")
    print("\n")

""" Друга частина лабораторної роботи """

# Перевибірка
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Навчання класифікатора, що уникає помилок II роду
y_predicted_resampled = createDict('y_predicted_resampled', endings, Any)
y_predicted_resampledKeys = list(y_predicted_resampled.keys())
for i in range(0, len(y_predicted_resampledKeys)):
    y_predicted_resampled[y_predicted_resampledKeys[i]] = classifiers[i].predict(X_test)

# y_predicted_resampled_dt = classifier_dt.predict(X_test)
# y_predicted_resampled['y_predicted_resampled_dt'] = classifier_dt.predict(X_test)

# Створення матриці невідповідностей
confusion_matrix_resampled = createDict('confusion_matrix_resampled', endings, Any)
confusion_matrix_resampledKeys = list(confusion_matrix_resampled.keys())
for i in range(0, len(confusion_matrix_resampledKeys)):
    confusion_matrix_resampled[confusion_matrix_resampledKeys[i]] = \
        confusion_matrix(y_test, y_predicted_resampled[y_predicted_resampledKeys[i]])

# Створення звіту категоризації
classification_report_balanced = createDict('classification_report_resampled', endings, Union[str, dict])
classification_report_balancedKeys = list(classification_report_balanced.keys())
for i in range(0, len(classification_report_balancedKeys)):
    classification_report_balanced[classification_report_balancedKeys[i]] = \
        classification_report(y_test, y_predicted_resampled[y_predicted_resampledKeys[i]])

# Виведення матриці і звіту
for i in range(0, len(confusion_matrix_resampledKeys)):
    print("Матриця невідповідностей для збалансованого класифікатора " + resultNames2[i])
    print(confusion_matrix_resampled[confusion_matrix_resampledKeys[i]])
    print("\nЗвіт категоризації для збалансованого класифікатора " + resultNames2[i])
    print(classification_report_balanced[classification_report_balancedKeys[i]])

# Побудова precision-recall і ROC-кривих для кожної моделі
draw_PR_ROC_Curve(classifiers, resultNames1, X_test, y_test)
