import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# Створення дікту з ключами, які триматимуть значення метрики для кожного класифікатора.
# e.g. accuracy[accuracy_dt] = 0.6798792989
def createDict(name, nameEndings, returnType):
    dictToReturn = {}
    for ending in nameEndings:
        newName = name + ending
        dictToReturn[newName] = returnType
    return dictToReturn


# Побудова precission-recall кривої
def plot_precision_recall_curve(y_true, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Повнота')
    plt.ylabel('Точність')
    plt.title(f'Precision-Recall Крива для {model_name}')
    plt.show()


# Побудова ROC-кривої
def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC крива (площа = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Хибно-позитивний відсоток')
    plt.ylabel('Вірно-позитивний відсоток')
    plt.title(f'ROC Крива для {model_name}')
    plt.legend(loc='lower right')
    plt.show()


# Виклик методів для побудови вищезазначених кривих для кожного класифікатора
def draw_PR_ROC_Curve(classifierList, resultNameList, xTest, yTest):
    for j in range(0, len(classifierList)):
        yProb = classifierList[j].predict_proba(xTest)[:, 1]
        plot_precision_recall_curve(yTest, yProb, resultNameList[j])
        plot_roc_curve(yTest, yProb, resultNameList[j])
