import numpy as np

precision1 = np.array([1.0,1.0,1.0,0.5,0.20])
recall1 = np.array([0.05,0.1,0.4,0.7,1.0])

precision2 = np.array([1.0,0.80,0.60,0.5,0.20])
recall2 = np.array([0.3,0.4,0.5,0.7,1.0])


def average_precision(precisions, recalls):
    recall_levels = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    highest_precisions = []

    for r in recall_levels:
        highest_precision = 0
        for precision, r_hat in zip(precisions, recalls):
            if r_hat >= r and precision >= highest_precision:
                highest_precision = precision
        highest_precisions.append(highest_precision)

    highest_precisions = np.array(highest_precisions)

    AP = np.average(highest_precisions)
    return AP  

if __name__ == "__main__":
    AP1 = average_precision(precision1, recall1)
    AP2 = average_precision(precision2, recall2)

    mAP = (AP1 + AP2)/2
    print(AP1,AP2,mAP)