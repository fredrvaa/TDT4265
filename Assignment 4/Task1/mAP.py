import numpy as np

precision1 = np.array([1.0,1.0,1.0,0.5,0.20])
recall1 = np.array([0.05,0.1,0.4,0.7,1.0])

precision2 = np.array([1.0,0.80,0.60,0.5,0.20])
recall2 = np.array([0.3,0.4,0.5,0.7,1.0])
print(recall2.shape)


def averape_precision(precision,recall):
    AP = 0
    for r in np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
        candidates = np.zeros((recall.shape[0],1))
        for i, r_hat in enumerate(recall):
            if r_hat >= r:
                candidates[i] = precision[i]
        AP += np.amax(candidates)
        print(np.amax(candidates))
    AP /= 11
    return AP

def main():
    AP1 = averape_precision(precision1, recall1)
    AP2 = averape_precision(precision2, recall2)

    mAP = (AP1 + AP2)/2
    print(AP1,AP2,mAP)

if __name__ == "__main__":
    main()