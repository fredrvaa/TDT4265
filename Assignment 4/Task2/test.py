matches = []
i = 1
for p in (1,2,3):
    for gt in (4,5,6):
        iou = i
        i += 1
        if iou >= 3:
            matches.append((gt,p,iou))
print(matches)