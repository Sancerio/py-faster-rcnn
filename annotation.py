import os
import numpy as np
filename = os.path.join('0_data.txt')
        # print 'Loading: {}'.format(filename)
with open(filename) as f:
     data = f.read()
import re

#print data

objs = re.findall('\d+ \d+ \d+ \d+; \d+ \d+ \d+ \d+; \d+ \d+ \d+ \d+; ', data)
num_objs = len(objs)
labs = re.findall('\d+.jpg', data)
print labs
#print objs
boxes = np.zeros((num_objs, 4), dtype=np.uint16)
gt_classes = np.zeros((num_objs), dtype=np.int32)
overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

#Load object bounding boxes into a data frame.
for ix, obj in enumerate(objs):
    # Make pixel indexes 0-based
	coor = re.findall('\d+', obj)
	# Just take the first annotation
	x1 = float(coor[0])
	y1 = float(coor[1])
	x2 = float(coor[2])
	y2 = float(coor[3])
	cls = self._class_to_ind['person']
	boxes[ix, :] = [x1, y1, x2, y2]
	gt_classes[ix] = cls
	overlaps[ix, cls] = 1.0

overlaps = scipy.sparse.csr_matrix(overlaps)
