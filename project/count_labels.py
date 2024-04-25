# -*- coding: utf-8 -*-
from load_data import set_labels, count_labels

vial_numbers = list({7, 8, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 41})
set_labels({'still', 'wave', 'nearDrops', 'smallDrops', 'drops', 'foam', 'useless'})

count_per_label = count_labels(vial_numbers)
print(count_per_label)