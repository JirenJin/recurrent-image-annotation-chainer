import json
import operator


dataset = 'iaprtc12'

with open(dataset + '/dictionary.json') as f:
    dictionary = json.load(f)
with open(dataset + '/train_annotations.json') as f:
    train_annotations = json.load(f)
    
# label is 1-index, while python is 0-index
frequency_dict = {label:0 for label in range(1, len(dictionary)+1)}
for annotation in train_annotations:
    for label in annotation:
        frequency_dict[label] += 1

sorted_frequency_dict = sorted(frequency_dict.items(), key=operator.itemgetter(1))
sorted_labels = [x[0] for x in sorted_frequency_dict]

rare_first_train_annotations = [sorted(annotation, key=lambda label: sorted_labels.index(label)) for annotation in train_annotations]

frequent_first_train_annotations = [sorted(annotation, key=lambda label: sorted_labels[::-1].index(label)) for annotation in train_annotations]

with open(dataset + '/rare_first_train_annotations.json', 'w') as f:
    json.dump(rare_first_train_annotations, f)
    
with open(dataset + '/frequent_first_train_annotations.json', 'w') as f:
    json.dump(frequent_first_train_annotations, f)
