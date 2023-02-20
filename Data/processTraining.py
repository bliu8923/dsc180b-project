import os
import re
import json

#Process the training log and generate data for plots/EDA/reports

def make_dict(fp,model_name):
    l, train, val = ([], ) * 3
    testbench = 0
    for fn in os.listdir(fp):
        if fn.startswith('train'):
            with open(os.path.join(fp,fn), 'r') as f:
                for line in f:
                    # read epoch from the lines
                    match = re.match(r'Epoch (\d+): ([a-zA-Z]+) loss: ([\d\.]+), train acc: ([\d\.]+), val acc: ([\d\.]+)', line)
                    if match:
                        epoch = int(match.group(1))
                        model_type = match.group(2)
                        loss = float(match.group(3))
                        train_acc = float(match.group(4))
                        val_acc = float(match.group(5))                
                        # Add to ditc
                        l.append(loss)
                        train.append(train_acc)
                        val.append(val_acc)
        elif fn.startswith('test'):
            with open(os.path.join(fp,fn), 'r') as f:
                for line in f:
                    match = re.match(r'Test Acc: ([\d\.]+)', line)
                    testbench = float(match.group(1))

    data_dict = {'loss':l,'train_acc':train,'val_acc':val,'model':model_name,'test_score':testbench}
    return data_dict

# Set the file path before proceed
path_to_files = r'...'
res = []
d = {}
for filename in os.listdir(path_to_files):
    for filename2 in os.listdir(os.path.join(path_to_files,filename)):
        #process
        d = make_dict(os.path.join(path_to_files,filename,filename2),filename2)
        res.append(d)

num_instance = str(len(res))#check

#save to json
with open(path_to_files+'J.json','w') as f:
    json.dump(res, f)
    print(num_instance+' Instances Saved')
