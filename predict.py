import torch
import time
from PIL import Image

import json
import matplotlib.pyplot as plt
import main

#获取索引到类名的映射，以便查看测试影像的输出类
idx_to_class = {v: k for k, v in main.train_datasets.class_to_idx.items()}
print(idx_to_class)

with open('cat_to_name.json', 'r', encoding='gbk') as f:
    cat_to_name = json.load(f)
print(cat_to_name)

def predict(model, test_image_name):

    transform = main.test_valid_transforms

    test_image = Image.open(test_image_name).convert('RGB')
    plt.imshow(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 124, 124).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 124, 124)

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        start = time.time()
        out = model(test_image_tensor)
        stop = time.time()
        print('cost time', stop - start)
    ps = torch.exp(out)
    topk, topclass = ps.topk(3, dim=1)
    names = []
    for i in range(3):
        names.append(cat_to_name[idx_to_class[topclass.cpu().numpy()[0][i]]])
        print("Predcition", i + 1, ":", names[i], ", Score: ",
              topk.cpu().numpy()[0][i])

if __name__ == '__main__':
    model = torch.load('trained_models/resnet50_model_23.pth')
    predict(model, '61.png')