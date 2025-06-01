import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18  # ✅ 加上这行
from torchvision import transforms
from PIL import Image

# 加载模型（根据你的模型结构来写）
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # 判断是否是state_dict还是完整模型
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("模型格式不支持")

    # 自动推断类别数
    fc_weight = state_dict['fc.weight']
    num_classes = fc_weight.shape[0]

    # 构建模型
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# 预测函数
def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return '新鲜' if predicted.item() == 0 else '腐烂'
