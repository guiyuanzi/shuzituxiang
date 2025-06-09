# Flask 图像处理应用

一个基于 Flask 的 Web 应用，支持多种图像处理功能，包括图像缩放、旋转、增强、平滑、锐化、OCR文字识别以及水果图像分类等。

## 📦 功能模块

| 功能        | 描述                             |
|-------------|----------------------------------|
| 缩放        | 按照用户设定比例缩放图像         |
| 旋转        | 根据输入角度旋转图像             |
| 分割        | 基于阈值和轮廓实现图像分割       |
| 增强        | 使用 CLAHE 提高图像对比度         |
| 平滑        | 应用高斯模糊减少图像噪声         |
| 锐化        | 使用拉普拉斯滤波增强边缘细节     |
| OCR         | 基于 Tesseract 实现中英文文字识别 |
| 水果识别    | 使用深度学习模型识别水果类别     |

## 🛠️ 安装说明

### 1. 克隆项目

```bash
git clone https://your-repo-url.git
cd your-repo-name
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate  # Windows 使用 venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 准备模型文件

请确保 `best_model.pth` 模型文件已经放置在项目根目录，或按需修改 `model_utils.py` 中的路径。

### 4. 启动应用

```bash
python app.py
```

浏览器访问：`http://localhost:5001`

## 📁 项目结构

```
├── app.py                  # 主程序入口
├── model_utils.py          # 模型加载与预测逻辑
├── templates/              # HTML 模板
├── static/
│   ├── uploads/            # 上传图片目录
│   └── result/             # 处理结果输出目录
├── requirements.txt        # Python依赖列表
└── best_model.pth          # 深度学习模型权重
```

## 🔧 依赖说明（部分）

- Flask
- OpenCV-Python
- NumPy
- pytesseract
- Torch / torchvision（用于模型推理）

## 📝 注意事项

- OCR 使用 Tesseract，需本地安装并配置好路径。
- 对于中文识别，请确保 `chi_sim` 语言包已安装。
- `model_utils.py` 需要根据你自己的模型格式进行调整。

## 📬 联系方式

15000278792@163.com