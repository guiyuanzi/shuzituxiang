from flask import Flask, render_template, request, send_from_directory, url_for
import os
import cv2
import numpy as np
import pytesseract
import uuid
from werkzeug.utils import secure_filename
from model_utils import load_model, predict_image  # 导入自定义模型加载与预测模块

app = Flask(__name__)

# 设置上传与结果图片保存路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'result')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 全局加载模型以供预测使用
model = load_model("best_model.pth")

@app.route('/')
def index():
    # 主页，显示操作导航
    return render_template('index.html')

@app.route('/resize', methods=['GET', 'POST'])
def resize():
    # 图像缩放
    if request.method == 'POST':
        file = request.files['image']
        scale = float(request.form.get('scale', 1.0))
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        resized = cv2.resize(img, None, fx=scale, fy=scale)
        result_filename = f"resized_{uuid.uuid4().hex}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, resized)

        return render_template('resize.html', result_img=result_filename)
    return render_template('resize.html')

@app.route('/rotate', methods=['GET', 'POST'])
def rotate():
    # 图像旋转
    if request.method == 'POST':
        file = request.files['image']
        angle = float(request.form.get('angle', 0))
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        img = cv2.imread(upload_path)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))

        result_filename = f"rotated_{uuid.uuid4().hex}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, rotated)

        return render_template('rotate.html', result_img=result_filename)
    return render_template('rotate.html')

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    # 图像分割（基于阈值 + 轮廓）
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('segment.html', error="请上传图像")

        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        img = cv2.imread(upload_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        result_filename = f"segmented_{uuid.uuid4().hex}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result)

        return render_template('segment.html', result_img=result_filename)
    return render_template('segment.html')

@app.route('/enhance', methods=['GET', 'POST'])
def enhance():
    # 图像增强（使用 CLAHE 增强亮度）
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        result = cv2.merge((cl, a, b))
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        result_filename = f"enhanced_{uuid.uuid4().hex}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result)

        return render_template('enhance.html', result_img=result_filename)
    return render_template('enhance.html')

@app.route('/sharpen', methods=['GET', 'POST'])
def sharpen():
    # 图像锐化（使用拉普拉斯内核）
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        result = cv2.filter2D(img, -1, kernel)

        result_filename = f"sharpened_{uuid.uuid4().hex}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result)

        return render_template('sharpen.html', result_img=result_filename)
    return render_template('sharpen.html')

@app.route('/smooth', methods=['GET', 'POST'])
def smooth():
    # 图像平滑（使用高斯模糊）
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        result = cv2.GaussianBlur(img, (7, 7), 0)

        result_filename = f"smoothed_{uuid.uuid4().hex}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, result)

        return render_template('smooth.html', result_img=result_filename)
    return render_template('smooth.html')

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    # 图像文字识别（OCR）
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')

        return render_template('ocr.html', text_result=text, result_img=filename)
    return render_template('ocr.html')

@app.route('/fruit', methods=['GET', 'POST'])
def fruit():
    # 使用模型识别水果类别
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        result = predict_image(model, path)

        return render_template('fruit.html', result=result, result_img=filename)
    return render_template('fruit.html')

if __name__ == '__main__':
    # 启动 Flask 服务，监听端口 5001
    app.run(debug=True, port=5001)
