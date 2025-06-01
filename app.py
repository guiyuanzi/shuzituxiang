from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import pytesseract
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'result')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 各模块模板
@app.route('/resize', methods=['GET', 'POST'])
def resize():
    if request.method == 'POST':
        file = request.files['image']
        scale = float(request.form.get('scale', 1.0))
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        
        img = cv2.imread(path)
        resized = cv2.resize(img, None, fx=scale, fy=scale)
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, resized)
        return render_template('resize.html', result_img=filename)
    return render_template('resize.html')

@app.route('/rotate', methods=['GET', 'POST'])
def rotate():
    if request.method == 'POST':
        file = request.files['image']
        angle = float(request.form.get('angle', 0))
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, rotated)
        return render_template('rotate.html', result_img=filename)
    return render_template('rotate.html')

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('segment.html', error="请上传图像")

        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        # --- 图像分割逻辑 ---
        img = cv2.imread(upload_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 查找轮廓并绘制
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(result_path, result)

        return render_template('segment.html', result_img=filename)
    return render_template('segment.html')


@app.route('/enhance', methods=['GET', 'POST'])
def enhance():
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
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, result)
        return render_template('enhance.html', result_img=filename)
    return render_template('enhance.html')

@app.route('/sharpen', methods=['GET', 'POST'])
def sharpen():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        result = cv2.filter2D(img, -1, kernel)
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, result)
        return render_template('sharpen.html', result_img=filename)
    return render_template('sharpen.html')

@app.route('/smooth', methods=['GET', 'POST'])
def smooth():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        result = cv2.GaussianBlur(img, (7, 7), 0)
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, result)
        return render_template('smooth.html', result_img=filename)
    return render_template('smooth.html')

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        return render_template('ocr.html', text_result=text, result_img=filename)
    return render_template('ocr.html')

from model_utils import load_model, predict_image  # 导入工具模块

model = load_model("best_model.pth")  # 全局加载一次模型

@app.route('/fruit', methods=['GET', 'POST'])
def fruit():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # 使用模型进行预测
        result = predict_image(model, path)

        return render_template('fruit.html', result=result, result_img=filename)
    return render_template('fruit.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
