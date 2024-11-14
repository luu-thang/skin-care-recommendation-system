from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sqlite3
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/my_model_v2.keras')

# Hàm lưu sản phẩm vào cơ sở dữ liệu
def add_product_to_db(name, description, effect, skin_type, image):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO products (name, description, effect, skin_type, image)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, description, effect, skin_type, image))
    conn.commit()
    conn.close()

# Hàm lấy sản phẩm phù hợp từ cơ sở dữ liệu
def get_products_for_skin_type(skin_type):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products WHERE skin_type = ?', (skin_type,))
    products = cursor.fetchall()
    conn.close()
    return products

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            # Phân tích loại da
            prediction = predict_skin_type(file_path)
            skin_types = ["Da Dầu", "Da Khô", "Da Mụn", "Da Bình Thường"]
            skin_type = skin_types[prediction]

            # Lấy danh sách sản phẩm từ cơ sở dữ liệu phù hợp với loại da
            suitable_products = get_products_for_skin_type(skin_type)

            return render_template('index.html', skin_type=skin_type, image=file.filename, products=suitable_products)

    return render_template('index.html', skin_type=None, products=[])

@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    if request.method == 'POST':
        product_name = request.form['product_name']
        product_description = request.form['product_description']
        product_effect = request.form['product_effect']
        skin_type = request.form['skin_type']
        product_image = request.files['product_image']
        if product_image:
            product_image_path = os.path.join('static', product_image.filename)
            product_image.save(product_image_path)

            # Lưu sản phẩm vào cơ sở dữ liệu
            add_product_to_db(product_name, product_description, product_effect, skin_type, product_image.filename)

            return redirect(url_for('view_products'))

    return render_template('add_product.html')

@app.route('/view_products', methods=['GET'])
def view_products():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM products')
    products = cursor.fetchall()
    conn.close()
    return render_template('view_products.html', products=products)

def predict_skin_type(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction)

if __name__ == '__main__':
    app.run(debug=True)
