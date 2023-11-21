import streamlit as st
from PIL import Image
from segment import process

st.title('Mnist demo')

image_file = st.file_uploader('Load an image with number', type=['png', 'jpg'])  # Добавление загрузчика файлов

if not image_file is None:                       # Выполнение блока, если загружено изображение
    image = Image.open(image_file)               # Открытие изображения
    results = process(image_file)                # Обработка изображения с помощью функции, реализованной в другом файле
    st.write('Таблица вероятностей цифры загруженной картинки:')
    st.write(results)