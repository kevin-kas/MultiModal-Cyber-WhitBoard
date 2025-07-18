import os

CYBER_DISPLAY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cyber Display</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css" rel="stylesheet">
    <style>
        body {{
            background: linear-gradient(135deg, #0a192f 0%, #172a45 50%, #1a3658 100%);
            animation: gradient 15s ease infinite;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            font-family: 'Segoe UI', system-ui;
            position: relative;
            overflow: hidden;
        }}

        .cyber-frame {{
            background: rgba(10, 25, 47, 0.9);
            border: 2px solid #64ffda;
            border-radius: 12px;
            box-shadow: 0 0 30px rgba(100, 255, 218, 0.3);
            padding: 20px;
            backdrop-filter: blur(8px);
            max-width: 80%;
            margin: 20px;
            transition: transform 0.3s ease;
            position: relative;
            z-index: 1;
        }}

        .cyber-image {{
            border: 1px solid #64ffda55;
            border-radius: 8px;
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}

        .cyber-title {{
            color: #64ffda;
            text-align: center;
            font-size: 1.5em;
            margin: 0 0 20px 0;
            text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .company-name {{
            position: fixed;
            top: 10px;
            left: 10px;
            color: #64ffda;
            font-size: 20px;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(100, 255, 218, 0.5);
            z-index: 2;
        }}

        @keyframes gradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    </style>
</head>
<body>
    <div class="company-name">∑ymΔig</div>
    <div class="cyber-frame">
        <h1 class="cyber-title">
            <i class="fas fa-{icon}"></i> {title}
        </h1>
        <img src="data:image/png;base64,{image_data}" class="cyber-image">
    </div>
    <script>
        // 动态粒子系统
        function createParticles() {{
            const container = document.createElement('div');
            container.style.cssText = `
                position: fixed;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
                top: 0;
                left: 0;
            `;

            for (let i = 0; i < 30; i++) {{
                const particle = document.createElement('div');
                particle.style.cssText = `
                    position: absolute;
                    width: 2px;
                    height: 2px;
                    background: #64ffda;
                    border-radius: 50%;
                    box-shadow: 0 0 10px #64ffda;
                    animation: particle ${{Math.random() * 5 + 3}}s linear ${{Math.random() * 5}}s infinite;
                    left: ${{Math.random() * 100}}%;
                `;
                container.appendChild(particle);
            }}
            document.body.appendChild(container);
        }}

        // 初始化
        document.addEventListener('DOMContentLoaded', () => {{
            createParticles();

            // 添加粒子动画关键帧
            const style = document.createElement('style');
            style.textContent = `
                @keyframes particle {{
                    0% {{ transform: translateY(-10vh) translateX(0); opacity: 0; }}
                    50% {{ opacity: 1; }}
                    100% {{ transform: translateY(110vh) translateX(100vw); opacity: 0; }}
                }}
            `;
            document.head.appendChild(style);
        }});
    </script>
</body>
</html>
"""


import webbrowser
from flask import Flask, render_template, request
import base64
software_page = Flask(__name__)
from io import BytesIO
import cv2
from Finder.predict import recognize
from VAE.generate import generate_fig
from Diffusion.generate import dif_generate as dif_generate1
from Calculator.Calculate import calculate
from Diffusion.labeled_Train import *

@software_page.route('/')
def index():
    return render_template('software_page.html')

@software_page.route('/get_values', methods=['POST'])
def get_values():
    data = request.get_json()
    canvas_data = data.get('canvasData')
    if canvas_data.startswith('data:image/png;base64,'):
        canvas_data = canvas_data.replace('data:image/png;base64,', '')
    try:
        img_data = base64.b64decode(canvas_data)
        with open('Calculator/images/canvas_image.png', 'wb') as f:
            f.write(img_data)
    except Exception as e:
        print(f"Error saving image: {e}")
    value1,value2=calculate()
    return {'value1': value1, 'value2': value2}

@software_page.route('/get_line',methods=['POST'])
def get_line():
    data = request.get_json()
    canvas_data = data.get('canvasData')
    try:
        if canvas_data.startswith('data:image/png;base64,'):
            canvas_data = canvas_data[len('data:image/png;base64,'):]
        img_data = base64.b64decode(canvas_data)
        image = Image.open(BytesIO(img_data)).convert('RGB')
        pixels = image.load()
        width, height = image.size
        for i in range(width):
            for j in range(height):
                r, g, b = pixels[i, j]
                pixels[i, j] = 255 - r, 255 - g, 255 - b
        gray_image = image.convert('L')
        binary_image = gray_image.point(lambda x: 0 if x < 128 else 255, '1')
        binary_array = np.array(binary_image).astype(np.uint8) * 255
        kernel = np.ones((7,7), np.uint8)
        dilated_array = cv2.dilate(binary_array, kernel, iterations=1)
        dilated_image = Image.fromarray(dilated_array).convert('RGB')
        with open('Finder/images/inverted_image.png', 'wb') as f:
            dilated_image.save(f, 'PNG')

    except Exception as e:
        print(f"Error processing image: {e}")
    # 保持你的原始返回值完全不变
    value1 = recognize()
    value2 = 'No Result'
    return {'value1': value1, 'value2': value2}

def generate_cyber_display(image_path, title, icon):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    html_content = CYBER_DISPLAY_TEMPLATE.format(
        image_data=image_data,
        title=title,
        icon=icon
    )
    output_path = os.path.join(os.path.dirname(image_path), 'cyber_display.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    return output_path


@software_page.route('/vae_generate', methods=['POST'])
def vae_generate():
    for i in os.listdir('VAE/output'):
        os.remove(os.path.join('VAE/output', i))
    data = request.get_json()
    text_value = data.get('textValue')
    print(text_value)
    generate_fig(text_value)
    image_path = os.path.join('VAE/output',os.listdir('VAE/output')[0])
    output_path = generate_cyber_display(image_path, "VAE GENERATION", "atom")
    webbrowser.open(f'file://{os.path.abspath(output_path)}')
    print("generated")
    return 'VAE Generate Figure Here'

@software_page.route('/dif_generate',methods=['POST'])
def dif_generate():
    for i in os.listdir('Diffusion/output'):
        os.remove(os.path.join('Diffusion/output', i))
    data = request.get_json()
    text_value = data.get('textValue')
    print(text_value)
    dif_generate1(text_value)
    # 展示图片
    image_path =os.path.join('Diffusion/output',os.listdir('Diffusion/output')[0])
    output_path = generate_cyber_display(image_path, "DIFFUSION GENERATION", "magic")
    webbrowser.open(f'file://{os.path.abspath(output_path)}')
    return 'Diffusion Generate completed'


if __name__ == '__main__':
    import threading
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000/')
    threading.Timer(1, open_browser).start()
    software_page.run(debug=True,use_reloader=False)
