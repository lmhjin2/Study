#########################################################################################################################################
# run_spacy_transformer_flask_langchain.py

from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import subprocess
from langchain_task import LangChain
import json
from PIL import Image
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['OUTPUT_FOLDER'] = 'static/outputs/'

def analyze_prompt(prompt):
    return LangChain().process_user_input(prompt)

def run_command(command):
    subprocess.run(command, shell=True, check=True, encoding="utf-8")

def generate_command(task, image_path, output_dir, bbox):
    if task['label'] == "날씨 변경":
        command = generate_weather_change_command(task, image_path, output_dir)
    elif task['label'] == "객체 제거":
        command = generate_object_removal_command(task, image_path, output_dir, bbox)
    return command

def generate_weather_change_command(task, image_path, output_dir):
    script = "grounded_sam_inpainting_2_demo_custom_mask.py"
    det_prompt = task['det_prompt']
    inpaint_prompt = task['inpainting_prompt']
    
    command = f"""
    CUDA_VISIBLE_DEVICES=0 python {script} \
    --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
    --sam_checkpoint Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
    --input_image {image_path} \
    --output_dir {output_dir} \
    --box_threshold 0.2 \
    --text_threshold 0.5 \
    --det_prompt "{det_prompt}" \
    --inpaint_prompt "{inpaint_prompt}" \
    --device "cuda"
    """
    return command

def generate_object_removal_command(task, image_path, output_dir, bbox):
    script = "grounded_sam_remove_select.py"
    det_prompt = task['det_prompt']
    inpaint_prompt = task['inpainting_prompt']
    
    command = f"""
    CUDA_VISIBLE_DEVICES=0 python {script} \
    --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
    --sam_checkpoint Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
    --input_image {image_path} \
    --output_dir {output_dir} \
    --box_threshold 0.2 \
    --text_threshold 0.5 \
    --det_prompt "{det_prompt}" \
    --inpaint_prompt "{inpaint_prompt}" \
    --device "cuda" \
    --bbox "{bbox}" 
    """
    return command

def main_workflow(prompt, image_path, bbox):
    output_dir = app.config['OUTPUT_FOLDER']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_pil = Image.open(image_path).convert("RGB")
    image_pil.save(os.path.join(output_dir, "inpainted_image.jpg"))

    tasks = analyze_prompt(prompt)
    tasks = json.loads(tasks)
    print(tasks)

    tasks['tasks'].sort(key=lambda x: x['label'] != '날씨 변경')

    image_paths = [image_path]

    for task in tasks['tasks']:
        command = generate_command(task, image_paths[-1], output_dir, bbox)
        run_command(command)
        if task['label'] == "날씨 변경":
           image_path = "inpainted_image.jpg"
        elif task['label'] == "객체 제거":
           image_path = "inpainted_image.png"

    return image_path, tasks['tasks']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    prompt = request.form['prompt']
    bbox = request.form.get('bbox', "0.0,0.0,0.0,0.0")
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return redirect(url_for('loading', filename=filename, prompt=prompt, bbox=bbox))

@app.route('/loading')
def loading():
    filename = request.args.get('filename')
    prompt = request.args.get('prompt')
    bbox = request.args.get('bbox')
    return render_template('loading.html', filename=filename, prompt=prompt, bbox=bbox)

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    filename = data['filename']
    prompt = data['prompt']
    bbox = data['bbox']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        output_image, tasks = main_workflow(prompt, file_path, bbox)
        return jsonify({'status': 'complete', 'output_image': output_image, 'tasks': tasks})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/result')
def result():
    filename = request.args.get('filename')
    prompt = request.args.get('prompt')
    output_image = request.args.get('output_image')
    tasks = json.loads(request.args.get('tasks'))
    return render_template('result.html', filename=filename, prompt=prompt, output_image=output_image, tasks=tasks)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    app.run(host='0.0.0.0', debug=True, port=8080)

#########################################################################################################################################
# langchain_task.py
import os
from dotenv import load_dotenv
import openai
import json

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

class LangChain:
    def __init__(self):
        self.examples = [
            {"input": "오늘 날씨가 좋다.", "tasks": []},
            {"input": "오늘 날씨가 좋고 사람들이 많네.", "tasks": []},
            {"input": "날씨가 왜이래? 사람들이 왜 이렇게 많아?.", "tasks": []},
            {"input": "나는 사과를 먹었다.", "tasks": []},
            {"input": "비가 오는 날씨로 바꿔야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "rainy sky"}]},
            {"input": "이 사진에서 자동차를 제거해야 한다.", "tasks": [{"label": "객체 제거", "det_prompt": "car", "inpainting_prompt": "remove car"}]},
            {"input": "천둥치는 날씨로 바꿔야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "thunder sky"}]},
            {"input": "맑은 날씨로 바꿔야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "clear sky"}]},
            {"input": "맑은 하늘로 변경해야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "clear sky"}]},
            {"input": "사람을 제거해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "여기 나온 사람들 제거해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "persons", "inpainting_prompt": "remove persons"}]},
            {"input": "물컵을 제거해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "cup", "inpainting_prompt": "remove cup"}]},
            {"input": "사람들을 제거 해주고 맑은 하늘로 변경해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}, {"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "clear sky"}]},
            {"input": "번개치는 하늘로 변경해주고 사람들을 제거 해줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a lightning sky"}, {"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "비오는 하늘로 변경해주고 사람들을 제거 해줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a rainy sky"}, {"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "지나가는 사람을 없애고, 맑은 하늘로 변경.", "tasks": [{"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}, {"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "clear sky"}]},
            {"input": "매우 맑고 화창한 하늘로 변경해주고, 사람들을 제거 해줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a clear sky"}, {"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]}
        ]

        self.prompt_template = """
        다음은 사용자 입력과 해당 작업을 JSON 형식으로 나타낸 예제입니다:
        예제:
        {examples}
        이제 사용자의 입력을 JSON 형식으로 처리하세요:
        사용자 입력: {input}
        JSON 형식으로 출력:
        """

    def process_user_input(self, user_input):
        examples = "\n".join([
            f'{{"input": "{ex["input"]}", "tasks": {json.dumps(ex["tasks"], ensure_ascii=False)}}}' 
            for ex in self.examples
        ])
        prompt_with_examples = self.prompt_template.format(examples=examples, input=user_input)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_with_examples}
            ],
            temperature=0.0,
            max_tokens=150,
            n=1,
            stop=None
        )
        
        prediction = response.choices[0].message.content.strip()

        try:
            parsed_result = json.loads(prediction)
        except json.JSONDecodeError:
            print("JSON 파싱에 실패했습니다.")
            parsed_result = {"error": "Invalid JSON format"}

        return json.dumps(parsed_result, ensure_ascii=False)

if __name__ == '__main__':
    lang = LangChain()
    result = lang.process_user_input("사람들을 제거해주고 맑은 하늘로 변경해줘.")
    print(result)

#########################################################################################################################################
