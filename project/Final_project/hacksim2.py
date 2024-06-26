#########################################################################################################################################
# 05.1 
def main_workflow(prompt, image_path, bbox):
    output_dir = app.config['OUTPUT_FOLDER'] 
    # app = flask / output 경로 설정
    if not os.path.exists(output_dir): # output_dir 이 존재하지 않으면
        os.makedirs(output_dir)        # output_dir 생성
        
    image_pil = Image.open(image_path).convert("RGB") # image_path 경로에 있는 이미지를 열고 RGB로 변환
    image_pil.save(os.path.join(output_dir, "inpainted_image.jpg")) # 변환된 이미지를 output_dir에 .jpg로 저장

    tasks = analyze_prompt(prompt) # 프롬프트 분석, tasks 생성
    tasks = json.loads(tasks)   # JSON 형식 문자열을 Python 데이터 구조로 변경
    print(tasks)
    
    tasks['tasks'].sort(key=lambda x: x['label'] != '날씨 변경') # tasks 리스트를 날씨 변경인 항목이 리스트의 앞으로 오게 정렬

    image_paths = [image_path] # image_paths 초기화, image_path 추가 
    
    for task in tasks['tasks']:
        command = generate_command(task, image_paths[-1], output_dir, bbox)
        # 날씨 변경인지 객체 제거인지 확인. 날씨는 bbox를 받지 않음
        run_command(command) 
        if task['label'] == "날씨 변경":
           image_path = "inpainted_image.jpg" # 날씨 변경은 jpg로 설정
        elif task['label'] == "객체 제거":
           image_path = "inpainted_image.png" # 객체 제거는 png로 설정

    return image_path, tasks['tasks'] 

@app.route('/upload', methods=['POST']) # 파일 업로드
# route는 Flask에서 특정 URL요청을 처리하기위해 사용됨
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400 
    file = request.files['file'] # request에서 파일을 불러옴
    prompt = request.form['prompt'] # request에서 프롬프트를 불러옴
    bbox = request.form.get('bbox', "0.0,0.0,0.0,0.0") # 박스를 불러옴, 옆은 기본값 좌표 
    if file.filename == '':
        return 'No selected file', 400
    if file: # 파일이 유효한지
        filename = file.filename # 파일이름 저장
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # 경로 설정
        file.save(file_path)
        
        return redirect(url_for('loading', filename=filename, prompt=prompt, bbox=bbox))

@app.route('/process', methods=['POST']) # 프로세스
def process():
    data = request.get_json() 
    filename = data['filename']
    prompt = data['prompt']
    bbox = data['bbox']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        output_image, tasks = main_workflow(prompt, file_path, bbox)
        return jsonify({'status': 'complete', 'output_image': output_image, 'tasks': tasks}) # JSON 형태로 반환
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

#########################################################################################################################################
# 05.2
# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
# FewShotPrompt : 예시 형태로 input, output 맞추기
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

#########################################################################################################################################
# 05.3
def analyze_prompt(prompt):
    return LangChain().process_user_input(prompt)

def run_command(command):
    subprocess.run(command, shell=True, check=True, encoding="utf-8")

def process_user_input(self, user_input):
    examples = "\n".join([
        f'{{"input": "{ex["input"]}", "tasks": {json.dumps(ex["tasks"], ensure_ascii=False)}}}' 
        for ex in self.examples
    ]) # 각 예제를 JSON형태로 변환 후, 하나의 문자열로 만듦

    prompt_with_examples = self.prompt_template.format(examples=examples, input=user_input)
    # 완성된 프롬프트 문자열 새로 저장

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_with_examples}
        ],
        temperature=0.0, # 모델의 창의성/무작위성 정도. 보통 0~1 
        max_tokens=150, 
        n=1, # 생성할 응답의 수
        stop=None # 특정 토큰을 생성하면 텍스트 생성 중지 / 기능 사용 안하는중
    )
    
