"""
## 1. GroundingDINO

# 이미지와 캡션을 입력으로 받아 GroundingDINO모델을 통해 물체 검출과 텍스트 라벨 정보를 추출하는 부분.
# 로짓과 박스를 필터링하고, 필터링된 결과로부터 텍스트 라벨을 생성해
# 최종적으로 박스와 텍스트 라벨을 반환함
"""
# 함수 정의와 초기 설정
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()  # 소문자
    caption = caption.strip()  # 앞뒤 공백 제거
    if not caption.endswith("."):
        caption = caption + "."  # 마침표로 끝나지 않으면, 마침표 추가
    
    # 모델을 이용한 예측 수행
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():  # 컨텍스트 내에서 예측을 수행해 로짓과 박스 생성
        outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0] # (nq, 256) / 확률값으로 변환
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]
    
    # filter output / 결과 필터링
    logit_filts = logits.clone() # 필터링을 위해 로짓 복사
    boxes_filts = boxes.clone() # 필터링을 위해 박스 복사
    filt_mask = logit_filts.max(dim=1)[0] > box_threshold  # box_threshold를 넘는지 판단하는 마스크
    logits_filt = logits_filt[filt_mask]  # num_filt, 256 / 필터링
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4 / 필터링
    logits_filt.shape[0]

    # get phrase / 텍스트 라벨 추출. 모델의 토크나이저로 캡션을 토크나이징
    tokenizer = model.tokenizer  
    tokenized = tokenizer(caption)
    
    # build pred / 예측된 프레이즈 생성
    # 필터링된 로짓과 박스를 순회하며 get_phrases_from_posmap 함수로 텍스트 라벨 추출
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if with_logits: # 참이면 최대값을 포함해 프레이즈를 리스트에 추가
            pred_phrases.append(pred_phrase + f" ({str(logit.max().item())[:4]})")
        else:           # 아니면 그냥 추가
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
    # 예측된 박스와 예측된 프레이즈를 반환


"""
## 2. Segment Anything
# GroundingDINO 모델에서 추출한 박스를 사용하여 Segment Anything 모델을 통해 segmentation을 수행하는 코드
# 이미지를 불러오고 박스를 이미지 크기에 맞게 조정한 후, SAM 모델로 segmentation을 수행하여 최종 마스크 생성
"""
# 초기 설정 및 이미지 로드
# SamPredictor 객체를 생성해서 SAM 모델 초기화.
# build_sam은 사전 학습된 모델 불러오는 부분.
predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
image = cv2.imread(image_path)  # 이미지 불러오기
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 에서 RGB로 변환
predictor.set_image(image)  # 변환된 이미지를 predictor에 설정?? 세팅??

# 이미지 크기 계산 및 박스 크기 조정
size = image_pil.size  # 이미지의 크기 가져오기
H, W = size[1], size[0]  # H와 W에 저장
for i in range(boxes_filt.size(0)):  
    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
    boxes_filt[i][2:] = boxes_filt[i][2:] / 2
    boxes_filt[i][2:] += boxes_filt[i][:2]
    # 박스 좌표값을 (x1, y1, x2, y2) 형식에서 (x, y, width, height) 형식으로 변환

# 박스 CPU로 이동 및 변환
boxes_filt = boxes_filt.cpu()
transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)  # 박스 좌표를 SAM모델에 맞게 변환 

# 세그멘테이션 수행
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes.to(device),
    multimask_output=False
)
# point_coords와 point_lables 를 None으로 설정해 박스 기반 세그멘테이션
# 그래서 얘네가 뭔데 ㅅㅂ
# multimask_output 을 False로 두어 하나의 마스크 출력을 반환함

"""
## 3. Mask 이미지 생성 코드
# 세그멘테이션된 결과를 마스크 이미지로 변환하는 과정
# 마스크 텐서를 합쳐 단일 마스크로 만들고, 이진화 한후, 넘파이 배열로 변환해 PIL 이미지 생성.
"""
if inpaint_mode == 'merge':
    masks = torch.sum(masks, dim=0).unsqueeze(0)  # masks 텐서를 합쳐 단일 마스크 생성.
    # dim=0 기준으로 합치고, 차원을 확장해 (1, H, W) 형태로 만듦
    masks = torch.where(masks > 0, True, False)  # 마스크 이진화. 마스크 값이 0보다 크면 True
    mask = masks[0][0].cpu().numpy()  # simply choose the first
    # 첫번째 마스크를 선택하고 CPU로 넘긴뒤, 넘파이 배열로 변환
    mask_pil = Image.fromarray(mask)
    # mask 넘파이 배열을 PIL이미지로 변환해 mask_pil로 저장
    image_pil = Image.fromarray(image)
    # image 넘파이 배열도 PIL 이미지로 변환해서 image_pil로 저장

"""
## 4. Stable Diffusion

# 마스크와 이미지를 입력으로 받아 Inpainting 수행 후 이미지를 저장
# Stable Diffusion에 넣기 위해 사이즈 조절 후, 원래 크기로 되돌림
"""

# 사전 학습된 가중치를 불러와 초기화
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, cache_dir=cache_dir
)
pipe = pipe.to("cuda")

# 이미지 및 마스크 크기 512x512로 조정 / Stable Diffusion의 input size로 맞추기 위함
image_pil = image_pil.resize((512, 512))
mask_pil = mask_pil.resize((512, 512))

# 마스크와 프롬프트 기반으로 Inpainting 수행
# prompt = "A sofa, high quality, detailed"
image = pipe(prompt=inpaint_prompt, image=image_pil, mask_image=mask_pil).images[0]
# pipe 호출해서 prompt, image_pil, mask_pil을 입력으로 받아 이미지 생성
# 생성된 결과를 첫번째 결과로 선택

image = image.resize(size)  # 이미지를 원래 크기로 재조정
image.save(os.path.join(output_dir, "grounded_sam_inpainting_output.jpg"))
# 지정된 위치에 이미지 저장

