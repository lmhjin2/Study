import requests
import csv
import time

START_ROUND = 1
END_ROUND = 1206

API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"

def get_lotto_numbers(draw_no):
    try:
        res = requests.get(API_URL.format(draw_no), timeout=5)
        res.raise_for_status()

        # JSON 파싱
        data = res.json()

        # 정상 데이터가 아닐 경우 스킵
        if data.get("returnValue") != "success":
            print(f"{draw_no}회차 데이터 없음")
            return None

        print(f"{draw_no}회차 수집 완료")

        return [
            data["drwNo"],
            data["drwNoDate"],
            data["drwtNo1"],
            data["drwtNo2"],
            data["drwtNo3"],
            data["drwtNo4"],
            data["drwtNo5"],
            data["drwtNo6"],
            data["bnusNo"],
        ]

    except Exception as e:
        print(f"{draw_no}회차 오류: {e}")
        return None


with open("lotto_1_to_1206.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["회차", "추첨일", "번호1", "번호2", "번호3", "번호4", "번호5", "번호6", "보너스"]
    )

    for round_no in range(START_ROUND, END_ROUND + 1):
        row = get_lotto_numbers(round_no)
        if row:
            writer.writerow(row)
        time.sleep(0.05)  # 서버 부담 방지


print("CSV 생성 완료")
