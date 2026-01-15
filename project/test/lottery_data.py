import json
import pandas as pd
import time
from urllib.request import urlopen, Request
from datetime import datetime

def get_lotto_data(start_no, end_no):
    lotto_list = []
    
    # 브라우저처럼 보이기 위한 헤더 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }

    print(f"{start_no}회차부터 {end_no}회차까지 수집을 시작합니다...")

    for drwNo in range(start_no, end_no + 1):
        url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}"
        
        try:
            # 요청 생성 및 데이터 읽기
            req = Request(url, headers=headers)
            with urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))

            if data.get('returnValue') == 'success':
                row = {
                    '회차': data['drwNo'],
                    '추첨일': data['drwNoDate'],
                    '번호1': data['drwtNo1'],
                    '번호2': data['drwtNo2'],
                    '번호3': data['drwtNo3'],
                    '번호4': data['drwtNo4'],
                    '번호5': data['drwtNo5'],
                    '번호6': data['drwtNo6'],
                    '보너스': data['bnusNo'],
                    '1등당첨금': data['firstWinamnt'],
                    '1등당첨자수': data['firstPrzwnerCo']
                }
                lotto_list.append(row)
            else:
                print(f"\n{drwNo}회차 데이터를 찾을 수 없습니다.")

            # 진행 상황 표시
            if drwNo % 100 == 0 or drwNo == end_no:
                print(f">> {drwNo}회차 완료 (현재 {len(lotto_list)}개 수집됨)")

            # 서버 부하 방지 (매우 중요)
            time.sleep(0.1)

        except Exception as e:
            print(f"\n{drwNo}회차에서 에러 발생: {e}")
            continue

    return lotto_list

def save_to_csv(data_list):
    if not data_list:
        print("수집된 데이터가 없어 파일을 생성하지 않습니다.")
        return

    df = pd.DataFrame(data_list)
    filename = f"lotto_1_to_1206_{datetime.now().strftime('%H%M%S')}.csv"
    
    # 한글 깨짐 방지를 위해 utf-8-sig 사용
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n[성공] 파일이 저장되었습니다: {filename}")

if __name__ == "__main__":
    # 1회부터 1206회까지 지정
    start_rev = 1
    end_rev = 1206
    
    results = get_lotto_data(start_rev, end_rev)
    save_to_csv(results)