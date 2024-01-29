import numpy as np

a = np.array(range(1,11))   # a = (1~10)
size = 5

def split_x(dataset, size):
    aaa = []            # aaa list만들기
    for i in range(len(dataset) - size + 1):        # dataset - 5 + 1
        subset = dataset[i : (i+size)]  # dataset에서 i 부터 i+5, 실질적으로는 i+4까지 subset 이라고 하기
        aaa.append(subset)          # subset에 담긴걸 aaa에 담기
      # aaa.append(dataset[i : (i+size)])  / # dataset에서 i 부터 i+5, 실질적으로는 i+4까지 aaa에 담기
    return np.array(aaa)   # 외주 맡긴거 돌려받기. 안주면 내가 못쓰자너. 쿠팡맨.

bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)  # (6, 5)

x = bbb[ : , : -1 ]  # "bbb" 배열에서 마지막 열을 제외한 모든 열을 가져와 "x"라는 이름의 새 배열에 저장.
    # 첫번째 : 은 행. 비어있으니 모든행 선택. / 두번째 : 은 열. : -1 이니까 마지막 하나 빼고 다.
y = bbb[ : , -1 ]   # "bbb" 배열에서 마지막 열만 가져와 "y"라는 이름의 새 배열에 저장.
    # 첫번째 :은 행. 모든행/ 두번째 -1 은 인덱스 번호 [-1]. 마지막 하나만 갖고오기
# print(x, y)
# print(x.shape, y.shape)  # (6, 4) (6,)
# =============================================================================================================================

    # print(bbb)
    # [[ 1  2  3  4  5]
    #  [ 2  3  4  5  6]
    #  [ 3  4  5  6  7]
    #  [ 4  5  6  7  8]
    #  [ 5  6  7  8  9]
    #  [ 6  7  8  9 10]]
    # print(bbb.shape)    # (6, 5)

    # print(x, y)
    #      x                  y
    # [[1 2 3 4]    # [ 5  6  7  8  9 10]
    #  [2 3 4 5]
    #  [3 4 5 6]
    #  [4 5 6 7]
    #  [5 6 7 8]
    #  [6 7 8 9]]

    # print(x.shape, y.shape)  # (6, 4) (6,)