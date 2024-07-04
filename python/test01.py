# a=['life','is','short']
# b="_".join(a)
# print(b)

########################################################################

# for i in range(2,10):
#     for j in range(1,10):
#         print(i*j, end=" ")
#     print('')

########################################################################

# max_width = max(len(str(9*i)) for i in range(1, 10))

# for i in range(2, 10):
#     line = " ".join(f"{i*j:^{max_width}}" for j in range(1, 10))
#     print(" " + line)  # 각 줄의 시작에 공백 추가

########################################################################

# # 각 열의 최대 너비 계산
# max_width = max(len(str(9*i)) for i in range(1, 10))

# for i in range(2, 10):
#     for j in range(1, 10):
#         # 각 결과를 가운데 정렬하여 출력
#         print(f"{i*j:^{max_width}}", end=" ")
#     print()  # 각 줄 끝에서 줄바꿈

########################################################################

# 각 열의 최대 너비 계산 ★★★★★
max_width = max(len(str(9*i)) for i in range(1, 10))
print(max_width)
for i in range(2, 10):
    for j in range(1, 10):
        # 각 결과를 오른쪽 정렬하여 출력
        print(f"{i*j:>{max_width}}", end=" ")
    print()  # 각 줄 끝에서 줄바꿈

