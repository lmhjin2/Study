import tensorflow as tf
# print(tf.__version__)   # 1.14.0
# print(tf.executing_eagerly())   # False  // 즉시 실행모드

# 즉시 실행모드 -> 텐서1의 그래프형태의 구성 없이 자연스러운 파이썬 문법으로 실행시킨다.

# tf.enable_eager_execution()   # 구 버전용 코드
# tf.disable_eager_execution()  # 구 버전용 코드

print(tf.executing_eagerly())   # True  // 즉시 실행모드
# tf.compat.v1.enable_eager_execution() # 최신 버전용 코드
tf.compat.v1.disable_eager_execution()  # 최신 버전용 코드

print(tf.executing_eagerly())   # True  // 즉시 실행모드


# sess = tf.Session()   # 구 버전용 코드
sess = tf.compat.v1.Session()   # 최신 버전용 코드

hello = tf.constant('Hello world')
print(sess.run(hello))  # 코드6번줄 쓰면 에러. 없으면 잘돌아감

#   가상환경    즉시 실행모드      사용 가능
#   1.14.0     disable(default)     가능  ★★★★★
#   1.14.0     enable               에러
#   2.9.0      disable              가능  ★★★★★
#   2.9.0      enable(default)      에러

###### 정리
# 1.14.0에서는 그냥 하면됨
# 2.9.0 에서는 tf.compat.v1.disable_eager_execution() 을 써줘야 함
# tf.compat.v1.disable_eager_execution() 을 쓰면 어디서든 됨

sess.close()

'''
Tensor 1은 '그래프연산' 모드
Tensor 2는 '즉시 실행' 모드

tf.compat.v1.enable_eager_execution()   # 즉시실행모드 켜
                -> Tensor 2의 디폴트

tf.compat.v1.disable_eager_execution()  # 즉시실행모드 꺼
                                        -> 그래프 연산 모드로 돌아감
                                        -> Tensor 1 코드를 쓸 수 있음

tf.executing_eagerly()  # True면 즉시실행모드, -> Tensor 2 코드만 써야함
                          False면 그래프 연산모드 -> Tensor 1 코드를 쓸 수 있다



'''

