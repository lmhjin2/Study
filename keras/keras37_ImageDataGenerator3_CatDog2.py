import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import time as tm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 데이터 생성기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_test = 'c://_data//image//cat_and_dog//Test//'
path_train = 'c://_data//image//cat_and_dog//Train//'

xy_train_generator = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    shuffle=True
)

xy_test_generator = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary'
)

# 모델
model = Sequential()
model.add(Conv2D(2, (16, 16), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(2, (12, 12), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(1, (8, 8), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 조기 종료
es = EarlyStopping(monitor='val_loss', mode='auto', patience=50, verbose=1, restore_best_weights=True)

# 모델 훈련
start_time = tm.time()
history = model.fit(
    xy_train_generator,
    steps_per_epoch=len(xy_train_generator), # xy_train_generator의 배치 수
    epochs=300,
    batch_size = 100,
    validation_data=xy_train_generator,
    validation_steps=len(xy_train_generator), # xy_test_generator의 배치 수
    callbacks=[es]
)
end_time = tm.time()
training_time = np.round(end_time - start_time, 2)

# 모델 평가
loss, accuracy = model.evaluate(xy_test_generator)
print(f"손실: {loss}, 정확도: {accuracy}")
print(f"훈련 시간: {training_time} 초")

print(xy_train_generator)
