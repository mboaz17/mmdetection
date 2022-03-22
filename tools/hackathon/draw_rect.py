import cv2

img = cv2.imread('/home/dalya/PycharmProjects/mmdet/mmdetection/data/car_damage/train/64.jpg', cv2.COLOR_BGR2RGB)

start_point = (399, 235)
end_point = (399 + 117, 235 + 93)
cv2.rectangle(img, start_point, end_point, color=(255, 0, 0))
cv2.imwrite('/home/dalya/PycharmProjects/mmdet/mmdetection/data/64_annotated.jpg', img)
# cv2.imshow('Image', img)
aaa=1