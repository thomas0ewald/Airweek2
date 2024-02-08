import numpy as np
from PIL import Image
import cv2
from cv_bridge import CvBridge


camera_data = np.load('camera_data.npy', allow_pickle=True)
scan_data = np.load('scan_data.npy', allow_pickle=True)


np.savetxt('asdf.txt', camera_data[1: 2], fmt='%s')


picture_encoded = camera_data[1: 2][0][1]

# picture_arr = np.array(picture_encoded, dtype=np.uint8)

cv_image = CvBridge.imgmsg_to_cv2(picture_encoded, "bgr8")

cv2.imwrite('oupuuuut.jpg', cv_image)




# print(picture_encoded[:10])

# num_pixels = len(picture_encoded) / 3
# print(num_pixels)


# np_data = np.array(picture_encoded, dtype=np.uint8)

# grouped_arrays = []
# for i in range(0, len(np_data), 3):
#     # Extrahieren von drei Elementen und Hinzufügen zu einem neuen Array
#     # Hinweis: np_array[i:i+3] schneidet das Array von Index i bis i+3 aus
#     new_array = np_data[i:i+3]
#     grouped_arrays.append(new_array)

# asdf = np.array(grouped_arrays, dtype=np.uint8).reshape(103968, 3)[2]
# print(asdf)


# Anzeigen der gruppierten Arrays
# for arr in grouped_arrays:
#     print(arr)


# np_data = np_data[:322 * 322 * 3]

# np_data = np_data.reshape((322, 322, 3))
# print(np_data)


# image = Image.fromarray(asdf, 'RGB')
# image.show()
# image.save('output_image.png')


# np.savetxt('picture.txt', picture_encoded, fmt='%s')
# print(picture_encoded[:10])


# 1. get r,g,b from array
# 2. combine r,g,b into single value
# 3. create image



# for i, element in enumerate(picture_encoded):
#     print(i, element)


# rgb_array = camera_data.reshape((height, width, 3))
# print(rgb_array)
# # Erstellen des Bildes aus dem NumPy-Array
# image = Image.fromarray(rgb_array, 'RGB')

# # Speichern des Bildes
# output_image_path = 'output_image.png'
# image.save(output_image_path)

# np_arr = np.array(np.round(picture_encoded))
# print(np_arr)







# print(picture_encoded('B'))
# print(picture_encoded('B'))

# picture_encoded = camera_data
# print(type(picture_encoded))
# print(picture_encoded.shape)

# # print(starter)
# example_image_array = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

# # new_arr = np.array(picture_encoded)
# # print(new_arr)

# image = Image.fromarray(example_image_array, 'RGB')

# # image = Image.fromarray(picture_encoded)
# image.show()
# image.save('output_image.png')


# count = 0

# for i in starter:
#     count += 1
#     print(i, end= '')
#     if count == 1000:
#         break

print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
# print(scan_data[:-1])