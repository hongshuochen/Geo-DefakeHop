import os
import cv2

if __name__ == '__main__':
    cities = ["Beijing", "Seattle", "Tacoma"]
    for city in cities:
        images = os.listdir("anti-deepfake-data and code/data/authentic/" + city)
        os.makedirs("anti-deepfake-data and code/data/authentic/" + city + "_128", exist_ok=True)
        for img in images:
            img_path = "anti-deepfake-data and code/data/authentic/"+ city+ "/" + img
            print(img_path)
            img = cv2.imread(img_path)
            img_a = img[0:128,0:128]
            cv2.imwrite(img_path.replace(city, city + "_128").replace(".png", "_0.png"), img_a)
            img_b = img[0:128,128:]
            cv2.imwrite(img_path.replace(city, city + "_128").replace(".png", "_1.png"), img_b)
            img_c = img[128:,0:128]
            cv2.imwrite(img_path.replace(city, city + "_128").replace(".png", "_2.png"), img_c)
            img_d = img[128:,128:]
            cv2.imwrite(img_path.replace(city, city + "_128").replace(".png", "_3.png"), img_d)
