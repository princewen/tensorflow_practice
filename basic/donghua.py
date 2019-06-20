from skimage import io
import matplotlib.pyplot as plt

num = 1


def show():
    global num


    while (num < 5):
        img = io.imread('pi1c/' + str(num) + '.jpeg')
        io.imshow(img)
        plt.ion()
        plt.pause(0.01)
        input("Press Enter to Continue")  # 之后改成识别输出
        num += 1



if __name__ == '__main__':
    show()
