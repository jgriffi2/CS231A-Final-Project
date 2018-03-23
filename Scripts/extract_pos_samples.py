import cv2
def main():
    vidcap = cv2.VideoCapture('posSamplesTable.MOV')
    success,image = vidcap.read()
    count = 0
    index = 0
    success = True
    while success:
      success,image = vidcap.read()
      if (index % 100 == 0):
          cv2.imwrite("Samples/PosSamples/frame%d.jpg" % count, image)     # save frame as JPEG file
          count += 1
      index += 1
if __name__ == "__main__":
    main()
