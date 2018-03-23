import cv2
def main():
    vidcap = cv2.VideoCapture('sample1.MOV')
    success,image = vidcap.read()
    count = 1153
    success = True
    outF = open("Samples/bg.txt", "a")
    while success:
      success,image = vidcap.read()
      cv2.imwrite("Samples/NegSamples/frame%d.jpg" % count, image)     # save frame as JPEG file
      outF.write("NegSample/frame%d.jpg" % count)
      outF.write("\n")
      count += 1
    outF.close()
if __name__ == "__main__":
    main()
