import cv2
import numpy as np
import matplotlib.pyplot as plt

def flatten(xss):
    return [x for xs in xss for x in xs]

def data_real_image(image_file):
    imRGB=cv2.imread(image_file)

    # convert image to HSV
    imRGB = cv2.cvtColor(imRGB,cv2.COLOR_BGR2RGB)
    imHSV = cv2.cvtColor(imRGB, cv2.COLOR_RGB2HSV)
    output=imRGB.copy()
    # split into channels
    h, s, v = cv2.split(imHSV)

    # blue mask
    bMin = (0, 0, 120)
    bMax = (255, 100, 255)
    blueMask = cv2.inRange(imRGB, bMin, bMax)
    #plt.imshow(blueMask)
    #plt.show()
    # green mask
    gMin = (0, 100, 0)
    gMax = (255, 255, 100)
    greenMask = cv2.inRange(imRGB, gMin, gMax)
    #plt.imshow(greenMask)
    #plt.show()
    # value threshold to remove reflected light
    #for k in [50, 100, 150]:
    vThresh = 100
    ret, thresh = cv2.threshold(v, vThresh, 255, cv2.THRESH_BINARY)
    # combine all the masks
    greenValMask = cv2.bitwise_and(thresh, greenMask)
    plt.imshow(greenValMask)
        #plt.title(f'k={k}')
    plt.show()


    #cv2.imshow('Green', greenValMask)
    # The oversaturation at the centre of the LED forms a really nice circle.
    # Pick this up with a Hough transform instead of using regionprops
    data = []
    imCentre = [imRGB.shape[1] / 2., imRGB.shape[0] / 2.]

    ims = [greenMask, blueMask]
    colour = [1, 0]
    colourName = ['Green', 'Blue']
    # Label the output image with the circles' colour and distance from centre
    # Calculate
    for i in range(0, 2):
        circles = cv2.HoughCircles(ims[i], cv2.HOUGH_GRADIENT, 1.5, 200, param1=100, param2=10, minRadius=1, maxRadius=20)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:

                xC, yC = imCentre  # from the centre of the image
                diffX = x - xC  # to the centre of each connected region
                diffY = y - yC
                diffR = np.sqrt(diffX ** 2 + diffY ** 2)
                diffR = np.round(diffR, decimals=3)
                data.append([diffR, diffX, diffY, colour[i]])

                cv2.circle(output, (x, y), r, (0, 255, 0), 1)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 2)

                cv2.putText(output, f'C({x},{y}),{colourName[i]}', (x + 50, y - 50), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=2)
        else:
            print("Houston, you have a problem.")
    sortedList = sorted(data)
    #qprint(len(sortedList))
    if len(sortedList) >=3:
        top3 = sortedList[:3]
        feature_vector = flatten(top3)
        print(feature_vector)
        #print(len(feature_vector))
    else:
        feature_vector = False
        print("The sky is not full of stars")
    #cv2.putText(output, feature_vector, imCentre, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255, 0, 0), thickness=4)

    return feature_vector, output
#k = cv2.waitKey(1)

image_file='starfield_frame_0.png'
feature_vector, output = data_real_image(image_file)
plt.imshow(output)
plt.show()
#if k % 256 == 27:
    # ESC pressed
 #   print("Escape hit, closing...")
  #  cv2.destroyAllWindows()