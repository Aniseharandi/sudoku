import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def draw_chart():

    image = np.zeros((450, 450, 3), np.uint8)
    image[:] = (255, 255, 255)

    cv2.rectangle(image, (0, 0), (450, 450), (0, 0, 0), 2)

    g = int(450 / 9)
    # horizontal lines:
    for j in range(3):

        for i in range(2):

            cv2.line(image, (0, g), (450, g), (0, 0, 0), 1)
            g += 50
        
        cv2.line(image, (0, g), (450, g), (0, 0, 0), 2)
        g += 50


    g = int(450 / 9)
    # vertical lines:
    for j in range(3):

        for i in range(2):

            cv2.line(image, (g, 0), (g, 450), (0, 0, 0), 1)
            g += 50
        
        cv2.line(image, (g, 0), (g, 450), (0, 0, 0), 2)
        g += 50

    return image


def predict_initial_numbers(image_path, thresh_amount):

    net = load_model("cnn9k.h5")

    img = cv2.imread(image_path)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(cnts[0])
    crop = img[y:y+h, x:x+w]
    img = cv2.resize(crop, (450, 450))


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_amount, 255, cv2.THRESH_BINARY_INV)
    cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    coor = []
    for i in range(len(cnts)):

        if (hier[0][i][3] == 0):

            x, y, w, h = cv2.boundingRect(cnts[i])
            coor.append([x, y])
    

    labels = []
    num_coor = []
    for i in range(len(cnts)):
        
        if (hier[0][i][3] == 0) and (hier[0][i][2] != -1):

            x, y, w, h = cv2.boundingRect(cnts[i])
            num_coor.append([x, y])

            roi = img[y+5: y+h-5, x+5: x+w-5]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (32, 32)) / 255
            roi = np.array([roi])

            out = net.predict(roi)[0]
            max_index = np.argmax(out)

            label = max_index + 1
            labels.append(label)
            cv2.putText(image, str(label), (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,0), 2)


    return coor, num_coor, labels


def fix_contour_randomness(coor):

    ord_xy = []
    rx = []
    ry = []

    # Ordered coordination of each row
    for t in range(9):

        ymin = 500
        for f in range(len(coor)):

            if coor[f][1] < ymin:

                ymin = coor[f][1]


        for i in range(len(coor)):
            
            if coor[i][1] > (ymin - 5) and coor[i][1] < (ymin + 5):

                rx.append(coor[i][0])
                ry.append(coor[i][1])


        for k in range(len(rx)):

            mins = 500
            for item in rx:

                if item < mins:

                    mins = item
                
            dx = rx.index(mins)
        
            ord_xy.append([mins, ry[dx]])
            coor.remove([mins, ry[dx]])
            rx.remove(mins)
            ry.remove(ry[dx])

    return ord_xy


def create_sudoku_array(num_coor, labels, ord_xy):

    ar = np.zeros((9, 9), dtype= np.uint8)

    ar_coor = []
    for s in range(9):

        for t in range(9):

            ar_coor.append([s, t])


    # Putting initial numbers in created array
    c = 0
    for item in num_coor:

        i = ord_xy.index(item)
        rc = ar_coor[i]
        ar[rc[0]][rc[1]] = labels[c]
        c += 1
    
    return ar


def solver(ar):

    def box_nums(n, k):

        # (n = 0, 3, 6) and (k = 0, 3, 6) -- possible quantities, to go through all 9 boxes.
        box = []
        for row in range(n, n+3):

            for col in range(k, k+3):

                if ar[row][col] != 0:

                    box.append(ar[row][col])

        return box


    def row_nums(n):

        # n = 0,...,8
        row = []
        for col in range(9):

            if ar[n][col] != 0:

                row.append(ar[n][col])

        return row


    def column_nums(k):

        # k = 0,...,8
        column = []
        for row in range(9):

            if ar[row][k] != 0:

                column.append(ar[row][k])

        return column


    def box_check(n, k):

        c = 0
        d = {}
        to_li = []
        place = []

        for row in range(n, n+3):

            for col in range(k, k+3):

                if ar[row][col] == 0:

                    li = []
                    place.append([row, col])

                    for digit in range(1, 10):

                        if (digit not in box_nums(n, k)) and (digit not in row_nums(row)) and (digit not in column_nums(col)):

                            li.append(digit)
                            to_li.append(digit)


                    if len(li) == 1:

                        ar[row][col] = li[0]
                        cv2.putText(image, str(li[0]), (15+col*50, 35+row*50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,0), 2)

                        to_li.pop()
                        place.pop()

                    if len(li) != 1:
                        d["{}".format(c)] = li
                        c += 1

        return to_li, d, place


    def row_check(n):

        c = 0
        d = {}
        place = []
        to_li = []

        for k in range(9):

            if ar[n][k] == 0:

                lst = []
                place.append([n, k])

                for digit in range(1, 10):

                    if (digit not in row_nums(n)) and (digit not in column_nums(k)):
                        
                        lst.append(digit)
                        to_li.append(digit)


                if len(lst) == 1:

                    ar[n][k] = lst[0]
                    cv2.putText(image, str(lst[0]), (15+k*50, 35+n*50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,0), 2)

                    to_li.pop()
                    place.pop()


                d["{}".format(c)] = lst
                c += 1

        return to_li, d, place


    def column_check(k):

        c = 0
        d = {}
        place = []
        to_li = []

        for r in range(9):

            if ar[r][k] == 0:

                lst = []
                place.append([r, k])

                for digit in range(1, 10):

                    if (digit not in row_nums(r)) and (digit not in column_nums(k)):
                        
                        lst.append(digit)
                        to_li.append(digit)
                        

                if len(lst) == 1:

                    ar[r][k] = lst[0]
                    cv2.putText(image, str(lst[0]), (15+k*50, 35+r*50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,0), 2)

                    to_li.pop()
                    place.pop()

                d["{}".format(c)] = lst
                c += 1

        return to_li, d, place


    def appreared_once(to_li, d, place):

        A = []
        keys = []

        for a in to_li:

            if to_li.count(a) == 1:

                for key, val in d.items():

                    if a in val:

                        keys.append(key)
                        A.append(a)


        if len(keys) > 0:

            for i in range(len(keys)):

                r_c = place[int(keys[i])]
                ar[r_c[0]][r_c[1]] = A[i]

                cv2.putText(image, str(A[i]), (15 + r_c[1]*50, 35 + r_c[0]*50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,255,0), 2)

        return ar


    def box_result(f1, n, k, f2):

        # f1 = box_check
        # f2 = appeared_once
        for i in range(5):

            to_li, d, place = f1(n, k)

        ar = f2(to_li, d, place)

        return ar


    def row_col_result(f1, n, f2):

        # f1 = row_check or column_check
        # f2 = appeared_once
        for i in range(5):

            to_li, d, place = f1(n)

        ar = f2(to_li, d, place)

        return ar


    for i in range(10):

        ar = box_result(box_check, 0, 0, appreared_once)
        ar = box_result(box_check, 0, 3, appreared_once)
        ar = box_result(box_check, 0, 6, appreared_once)
        ar = box_result(box_check, 3, 0, appreared_once)
        ar = box_result(box_check, 3, 3, appreared_once)
        ar = box_result(box_check, 3, 6, appreared_once)
        ar = box_result(box_check, 6, 0, appreared_once)
        ar = box_result(box_check, 6, 3, appreared_once)
        ar = box_result(box_check, 6, 6, appreared_once)


    for i in range(5):

        for k in range(9):

            ar = row_col_result(row_check, k, appreared_once)
            ar = row_col_result(column_check, k, appreared_once)


image = draw_chart()

# The correct amount of "thresh_amount" plays a crucial role to solve the puzzle.
# If the input image has light gray lines: thresh_amount = 250 or 240
# If the input image has black lines: thresh_amount = 200 or 220
thresh_amount = 240
coor, num_coor, labels = predict_initial_numbers("zm12.png", thresh_amount)

ord_xy = fix_contour_randomness(coor)

ar = create_sudoku_array(num_coor, labels, ord_xy)

solver(ar)


cv2.imshow('sudoku', image)
cv2.waitKey()
cv2.destroyAllWindows()

