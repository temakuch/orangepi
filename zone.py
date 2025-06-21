import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import threading as td
import open_file
from upscale import upscale_nn
from detection import detectVehicleCoords
import dlib
import math

tracker = dlib.correlation_tracker()
track_flag = False

cut_img = None
res = 0
old_res = 0
k = 0

startX, startY, endX, endY = 0, 0, 0, 0
lenX, lenY = 0, 0

# def callbacking(temp_res):
#     global res
#     if isinstance(temp_res, np.ndarray):
#         res = temp_res

# def show_result():
#     global res, old_res
#     while True:
#         if isinstance(res, np.ndarray):
#             if not np.array_equal(res, old_res):
#                 im = res[:, :, ::-1]
#                 cv.imshow('Upscaled', im)
#                 cv.waitKey(1)
#                 old_res = res

if __name__ == "__main__":
    filepath_window = open_file.FilePathWindow()
    if open_file.filename is not None:
        filepath_window.destroy()
    filepath_window.mainloop()

    video = cv.VideoCapture(0)

    frames = video.get(cv.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv.CAP_PROP_FPS)
    fr_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    fr_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    if not fps:
        fps = 30

    ret, first_frame = video.read()

    # Центр кадру та розміри зони
    center_w, center_h = fr_width // 2, fr_height // 2
    zone_width, zone_height = 200, 200
    zone_startX = center_w - zone_width // 2
    zone_startY = center_h - zone_height // 2
    zone_endX = center_w + zone_width // 2
    zone_endY = center_h + zone_height // 2

    cut_img = first_frame[zone_startY:zone_endY, zone_startX:zone_endX]

    # pool = multiprocessing.Pool(processes=2)
    # res = pool.apply_async(upscale_nn, args=(cut_img,), callback=callbacking)

    # t = td.Thread(target=show_result, daemon=True)
    # t.start()

    while True:
        ret, frame = video.read()
        if not ret:
            break
        clear_frame = frame.copy()
        cv.namedWindow('Frame')

        if track_flag:
            tracker.update(frame)
            pos = tracker.get_position()
            X1, Y1, X2, Y2 = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
            cv.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 2)

            dx, dy = (X1 - startX) - lenX, (Y1 - startY) - lenY
            startX += dx
            startY += dy
            endX += dx
            endY += dy

        # Приціл
        cv.line(frame, (center_w - 10, center_h), (center_w + 10, center_h), (0, 0, 255), 1)
        cv.line(frame, (center_w, center_h - 10), (center_w, center_h + 10), (0, 0, 255), 1)

        # Формування зони для наступного апскейлу
        if track_flag and (startX != endX and startY != endY):
            cut_img = clear_frame[startY:endY, startX:endX]
        else:
            cut_img = clear_frame[zone_startY:zone_endY, zone_startX:zone_endX]

        # Запуск нового апскейлу лише якщо попередній завершено
        # if not pool._cache:
        #     pool.apply_async(upscale_nn, args=(cut_img,), callback=callbacking)

        cv.imshow('Frame', frame)
        k = cv.waitKey(round(1000 / fps))

        if k == 27:
            break
        if k == ord("p"):
            cv.waitKey(-1)

        if k == ord(" "):
            zone_roi = clear_frame[zone_startY:zone_endY, zone_startX:zone_endX]
            x1, y1, x2, y2 = detectVehicleCoords(zone_roi)
            if (x1, y1, x2, y2) != (0, 0, 0, 0):
                abs_x1, abs_y1 = zone_startX + x1, zone_startY + y1
                abs_x2, abs_y2 = zone_startX + x2, zone_startY + y2
                box = dlib.rectangle(abs_x1, abs_y1, abs_x2, abs_y2)
                tracker.start_track(frame, box)
                track_flag = True
                startX, startY, endX, endY = abs_x1, abs_y1, abs_x2, abs_y2
                lenX, lenY = abs_x2 - abs_x1, abs_y2 - abs_y1

    cv.destroyAllWindows()
