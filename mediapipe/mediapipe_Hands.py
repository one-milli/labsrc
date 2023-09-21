import os
import sys
import PySpin
import cv2
import mediapipe as mp


def initialize_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        raise ValueError("No camera found.")
    cam = cam_list.GetByIndex(0)
    cam.Init()
    nodemap = cam.GetNodeMap()
    # ここでカメラの設定を行う

    return cam, nodemap, system, cam_list


def main():
    result = True
    # カメラの初期化
    cam, nodemap, system, cam_list = initialize_camera()

    # result &= run_single_camera(cam)
    cam.BeginAcquisition()
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(
        PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Webカメラから入力
    # cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while True:
            image_result = cam.GetNextImage()
            image_converted = processor.Convert(
                image_result, PySpin.PixelFormat_BGR8)
            image = image_converted.GetNDArray()
            """ success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue """
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # 検出された手の骨格をカメラ画像に重ねて描画
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    # cap.release()

    # カメラの解放
    del cam
    # Clear camera list before releasing system
    cam_list.Clear()
    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result


if __name__ == "__main__":
    main()
