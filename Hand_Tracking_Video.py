import cv2
import mediapipe as mp

a = 3
nombre_video = "VID_test_01.mp4"

# distancia entre dos puntos
def distancia(x1, y1, x2, y2):
    resultado = ((x2-x1)**2 + (y2-y1)**2) ** (1/2)
    return resultado

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    video_file = nombre_video
    cap = cv2.VideoCapture(video_file)

    # Configuracion ventana
    width = int(1080/a)
    height = int(1920/a)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame =cv2.resize(frame, (width, height))

            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar conecciones
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    #################################
                    base_palma_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width
                    base_palma_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height

                    indice_punta_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width
                    indice_punta_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height
                    indice_raiz_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width
                    indice_raiz_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height
                    
                    indice = distancia(indice_punta_x,indice_punta_y,indice_raiz_x,indice_raiz_y)
                    referencia = distancia(indice_raiz_x,indice_raiz_y,base_palma_x,base_palma_y)
                    
                    print(indice / referencia)
                    #################################

            cv2.imshow('Video', frame)

            # Si se presiona la tecla 'q', sal del bucle
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
