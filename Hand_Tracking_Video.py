import cv2
import mediapipe as mp

a = 3
nombre_video = "VID_test_01.mp4"
umbral = 0.7

# distancia entre dos puntos
def distancia(x1, y1, x2, y2):
    resultado = ((x2-x1)**2 + (y2-y1)**2) ** (1/2)
    return resultado

# Situacion del dedo
def dedo_status(dedo):
    if dedo >= umbral:
        return True
    elif dedo < umbral:
        return False

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

    visualizacion_cuadrados = int(width / 5)

    while cap.isOpened():
        ret, frame = cap.read()
        
        status_mano = []
        grosor = [2, 2, 2, 2]

        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame =cv2.resize(frame, (width, height))

            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar conexiones
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    #################################
                    base_palma_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width
                    base_palma_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height

                    indice_punta_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width
                    indice_punta_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height
                    indice_raiz_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width
                    indice_raiz_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height
                    
                    medio_punta_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width
                    medio_punta_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height
                    medio_raiz_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width
                    medio_raiz_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height
                    
                    anular_punta_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width
                    anular_punta_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height
                    anular_raiz_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * width
                    anular_raiz_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * height
                    
                    menique_punta_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width
                    menique_punta_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height
                    menique_raiz_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width
                    menique_raiz_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height
                    
                    indice = distancia(indice_punta_x,indice_punta_y,indice_raiz_x,indice_raiz_y)
                    referencia_indice = distancia(indice_raiz_x,indice_raiz_y,base_palma_x,base_palma_y)
                    indice = (indice/referencia_indice)
                    status_mano.append(dedo_status(indice))

                    medio = distancia(medio_punta_x,medio_punta_y,medio_raiz_x,medio_raiz_y)
                    referencia_medio = distancia(medio_raiz_x,medio_raiz_y,base_palma_x,base_palma_y)
                    medio = (medio/referencia_medio)
                    status_mano.append(dedo_status(medio))

                    anular = distancia(anular_punta_x,anular_punta_y,anular_raiz_x,anular_raiz_y)
                    referencia_anular = distancia(anular_raiz_x,anular_raiz_y,base_palma_x,base_palma_y)
                    anular = anular/referencia_anular
                    status_mano.append(dedo_status(anular))
                    
                    menique = distancia(menique_punta_x,menique_punta_y,menique_raiz_x,menique_raiz_y)
                    referencia_menique = distancia(menique_raiz_x,menique_raiz_y,base_palma_x,base_palma_y)
                    menique = menique/referencia_menique
                    status_mano.append(dedo_status(menique))

                    for (i,status_mano) in enumerate(status_mano):
                       if status_mano == True:
                           grosor[i] = -1

                    #################################

            # Visualizacion de cuadros
            cv2.rectangle(frame, (0, 0), (visualizacion_cuadrados, visualizacion_cuadrados), (0, 0, 225), grosor[0])
            cv2.rectangle(frame, (visualizacion_cuadrados, 0), (visualizacion_cuadrados*2, visualizacion_cuadrados), (0, 225, 0), grosor[1])
            cv2.rectangle(frame, (visualizacion_cuadrados*2, 0), (visualizacion_cuadrados*3, visualizacion_cuadrados), (0, 225, 255), grosor[2])
            cv2.rectangle(frame, (visualizacion_cuadrados*3, 0), (visualizacion_cuadrados*4, visualizacion_cuadrados), ( 225,0 , 0), grosor[3])
            
            cv2.imshow('Video', frame)

            # Si se presiona la tecla 'q', sal del bucle
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Libera los recursos y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
