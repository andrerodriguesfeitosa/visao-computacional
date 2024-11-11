import cv2
import mediapipe as mp
import numpy as np
import pygame

# Inicializa o mixer de áudio
pygame.mixer.init()

# Carrega os arquivos de áudio
audio_file = pygame.mixer.Sound("wow_8.mp3")
audio_file2 = pygame.mixer.Sound("discord-notification.mp3")

# Carrega a imagem do chapéu com transparência
hat_image = cv2.imread("chapeu.png", -1)

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Funções de cálculo dos olhos e boca
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * np.linalg.norm(face_esq[4] - face_esq[5]))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * np.linalg.norm(face_dir[4] - face_dir[5]))
        return (ear_esq + ear_dir) / 2
    except:
        return 0.0

def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]
        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * np.linalg.norm(face_boca[6] - face_boca[7]))
        return mar
    except:
        return 0.0

# Definição dos limiares
ear_limiar = 0.17
mar_limiar = 0.4

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

som_tocado = False

# Inicializa os canais de som
channel_ear = pygame.mixer.Channel(0)
channel_mar = pygame.mixer.Channel(1)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Frame vazio da câmera ignorado.')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if resultado.multi_face_landmarks:
            for face_landmarks in resultado.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                )

                face = face_landmarks.landmark
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                mar = calculo_mar(face, p_boca)

                # Exibir EAR e MAR
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

                # Condição para tocar o som ao fechar os olhos
                if ear < ear_limiar:
                    if not channel_ear.get_busy():
                        channel_ear.play(audio_file2)
                else:
                    channel_ear.stop()

                # Condição para tocar o som ao abrir a boca
                if mar >= mar_limiar:
                    if not som_tocado:
                        channel_mar.play(audio_file)
                        som_tocado = True
                else:
                    #channel_mar.stop()
                    som_tocado = False

                # Calcula o ponto médio entre os olhos
                altura, largura = frame.shape[:2]
                olho_esq = int(face[33].x * largura), int(face[33].y * altura)
                olho_dir = int(face[263].x * largura), int(face[263].y * altura)
                ponto_medio_olhos = ((olho_esq[0] + olho_dir[0]) // 2, (olho_esq[1] + olho_dir[1]) // 2)

                # Calcular a largura entre os olhos para redimensionar o chapéu
                distancia_olhos = np.linalg.norm(np.array(olho_esq) - np.array(olho_dir))
                escala_chapeu = distancia_olhos / hat_image.shape[1] * 2.9  # Ajuste de escala
                chapeu_redimensionado = cv2.resize(hat_image, (int(hat_image.shape[1] * escala_chapeu), int(hat_image.shape[0] * escala_chapeu)))

                # Calcula a posição para sobrepor o chapéu
                pos_x = ponto_medio_olhos[0] - chapeu_redimensionado.shape[1] // 2
                pos_y = ponto_medio_olhos[1] - int(chapeu_redimensionado.shape[0] * 1.0)

                # Ajuste de sobreposição do chapéu
                y1, y2 = max(0, pos_y), min(altura, pos_y + chapeu_redimensionado.shape[0])
                x1, x2 = max(0, pos_x), min(largura, pos_x + chapeu_redimensionado.shape[1])
                chapeu_corte = chapeu_redimensionado[0:(y2 - y1), 0:(x2 - x1)]

                if chapeu_corte.shape[0] > 0 and chapeu_corte.shape[1] > 0:
                    alpha_chapeu = chapeu_corte[:, :, 3] / 255.0
                    for c in range(0, 3):
                        frame[y1:y2, x1:x2, c] = (1.0 - alpha_chapeu) * frame[y1:y2, x1:x2, c] + alpha_chapeu * chapeu_corte[:, :, c]

        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
