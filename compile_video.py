## Se necessário mudar versão do torch e torchvision
## torch==2.3.1 torchvision==0.18.1

from pathlib import Path
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

ROOT = Path(__file__).parent
ROOT_FILE = ROOT / "test_files/track_video_car01.mp4"

model = YOLO("YOLOweights\\yolov8x.pt")
cap = cv2.VideoCapture(str(ROOT_FILE))

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# classes que serão identificadas(COCO)
class_to_count = [2, 3, 7] #(2: carro, 3:motocicleta, 7:caminhão)
class_names = {2: "Car", 3: "Motorcycle", 7: "Truck"}

# linha tracker
line_points = [(20, 400), (1500, 400)]

video_writer = cv2.VideoWriter("compiled_files\\object_counting_output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))


# Iniciar Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(
    classes_names=model.names,
    reg_pts=line_points,
    view_img=True,
    draw_tracks=True,
    view_in_counts=False,
    view_out_counts=False,
)

# Dicionários para armazenar contagens por classe
counts = {
    'right': {cls_id: 0 for cls_id in class_to_count},
    'left': {cls_id: 0 for cls_id in class_to_count}
}

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    tracks = model.track(
        im0, 
        persist=True, 
        show=False, 
        classes=class_to_count
    )
    im0 = counter.start_counting(im0, tracks)

    # Guarda a quantidade de veiculos que passou na via
    counter_vehicles = counter.in_counts + counter.out_counts
    total_vehicles = counter_vehicles
    
    # Atualizar contagens por classe e direção (direita/esquerda)
    # Com base nos dados retornados pelo ObjectCounter
    for class_name, direction_counts in counter.class_wise_count.items():
        # class_name é o nome da classe (ex: 'car), 
        # direction_counts é um dicionário com 'IN' e 'OUT'

        for class_id, model_class_name in model.names.items():
            # Percorre as classes do modelo para mapear nome -> id

            if class_name == model_class_name:
                
                for direction, count in direction_counts.items():

                    if direction == 'IN':
                        counts['right'][class_id] = count
                    elif direction == 'OUT':
                        counts['left'][class_id] = count


    # Desenhar fundo semi-transparente para os textos
    overlay = im0.copy()
    cv2.rectangle(overlay, (10, 10), (500, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, im0, 0.4, 0, im0)

    # Adicionar Titulo
    cv2.putText(im0, "Traffic Counter", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #Adicionar contagens por classe
    y_offset = 90
    for cls_id in class_to_count:

        text = (f"{class_names[cls_id]} "
        f"( Right: {counts['right'][cls_id]} | "
        f"Left: {counts['left'][cls_id]} )")

        cv2.putText(im0, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset += 40 # Espaçamento entre linhas

    # Mostra na tela a quantidade de veiculos que passa na via
    total_vehicles_text = f"Total vehicles: {total_vehicles}"
    cv2.putText(im0, total_vehicles_text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    video_writer.write(im0)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
