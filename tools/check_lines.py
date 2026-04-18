import argparse
import sys
from pathlib import Path

import cv2


def inspect_line(video_path: Path) -> int:
    """
    Opens a video, captures its first frame, and initializes an interactive UI
    to display current line coordinates and capture new ones via mouse clicks.

    Args:
        video_path (Path): Path to the target video file.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    if not video_path.exists():
        print(f"Erro: O ficheiro {video_path} não foi encontrado.")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()

    # We only need the first frame for static calibration. Release hardware immeadiately
    cap.release()

    if not success:
        print("Erro: Não foi possível ler o primeiro frame do vídeo.")
        return 1

    # Hardcoded reference line to visualize the default configuration
    current_line = [(20, 400), (1500, 400)]

    cv2.line(frame, current_line[0], current_line[1], (0, 0, 255), 2)
    cv2.putText(
        frame,
        "Linha Atual",
        (current_line[0][0], current_line[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    def mouse_click(
            event: int,
            x: int,
            y: int,
            flags: int,
            param: object) -> None:
        """
        OpenCV mouse callback event handler.
        Note: 'flags' and 'param' are unused but required by
        the cv2.setMouseCallback signature.
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"[NOVA COORDENADA] X: {x} | Y: {y}")

            # Draw a green dot ai the clicked location for visual feedback
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Calibrador de Linha", frame)

    cv2.imshow("Calibrador de Linha", frame)
    cv2.setMouseCallback("Calibrador de Linha", mouse_click)

    print("=" * 40)
    print("MODO DE CALIBRAÇÃO ATIVO")
    print("1. Veja onde está a linha vermelha.")
    print("2. Clique no ecrã para descobrir novas coordenadas.")
    print("3. Pressione 'q' ou 'ESC' na janela do vídeo para fechar.")
    print("=" * 40)

    # UI render loop
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Break loop if 'q' (113) or 'ESC'(27) is pressed
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    return 0

def main() -> int:
    """
    Parses command-line arguments and triggers the calibration tool.

    Returns:
        int: System exit code.
    """
    parser = argparse.ArgumentParser(description="Verificador de Linha de Contagem")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Caminho para o vídeo"
    )
    args = parser.parse_args()

    return inspect_line(args.video)


if __name__ == "__main__":
    sys.exit(main())
