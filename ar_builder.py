import cv2
import mediapipe as mp
import math
import time
import os
import urllib.request


# =========================================================
# SETTINGS
# =========================================================

CAMERA_NUMBER = 0

DEFAULT_BLOCK_SIZE = 55
MIN_BLOCK_SIZE = 25
MAX_BLOCK_SIZE = 120

PINCH_DISTANCE = 0.055
GRAB_DISTANCE = 75

MODEL_FILE = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)


# =========================================================
# DOWNLOAD MODEL
# =========================================================

if not os.path.exists(MODEL_FILE):
    print("Downloading hand tracking model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_FILE
    )

    print("Model downloaded.")


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


landmarker = HandLandmarker.create_from_options(
    options
)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(CAMERA_NUMBER)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# =========================================================
# BLOCK DATA
# =========================================================

blocks = []

selected_block = None

pinch_active = False

block_size = DEFAULT_BLOCK_SIZE

grid_mode = False

last_create_time = 0

glow_phase = 0


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def distance_points(x1, y1, x2, y2):

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def snap_to_grid(value, grid=50):

    return round(value / grid) * grid


def find_nearest_block(x, y):

    nearest = None

    nearest_distance = 999999

    for i, block in enumerate(blocks):

        dist = distance_points(
            x,
            y,
            block["x"],
            block["y"]
        )

        if dist < GRAB_DISTANCE:

            if dist < nearest_distance:

                nearest = i

                nearest_distance = dist

    return nearest


# =========================================================
# DRAW GLOWING CUBE
# =========================================================

def draw_cube(
    image,
    x,
    y,
    size,
    selected=False,
    hovered=False
):

    global glow_phase

    glow_phase += 0.03

    glow = int(
        180 +
        math.sin(glow_phase) * 70
    )

    glow = max(100, min(255, glow))

    offset = int(size * 0.35)

    half = size // 2


    front_tl = (
        x - half,
        y - half
    )

    front_tr = (
        x + half,
        y - half
    )

    front_bl = (
        x - half,
        y + half
    )

    front_br = (
        x + half,
        y + half
    )


    back_tl = (
        x - half + offset,
        y - half - offset
    )

    back_tr = (
        x + half + offset,
        y - half - offset
    )

    back_bl = (
        x - half + offset,
        y + half - offset
    )

    back_br = (
        x + half + offset,
        y + half - offset
    )


    if selected:

        color = (0, 255, 255)

    elif hovered:

        color = (255, 255, 255)

    else:

        color = (255, glow, 0)


    overlay = image.copy()


    cv2.rectangle(
        overlay,
        front_tl,
        front_br,
        color,
        -1
    )


    cv2.addWeighted(
        overlay,
        0.12,
        image,
        0.88,
        0,
        image
    )


    thickness = 3 if selected else 2


    # front

    cv2.rectangle(
        image,
        front_tl,
        front_br,
        color,
        thickness
    )


    # back

    cv2.rectangle(
        image,
        back_tl,
        back_br,
        color,
        thickness
    )


    # connectors

    cv2.line(
        image,
        front_tl,
        back_tl,
        color,
        thickness
    )

    cv2.line(
        image,
        front_tr,
        back_tr,
        color,
        thickness
    )

    cv2.line(
        image,
        front_bl,
        back_bl,
        color,
        thickness
    )

    cv2.line(
        image,
        front_br,
        back_br,
        color,
        thickness
    )


    # center point

    cv2.circle(
        image,
        (x, y),
        4,
        color,
        -1
    )


# =========================================================
# DRAW GRID
# =========================================================

def draw_grid(frame):

    h, w, _ = frame.shape

    spacing = 50

    for x in range(
        0,
        w,
        spacing
    ):

        cv2.line(
            frame,
            (x, 0),
            (x, h),
            (40, 40, 40),
            1
        )


    for y in range(
        0,
        h,
        spacing
    ):

        cv2.line(
            frame,
            (0, y),
            (w, y),
            (40, 40, 40),
            1
        )


# =========================================================
# HUD
# =========================================================

def draw_hud(frame, width, height):

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (20, 15),
        (500, 160),
        (0, 0, 0),
        -1
    )


    cv2.addWeighted(
        overlay,
        0.40,
        frame,
        0.60,
        0,
        frame
    )


    cv2.putText(
        frame,
        "STARK AR BUILDER // MK-II",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 220, 0),
        2
    )


    cv2.putText(
        frame,
        f"BLOCKS: {len(blocks)}",
        (40, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"SIZE: {block_size}",
        (40, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    grid_text = (
        "GRID: ON"
        if grid_mode
        else
        "GRID: OFF"
    )


    cv2.putText(
        frame,
        grid_text,
        (40, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "PINCH = CREATE / GRAB",
        (width - 360, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "G = GRID",
        (width - 360, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "+ / - = BLOCK SIZE",
        (width - 360, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "D = DELETE | C = CLEAR | Q = QUIT",
        (width - 360, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


# =========================================================
# MAIN
# =========================================================

start_time = time.perf_counter()


while True:

    success, frame = camera.read()

    if not success:

        break


    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    if grid_mode:

        draw_grid(frame)


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp_ms = int(
        (
            time.perf_counter()
            - start_time
        )
        * 1000
    )


    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    hovered_block = None

    hand_detected = False


    # =====================================================
    # PROCESS HANDS
    # =====================================================

    if result.hand_landmarks:

        for hand_number, hand in enumerate(
            result.hand_landmarks
        ):

            hand_detected = True


            thumb = hand[4]

            index = hand[8]


            thumb_x = int(
                thumb.x * width
            )

            thumb_y = int(
                thumb.y * height
            )


            index_x = int(
                index.x * width
            )

            index_y = int(
                index.y * height
            )


            cursor_x = (
                thumb_x +
                index_x
            ) // 2


            cursor_y = (
                thumb_y +
                index_y
            ) // 2


            # =================================================
            # DRAW HAND
            # =================================================

            for connection in (
                mp.tasks.vision
                .HandLandmarksConnections
                .HAND_CONNECTIONS
            ):

                p1 = hand[
                    connection.start
                ]

                p2 = hand[
                    connection.end
                ]


                x1 = int(
                    p1.x * width
                )

                y1 = int(
                    p1.y * height
                )


                x2 = int(
                    p2.x * width
                )

                y2 = int(
                    p2.y * height
                )


                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 180, 0),
                    2
                )


            for landmark in hand:

                lx = int(
                    landmark.x * width
                )

                ly = int(
                    landmark.y * height
                )


                cv2.circle(
                    frame,
                    (lx, ly),
                    3,
                    (255, 255, 255),
                    -1
                )


            # =================================================
            # PINCH
            # =================================================

            pinch_distance = math.sqrt(

                (
                    thumb.x -
                    index.x
                ) ** 2

                +

                (
                    thumb.y -
                    index.y
                ) ** 2

            )


            nearest = find_nearest_block(
                cursor_x,
                cursor_y
            )


            if nearest is not None:

                hovered_block = nearest


            # cursor

            cv2.circle(
                frame,
                (
                    cursor_x,
                    cursor_y
                ),
                10,
                (255, 255, 0),
                2
            )


            cv2.line(
                frame,
                (
                    thumb_x,
                    thumb_y
                ),
                (
                    index_x,
                    index_y
                ),
                (255, 255, 0),
                2
            )


            # =================================================
            # PINCH ACTIVE
            # =================================================

            if pinch_distance < PINCH_DISTANCE:

                cv2.circle(
                    frame,
                    (
                        cursor_x,
                        cursor_y
                    ),
                    14,
                    (255, 255, 255),
                    2
                )


                # ---------------------------------------------
                # GRAB BLOCK
                # ---------------------------------------------

                if selected_block is None:

                    if nearest is not None:

                        selected_block = nearest


                    elif not pinch_active:

                        now = time.time()


                        if (
                            now -
                            last_create_time
                            > 0.25
                        ):

                            new_x = cursor_x

                            new_y = cursor_y


                            if grid_mode:

                                new_x = snap_to_grid(
                                    new_x
                                )

                                new_y = snap_to_grid(
                                    new_y
                                )


                            blocks.append(
                                {
                                    "x": new_x,
                                    "y": new_y,
                                    "size": block_size
                                }
                            )


                            selected_block = (
                                len(blocks) - 1
                            )


                            last_create_time = now


                # ---------------------------------------------
                # MOVE BLOCK
                # ---------------------------------------------

                if selected_block is not None:

                    if (
                        selected_block
                        < len(blocks)
                    ):

                        move_x = cursor_x

                        move_y = cursor_y


                        if grid_mode:

                            move_x = snap_to_grid(
                                move_x
                            )

                            move_y = snap_to_grid(
                                move_y
                            )


                        blocks[
                            selected_block
                        ]["x"] = move_x


                        blocks[
                            selected_block
                        ]["y"] = move_y


                pinch_active = True


            else:

                pinch_active = False

                selected_block = None


    else:

        pinch_active = False

        selected_block = None


    # =====================================================
    # DRAW BLOCKS
    # =====================================================

    for i, block in enumerate(blocks):

        selected = (
            i == selected_block
        )

        hovered = (
            i == hovered_block
        )


        draw_cube(
            frame,
            block["x"],
            block["y"],
            block["size"],
            selected,
            hovered
        )


    # =====================================================
    # PREVIEW
    # =====================================================

    if (
        hand_detected
        and
        not pinch_active
    ):

        pass


    # =====================================================
    # HUD
    # =====================================================

    draw_hud(
        frame,
        width,
        height
    )


    cv2.imshow(
        "Iron Man AR Builder MK-II",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # CONTROLS
    # =====================================================

    if key == ord("q"):

        break


    if key == ord("c"):

        blocks.clear()

        selected_block = None


    if key == ord("g"):

        grid_mode = not grid_mode


    if key == ord("+") or key == ord("="):

        block_size += 10

        block_size = min(
            block_size,
            MAX_BLOCK_SIZE
        )


    if key == ord("-"):

        block_size -= 10

        block_size = max(
            block_size,
            MIN_BLOCK_SIZE
        )


    if key == ord("d"):

        if hovered_block is not None:

            if (
                hovered_block
                < len(blocks)
            ):

                blocks.pop(
                    hovered_block
                )


# =========================================================
# CLEANUP
# =========================================================

camera.release()

landmarker.close()

cv2.destroyAllWindows()
