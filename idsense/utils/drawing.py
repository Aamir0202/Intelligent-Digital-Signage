import cv2 as cv

FONT_SCALE = 0.6
FONT_THICKNESS = 1
TEXT_COLOR = (255, 64, 0)
BOX_RADIUS = 8
BOX_COLOR = (0, 191, 255)
BORDER_THICKNESS = 1


def draw_text(frame, text, position, color=TEXT_COLOR):
    """Draws text on the given frame at the specified position."""

    return cv.putText(
        frame,
        text,
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
        cv.LINE_AA,
    )


def draw_rounded_rectangle(
    frame,
    top_left,
    bottom_right,
    radius=BOX_RADIUS,
    color=BOX_COLOR,
    thickness=BORDER_THICKNESS,
):
    """Draws a rounded rectangle on the given frame."""

    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])

    if thickness < 0:
        top_left_main_rect = (int(p1[0] + radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + radius)
        bottom_right_rect_left = (p4[0] + radius, p4[1] - radius)

        top_left_rect_right = (p2[0] - radius, p2[1] + radius)
        bottom_right_rect_right = (p3[0], p3[1] - radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right],
        ]

        [cv.rectangle(frame, rect[0], rect[1], color, thickness) for rect in all_rects]

    cv.line(
        frame,
        (p1[0] + radius, p1[1]),
        (p2[0] - radius, p2[1]),
        color,
        abs(thickness),
        cv.LINE_AA,
    )
    cv.line(
        frame,
        (p2[0], p2[1] + radius),
        (p3[0], p3[1] - radius),
        color,
        abs(thickness),
        cv.LINE_AA,
    )
    cv.line(
        frame,
        (p3[0] - radius, p4[1]),
        (p4[0] + radius, p3[1]),
        color,
        abs(thickness),
        cv.LINE_AA,
    )
    cv.line(
        frame,
        (p4[0], p4[1] - radius),
        (p1[0], p1[1] + radius),
        color,
        abs(thickness),
        cv.LINE_AA,
    )

    cv.ellipse(
        frame,
        (p1[0] + radius, p1[1] + radius),
        (radius, radius),
        180.0,
        0,
        90,
        color,
        thickness,
        cv.LINE_AA,
    )
    cv.ellipse(
        frame,
        (p2[0] - radius, p2[1] + radius),
        (radius, radius),
        270.0,
        0,
        90,
        color,
        thickness,
        cv.LINE_AA,
    )
    cv.ellipse(
        frame,
        (p3[0] - radius, p3[1] - radius),
        (radius, radius),
        0.0,
        0,
        90,
        color,
        thickness,
        cv.LINE_AA,
    )
    cv.ellipse(
        frame,
        (p4[0] + radius, p4[1] - radius),
        (radius, radius),
        90.0,
        0,
        90,
        color,
        thickness,
        cv.LINE_AA,
    )

    return frame
